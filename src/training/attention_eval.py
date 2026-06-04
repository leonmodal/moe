"""Attention-pattern evaluation against a known ground-truth attention map.

Designed for the synthetic linear-map / cellular-automata tasks where the
"correct" attention pattern is given by construction:

  * Linear map: ground-truth A is an S x S binary matrix. To predict token
    at position S+i (the i-th output cell of x_1), the model must attend to
    the input positions {j : A[i, j] == 1}. We construct a per-position
    target distribution over the seq_len-token context and compare to the
    model's learned softmax pattern.

  * Cellular automata: ground-truth pattern is a local 3-window on the
    previous state. For position t with t mod S = i, the targets are the
    three input positions {prev_state_offset + (i-1) % S,
    prev_state_offset + i, prev_state_offset + (i+1) % S}.

For each depth we compute two views of the model's effective attention:

  * `agg`: routing-weighted aggregate. For each (k_expert, v_expert) pair
    that fired this batch we multiply its per-token attention map by the
    `group_mask` (1 if this token used this pair) and the qk routing weight,
    sum across pairs, and average across tokens routed to the same output
    position. The result approximates "what the model effectively attended
    to" for each query position.

  * `best`: per-expert best-match KL. For each ground-truth row r we find
    the expert whose mean attention map row most closely matches r (lowest
    KL), and report the mean best-KL. Measures whether *some* head learned
    the pattern, even if routing hasn't yet aligned to surface it.

Metrics logged per depth (under `attn_eval/depth_{d}/...`):
  * `kl_agg_mean`, `kl_best_mean`: average KL across query positions.
  * `iou_agg_mean`: mean IoU between top-`nnz` aggregate-attention positions
    and the ground-truth nonzero positions.
  * `entropy_agg_mean`: mean entropy of the aggregate attention.

Also writes a per-depth heatmap PNG comparing `agg`, `best`, and ground-truth
to `<output_dir>/attn_eval/step_<step>/depth_<d>.png` at checkpoint cadence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _build_linear_map_targets(
    ground_truth_A: np.ndarray, seq_len: int
) -> np.ndarray:
    """Return [seq_len, seq_len] binary target matrix for linear map.

    For positions 0..S-1 (the input state x_0) there is no learnable target,
    so we leave those rows as zeros and the eval code skips them (`row_mask`).
    For positions S..2S-1 (predicting cells of x_1), the row equals A[i].
    """
    S = ground_truth_A.shape[0]
    target = np.zeros((seq_len, seq_len), dtype=np.float32)
    for t in range(S, seq_len):
        i = t - S  # which cell of x_1 we're predicting
        # The query at position t-1 produces logits for position t under
        # next-token prediction. So the attention pattern of interest is at
        # query position t-1 (which writes the cell). The keys range over
        # positions 0..S-1 (the input state).
        if i >= S:
            continue
        # The model can attend at query position t-1 over keys [0..S-1]
        # (and itself at t-1, but the linear-map task only needs positions
        # in x_0). The relevant nonzeros are A[i, j] for j in [0..S-1].
        target[t - 1, :S] = ground_truth_A[i]
    return target


def _build_cellular_automata_targets(
    window_offsets: np.ndarray, seq_len: int, state_size: int
) -> np.ndarray:
    """Return [seq_len, seq_len] binary target matrix for cellular automata.

    For position t = s * S + i (predicting cell i of state s, given s >= 1),
    the relevant keys are at positions (s-1) * S + ((i + off) mod S) for
    `off` in `window_offsets`. The query writing this prediction is at
    position t-1.
    """
    S = state_size
    target = np.zeros((seq_len, seq_len), dtype=np.float32)
    T = seq_len // S
    for s in range(1, T):
        prev_offset = (s - 1) * S
        for i in range(S):
            t = s * S + i
            query = t - 1
            for off in window_offsets:
                key = prev_offset + ((i + int(off)) % S)
                target[query, key] = 1.0
    return target


def build_ground_truth_targets(dataset) -> tuple[np.ndarray, np.ndarray] | None:
    """Construct (target_matrix, row_mask) from a synthetic dataset.

    Returns `None` if the dataset has no ground-truth attention pattern.
    The row mask is 1 for query positions where any target row is set
    (i.e., positions for which the eval has a meaningful comparison).
    """
    A = getattr(dataset, "ground_truth_A", None)
    if A is None:
        return None
    cfg = getattr(dataset, "config", None)
    if cfg is None:
        return None
    seq_len = int(getattr(dataset, "seq_len", 0))
    if seq_len <= 0:
        return None
    if cfg.task == "linear_map":
        target = _build_linear_map_targets(A, seq_len)
    elif cfg.task == "cellular_automata":
        target = _build_cellular_automata_targets(A, seq_len, cfg.state_size)
    else:
        return None
    row_mask = target.sum(axis=-1) > 0
    return target, row_mask


def _aggregate_attention(
    depth_maps: list[dict],
    qk_w: torch.Tensor | None,
    num_kv_groups: int,
) -> torch.Tensor:
    """Average per-(k_expert, v_expert) maps weighted by routing.

    Returns a tensor of shape `[B, num_heads, T, T]` representing the
    routing-weighted effective attention pattern for this depth. If `qk_w`
    is None (no router weights captured), falls back to unweighted average
    using only the group masks.
    """
    if not depth_maps:
        return None
    # Sum (weights * head_mask * attn_weights) across experts. Use a counter
    # for normalization (some query positions might not be routed to any
    # captured pair; their aggregate stays zero, which we treat as missing).
    total = None
    norm = None
    for entry in depth_maps:
        weights = entry["weights"].float()  # [B, num_heads, T, T]
        group_mask = entry["group_mask"].float()  # [B, num_kv_heads, T]
        head_mask = group_mask.unsqueeze(-1).repeat_interleave(num_kv_groups, dim=1)
        head_mask_qk = head_mask.squeeze(-1).unsqueeze(-1)  # [B, num_heads, T, 1]
        if qk_w is not None:
            # qk_w is [B*T, num_kv_heads] or similar; reshape to [B, num_heads, T, 1].
            qk_w_local = qk_w.float()
            # Best-effort reshape: if qk_w is per-(token, kv_head), expand to heads.
            if qk_w_local.dim() == 2:
                B = weights.shape[0]
                T_ = weights.shape[2]
                qk_w_local = (
                    qk_w_local.view(B, T_, -1)
                    .permute(0, 2, 1)
                    .unsqueeze(-1)
                    .repeat_interleave(num_kv_groups, dim=1)
                )
            contribution = weights * head_mask_qk * qk_w_local
            mass = head_mask_qk * qk_w_local
        else:
            contribution = weights * head_mask_qk
            mass = head_mask_qk
        if total is None:
            total = contribution
            norm = mass
        else:
            total = total + contribution
            norm = norm + mass
    # Normalize per (B, head, query) so the aggregate is a proper distribution
    # for any query routed to any captured expert.
    eps = 1e-9
    return total / (norm + eps)


def _safe_kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with eps smoothing to avoid divide-by-zero."""
    p = p + 1e-9
    q = q + 1e-9
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())


def _best_match_kl_per_row(
    depth_maps: list[dict],
    targets: np.ndarray,
    row_mask: np.ndarray,
    num_kv_groups: int,
) -> float:
    """Mean over learnable rows of min-over-experts KL(expert_row || target_row)."""
    if not depth_maps:
        return float("nan")
    # Build per-expert averaged attention maps across the batch and heads
    # (mean over the positions routed to that expert; falls back to overall
    # mean if no tokens routed to it).
    expert_rows: list[np.ndarray] = []  # each [T, T]
    for entry in depth_maps:
        weights = entry["weights"].numpy()  # [B, num_heads, T, T]
        group_mask = entry["group_mask"].numpy()  # [B, num_kv_heads, T]
        head_mask = np.repeat(group_mask, num_kv_groups, axis=1)  # [B, num_heads, T]
        mass = head_mask.sum(axis=(0, 1))  # [T]
        # Weighted mean over routed (batch, head, token=query position):
        weighted = weights * head_mask[..., None]
        summed = weighted.sum(axis=(0, 1))  # [T, T]
        denom = mass[:, None] + 1e-9
        expert_rows.append(summed / denom)
    expert_rows_arr = np.stack(expert_rows, axis=0)  # [num_pairs, T, T]
    learnable_idx = np.flatnonzero(row_mask)
    kls = []
    for r in learnable_idx:
        target_row = targets[r]
        # min KL across experts at row r
        per_expert = [
            _safe_kl(expert_rows_arr[e, r], target_row)
            for e in range(expert_rows_arr.shape[0])
        ]
        if per_expert:
            kls.append(min(per_expert))
    if not kls:
        return float("nan")
    return float(np.mean(kls))


def _iou_topk(agg_rows: np.ndarray, target_rows: np.ndarray, row_mask: np.ndarray) -> float:
    """Mean IoU between top-`nnz_per_row` agg positions and target nonzeros."""
    learnable_idx = np.flatnonzero(row_mask)
    ious = []
    for r in learnable_idx:
        target = target_rows[r]
        nnz = int(target.sum())
        if nnz <= 0:
            continue
        top_idx = set(np.argsort(-agg_rows[r])[:nnz].tolist())
        target_idx = set(np.flatnonzero(target).tolist())
        union = top_idx | target_idx
        intersect = top_idx & target_idx
        if not union:
            continue
        ious.append(len(intersect) / len(union))
    if not ious:
        return float("nan")
    return float(np.mean(ious))


def _entropy(agg_rows: np.ndarray, row_mask: np.ndarray) -> float:
    learnable_idx = np.flatnonzero(row_mask)
    if learnable_idx.size == 0:
        return float("nan")
    p = agg_rows[learnable_idx]
    p = p + 1e-9
    p = p / p.sum(axis=-1, keepdims=True)
    return float(-(p * np.log(p)).sum(axis=-1).mean())


def _maybe_save_heatmaps(
    output_dir: str | None,
    step: int,
    *,
    targets: np.ndarray,
    agg_rows_per_depth: list[np.ndarray],
    best_rows_per_depth: list[np.ndarray],
    row_mask: np.ndarray,
) -> None:
    if output_dir is None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    eval_dir = Path(output_dir) / "attn_eval" / f"step_{step:08d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    learnable_idx = np.flatnonzero(row_mask)
    if learnable_idx.size == 0:
        return
    truncated_target = targets[learnable_idx][:, : targets.shape[1]]

    for d, (agg_rows, best_rows) in enumerate(zip(agg_rows_per_depth, best_rows_per_depth)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, mat, title in zip(
            axes,
            [agg_rows[learnable_idx], best_rows[learnable_idx], truncated_target],
            ["agg (routing-weighted)", "best-match per row", "ground truth"],
        ):
            ax.imshow(mat, aspect="auto", cmap="magma")
            ax.set_title(title)
            ax.set_xlabel("key position")
            ax.set_ylabel("query position (learnable)")
        fig.suptitle(f"depth {d} attention vs ground truth (step {step})")
        fig.tight_layout()
        fig.savefig(eval_dir / f"depth_{d:02d}.png", dpi=120)
        plt.close(fig)


def evaluate_attention_against_ground_truth(
    *,
    model: torch.nn.Module,
    eval_batch: dict[str, torch.Tensor],
    targets: np.ndarray,
    row_mask: np.ndarray,
    device: torch.device,
    output_dir: str | None,
    step: int,
    save_heatmaps: bool,
) -> dict[str, float]:
    """Run one forward with attention capture and compute per-depth metrics.

    Returns a flat dict of metric names -> floats, prefixed with `attn_eval/`.
    """
    from .distributed import unwrap_model

    raw_model = unwrap_model(model)
    inner = getattr(raw_model, "model", raw_model)
    # Per-layer expert bank: every depth has its own AttentionExpertBank.
    # Toggle the capture flag on each one so the model's per-depth forward
    # captures regardless of which bank instance runs at that depth.
    banks_list = getattr(inner, "attn_banks_per_depth", None)
    if banks_list is not None:
        attn_banks = list(banks_list)
    else:
        single = getattr(inner, "attn_bank", None)
        attn_banks = [single] if single is not None else []
    if not attn_banks:
        return {}

    was_training = model.training
    model.eval()
    prev_flags = [getattr(b, "capture_attention_maps", False) for b in attn_banks]
    for b in attn_banks:
        b.capture_attention_maps = True
    # Force capture on the model regardless of recurrent vs non-recurrent path:
    # the bank stores per-(k_expert, v_expert) maps internally, and the model
    # forward appends them to `_all_attention_maps` after each depth.
    inner._all_attention_maps = []

    try:
        with torch.no_grad():
            input_ids = eval_batch["input_ids"].to(device)
            _ = model(input_ids=input_ids, labels=input_ids, return_logits=False)
        depth_attention = list(getattr(inner, "_all_attention_maps", []))
    finally:
        for b, prev in zip(attn_banks, prev_flags):
            b.capture_attention_maps = prev

    if was_training:
        model.train()

    if not depth_attention:
        return {}

    num_kv_groups = int(getattr(attn_banks[0], "num_kv_groups", 1))

    metrics: dict[str, float] = {}
    agg_rows_per_depth: list[np.ndarray] = []
    best_rows_per_depth: list[np.ndarray] = []

    for d, depth_maps in enumerate(depth_attention):
        if not depth_maps:
            agg_rows_per_depth.append(np.zeros_like(targets))
            best_rows_per_depth.append(np.zeros_like(targets))
            continue

        agg = _aggregate_attention(depth_maps, qk_w=None, num_kv_groups=num_kv_groups)
        # Average over batch and head dims to get a [T, T] map for diagnostics.
        agg_mat = agg.mean(dim=(0, 1)).numpy()  # [T, T]
        agg_rows_per_depth.append(agg_mat)

        # Per-row best expert map (already a [T, T] matrix per expert; we
        # take the min-KL per row but also surface the visual best per row
        # for the heatmap).
        per_expert_rows: list[np.ndarray] = []
        for entry in depth_maps:
            weights = entry["weights"].numpy()
            group_mask = entry["group_mask"].numpy()
            head_mask = np.repeat(group_mask, num_kv_groups, axis=1)
            mass = head_mask.sum(axis=(0, 1))[:, None] + 1e-9
            per_expert_rows.append(
                (weights * head_mask[..., None]).sum(axis=(0, 1)) / mass
            )
        per_expert_arr = np.stack(per_expert_rows, axis=0)  # [E, T, T]

        # Visual best: for each row, pick the expert with min KL to target.
        best_visual = np.zeros_like(agg_mat)
        learnable_idx = np.flatnonzero(row_mask)
        for r in learnable_idx:
            target_row = targets[r]
            kls = [_safe_kl(per_expert_arr[e, r], target_row) for e in range(per_expert_arr.shape[0])]
            best_visual[r] = per_expert_arr[int(np.argmin(kls)), r]
        best_rows_per_depth.append(best_visual)

        # Metrics
        kl_agg = []
        for r in learnable_idx:
            kl_agg.append(_safe_kl(agg_mat[r], targets[r]))
        prefix = f"attn_eval/depth_{d}"
        metrics[f"{prefix}/kl_agg_mean"] = float(np.mean(kl_agg)) if kl_agg else float("nan")
        metrics[f"{prefix}/kl_best_mean"] = _best_match_kl_per_row(
            depth_maps, targets, row_mask, num_kv_groups
        )
        metrics[f"{prefix}/iou_agg_mean"] = _iou_topk(agg_mat, targets, row_mask)
        metrics[f"{prefix}/entropy_agg_mean"] = _entropy(agg_mat, row_mask)
        metrics[f"{prefix}/num_active_pairs"] = float(len(depth_maps))

    # Roll-up across depths
    for stat in ("kl_agg_mean", "kl_best_mean", "iou_agg_mean", "entropy_agg_mean"):
        values = [
            metrics[k] for k in metrics if k.endswith("/" + stat) and not math.isnan(metrics[k])
        ]
        if values:
            metrics[f"attn_eval/overall/{stat}"] = float(np.mean(values))

    if save_heatmaps:
        _maybe_save_heatmaps(
            output_dir,
            step,
            targets=targets,
            agg_rows_per_depth=agg_rows_per_depth,
            best_rows_per_depth=best_rows_per_depth,
            row_mask=row_mask,
        )

    return metrics
