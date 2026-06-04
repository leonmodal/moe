"""Post-hoc attention-heatmap extractor for dense / standard_moe runs.

The moe_everything attention bank has a built-in `capture_attention_maps`
flag (see `src/models/moe_everything/attention_bank.py`). Dense and
standard_moe models use HF's `Qwen3Attention` layer which doesn't expose
attention weights through the model output. This script registers PyTorch
forward hooks on every `Qwen3Attention` module to capture the per-layer
attention weights (forcing `attn_implementation='eager'` so weights are
actually computed and returned by the layer), runs one eval forward, and
writes the same heatmap PNG format as `attention_eval.evaluate_attention_against_ground_truth`.

Designed to run inside Modal so it can read checkpoints directly from the
shared `moe-checkpoints` volume — see `modal_synthetic.regen_heatmaps` for
the launcher.

Usage (inside container):
    python scripts/extract_heatmaps_via_hooks.py \
        --config /root/moe/configs/synthetic/baseline_dense_1L_linearmap_30k.yaml \
        --checkpoint /checkpoints/synthetic/baseline_dense_1L_linearmap_30k/checkpoint-30000 \
        --output-dir /checkpoints/synthetic/baseline_dense_1L_linearmap_30k/attn_eval/step_00030000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make sure the repo root is on sys.path when invoked from inside the Modal
# container (where this lives at /root/moe/scripts/).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.training.config import load_config
from src.training.balancing_fields import normalize_balancing_config
from src.training.model_factory import build_model
from src.training.data import build_eval_dataset
from src.training.attention_eval import (
    build_ground_truth_targets,
    _aggregate_attention,
    _best_match_kl_per_row,
    _iou_topk,
    _entropy,
    _safe_kl,
    _maybe_save_heatmaps,
)


def _find_attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return all attention sub-modules in order of forward-pass execution."""
    # Class-name match keeps us model-agnostic: matches Qwen3Attention,
    # Qwen3MoeAttention, and any subclasses thereof.
    attn_modules = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls.endswith("Attention") and hasattr(module, "q_proj"):
            attn_modules.append((name, module))
    # Sort by name so depth order is consistent (depends on the model's
    # `model.layers.<i>.self_attn` layout — Qwen3 uses this scheme).
    attn_modules.sort(key=lambda nm: nm[0])
    return [m for _, m in attn_modules]


def _capture_hooks(model: torch.nn.Module) -> tuple[list, list]:
    """Install forward hooks that capture attention weights per layer.

    Returns (captured_list, handles). After running the forward pass,
    `captured_list[d]` holds the `attn_weights` tensor from depth d (or
    None if the layer didn't return one — e.g. SDPA path).
    """
    attn_modules = _find_attention_modules(model)
    captured: list[torch.Tensor | None] = [None] * len(attn_modules)

    def _make_hook(idx):
        def _hook(module, args, output):
            # Qwen3Attention returns (attn_output, attn_weights). With
            # attn_implementation='eager', attn_weights is a real tensor.
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                w = output[1]
                if isinstance(w, torch.Tensor):
                    captured[idx] = w.detach().cpu()
        return _hook

    handles = [m.register_forward_hook(_make_hook(i)) for i, m in enumerate(attn_modules)]
    return captured, handles


def _load_checkpoint_into(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load a training checkpoint's model state_dict, tolerating wrapper prefixes."""
    # Look for the canonical model state file the trainer writes.
    candidate_files = [
        ckpt_path / "model.safetensors",
        ckpt_path / "pytorch_model.bin",
        ckpt_path / "model.pt",
    ]
    state = None
    for p in candidate_files:
        if p.exists():
            if p.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(str(p))
            else:
                state = torch.load(p, map_location="cpu", weights_only=False)
            print(f"Loaded weights from {p}", flush=True)
            break
    if state is None:
        # Some training runs save the full Trainer state in a single .pt
        # under the checkpoint dir. Look for it.
        for p in ckpt_path.glob("*.pt"):
            blob = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(blob, dict) and "model" in blob:
                state = blob["model"]
                print(f"Loaded weights from {p} (model key)", flush=True)
                break
    if state is None:
        raise FileNotFoundError(f"No model state file found under {ckpt_path}")

    # Strip common DDP / FSDP wrapper prefixes.
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "_orig_mod.", "model._orig_mod."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (first: {missing[:3]})", flush=True)
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (first: {unexpected[:3]})", flush=True)


def _build_depth_maps_from_captured(
    captured: list[torch.Tensor | None],
) -> list[list[dict]]:
    """Wrap each layer's [B, H, T, T] attention tensor into the format
    `_aggregate_attention` and the heatmap code expect: a list of "expert
    pair" dicts per depth. Dense/standard_moe have one effective expert per
    layer, so each depth gets exactly one entry with a full-ones group_mask.
    """
    depth_maps: list[list[dict]] = []
    for w in captured:
        if w is None:
            depth_maps.append([])
            continue
        B, H, T, _ = w.shape
        # Synthesize an all-ones group_mask with shape [B, num_kv_heads, T].
        # We don't have num_kv_heads available here without the model config,
        # but the aggregator only needs head_mask = group_mask
        # repeat_interleaved by num_kv_groups; passing num_kv_groups=1 and a
        # single-head mask reduces to a uniform-weight average.
        group_mask = torch.ones(B, H, T, dtype=torch.bool)
        depth_maps.append([
            {
                "k_expert": 0,
                "v_expert": 0,
                "weights": w.float(),  # [B, H, T, T]
                "group_mask": group_mask,
            }
        ])
    return depth_maps


def run(config_path: str, checkpoint_path: str, output_dir: str) -> None:
    cfg = load_config(config_path)
    normalize_balancing_config(cfg)
    # Force eager attention so Qwen3Attention returns real attn_weights via
    # eager_attention_forward (SDPA returns None).
    cfg["model"]["attn_implementation"] = "eager"

    model, model_cfg = build_model(cfg)
    _load_checkpoint_into(model, Path(checkpoint_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eval_dataset = build_eval_dataset(cfg, tokenizer=None, rank=0, world_size=1)
    if eval_dataset is None:
        raise RuntimeError("eval dataset not available")
    targets, row_mask = build_ground_truth_targets(eval_dataset)
    if targets is None:
        raise RuntimeError("ground truth targets unavailable")

    # Form one eval batch.
    batch_size = int(cfg.get("eval", {}).get("batch_size", 8))
    it = iter(eval_dataset)
    input_ids = torch.stack([next(it)["input_ids"] for _ in range(batch_size)]).to(device)

    captured, handles = _capture_hooks(model)
    try:
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids, labels=input_ids, return_logits=False)
            except TypeError:
                _ = model(input_ids=input_ids, labels=input_ids)
    finally:
        for h in handles:
            h.remove()

    depth_attention = _build_depth_maps_from_captured(captured)
    num_active = sum(1 for d in depth_attention if d)
    print(f"Captured attention maps for {num_active}/{len(depth_attention)} depths", flush=True)

    # Compute metrics + heatmaps using the same helpers as the in-training eval.
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, float] = {}
    agg_rows_per_depth: list[np.ndarray] = []
    best_rows_per_depth: list[np.ndarray] = []
    num_kv_groups = 1  # dense/standard_moe path: heads already expanded by hook

    for d, depth_maps in enumerate(depth_attention):
        if not depth_maps:
            agg_rows_per_depth.append(np.zeros_like(targets))
            best_rows_per_depth.append(np.zeros_like(targets))
            continue

        agg = _aggregate_attention(depth_maps, qk_w=None, num_kv_groups=num_kv_groups)
        agg_mat = agg.mean(dim=(0, 1)).numpy()
        agg_rows_per_depth.append(agg_mat)

        # Per-expert best-match — with a single "expert" per depth this just
        # equals the depth's mean row, but keeping the same code path so the
        # heatmap structure matches the moe_everything output.
        per_expert_rows = []
        for entry in depth_maps:
            weights = entry["weights"].numpy()
            group_mask = entry["group_mask"].numpy()
            head_mask = np.repeat(group_mask, num_kv_groups, axis=1)
            mass = head_mask.sum(axis=(0, 1))[:, None] + 1e-9
            per_expert_rows.append((weights * head_mask[..., None]).sum(axis=(0, 1)) / mass)
        per_expert_arr = np.stack(per_expert_rows, axis=0)
        best_visual = np.zeros_like(agg_mat)
        learnable_idx = np.flatnonzero(row_mask)
        for r in learnable_idx:
            kls = [_safe_kl(per_expert_arr[e, r], targets[r]) for e in range(per_expert_arr.shape[0])]
            best_visual[r] = per_expert_arr[int(np.argmin(kls)), r]
        best_rows_per_depth.append(best_visual)

        kl_agg = [_safe_kl(agg_mat[r], targets[r]) for r in learnable_idx]
        prefix = f"attn_eval/depth_{d}"
        metrics[f"{prefix}/kl_agg_mean"] = float(np.mean(kl_agg)) if kl_agg else float("nan")
        metrics[f"{prefix}/kl_best_mean"] = _best_match_kl_per_row(
            depth_maps, targets, row_mask, num_kv_groups
        )
        metrics[f"{prefix}/iou_agg_mean"] = _iou_topk(agg_mat, targets, row_mask)
        metrics[f"{prefix}/entropy_agg_mean"] = _entropy(agg_mat, row_mask)
        metrics[f"{prefix}/num_active_pairs"] = float(len(depth_maps))

    # Roll-ups
    import math
    for stat in ("kl_agg_mean", "kl_best_mean", "iou_agg_mean", "entropy_agg_mean"):
        values = [v for k, v in metrics.items() if k.endswith("/" + stat) and not math.isnan(v)]
        if values:
            metrics[f"attn_eval/overall/{stat}"] = float(np.mean(values))

    print("Final metrics:", flush=True)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}", flush=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir / 'metrics.json'}", flush=True)

    # The heatmap helper writes per-depth PNGs into the parent
    # `attn_eval/step_<N>/` directory; pass it the output_dir's parent so it
    # matches the moe_everything output layout.
    _maybe_save_heatmaps(
        str(out_dir.parent.parent),  # path that contains attn_eval/
        step=int(out_dir.name.replace("step_", "").lstrip("0") or 0),
        targets=targets,
        agg_rows_per_depth=agg_rows_per_depth,
        best_rows_per_depth=best_rows_per_depth,
        row_mask=row_mask,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run(args.config, args.checkpoint, args.output_dir)
