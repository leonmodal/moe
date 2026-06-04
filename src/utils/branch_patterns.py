"""Branch-route artifact export for MoE-Everything.

The training loop already logs aggregate ATTN-vs-MLP branch fractions.
This module preserves the token axis so a routing snapshot can answer:

* which full-depth branch paths are common?
* which individual tokens follow which path?
* how confident was the branch router at each depth?
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BRANCH_LABELS = {0: "A", 1: "M"}
BRANCH_NAMES = {"A": "attn", "M": "mlp"}


def _inner_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _selection_rows(selected: list[Any] | tuple[Any, ...] | None) -> np.ndarray | None:
    if not selected:
        return None

    rows: list[np.ndarray] = []
    min_tokens: int | None = None
    for item in selected:
        arr = _to_numpy(item)
        if arr is None:
            continue
        if arr.ndim >= 1 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        flat = arr.reshape(-1).astype(np.int64, copy=False)
        rows.append(flat)
        min_tokens = flat.shape[0] if min_tokens is None else min(min_tokens, flat.shape[0])

    if not rows or min_tokens is None or min_tokens == 0:
        return None
    return np.stack([row[:min_tokens] for row in rows], axis=0)


def _expert_selection_rows(
    selected: list[Any] | tuple[Any, ...] | None,
    num_tokens: int,
) -> list[np.ndarray | None]:
    """Return per-depth selected expert ids as ``[(tokens, top_k), ...]``.

    MLP selections are naturally ``(tokens, top_k)``. Attention selections are
    usually ``(tokens, routed_slots)`` after the model flattens per-head routes.
    Keeping the second axis intact makes the exported routes readable without
    assuming how many experts/heads the config uses.
    """
    if not selected:
        return []

    rows: list[np.ndarray | None] = []
    for item in selected:
        arr = _to_numpy(item)
        if arr is None or arr.size == 0:
            rows.append(None)
            continue
        if arr.ndim == 1:
            flat = arr.reshape(-1, 1)
        elif arr.shape[0] == num_tokens:
            flat = arr.reshape(num_tokens, -1)
        elif arr.size % max(1, num_tokens) == 0:
            flat = arr.reshape(num_tokens, -1)
        else:
            flat = arr.reshape(-1, arr.shape[-1])
        rows.append(flat[:num_tokens].astype(np.int64, copy=False))
    return rows


def _mask_rows(
    masks: list[Any] | tuple[Any, ...] | None,
    num_tokens: int,
) -> list[np.ndarray | None]:
    if not masks:
        return []

    rows: list[np.ndarray | None] = []
    for item in masks:
        arr = _to_numpy(item)
        if arr is None or arr.size == 0:
            rows.append(None)
            continue
        rows.append(arr.reshape(-1)[:num_tokens].astype(bool, copy=False))
    return rows


def _attention_selection_rows(
    attention_info: list[Any] | tuple[Any, ...] | None,
    num_tokens: int,
    route_name_aliases: dict[str, str] | None = None,
) -> list[dict[str, dict[str, np.ndarray | None]]]:
    if not attention_info:
        return []

    aliases = route_name_aliases or {}
    per_depth: list[dict[str, dict[str, np.ndarray | None]]] = []
    for depth_info in attention_info:
        depth_routes: dict[str, dict[str, np.ndarray | None]] = {}
        if not isinstance(depth_info, dict):
            per_depth.append(depth_routes)
            continue
        for name, info in depth_info.items():
            if not isinstance(info, dict):
                continue
            arr = _to_numpy(info.get("selected_experts"))
            if arr is None or arr.size == 0:
                continue
            if arr.ndim == 1:
                flat = arr.reshape(-1, 1)
            elif arr.shape[0] == num_tokens:
                flat = arr.reshape(num_tokens, -1)
            elif arr.size % max(1, num_tokens) == 0:
                flat = arr.reshape(num_tokens, -1)
            else:
                flat = arr.reshape(-1, arr.shape[-1])
            mask = _to_numpy(info.get("token_mask"))
            mask_flat = None
            if mask is not None and mask.size > 0:
                mask_flat = mask.reshape(-1)[:num_tokens].astype(bool, copy=False)
            route_name = aliases.get(str(name), str(name))
            depth_routes[route_name] = {
                "routes": flat[:num_tokens].astype(np.int64, copy=False),
                "mask": mask_flat,
            }
        per_depth.append(depth_routes)
    return per_depth


def _attention_route_name_aliases(inner: Any) -> dict[str, str]:
    attn_bank = getattr(inner, "attn_bank", None)
    bundle = getattr(attn_bank, "routing_bundle", None)
    aliases: dict[str, str] = {}
    if bundle == "qkvo":
        aliases["qk"] = "qkvo"
    elif bundle == "qkv_o":
        aliases["qk"] = "qkv"
    elif bundle == "qk_vo":
        aliases["v"] = "vo"
    return aliases


def _prob_rows(probs: list[Any] | tuple[Any, ...] | None, num_tokens: int) -> np.ndarray | None:
    if not probs:
        return None

    rows: list[np.ndarray] = []
    for item in probs:
        arr = _to_numpy(item)
        if arr is None or arr.shape[-1] < 2:
            continue
        flat = arr.reshape(-1, arr.shape[-1])[:num_tokens, :2].astype(np.float64, copy=False)
        rows.append(flat)

    if not rows:
        return None
    min_depths = min(len(rows), 10**9)
    return np.stack(rows[:min_depths], axis=0)


def _batch_position(index: int, seq_len: int | None) -> tuple[int | None, int]:
    if not seq_len or seq_len <= 0:
        return None, index
    return index // seq_len, index % seq_len


def _token_text(tokenizer: Any, token_id: int | None) -> str | None:
    if tokenizer is None or token_id is None:
        return None
    try:
        text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        return None
    return text.replace("\n", "\\n").replace("\t", "\\t")


def _round(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _md_escape(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("|", "&#124;")
        .replace("\n", "<br>")
    )


def _md_number(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _md_escape(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    out = [
        "| " + " | ".join(_md_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_md_escape(value) for value in row) + " |")
    return "\n".join(out) + "\n"


def _patterns(selection: np.ndarray) -> list[str]:
    depth, num_tokens = selection.shape
    out: list[str] = []
    for token_idx in range(num_tokens):
        chars = []
        for d in range(depth):
            chars.append(BRANCH_LABELS.get(int(selection[d, token_idx]), "?"))
        out.append("".join(chars))
    return out


def _format_experts(values: np.ndarray | list[int] | tuple[int, ...] | None, prefix: str) -> str | None:
    if values is None:
        return None
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return None
    return "|".join(f"{prefix}{int(v)}" for v in arr.tolist())


def _format_attention_route(
    attention_routes: list[dict[str, dict[str, np.ndarray | None]]],
    depth_idx: int,
    token_idx: int,
) -> str | None:
    if depth_idx >= len(attention_routes):
        return None
    routes = attention_routes[depth_idx]
    if not routes:
        return None

    ordered_names = [name for name in ("qkvo", "qkv", "qk", "q", "k", "v", "o") if name in routes]
    ordered_names.extend(sorted(name for name in routes if name not in ordered_names))
    parts = []
    for name in ordered_names:
        entry = routes[name]
        arr = entry.get("routes")
        mask = entry.get("mask")
        if arr is None:
            continue
        if mask is not None and token_idx < mask.shape[0] and not bool(mask[token_idx]):
            continue
        if token_idx >= arr.shape[0]:
            continue
        formatted = _format_experts(arr[token_idx], "A")
        if formatted is not None:
            parts.append((name, formatted))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0][1]
    return ";".join(f"{name}={value}" for name, value in parts)


def _format_mlp_route(
    mlp_routes: list[np.ndarray | None],
    mlp_masks: list[np.ndarray | None],
    depth_idx: int,
    token_idx: int,
) -> str | None:
    if depth_idx >= len(mlp_routes):
        return None
    route = mlp_routes[depth_idx]
    if route is None or token_idx >= route.shape[0]:
        return None
    if depth_idx < len(mlp_masks):
        mask = mlp_masks[depth_idx]
        if mask is not None and token_idx < mask.shape[0] and not bool(mask[token_idx]):
            return None
    return _format_experts(route[token_idx], "M")


def _depth_route_records(
    *,
    selection: np.ndarray,
    token_idx: int,
    mlp_routes: list[np.ndarray | None],
    mlp_masks: list[np.ndarray | None],
    attention_routes: list[dict[str, dict[str, np.ndarray | None]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for depth_idx in range(selection.shape[0]):
        branch = BRANCH_LABELS.get(int(selection[depth_idx, token_idx]), "?")
        attn = _format_attention_route(attention_routes, depth_idx, token_idx)
        mlp = _format_mlp_route(mlp_routes, mlp_masks, depth_idx, token_idx)
        active = attn if branch == "A" else mlp if branch == "M" else None
        records.append({
            "depth": int(depth_idx),
            "branch": branch,
            "attention_experts": attn,
            "mlp_experts": mlp,
            "route": active,
        })
    return records


def _prob_metrics_for_tokens(
    probs: np.ndarray | None,
    selection: np.ndarray,
    token_indices: list[int] | np.ndarray,
) -> dict[str, float | None]:
    if probs is None or probs.shape[0] == 0 or len(token_indices) == 0:
        return {
            "mean_attn_score": None,
            "mean_mlp_score": None,
            "mean_selected_score": None,
            "mean_score_margin": None,
            "mean_entropy": None,
        }

    d = min(probs.shape[0], selection.shape[0])
    idx = np.asarray(token_indices, dtype=np.int64)
    p = probs[:d, idx, :2]
    chosen = selection[:d, idx]
    selected_scores = np.take_along_axis(p, chosen[..., None], axis=-1)[..., 0]
    sums = np.clip(p.sum(axis=-1, keepdims=True), 1e-20, None)
    normalized = np.clip(p / sums, 1e-20, 1.0)
    entropy = -(normalized * np.log(normalized)).sum(axis=-1)
    return {
        "mean_attn_score": _round(p[..., 0].mean()),
        "mean_mlp_score": _round(p[..., 1].mean()),
        "mean_selected_score": _round(selected_scores.mean()),
        "mean_score_margin": _round(np.abs(p[..., 0] - p[..., 1]).mean()),
        "mean_entropy": _round(entropy.mean()),
    }


def _depth_summary(selection: np.ndarray, probs: np.ndarray | None) -> list[dict[str, Any]]:
    rows = []
    depth, num_tokens = selection.shape
    for d in range(depth):
        chosen = selection[d]
        attn = int((chosen == 0).sum())
        mlp = int((chosen == 1).sum())
        row: dict[str, Any] = {
            "depth": d,
            "tokens": int(num_tokens),
            "attn_tokens": attn,
            "mlp_tokens": mlp,
            "attn_fraction": _round(attn / max(1, num_tokens)),
            "mlp_fraction": _round(mlp / max(1, num_tokens)),
        }
        if probs is not None and d < probs.shape[0]:
            p = probs[d, :, :2]
            sums = np.clip(p.sum(axis=-1, keepdims=True), 1e-20, None)
            normalized = np.clip(p / sums, 1e-20, 1.0)
            entropy = -(normalized * np.log(normalized)).sum(axis=-1)
            row.update({
                "mean_attn_score": _round(p[:, 0].mean()),
                "mean_mlp_score": _round(p[:, 1].mean()),
                "mean_score_margin": _round(np.abs(p[:, 0] - p[:, 1]).mean()),
                "mean_entropy": _round(entropy.mean()),
            })
        rows.append(row)
    return rows


def _token_record(
    *,
    index: int,
    pattern: str,
    selection: np.ndarray,
    probs: np.ndarray | None,
    mlp_routes: list[np.ndarray | None],
    mlp_masks: list[np.ndarray | None],
    attention_routes: list[dict[str, dict[str, np.ndarray | None]]],
    token_ids: np.ndarray | None,
    seq_len: int | None,
    tokenizer: Any,
    include_probs: bool,
) -> dict[str, Any]:
    batch, position = _batch_position(index, seq_len)
    token_id = int(token_ids[index]) if token_ids is not None and index < len(token_ids) else None
    record: dict[str, Any] = {
        "flat_index": int(index),
        "batch": batch,
        "position": int(position),
        "token_id": token_id,
        "token_text": _token_text(tokenizer, token_id),
        "pattern": pattern,
        "branches": [BRANCH_NAMES.get(ch, "unknown") for ch in pattern],
    }
    depth_routes = _depth_route_records(
        selection=selection,
        token_idx=index,
        mlp_routes=mlp_routes,
        mlp_masks=mlp_masks,
        attention_routes=attention_routes,
    )
    record["depth_routes"] = depth_routes
    record["expert_pattern"] = " ".join(
        str(item["route"]) if item.get("route") else str(item["branch"])
        for item in depth_routes
    )
    record.update(_prob_metrics_for_tokens(probs, selection, [index]))
    if include_probs and probs is not None:
        d = min(probs.shape[0], selection.shape[0])
        record["attn_scores_by_depth"] = [_round(v) for v in probs[:d, index, 0].tolist()]
        record["mlp_scores_by_depth"] = [_round(v) for v in probs[:d, index, 1].tolist()]
    return record


def build_branch_route_artifacts(
    model: Any,
    *,
    step: int,
    input_ids: Any = None,
    tokenizer: Any = None,
    top_k: int = 32,
    max_tokens: int = 512,
    max_examples_per_pattern: int = 8,
) -> dict[str, Any] | None:
    """Build serializable token-level branch routing artifacts.

    Returns None when the model has not run a MoE-Everything branch forward.
    """

    inner = _inner_model(model)
    selection = _selection_rows(getattr(inner, "_all_branch_selected_experts", None))
    if selection is None:
        return None

    num_depths, num_tokens = selection.shape
    probs = _prob_rows(getattr(inner, "_all_branch_probs", None), num_tokens)
    if probs is not None and probs.shape[0] != num_depths:
        common_depths = min(num_depths, probs.shape[0])
        selection = selection[:common_depths]
        probs = probs[:common_depths]
        num_depths = common_depths

    input_arr = _to_numpy(input_ids)
    token_ids = input_arr.reshape(-1)[:num_tokens].astype(np.int64, copy=False) if input_arr is not None and input_arr.size > 0 else None
    if token_ids is not None:
        num_tokens = min(num_tokens, token_ids.shape[0])
        selection = selection[:, :num_tokens]
        probs = probs[:, :num_tokens, :] if probs is not None else None

    input_shape = list(input_arr.shape) if input_arr is not None else None
    seq_len = input_shape[-1] if input_shape and len(input_shape) >= 2 else None
    mlp_routes = _expert_selection_rows(
        getattr(inner, "_all_mlp_selected_experts", None),
        num_tokens,
    )
    mlp_masks = _mask_rows(getattr(inner, "_all_mlp_token_masks", None), num_tokens)
    attention_routes = _attention_selection_rows(
        getattr(inner, "_all_attn_router_info", None),
        num_tokens,
        route_name_aliases=_attention_route_name_aliases(inner),
    )
    patterns = _patterns(selection)
    counts = Counter(patterns)
    pattern_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, pattern in enumerate(patterns):
        if len(pattern_to_indices[pattern]) < max_examples_per_pattern:
            pattern_to_indices[pattern].append(idx)

    top_patterns = []
    for pattern, count in counts.most_common(top_k):
        example_indices = pattern_to_indices[pattern]
        examples = [
            _token_record(
                index=idx,
                pattern=pattern,
                selection=selection,
                probs=probs,
                mlp_routes=mlp_routes,
                mlp_masks=mlp_masks,
                attention_routes=attention_routes,
                token_ids=token_ids,
                seq_len=seq_len,
                tokenizer=tokenizer,
                include_probs=False,
            )
            for idx in example_indices
        ]
        row: dict[str, Any] = {
            "pattern": pattern,
            "count": int(count),
            "share": _round(count / max(1, num_tokens)),
            "attn_depths": int(pattern.count("A")),
            "mlp_depths": int(pattern.count("M")),
            "examples": examples,
        }
        all_indices = [i for i, p in enumerate(patterns) if p == pattern]
        row.update(_prob_metrics_for_tokens(probs, selection, all_indices))
        top_patterns.append(row)
    pattern_stats = {
        pattern: {
            "pattern_rank": rank,
            "pattern_count": int(count),
            "pattern_share": _round(count / max(1, num_tokens)),
            "pattern_attn_depths": int(pattern.count("A")),
            "pattern_mlp_depths": int(pattern.count("M")),
        }
        for rank, (pattern, count) in enumerate(counts.most_common(), start=1)
    }

    sample_count = min(max_tokens, num_tokens)
    sample_tokens = [
        _token_record(
            index=idx,
            pattern=patterns[idx],
            selection=selection,
            probs=probs,
            mlp_routes=mlp_routes,
            mlp_masks=mlp_masks,
            attention_routes=attention_routes,
            token_ids=token_ids,
            seq_len=seq_len,
            tokenizer=tokenizer,
            include_probs=True,
        )
        for idx in range(sample_count)
    ]

    return {
        "step": int(step),
        "kind": "moe_everything_branch_routes",
        "legend": {
            "A": "attention branch",
            "M": "MLP branch",
            "pattern_order": "left-to-right is increasing depth",
            "expert_pattern": "per-depth active route; attention entries are per routed head/slot, MLP entries are top-k experts",
        },
        "num_depths": int(num_depths),
        "num_tokens": int(num_tokens),
        "num_unique_patterns": int(len(counts)),
        "input_shape": input_shape,
        "depth_summary": _depth_summary(selection, probs),
        "pattern_stats": pattern_stats,
        "top_patterns": top_patterns,
        "sample_tokens": sample_tokens,
    }


def _write_token_routes_csv(path: Path, payload: dict[str, Any]) -> None:
    num_depths = int(payload.get("num_depths", 0) or 0)
    pattern_lookup = payload.get("pattern_stats", {})
    base_fields = [
        "flat_index",
        "batch",
        "position",
        "token_id",
        "token_text",
        "pattern",
        "pattern_rank",
        "pattern_count",
        "pattern_share",
        "pattern_attn_depths",
        "pattern_mlp_depths",
        "expert_pattern",
        "mean_selected_score",
        "mean_score_margin",
        "mean_entropy",
    ]
    depth_fields: list[str] = []
    for depth_idx in range(num_depths):
        stem = f"layer_{depth_idx:02d}"
        depth_fields.extend([
            stem,
            f"{stem}_branch",
            f"{stem}_attn",
            f"{stem}_mlp",
        ])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + depth_fields)
        writer.writeheader()
        for row in payload["sample_tokens"]:
            out = {field: row.get(field) for field in base_fields}
            out.update(pattern_lookup.get(row.get("pattern"), {}))
            routes = row.get("depth_routes", [])
            for depth_idx in range(num_depths):
                stem = f"layer_{depth_idx:02d}"
                route = routes[depth_idx] if depth_idx < len(routes) else {}
                out[stem] = route.get("route") or route.get("branch")
                out[f"{stem}_branch"] = route.get("branch")
                out[f"{stem}_attn"] = route.get("attention_experts")
                out[f"{stem}_mlp"] = route.get("mlp_experts")
            writer.writerow(out)


def _write_token_routes_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Token Routes",
        "",
        f"- Step: `{payload.get('step')}`",
        f"- Tokens shown: `{len(payload.get('sample_tokens', []))}`",
        f"- Total tokens in snapshot: `{payload.get('num_tokens')}`",
        f"- Depths: `{payload.get('num_depths')}`",
        "",
        "Legend: `A` means attention branch, `M` means MLP branch. Route entries are ordered by increasing depth.",
        "",
    ]

    rows = []
    for row in payload.get("sample_tokens", []):
        route_lines = []
        for item in row.get("depth_routes", []):
            depth = item.get("depth")
            branch = item.get("branch")
            route = item.get("route") or branch
            route_lines.append(f"{int(depth):02d}:{branch} {route}")
        rows.append([
            row.get("flat_index"),
            row.get("batch"),
            row.get("position"),
            row.get("token_text") if row.get("token_text") is not None else row.get("token_id"),
            f"`{row.get('pattern')}`",
            _md_number(row.get("mean_selected_score")),
            _md_number(row.get("mean_score_margin")),
            _md_number(row.get("mean_entropy")),
            "<br>".join(route_lines),
        ])

    lines.append(_md_table(
        [
            "Index",
            "Batch",
            "Pos",
            "Token",
            "Pattern",
            "Selected score",
            "Margin",
            "Entropy",
            "Route by depth",
        ],
        rows,
    ))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_top_patterns_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "rank",
        "pattern",
        "count",
        "share",
        "attn_depths",
        "mlp_depths",
        "mean_selected_score",
        "mean_score_margin",
        "mean_entropy",
        "examples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(payload.get("top_patterns", []), start=1):
            examples = []
            for example in row.get("examples", []):
                token = example.get("token_text")
                pos = example.get("position")
                if token is None:
                    examples.append(str(pos))
                else:
                    examples.append(f"{pos}:{token}")
            writer.writerow({
                "rank": rank,
                "pattern": row.get("pattern"),
                "count": row.get("count"),
                "share": row.get("share"),
                "attn_depths": row.get("attn_depths"),
                "mlp_depths": row.get("mlp_depths"),
                "mean_selected_score": row.get("mean_selected_score"),
                "mean_score_margin": row.get("mean_score_margin"),
                "mean_entropy": row.get("mean_entropy"),
                "examples": "|".join(examples),
            })


def _write_depth_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "depth",
        "tokens",
        "attn_tokens",
        "mlp_tokens",
        "attn_fraction",
        "mlp_fraction",
        "attn_to_mlp_ratio",
        "mean_attn_score",
        "mean_mlp_score",
        "mean_score_margin",
        "mean_entropy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in payload.get("depth_summary", []):
            attn = float(row.get("attn_fraction") or 0.0)
            mlp = float(row.get("mlp_fraction") or 0.0)
            out = {field: row.get(field) for field in fields}
            out["attn_to_mlp_ratio"] = _round(attn / mlp) if mlp > 0 else None
            writer.writerow(out)


def _write_branch_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Branch Route Snapshot",
        "",
        f"- Step: `{payload.get('step')}`",
        f"- Tokens: `{payload.get('num_tokens')}`",
        f"- Depths: `{payload.get('num_depths')}`",
        f"- Unique branch patterns: `{payload.get('num_unique_patterns')}`",
        "",
        "Legend: `A` means attention branch, `M` means MLP branch. Pattern characters are ordered by increasing depth.",
        "",
        "Plots in this folder: `top_patterns.png`, `top_pattern_matrix.png`, `branch_depth_ratios.png`.",
        "",
        "## Top Branch Patterns",
        "",
    ]

    top_rows = []
    for rank, row in enumerate(payload.get("top_patterns", [])[:32], start=1):
        examples = []
        for example in row.get("examples", [])[:6]:
            token = example.get("token_text")
            position = example.get("position")
            examples.append(f"{position}:{token}" if token is not None else str(position))
        top_rows.append([
            rank,
            f"`{row.get('pattern')}`",
            row.get("count"),
            _md_number(row.get("share")),
            row.get("attn_depths"),
            row.get("mlp_depths"),
            _md_number(row.get("mean_selected_score")),
            _md_number(row.get("mean_score_margin")),
            _md_number(row.get("mean_entropy")),
            "<br>".join(examples),
        ])
    lines.append(_md_table(
        [
            "Rank",
            "Pattern",
            "Count",
            "Share",
            "A depths",
            "M depths",
            "Selected score",
            "Score margin",
            "Entropy",
            "Example tokens",
        ],
        top_rows,
    ))

    lines.extend(["", "## Branch Ratio By Depth", ""])
    depth_rows = []
    for row in payload.get("depth_summary", []):
        attn = float(row.get("attn_fraction") or 0.0)
        mlp = float(row.get("mlp_fraction") or 0.0)
        ratio = _round(attn / mlp) if mlp > 0 else None
        depth_rows.append([
            row.get("depth"),
            row.get("tokens"),
            row.get("attn_tokens"),
            row.get("mlp_tokens"),
            _md_number(row.get("attn_fraction")),
            _md_number(row.get("mlp_fraction")),
            _md_number(ratio),
            _md_number(row.get("mean_score_margin")),
            _md_number(row.get("mean_entropy")),
        ])
    lines.append(_md_table(
        [
            "Depth",
            "Tokens",
            "A tokens",
            "M tokens",
            "A frac",
            "M frac",
            "A/M",
            "Margin",
            "Entropy",
        ],
        depth_rows,
    ))

    lines.extend(["", "## Sample Token Routes", ""])
    token_rows = []
    for row in payload.get("sample_tokens", [])[:64]:
        route_lines = []
        for item in row.get("depth_routes", []):
            depth = item.get("depth")
            branch = item.get("branch")
            route = item.get("route") or branch
            route_lines.append(f"{int(depth):02d}:{branch} {route}")
        token_rows.append([
            row.get("flat_index"),
            row.get("batch"),
            row.get("position"),
            row.get("token_text") if row.get("token_text") is not None else row.get("token_id"),
            f"`{row.get('pattern')}`",
            _md_number(row.get("mean_selected_score")),
            _md_number(row.get("mean_entropy")),
            "<br>".join(route_lines),
        ])
    lines.append(_md_table(
        ["Index", "Batch", "Pos", "Token", "Pattern", "Selected score", "Entropy", "Route by depth"],
        token_rows,
    ))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _plot_branch_depth_summary(out_dir: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("depth_summary", [])
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    depths = np.array([int(row["depth"]) for row in rows])
    attn = np.array([float(row.get("attn_fraction") or 0.0) for row in rows])
    mlp = np.array([float(row.get("mlp_fraction") or 0.0) for row in rows])
    ratio = np.divide(attn, mlp, out=np.full_like(attn, np.nan), where=mlp > 0)

    has_entropy = any(row.get("mean_entropy") is not None for row in rows)
    n_rows = 3 if has_entropy else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3.0 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    axes[0].plot(depths, attn, marker="o", label="Attention", color="#1f77b4", linewidth=2)
    axes[0].plot(depths, mlp, marker="o", label="MLP", color="#ff7f0e", linewidth=2)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Token fraction")
    axes[0].set_title(f"Branch Ratio by Depth (step {payload.get('step')})")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(depths, ratio, marker="o", color="#2c3e50", linewidth=2)
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Attn / MLP")
    axes[1].grid(alpha=0.25)

    if has_entropy:
        entropy = np.array([
            float(row.get("mean_entropy")) if row.get("mean_entropy") is not None else np.nan
            for row in rows
        ])
        margin = np.array([
            float(row.get("mean_score_margin")) if row.get("mean_score_margin") is not None else np.nan
            for row in rows
        ])
        axes[2].plot(depths, entropy, marker="o", label="Entropy", color="#9467bd", linewidth=2)
        axes[2].plot(depths, margin, marker="o", label="Score margin", color="#2ca02c", linewidth=2)
        axes[2].set_ylabel("Score statistic")
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Depth")
    axes[-1].set_xticks(depths)
    fig.tight_layout()
    fig.savefig(out_dir / "branch_depth_ratios.png", dpi=140)
    plt.close(fig)


def _plot_top_patterns(out_dir: Path, payload: dict[str, Any], max_patterns: int = 20) -> None:
    rows = payload.get("top_patterns", [])[:max_patterns]
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [str(row["pattern"]) for row in rows]
    shares = np.array([float(row.get("share") or 0.0) for row in rows])
    counts = [int(row.get("count") or 0) for row in rows]

    y = np.arange(len(rows))
    fig_h = max(4.0, 0.36 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    bars = ax.barh(y, shares, color="#4c78a8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Share of tokens")
    ax.set_title(f"Most Common Branch Patterns (step {payload.get('step')})")
    ax.grid(axis="x", alpha=0.25)
    max_share = float(shares.max()) if shares.size else 0.0
    ax.set_xlim(0.0, min(1.0, max(0.05, max_share * 1.18)))
    for bar, count, share in zip(bars, counts, shares):
        ax.text(
            bar.get_width() + max(0.002, max_share * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{share:.3f} ({count})",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "top_patterns.png", dpi=140)
    plt.close(fig)


def _plot_top_pattern_matrix(out_dir: Path, payload: dict[str, Any], max_patterns: int = 20) -> None:
    rows = payload.get("top_patterns", [])[:max_patterns]
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception:
        return

    patterns = [str(row["pattern"]) for row in rows]
    if not patterns:
        return
    depth = max(len(pattern) for pattern in patterns)
    mat = np.full((len(patterns), depth), np.nan)
    for i, pattern in enumerate(patterns):
        for j, ch in enumerate(pattern):
            if ch == "A":
                mat[i, j] = 0
            elif ch == "M":
                mat[i, j] = 1

    labels = [
        f"{row['pattern']}  {float(row.get('share') or 0.0):.3f}"
        for row in rows
    ]
    fig_h = max(4.0, 0.36 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(max(9.0, depth * 0.45 + 4), fig_h))
    cmap = ListedColormap(["#1f77b4", "#ff7f0e"])
    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=8)
    ax.set_xticks(np.arange(depth))
    ax.set_xlabel("Depth")
    ax.set_title(f"Top Branch Pattern Structure (A=attention, M=MLP, step {payload.get('step')})")
    for i in range(len(rows)):
        for j in range(depth):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, "A" if value == 0 else "M", ha="center", va="center", color="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "top_pattern_matrix.png", dpi=140)
    plt.close(fig)


def _write_branch_plots(out_dir: Path, payload: dict[str, Any]) -> None:
    _plot_branch_depth_summary(out_dir, payload)
    _plot_top_patterns(out_dir, payload)
    _plot_top_pattern_matrix(out_dir, payload)


def save_branch_route_artifacts(
    model: Any,
    *,
    step_dir: str,
    step: int,
    input_ids: Any = None,
    tokenizer: Any = None,
    top_k: int = 32,
    max_tokens: int = 512,
    max_examples_per_pattern: int = 8,
) -> bool:
    """Write the canonical branch-route table under ``step_dir/branch_patterns``."""

    payload = build_branch_route_artifacts(
        model,
        step=step,
        input_ids=input_ids,
        tokenizer=tokenizer,
        top_k=top_k,
        max_tokens=max_tokens,
        max_examples_per_pattern=max_examples_per_pattern,
    )
    if payload is None:
        return False

    out_dir = Path(step_dir) / "branch_patterns"
    os.makedirs(out_dir, exist_ok=True)
    _write_token_routes_csv(out_dir / "token_routes.csv", payload)
    _write_token_routes_md(out_dir / "token_routes.md", payload)
    _write_top_patterns_csv(out_dir / "top_patterns.csv", payload)
    _write_depth_summary_csv(out_dir / "depth_summary.csv", payload)
    _write_branch_summary_md(out_dir / "summary.md", payload)
    _write_branch_plots(out_dir, payload)
    return True
