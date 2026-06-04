"""Load-balance artifact export for routed MoE snapshots.

These artifacts are computed from the model's most recent routing caches, not
from ``local_tokens_per_expert``. The bias-update path zeros those transient
buffers after every optimizer step, while the selected-expert caches remain
available for routing snapshots and eval snapshots.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def _num_experts_for(model: Any, pool: str) -> int:
    raw = model
    inner = _inner_model(raw)
    config = getattr(raw, "config", getattr(inner, "config", None))
    attn_bank = getattr(inner, "attn_bank", None)

    if pool == "branch":
        return 2
    if pool == "mlp_boundary":
        return int(
            getattr(config, "boundary_num_experts", None)
            or getattr(config, "num_experts", 0)
            or 0
        )
    if pool == "mlp_recurrent":
        return int(getattr(config, "num_experts", 0) or 0)
    if pool == "mlp":
        return int(getattr(config, "num_experts", 0) or 0)
    if pool in {"k", "v"} and attn_bank is not None:
        return int(getattr(attn_bank, "num_kv_experts", getattr(config, "num_attn_experts", 0)) or 0)
    if pool == "o" and attn_bank is not None:
        return int(getattr(attn_bank, "num_o_experts", getattr(config, "num_attn_experts", 0)) or 0)
    return int(getattr(config, "num_attn_experts", 0) or 0)


def _pool_label(name: str) -> str:
    if name.startswith("attn:"):
        return name.split(":", 1)[1]
    return name


def _pool_family(name: str) -> str:
    if name.startswith("attn:"):
        return "attn"
    if name == "mlp":
        return "mlp"
    if name == "branch":
        return "branch"
    return _safe_filename(name)


def _pool_output_dir(root: Path, pool: str) -> Path:
    if pool.startswith("attn:"):
        return root / "attn" / _safe_filename(_pool_label(pool))
    return root / _safe_filename(pool)


def _canonical_attn_route_name(attn_bank: Any, name: str) -> str:
    """Name shared attention routes by the projections they actually serve."""

    bundle = getattr(attn_bank, "routing_bundle", None)
    if name == "qk":
        if bundle == "qkvo":
            return "qkvo"
        if bundle == "qkv_o":
            return "qkv"
    if name == "v" and bundle == "qk_vo":
        return "vo"
    return name


def _include_branch_pool(inner: Any) -> bool:
    return (
        getattr(inner, "branch_balancing", None) != "fixed_alternating"
        and getattr(inner, "sanity_check_mode", None) != "alternating_global_moe"
    )


def _rows_from_selected(selected: Any) -> np.ndarray | None:
    arr = _to_numpy(selected)
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.int64, copy=False)
    return arr.reshape(-1, arr.shape[-1]).astype(np.int64, copy=False)


def _mask_for_rows(mask: Any, row_count: int) -> np.ndarray | None:
    arr = _to_numpy(mask)
    if arr is None or arr.size == 0:
        return None
    flat = arr.reshape(-1).astype(bool, copy=False)
    if flat.size == row_count:
        return flat
    if flat.size > 0 and row_count % flat.size == 0:
        return np.repeat(flat, row_count // flat.size)
    return flat[:row_count]


def _counts_from_rows(
    rows: np.ndarray | None,
    *,
    num_experts: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    counts = np.zeros(max(0, num_experts), dtype=np.int64)
    if rows is None or num_experts <= 0:
        return counts, 0, 0
    if mask is not None:
        n = min(rows.shape[0], mask.shape[0])
        rows = rows[:n][mask[:n]]
    if rows.size == 0:
        return counts, int(rows.shape[0]), 0

    flat = rows.reshape(-1)
    flat = flat[(flat >= 0) & (flat < num_experts)]
    if flat.size > 0:
        counts += np.bincount(flat, minlength=num_experts).astype(np.int64, copy=False)
    return counts, int(rows.shape[0]), int(flat.size)


def _summary_from_counts(
    *,
    pool: str,
    depth: int | None,
    counts: np.ndarray,
    active_tokens: int,
    assignments: int,
) -> dict[str, Any]:
    num_experts = int(counts.shape[0])
    ideal = 1.0 / num_experts if num_experts > 0 else None
    if assignments > 0 and num_experts > 0:
        fracs = counts.astype(np.float64) / float(assignments)
        mean = 1.0 / num_experts
        cv = float(fracs.std() / mean) if mean > 0 else None
        nonzero = fracs[fracs > 0]
        max_expert = int(fracs.argmax())
        max_fraction = float(fracs[max_expert])
        entropy = float(-(nonzero * np.log(nonzero)).sum()) if nonzero.size else 0.0
        norm_entropy = entropy / math.log(num_experts) if num_experts > 1 else 1.0
    else:
        fracs = np.zeros(num_experts, dtype=np.float64)
        cv = None
        max_expert = None
        max_fraction = None
        norm_entropy = None

    active_experts = int((counts > 0).sum())
    return {
        "pool": pool,
        "depth": "" if depth is None else int(depth),
        "num_experts": num_experts,
        "active_tokens": int(active_tokens),
        "assignments": int(assignments),
        "active_experts": active_experts,
        "active_expert_fraction": round(active_experts / num_experts, 6) if num_experts else None,
        "ideal_fraction": round(ideal, 8) if ideal is not None else None,
        "max_expert": max_expert,
        "max_fraction": round(max_fraction, 8) if max_fraction is not None else None,
        "max_over_ideal": round(max_fraction / ideal, 6) if max_fraction is not None and ideal else None,
        "cv": round(cv, 6) if cv is not None else None,
        "normalized_entropy": round(norm_entropy, 6) if norm_entropy is not None else None,
        "_counts": counts,
        "_fractions": fracs,
    }


def _append_pool(
    pools: dict[str, dict[str, Any]],
    *,
    pool: str,
    depth: int,
    counts: np.ndarray,
    active_tokens: int,
    assignments: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    entry = pools.setdefault(pool, {"per_depth": [], "global_counts": None})
    row = _summary_from_counts(
        pool=pool,
        depth=depth,
        counts=counts,
        active_tokens=active_tokens,
        assignments=assignments,
    )
    if metadata:
        row.update(metadata)
    entry["per_depth"].append(row)
    if entry["global_counts"] is None:
        entry["global_counts"] = counts.astype(np.int64, copy=True)
    else:
        entry["global_counts"] = entry["global_counts"] + counts
    entry["global_tokens"] = int(entry.get("global_tokens", 0)) + int(active_tokens)
    entry["global_assignments"] = int(entry.get("global_assignments", 0)) + int(assignments)


def _has_bias(router: Any) -> bool:
    return hasattr(router, "expert_bias") and _to_numpy(getattr(router, "expert_bias", None)) is not None


def _iter_router_collection(collection: Any):
    """Yield ``(depth, slot, router)`` from shared or per-depth router lists."""

    if collection is None:
        return
    if _has_bias(collection):
        yield None, None, collection
        return
    try:
        items = list(collection)
    except TypeError:
        return
    if not items:
        return
    if all(_has_bias(item) for item in items):
        for slot, router in enumerate(items):
            yield None, slot, router
        return
    for depth, item in enumerate(items):
        if _has_bias(item):
            yield depth, None, item
            continue
        try:
            subitems = list(item)
        except TypeError:
            continue
        for slot, router in enumerate(subitems):
            if _has_bias(router):
                yield depth, slot, router


def _append_bias_rows(
    rows: list[dict[str, Any]],
    *,
    pool: str,
    depth: int | None,
    slot: int | None,
    router: Any,
) -> None:
    bias = _to_numpy(getattr(router, "expert_bias", None))
    if bias is None or bias.size == 0:
        return
    flat = bias.reshape(-1).astype(np.float64, copy=False)
    depth_value = "" if depth is None else int(depth)
    slot_value = "" if slot is None else int(slot)
    if depth is None and slot is None:
        router_name = "shared"
    elif depth is None:
        router_name = f"slot_{int(slot):02d}"
    elif slot is None:
        router_name = f"depth_{int(depth):02d}"
    else:
        router_name = f"depth_{int(depth):02d}/slot_{int(slot):02d}"
    for expert_idx, value in enumerate(flat.tolist()):
        rows.append({
            "pool": pool,
            "family": _pool_family(pool),
            "depth": depth_value,
            "slot": slot_value,
            "router": router_name,
            "expert": int(expert_idx),
            "bias": round(float(value), 8),
        })


def _collect_bias_rows(model: Any, pools: set[str]) -> list[dict[str, Any]]:
    raw = model
    inner = _inner_model(raw)
    rows: list[dict[str, Any]] = []

    mlp_bank = getattr(inner, "mlp_bank", None)
    if mlp_bank is not None:
        for mlp_pool in ("mlp", "mlp_recurrent"):
            if mlp_pool not in pools:
                continue
            if hasattr(mlp_bank, "gates"):
                for depth, router in enumerate(getattr(mlp_bank, "gates")):
                    _append_bias_rows(rows, pool=mlp_pool, depth=depth, slot=None, router=router)
            else:
                _append_bias_rows(rows, pool=mlp_pool, depth=None, slot=None, router=getattr(mlp_bank, "gate", None))

    attn_bank = getattr(inner, "attn_bank", None)
    if attn_bank is not None:
        router_attrs = (
            ("q_routers", "q"),
            ("k_routers", "k"),
            ("v_routers", "v"),
            ("o_routers", "o"),
            ("qk_routers", "qk"),
            ("qkv_routers", "qkv"),
            ("qkvo_routers", "qkvo"),
            ("vo_routers", "vo"),
        )
        for attr, raw_name in router_attrs:
            if not hasattr(attn_bank, attr):
                continue
            route_name = _canonical_attn_route_name(attn_bank, raw_name)
            pool = f"attn:{route_name}"
            if pool not in pools:
                continue
            for depth, slot, router in _iter_router_collection(getattr(attn_bank, attr)):
                _append_bias_rows(rows, pool=pool, depth=depth, slot=slot, router=router)

    if "branch" in pools and _include_branch_pool(inner):
        branch_routers = getattr(inner, "branch_routers", None)
        if branch_routers is not None:
            for depth, router in enumerate(branch_routers):
                _append_bias_rows(rows, pool="branch", depth=depth, slot=None, router=router)
        else:
            _append_bias_rows(rows, pool="branch", depth=None, slot=None, router=getattr(inner, "branch_router", None))

    if hasattr(inner, "recurrent_blocks"):
        if "mlp_boundary" in pools:
            for depth, block in enumerate(getattr(inner, "prelude_blocks", []) or []):
                router = getattr(getattr(block, "mlp", None), "gate", None)
                _append_bias_rows(
                    rows,
                    pool="mlp_boundary",
                    depth=depth,
                    slot=None,
                    router=router,
                )
            coda_offset = int(getattr(getattr(inner, "config", None), "prelude_layers", 0) or 0)
            coda_offset += int(getattr(getattr(inner, "config", None), "recurrent_layers", 0) or 0)
            for idx, block in enumerate(getattr(inner, "coda_blocks", []) or []):
                router = getattr(getattr(block, "mlp", None), "gate", None)
                _append_bias_rows(
                    rows,
                    pool="mlp_boundary",
                    depth=coda_offset + idx,
                    slot=None,
                    router=router,
                )
        if "mlp_recurrent" in pools:
            shared = getattr(inner, "shared_recurrent_bias", None)
            if shared is not None:
                _append_bias_rows(
                    rows,
                    pool="mlp_recurrent",
                    depth=None,
                    slot=None,
                    router=shared,
                )
            else:
                offset = int(getattr(getattr(inner, "config", None), "prelude_layers", 0) or 0)
                for idx, block in enumerate(getattr(inner, "recurrent_blocks", []) or []):
                    router = getattr(getattr(block, "mlp", None), "gate", None)
                    _append_bias_rows(
                        rows,
                        pool="mlp_recurrent",
                        depth=offset + idx,
                        slot=None,
                        router=router,
                    )

    return rows


def _bias_pool_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([float(row["bias"]) for row in rows], dtype=np.float64)
    experts = sorted({int(row["expert"]) for row in rows})
    routers = sorted({str(row["router"]) for row in rows})
    if values.size == 0:
        summary = {
            "num_routers": 0,
            "num_experts": 0,
            "num_values": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "mean_abs": None,
            "max_abs": None,
            "positive": 0,
            "negative": 0,
            "zero": 0,
        }
    else:
        summary = {
            "num_routers": len(routers),
            "num_experts": len(experts),
            "num_values": int(values.size),
            "mean": round(float(values.mean()), 8),
            "std": round(float(values.std()), 8),
            "min": round(float(values.min()), 8),
            "max": round(float(values.max()), 8),
            "mean_abs": round(float(np.abs(values).mean()), 8),
            "max_abs": round(float(np.abs(values).max()), 8),
            "positive": int((values > 0).sum()),
            "negative": int((values < 0).sum()),
            "zero": int((values == 0).sum()),
        }
    return {
        "rows": rows,
        "summary": summary,
        "_values": values,
    }


def _build_bias_payload(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pool[str(row["pool"])].append(row)
    return {pool: _bias_pool_payload(pool_rows) for pool, pool_rows in sorted(by_pool.items())}


def _collapse_bias_rows_to_global(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report bank-level bias as one global vector per pool.

    MoE-Everything can instantiate one router module per layer/head slot while
    using a single global load-balancing bias per shared expert bank. The model
    stores that vector on every router for runtime convenience; artifacts should
    show the conceptual bank-level vector, not duplicate identical rows by
    depth/slot.
    """

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["pool"]), int(row["expert"]))].append(row)

    out: list[dict[str, Any]] = []
    for (pool, expert), group in sorted(grouped.items()):
        values = [float(row["bias"]) for row in group]
        family = group[0].get("family", _pool_family(pool))
        out.append({
            "pool": pool,
            "family": family,
            "depth": "",
            "slot": "",
            "router": "global",
            "expert": expert,
            "bias": round(float(np.mean(values)), 8),
        })
    return out


def _collapse_bias_rows_for_model(model: Any) -> bool:
    raw = model
    inner = _inner_model(raw)
    config = getattr(raw, "config", getattr(inner, "config", None))
    return (
        bool(getattr(config, "global_router_update", False))
        and getattr(config, "model_type", None) in {"moe_everything", "recurrent_moe_everything"}
    )


def build_load_balance_artifacts(model: Any, *, step: int) -> dict[str, Any] | None:
    """Build load-balance summaries from the latest model routing snapshot."""

    raw = model
    inner = _inner_model(raw)
    pools: dict[str, dict[str, Any]] = {}

    mlp_selected = getattr(inner, "_all_mlp_selected_experts", None) or []
    mlp_masks = getattr(inner, "_all_mlp_token_masks", None) or []
    mlp_scopes = getattr(inner, "_all_mlp_router_scopes", None) or []
    per_pool_depth: dict[str, int] = defaultdict(int)
    for depth, selected in enumerate(mlp_selected):
        rows = _rows_from_selected(selected)
        if rows is None:
            continue
        scope = mlp_scopes[depth] if depth < len(mlp_scopes) else {}
        pool = str(scope.get("pool", "mlp")) if isinstance(scope, dict) else "mlp"
        num_experts = (
            int(scope.get("num_experts", 0) or 0)
            if isinstance(scope, dict)
            else 0
        )
        if num_experts <= 0:
            num_experts = _num_experts_for(raw, pool)
        mask = _mask_for_rows(mlp_masks[depth], rows.shape[0]) if depth < len(mlp_masks) else None
        counts, tokens, assignments = _counts_from_rows(rows, num_experts=num_experts, mask=mask)
        pool_depth = per_pool_depth[pool]
        per_pool_depth[pool] += 1
        metadata = None
        if pool == "mlp_recurrent":
            cfg = getattr(inner, "config", None)
            recurrent_layers = max(1, int(getattr(cfg, "recurrent_layers", 0) or 0))
            prelude_layers = int(getattr(cfg, "prelude_layers", 0) or 0)
            block_index = (
                scope.get("block_index")
                if isinstance(scope, dict)
                else None
            )
            if block_index is None:
                logical_block = pool_depth % recurrent_layers
            else:
                logical_block = int(block_index) - prelude_layers
                logical_block = max(0, min(recurrent_layers - 1, logical_block))
            if isinstance(scope, dict) and "hrm_cycle" in scope:
                metadata = {
                    "_loop": int(scope.get("hrm_cycle", 0) or 0),
                    "_block": logical_block,
                    "_block_index": block_index,
                    "_hrm_module": scope.get("hrm_module", ""),
                    "_hrm_l_index": scope.get("hrm_l_index", ""),
                    "_hrm_local_block": scope.get("hrm_local_block", ""),
                }
            else:
                metadata = {
                    "_loop": pool_depth // recurrent_layers,
                    "_block": logical_block,
                    "_block_index": block_index,
                }
        _append_pool(
            pools,
            pool=pool,
            depth=pool_depth,
            counts=counts,
            active_tokens=tokens,
            assignments=assignments,
            metadata=metadata,
        )

    attn_bank = getattr(inner, "attn_bank", None)
    attn_info = getattr(inner, "_all_attn_router_info", None) or []
    for depth, depth_info in enumerate(attn_info):
        if not isinstance(depth_info, dict):
            continue
        for name, info in depth_info.items():
            if not isinstance(info, dict):
                continue
            rows = _rows_from_selected(info.get("selected_experts"))
            if rows is None:
                continue
            route_name = _canonical_attn_route_name(attn_bank, str(name)) if attn_bank is not None else str(name)
            pool_name = f"attn:{route_name}"
            pool_key = _pool_label(pool_name)
            num_experts = _num_experts_for(raw, pool_key)
            mask = _mask_for_rows(info.get("token_mask"), rows.shape[0])
            counts, tokens, assignments = _counts_from_rows(rows, num_experts=num_experts, mask=mask)
            _append_pool(
                pools,
                pool=pool_name,
                depth=depth,
                counts=counts,
                active_tokens=tokens,
                assignments=assignments,
            )

    if _include_branch_pool(inner):
        branch_selected = getattr(inner, "_all_branch_selected_experts", None) or []
        for depth, selected in enumerate(branch_selected):
            rows = _rows_from_selected(selected)
            if rows is None:
                continue
            counts, tokens, assignments = _counts_from_rows(rows, num_experts=2)
            _append_pool(
                pools,
                pool="branch",
                depth=depth,
                counts=counts,
                active_tokens=tokens,
                assignments=assignments,
            )

    if not pools:
        return None

    summary_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    output_pools: dict[str, dict[str, Any]] = {}

    for pool, entry in sorted(pools.items()):
        global_counts = entry["global_counts"]
        global_summary = _summary_from_counts(
            pool=pool,
            depth=None,
            counts=global_counts,
            active_tokens=int(entry.get("global_tokens", 0)),
            assignments=int(entry.get("global_assignments", 0)),
        )
        output_pools[pool] = {
            "global": global_summary,
            "per_depth": entry["per_depth"],
        }
        summary_rows.append(global_summary)
        summary_rows.extend(entry["per_depth"])

        for scope, row in [("global", global_summary)] + [("depth", r) for r in entry["per_depth"]]:
            counts = row["_counts"]
            fracs = row["_fractions"]
            for expert_idx, (count, frac) in enumerate(zip(counts.tolist(), fracs.tolist())):
                expert_rows.append({
                    "pool": pool,
                    "scope": scope,
                    "depth": row["depth"],
                    "expert": expert_idx,
                    "count": int(count),
                    "fraction": round(float(frac), 8),
                })

    bias_rows = _collect_bias_rows(raw, set(output_pools))
    if _collapse_bias_rows_for_model(raw):
        bias_rows = _collapse_bias_rows_to_global(bias_rows)
    bias_pools = _build_bias_payload(bias_rows)

    return {
        "step": int(step),
        "kind": "moe_load_balance",
        "summary_rows": summary_rows,
        "expert_rows": expert_rows,
        "bias_rows": bias_rows,
        "bias_pools": bias_pools,
        "pools": output_pools,
    }


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


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


def _write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "pool",
        "depth",
        "num_experts",
        "active_tokens",
        "assignments",
        "active_experts",
        "active_expert_fraction",
        "ideal_fraction",
        "max_expert",
        "max_fraction",
        "max_over_ideal",
        "cv",
        "normalized_entropy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in payload.get("summary_rows", []):
            writer.writerow({field: _public_row(row).get(field) for field in fields})


def _write_expert_load_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = ["pool", "scope", "depth", "expert", "count", "fraction"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload.get("expert_rows", []))


def _write_bias_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["pool", "family", "depth", "slot", "router", "expert", "bias"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bias_summary_rows(payload: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for pool, entry in sorted(payload.get("bias_pools", {}).items()):
        summary = entry.get("summary", {})
        rows.append([
            pool,
            summary.get("num_routers"),
            summary.get("num_experts"),
            summary.get("num_values"),
            _md_number(summary.get("mean")),
            _md_number(summary.get("std")),
            _md_number(summary.get("min")),
            _md_number(summary.get("max")),
            _md_number(summary.get("mean_abs")),
            _md_number(summary.get("max_abs")),
        ])
    return rows


def _write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Load Balance Snapshot",
        "",
        f"- Step: `{payload.get('step')}`",
        "",
        (
            "This folder is an overview. Detailed pool artifacts live under "
            "`attn/<route>/`, `mlp/`, and `branch/` when those pools are active."
        ),
        "",
        "## Global Pool Summary",
        "",
    ]

    global_rows = [
        row for row in payload.get("summary_rows", [])
        if row.get("depth") == ""
    ]
    lines.append(_md_table(
        [
            "Pool",
            "Experts",
            "Active tokens",
            "Assignments",
            "Active experts",
            "Active %",
            "Max expert",
            "Max/ideal",
            "CV",
            "Entropy",
        ],
        [
            [
                row.get("pool"),
                row.get("num_experts"),
                row.get("active_tokens"),
                row.get("assignments"),
                row.get("active_experts"),
                _md_number(row.get("active_expert_fraction")),
                row.get("max_expert"),
                _md_number(row.get("max_over_ideal")),
                _md_number(row.get("cv")),
                _md_number(row.get("normalized_entropy")),
            ]
            for row in global_rows
        ],
    ))

    expert_rows = payload.get("expert_rows", [])
    top_expert_rows = []
    for pool in sorted({row.get("pool") for row in expert_rows}):
        pool_rows = [
            row for row in expert_rows
            if row.get("pool") == pool and row.get("scope") == "global" and int(row.get("count") or 0) > 0
        ]
        pool_rows.sort(key=lambda row: (float(row.get("fraction") or 0.0), int(row.get("count") or 0)), reverse=True)
        for row in pool_rows[:12]:
            top_expert_rows.append([
                row.get("pool"),
                row.get("expert"),
                row.get("count"),
                _md_number(row.get("fraction")),
            ])
    lines.extend(["", "## Top Loaded Experts", ""])
    lines.append(_md_table(["Pool", "Expert", "Count", "Fraction"], top_expert_rows))

    lines.extend(["", "## Bias Summary", ""])
    lines.append(_md_table(
        [
            "Pool",
            "Routers",
            "Experts",
            "Values",
            "Mean",
            "Std",
            "Min",
            "Max",
            "Mean |bias|",
            "Max |bias|",
        ],
        _bias_summary_rows(payload),
    ))

    for pool, entry in sorted(payload.get("pools", {}).items()):
        lines.extend(["", f"## Per-Depth Summary: `{pool}`", ""])
        depth_rows = []
        for row in entry.get("per_depth", []):
            depth_rows.append([
                row.get("depth"),
                row.get("active_tokens"),
                row.get("assignments"),
                row.get("active_experts"),
                _md_number(row.get("active_expert_fraction")),
                row.get("max_expert"),
                _md_number(row.get("max_over_ideal")),
                _md_number(row.get("cv")),
                _md_number(row.get("normalized_entropy")),
            ])
        lines.append(_md_table(
            [
                "Depth",
                "Active tokens",
                "Assignments",
                "Active experts",
                "Active %",
                "Max expert",
                "Max/ideal",
                "CV",
                "Entropy",
            ],
            depth_rows,
        ))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_pool_summary_csv(path: Path, pool: str, entry: dict[str, Any]) -> None:
    fields = [
        "pool",
        "depth",
        "num_experts",
        "active_tokens",
        "assignments",
        "active_experts",
        "active_expert_fraction",
        "ideal_fraction",
        "max_expert",
        "max_fraction",
        "max_over_ideal",
        "cv",
        "normalized_entropy",
    ]
    rows = [entry.get("global", {})] + list(entry.get("per_depth", []))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _public_row(row).get(field) for field in fields})


def _write_pool_expert_csv(path: Path, pool: str, payload: dict[str, Any]) -> None:
    fields = ["pool", "scope", "depth", "expert", "count", "fraction"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in payload.get("expert_rows", []):
            if row.get("pool") == pool:
                writer.writerow(row)


def _top_expert_rows_for_pool(pool: str, payload: dict[str, Any]) -> list[list[Any]]:
    pool_rows = [
        row for row in payload.get("expert_rows", [])
        if row.get("pool") == pool and row.get("scope") == "global" and int(row.get("count") or 0) > 0
    ]
    pool_rows.sort(key=lambda row: (float(row.get("fraction") or 0.0), int(row.get("count") or 0)), reverse=True)
    return [
        [row.get("expert"), row.get("count"), _md_number(row.get("fraction"))]
        for row in pool_rows[:16]
    ]


def _top_bias_rows(rows: list[dict[str, Any]], *, reverse: bool) -> list[list[Any]]:
    ordered = sorted(rows, key=lambda row: float(row.get("bias") or 0.0), reverse=reverse)
    return [
        [
            row.get("router"),
            row.get("depth"),
            row.get("slot"),
            row.get("expert"),
            _md_number(row.get("bias")),
        ]
        for row in ordered[:12]
    ]


def _write_pool_summary_md(path: Path, pool: str, payload: dict[str, Any]) -> None:
    entry = payload["pools"][pool]
    global_row = entry.get("global", {})
    bias_entry = payload.get("bias_pools", {}).get(pool)
    lines = [
        f"# Load Balance: `{pool}`",
        "",
        f"- Step: `{payload.get('step')}`",
        "",
        (
            "Files in this folder: `summary.md`, `summary.csv`, `expert_load.csv`, "
            "`heatmap.png`, `global_histogram.png`, `per_layer_histograms.png`, "
            "plus `bias.*` files when router bias exists."
        ),
        "",
        "## Global Load",
        "",
    ]
    lines.append(_md_table(
        [
            "Experts",
            "Active tokens",
            "Assignments",
            "Active experts",
            "Active %",
            "Max expert",
            "Max/ideal",
            "CV",
            "Entropy",
        ],
        [[
            global_row.get("num_experts"),
            global_row.get("active_tokens"),
            global_row.get("assignments"),
            global_row.get("active_experts"),
            _md_number(global_row.get("active_expert_fraction")),
            global_row.get("max_expert"),
            _md_number(global_row.get("max_over_ideal")),
            _md_number(global_row.get("cv")),
            _md_number(global_row.get("normalized_entropy")),
        ]],
    ))
    lines.extend(["", "## Top Loaded Experts", ""])
    lines.append(_md_table(["Expert", "Count", "Fraction"], _top_expert_rows_for_pool(pool, payload)))
    lines.extend(["", "## Per-Depth Load", ""])
    lines.append(_md_table(
        [
            "Depth",
            "Active tokens",
            "Assignments",
            "Active experts",
            "Active %",
            "Max expert",
            "Max/ideal",
            "CV",
            "Entropy",
        ],
        [[
            row.get("depth"),
            row.get("active_tokens"),
            row.get("assignments"),
            row.get("active_experts"),
            _md_number(row.get("active_expert_fraction")),
            row.get("max_expert"),
            _md_number(row.get("max_over_ideal")),
            _md_number(row.get("cv")),
            _md_number(row.get("normalized_entropy")),
        ] for row in entry.get("per_depth", [])],
    ))

    if bias_entry is not None:
        summary = bias_entry.get("summary", {})
        lines.extend(["", "## Router Bias", ""])
        lines.append(_md_table(
            ["Routers", "Experts", "Values", "Mean", "Std", "Min", "Max", "Mean |bias|", "Max |bias|"],
            [[
                summary.get("num_routers"),
                summary.get("num_experts"),
                summary.get("num_values"),
                _md_number(summary.get("mean")),
                _md_number(summary.get("std")),
                _md_number(summary.get("min")),
                _md_number(summary.get("max")),
                _md_number(summary.get("mean_abs")),
                _md_number(summary.get("max_abs")),
            ]],
        ))
        lines.extend(["", "### Most Positive Biases", ""])
        lines.append(_md_table(["Router", "Depth", "Slot", "Expert", "Bias"], _top_bias_rows(bias_entry.get("rows", []), reverse=True)))
        lines.extend(["", "### Most Negative Biases", ""])
        lines.append(_md_table(["Router", "Depth", "Slot", "Expert", "Bias"], _top_bias_rows(bias_entry.get("rows", []), reverse=False)))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _plot_summary(out_dir: Path, payload: dict[str, Any]) -> None:
    global_rows = [
        row for row in payload.get("summary_rows", [])
        if row.get("depth") == "" and row.get("max_over_ideal") is not None
    ]
    if not global_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [str(row["pool"]) for row in global_rows]
    max_over = np.array([float(row["max_over_ideal"]) for row in global_rows])
    entropy = np.array([
        float(row["normalized_entropy"]) if row.get("normalized_entropy") is not None else np.nan
        for row in global_rows
    ])
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.38 * len(labels) + 1.5)))
    axes[0].barh(y, max_over, color="#4c78a8")
    axes[0].axvline(1.0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Max expert fraction / ideal")
    axes[0].set_title("Peak Load Skew")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y, entropy, color="#59a14f")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Normalized entropy")
    axes[1].set_title("Expert Usage Entropy")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(f"Load Balance Summary (step {payload.get('step')})")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=140)
    plt.close(fig)


def _safe_filename(pool: str) -> str:
    return pool.replace(":", "_").replace("/", "_")


def _plot_pool_heatmap(out_dir: Path, pool: str, entry: dict[str, Any], *, step: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = entry.get("per_depth", [])
    if not rows:
        return
    mat = np.stack([row["_fractions"] for row in rows], axis=0)
    if mat.size == 0:
        return
    num_depths, num_experts = mat.shape
    fig_w = max(7.0, min(22.0, num_experts * 0.16 + 3.0))
    fig_h = max(4.0, num_depths * 0.32 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Assignment fraction")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Depth")
    ax.set_title(f"{pool} Load Heatmap (step {step})")
    ax.set_yticks(np.arange(num_depths))
    ax.set_yticklabels([str(row["depth"]) for row in rows])
    xtick_step = max(1, num_experts // 16)
    ax.set_xticks(np.arange(0, num_experts, xtick_step))
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=140)
    plt.close(fig)


def _plot_single_histogram(
    ax: Any,
    *,
    fractions: np.ndarray,
    title: str,
    color: str = "#4c78a8",
    show_legend: bool = True,
) -> None:
    num_experts = int(fractions.shape[0])
    if num_experts <= 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No experts", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(num_experts)
    ideal = 1.0 / num_experts
    active = int((fractions > 0).sum())
    ax.bar(x, fractions, color=color, width=0.8)
    ax.axhline(ideal, color="#333333", linestyle="--", linewidth=1, label=f"ideal={ideal:.4f}")
    ax.set_title(f"{title} ({active}/{num_experts} active)")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Assignment fraction")
    xtick_step = max(1, num_experts // 16)
    ax.set_xticks(np.arange(0, num_experts, xtick_step))
    ax.grid(axis="y", alpha=0.2)
    if show_legend:
        ax.legend(fontsize=8)


def _recurrent_depth_grid(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int] | None:
    depth_rows = entry.get("per_depth", [])
    if not depth_rows:
        return None
    if not all("_loop" in row and "_block" in row for row in depth_rows):
        return None
    max_loop = max(int(row["_loop"]) for row in depth_rows)
    max_block = max(int(row["_block"]) for row in depth_rows)
    return depth_rows, max_loop + 1, max_block + 1


def _plot_recurrent_block_heatmaps(
    out_dir: Path,
    pool: str,
    entry: dict[str, Any],
    *,
    step: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    grid = _recurrent_depth_grid(entry)
    if grid is None:
        return
    depth_rows, num_loops, num_blocks = grid
    first_fracs = np.asarray(depth_rows[0].get("_fractions", []), dtype=np.float64)
    num_experts = int(first_fracs.shape[0])
    if num_experts <= 0:
        return

    for block_idx in range(num_blocks):
        mat = np.zeros((num_loops, num_experts), dtype=np.float64)
        has_row = np.zeros(num_loops, dtype=bool)
        for row in depth_rows:
            if int(row["_block"]) != block_idx:
                continue
            loop_idx = int(row["_loop"])
            fracs = np.asarray(row.get("_fractions", []), dtype=np.float64)
            if fracs.shape[0] != num_experts:
                continue
            mat[loop_idx] = fracs
            has_row[loop_idx] = True
        if not has_row.any():
            continue

        fig_w = max(8.0, min(24.0, num_experts * 0.12 + 4.0))
        fig_h = max(3.5, min(40.0, num_loops * 0.35 + 2.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="YlOrRd")
        fig.colorbar(im, ax=ax, label="Assignment fraction")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Loop")
        ax.set_title(f"{pool} Block {block_idx} Load Heatmap (step {step})")
        ax.set_yticks(np.arange(num_loops))
        ax.set_yticklabels([str(i) for i in range(num_loops)])
        xtick_step = max(1, num_experts // 16)
        ax.set_xticks(np.arange(0, num_experts, xtick_step))
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_block_{block_idx:02d}.png", dpi=140)
        plt.close(fig)


def _plot_pool_histograms(out_dir: Path, pool: str, entry: dict[str, Any], *, step: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    global_row = entry.get("global", {})
    global_fracs = np.asarray(global_row.get("_fractions", []), dtype=np.float64)
    if global_fracs.size > 0:
        fig, ax = plt.subplots(figsize=(max(8.0, min(22.0, global_fracs.size * 0.12 + 4.0)), 4.8))
        _plot_single_histogram(
            ax,
            fractions=global_fracs,
            title=f"{pool} Global Expert Load (step {step})",
        )
        fig.tight_layout()
        fig.savefig(out_dir / "global_histogram.png", dpi=140)
        plt.close(fig)

    depth_rows = entry.get("per_depth", [])
    if not depth_rows:
        return
    first_fracs = np.asarray(depth_rows[0].get("_fractions", []), dtype=np.float64)
    grid = _recurrent_depth_grid(entry)
    if grid is not None:
        depth_rows, rows, cols = grid
        fig_w = max(14.0, min(34.0, first_fracs.size * 0.025 * cols + 4.0))
        fig_h = max(4.0, rows * 2.2 + 1.5)
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for row in depth_rows:
            loop_idx = int(row["_loop"])
            block_idx = int(row["_block"])
            ax = axes[loop_idx, block_idx]
            ax.set_visible(True)
            fracs = np.asarray(row.get("_fractions", []), dtype=np.float64)
            _plot_single_histogram(
                ax,
                fractions=fracs,
                title=f"L{loop_idx} B{block_idx}",
                color="#4c78a8",
                show_legend=False,
            )
            if block_idx == 0:
                ax.set_ylabel(f"Loop {loop_idx}")
            if loop_idx == 0:
                ax.set_title(f"Block {block_idx}")
        fig.suptitle(
            f"{pool} Expert Load Histograms by Recurrent Loop (step {step})",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / "per_layer_histograms.png", dpi=140)
        plt.close(fig)
        return

    num_depths = len(depth_rows)
    cols = min(4, num_depths)
    rows = int(math.ceil(num_depths / cols))
    fig_w = max(10.0, min(24.0, first_fracs.size * 0.08 * cols + 3.0))
    fig_h = max(4.0, rows * 3.2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for plot_idx, row in enumerate(depth_rows):
        ax = axes.flat[plot_idx]
        ax.set_visible(True)
        fracs = np.asarray(row.get("_fractions", []), dtype=np.float64)
        _plot_single_histogram(
            ax,
            fractions=fracs,
            title=f"Depth {row.get('depth')}",
            color="#4c78a8",
        )
    fig.suptitle(f"{pool} Per-Layer Expert Load Histograms (step {step})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "per_layer_histograms.png", dpi=140)
    plt.close(fig)


def _plot_bias_artifacts(out_dir: Path, pool: str, entry: dict[str, Any], *, step: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = entry.get("rows", [])
    values = np.asarray([float(row["bias"]) for row in rows], dtype=np.float64)
    if values.size == 0:
        return

    num_experts = max(int(row["expert"]) for row in rows) + 1
    by_expert: list[list[float]] = [[] for _ in range(num_experts)]
    for row in rows:
        by_expert[int(row["expert"])].append(float(row["bias"]))
    expert_bias = np.array([np.mean(v) if v else np.nan for v in by_expert], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(8.0, min(22.0, num_experts * 0.12 + 4.0)), 4.8))
    x = np.arange(num_experts)
    plot_bias = np.nan_to_num(expert_bias, nan=0.0)
    colors = np.where(plot_bias < 0.0, "#d62728", "#2ca02c")
    ax.bar(x, plot_bias, color=colors, width=0.8)
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_title(f"{pool} Global Bias By Expert (step {step})")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Global bias")
    ax.set_xticks(np.arange(0, num_experts, max(1, num_experts // 16)))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "bias_by_expert.png", dpi=140)
    plt.close(fig)


def save_load_balance_artifacts(
    model: Any,
    *,
    step_dir: str,
    step: int,
) -> bool:
    payload = build_load_balance_artifacts(model, step=step)
    if payload is None:
        return False

    out_dir = Path(step_dir) / "load_balancing"
    os.makedirs(out_dir, exist_ok=True)
    _write_summary_csv(out_dir / "summary.csv", payload)
    _write_expert_load_csv(out_dir / "expert_load.csv", payload)
    if payload.get("bias_rows"):
        _write_bias_csv(out_dir / "bias.csv", payload["bias_rows"])
    _write_summary_md(out_dir / "summary.md", payload)
    _plot_summary(out_dir, payload)
    for pool, entry in sorted(payload.get("pools", {}).items()):
        pool_dir = _pool_output_dir(out_dir, pool)
        os.makedirs(pool_dir, exist_ok=True)
        _write_pool_summary_csv(pool_dir / "summary.csv", pool, entry)
        _write_pool_expert_csv(pool_dir / "expert_load.csv", pool, payload)
        _write_pool_summary_md(pool_dir / "summary.md", pool, payload)
        _plot_pool_heatmap(pool_dir, pool, entry, step=int(payload.get("step", step)))
        _plot_recurrent_block_heatmaps(pool_dir, pool, entry, step=int(payload.get("step", step)))
        _plot_pool_histograms(pool_dir, pool, entry, step=int(payload.get("step", step)))
        bias_entry = payload.get("bias_pools", {}).get(pool)
        if bias_entry is not None:
            _write_bias_csv(pool_dir / "bias.csv", bias_entry.get("rows", []))
            _plot_bias_artifacts(pool_dir, pool, bias_entry, step=int(payload.get("step", step)))
    return True
