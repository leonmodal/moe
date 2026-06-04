"""Diagnostics for recurrent MoE loop activity and routing stability."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _inner_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().cpu().reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (float, int)) and math.isfinite(float(row[key]))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _last(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            return float(value)
    return None


def _loop_rows(inner: Any, *, step: int) -> list[dict[str, Any]]:
    entries = getattr(inner, "_recurrent_loop_diagnostics", None) or []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row = {
            "step": int(step),
            "loop": int(entry.get("loop", len(rows))),
            "phase": str(entry.get("phase", "")),
        }
        for key in (
            "residual_rms",
            "relative_residual_rms",
            "cosine",
            "last_token_residual_rms",
            "last_token_relative_residual_rms",
            "last_token_cosine",
        ):
            value = _scalar(entry.get(key))
            if value is not None:
                row[key] = value
        rows.append(row)
    return rows


def _last_token_indices(input_ids: torch.Tensor | None, total_tokens: int, *, device: torch.device) -> torch.Tensor | None:
    if input_ids is None or input_ids.ndim != 2:
        return None
    batch, seq_len = input_ids.shape
    if int(batch * seq_len) != int(total_tokens):
        return None
    return torch.arange(seq_len - 1, total_tokens, seq_len, device=device)


def _route_agreement(
    prev_selected: torch.Tensor,
    next_selected: torch.Tensor,
    *,
    indices: torch.Tensor | None = None,
) -> dict[str, float] | None:
    if prev_selected is None or next_selected is None:
        return None
    if prev_selected.ndim == 1:
        prev_selected = prev_selected.reshape(-1, 1)
    if next_selected.ndim == 1:
        next_selected = next_selected.reshape(-1, 1)
    if prev_selected.ndim != 2 or next_selected.ndim != 2:
        return None
    if prev_selected.shape != next_selected.shape or prev_selected.shape[0] == 0:
        return None

    prev = prev_selected.detach()
    nxt = next_selected.detach().to(device=prev.device)
    if indices is not None:
        idx = indices.to(device=prev.device)
        if idx.numel() == 0:
            return None
        prev = prev.index_select(0, idx)
        nxt = nxt.index_select(0, idx)
    if prev.shape[0] == 0:
        return None

    with torch.no_grad():
        top1_agreement = (prev[:, 0] == nxt[:, 0]).float().mean()
        intersection = (
            prev.unsqueeze(2) == nxt.unsqueeze(1)
        ).any(dim=2).sum(dim=1).float()
        union = (
            prev.shape[1] + nxt.shape[1] - intersection
        ).clamp_min(1.0)
        jaccard = (intersection / union).mean()

    top1 = float(top1_agreement.cpu().item())
    jac = float(jaccard.cpu().item())
    return {
        "top1_agreement": top1,
        "top1_switch_rate": 1.0 - top1,
        "jaccard": jac,
        "jaccard_distance": 1.0 - jac,
    }


def _routing_rows(
    inner: Any,
    *,
    step: int,
    input_ids: torch.Tensor | None,
) -> list[dict[str, Any]]:
    selected = getattr(inner, "_all_mlp_selected_experts", None) or []
    scopes = getattr(inner, "_all_mlp_router_scopes", None) or []
    config = getattr(inner, "config", None)
    if getattr(config, "recurrent_loop", "flat") == "hrm":
        return []
    recurrent_layers = int(getattr(config, "recurrent_layers", 0) or 0)
    if recurrent_layers <= 0 or not selected:
        return []

    prelude_layers = int(getattr(config, "prelude_layers", 0) or 0)
    recurrent_items: list[dict[str, Any]] = []
    for depth, selected_experts in enumerate(selected):
        if depth >= len(scopes):
            continue
        scope = scopes[depth] or {}
        if not isinstance(scope, dict) or scope.get("pool") != "mlp_recurrent":
            continue
        block_index = scope.get("block_index")
        if block_index is None:
            block = len(recurrent_items) % recurrent_layers
        else:
            block = int(block_index) - prelude_layers
            block = max(0, min(recurrent_layers - 1, block))
        loop = len(recurrent_items) // recurrent_layers
        recurrent_items.append({
            "loop": loop,
            "block": block,
            "selected": selected_experts,
        })

    by_loop_block = {
        (int(item["loop"]), int(item["block"])): item["selected"]
        for item in recurrent_items
    }
    loop_indices = sorted({int(item["loop"]) for item in recurrent_items})
    rows: list[dict[str, Any]] = []
    for loop in loop_indices:
        next_loop = loop + 1
        if next_loop not in loop_indices:
            continue
        for block in range(recurrent_layers):
            prev_selected = by_loop_block.get((loop, block))
            next_selected = by_loop_block.get((next_loop, block))
            if prev_selected is None or next_selected is None:
                continue
            total_tokens = int(prev_selected.shape[0])
            last_indices = _last_token_indices(
                input_ids,
                total_tokens,
                device=prev_selected.device,
            )
            all_metrics = _route_agreement(prev_selected, next_selected)
            if all_metrics is None:
                continue
            last_metrics = _route_agreement(
                prev_selected,
                next_selected,
                indices=last_indices,
            )
            row = {
                "step": int(step),
                "loop": int(loop),
                "next_loop": int(next_loop),
                "block": int(block),
                **all_metrics,
            }
            if last_metrics is not None:
                row.update({
                    f"last_token_{key}": value
                    for key, value in last_metrics.items()
                })
            rows.append(row)
    return rows


def _scope_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hrm_l_routing_rows(
    inner: Any,
    *,
    step: int,
    input_ids: torch.Tensor | None,
) -> list[dict[str, Any]]:
    selected = getattr(inner, "_all_mlp_selected_experts", None) or []
    scopes = getattr(inner, "_all_mlp_router_scopes", None) or []
    config = getattr(inner, "config", None)
    if getattr(config, "recurrent_loop", "flat") != "hrm" or not selected:
        return []

    prelude_layers = int(getattr(config, "prelude_layers", 0) or 0)
    by_cycle_l_block: dict[tuple[int, int, int], dict[str, Any]] = {}
    l_indices_by_cycle: dict[int, set[int]] = defaultdict(set)
    blocks_by_cycle_l: dict[tuple[int, int], set[int]] = defaultdict(set)

    for depth, selected_experts in enumerate(selected):
        if depth >= len(scopes):
            continue
        scope = scopes[depth] or {}
        if (
            not isinstance(scope, dict)
            or scope.get("pool") != "mlp_recurrent"
            or scope.get("hrm_module") != "L"
        ):
            continue
        cycle = _scope_int(scope.get("hrm_cycle"))
        l_index = _scope_int(scope.get("hrm_l_index"))
        if cycle is None or l_index is None:
            continue
        block = _scope_int(scope.get("hrm_local_block"))
        block_index = _scope_int(scope.get("block_index"))
        if block is None:
            if block_index is None:
                continue
            block = max(0, block_index - prelude_layers)
        key = (cycle, l_index, block)
        by_cycle_l_block[key] = {
            "selected": selected_experts,
            "block_index": "" if block_index is None else block_index,
        }
        l_indices_by_cycle[cycle].add(l_index)
        blocks_by_cycle_l[(cycle, l_index)].add(block)

    rows: list[dict[str, Any]] = []
    transition_index = 0
    for cycle in sorted(l_indices_by_cycle):
        l_indices = sorted(l_indices_by_cycle[cycle])
        for transition_in_cycle, (l_index, next_l_index) in enumerate(
            zip(l_indices, l_indices[1:])
        ):
            blocks = sorted(
                blocks_by_cycle_l[(cycle, l_index)]
                & blocks_by_cycle_l[(cycle, next_l_index)]
            )
            wrote_transition = False
            for block in blocks:
                prev_record = by_cycle_l_block.get((cycle, l_index, block))
                next_record = by_cycle_l_block.get((cycle, next_l_index, block))
                if prev_record is None or next_record is None:
                    continue
                prev_selected = prev_record["selected"]
                next_selected = next_record["selected"]
                all_metrics = _route_agreement(prev_selected, next_selected)
                if all_metrics is None:
                    continue
                total_tokens = int(prev_selected.shape[0])
                last_indices = _last_token_indices(
                    input_ids,
                    total_tokens,
                    device=prev_selected.device,
                )
                last_metrics = _route_agreement(
                    prev_selected,
                    next_selected,
                    indices=last_indices,
                )
                row = {
                    "step": int(step),
                    "transition_index": int(transition_index),
                    "transition_in_cycle": int(transition_in_cycle),
                    "cycle": int(cycle),
                    "l_index": int(l_index),
                    "next_l_index": int(next_l_index),
                    "block": int(block),
                    "block_index": prev_record.get("block_index", ""),
                    **all_metrics,
                }
                if last_metrics is not None:
                    row.update({
                        f"last_token_{key}": value
                        for key, value in last_metrics.items()
                    })
                rows.append(row)
                wrote_transition = True
            if wrote_transition:
                transition_index += 1
    return rows


def _hrm_h_routing_rows(
    inner: Any,
    *,
    step: int,
    input_ids: torch.Tensor | None,
) -> list[dict[str, Any]]:
    selected = getattr(inner, "_all_mlp_selected_experts", None) or []
    scopes = getattr(inner, "_all_mlp_router_scopes", None) or []
    config = getattr(inner, "config", None)
    if getattr(config, "recurrent_loop", "flat") != "hrm" or not selected:
        return []

    prelude_layers = int(getattr(config, "prelude_layers", 0) or 0)
    l_layers = int(getattr(config, "hrm_l_layers", 0) or 0)
    by_cycle_block: dict[tuple[int, int], dict[str, Any]] = {}
    blocks_by_cycle: dict[int, set[int]] = defaultdict(set)

    for depth, selected_experts in enumerate(selected):
        if depth >= len(scopes):
            continue
        scope = scopes[depth] or {}
        if (
            not isinstance(scope, dict)
            or scope.get("pool") != "mlp_recurrent"
            or scope.get("hrm_module") != "H"
        ):
            continue
        cycle = _scope_int(scope.get("hrm_cycle"))
        if cycle is None:
            continue
        block = _scope_int(scope.get("hrm_local_block"))
        block_index = _scope_int(scope.get("block_index"))
        if block is None:
            if block_index is None:
                continue
            block = max(0, block_index - prelude_layers - l_layers)
        by_cycle_block[(cycle, block)] = {
            "selected": selected_experts,
            "block_index": "" if block_index is None else block_index,
        }
        blocks_by_cycle[cycle].add(block)

    rows: list[dict[str, Any]] = []
    cycles = sorted(blocks_by_cycle)
    for transition_index, (cycle, next_cycle) in enumerate(zip(cycles, cycles[1:])):
        blocks = sorted(blocks_by_cycle[cycle] & blocks_by_cycle[next_cycle])
        for block in blocks:
            prev_record = by_cycle_block.get((cycle, block))
            next_record = by_cycle_block.get((next_cycle, block))
            if prev_record is None or next_record is None:
                continue
            prev_selected = prev_record["selected"]
            next_selected = next_record["selected"]
            all_metrics = _route_agreement(prev_selected, next_selected)
            if all_metrics is None:
                continue
            total_tokens = int(prev_selected.shape[0])
            last_indices = _last_token_indices(
                input_ids,
                total_tokens,
                device=prev_selected.device,
            )
            last_metrics = _route_agreement(
                prev_selected,
                next_selected,
                indices=last_indices,
            )
            row = {
                "step": int(step),
                "transition_index": int(transition_index),
                "cycle": int(cycle),
                "next_cycle": int(next_cycle),
                "block": int(block),
                "block_index": prev_record.get("block_index", ""),
                **all_metrics,
            }
            if last_metrics is not None:
                row.update({
                    f"last_token_{key}": value
                    for key, value in last_metrics.items()
                })
            rows.append(row)
    return rows


def _logical_block(scope: dict[str, Any], config: Any) -> int:
    recurrent_layers = max(1, int(getattr(config, "recurrent_layers", 0) or 0))
    prelude_layers = int(getattr(config, "prelude_layers", 0) or 0)
    block_index = _scope_int(scope.get("block_index"))
    if block_index is None:
        return 0
    block = block_index - prelude_layers
    return max(0, min(recurrent_layers - 1, block))


def _cycle_value(scope: dict[str, Any]) -> int | None:
    if "hrm_cycle" in scope:
        return _scope_int(scope.get("hrm_cycle"))
    return _scope_int(scope.get("loop"))


def _cycle_label(scope: dict[str, Any]) -> str:
    if "hrm_cycle" in scope:
        cycle = _scope_int(scope.get("hrm_cycle"))
        return f"c{cycle}" if cycle is not None else "cycle"
    loop = _scope_int(scope.get("loop"))
    return f"loop{loop}" if loop is not None else "loop"


def _computation_label(scope: dict[str, Any], config: Any) -> str:
    block = _logical_block(scope, config)
    if "hrm_cycle" in scope:
        cycle = _scope_int(scope.get("hrm_cycle"))
        module = str(scope.get("hrm_module", ""))
        if module == "L":
            l_index = _scope_int(scope.get("hrm_l_index"))
            return f"c{cycle} L{l_index} layer{block}"
        if module == "H":
            return f"c{cycle} H layer{block}"
        return f"c{cycle} layer{block}"
    loop = _scope_int(scope.get("loop"))
    return f"loop{loop} layer{block}" if loop is not None else f"layer{block}"


def _axis_label(scope: dict[str, Any], config: Any) -> str:
    block = _logical_block(scope, config)
    if "hrm_cycle" in scope:
        cycle = _scope_int(scope.get("hrm_cycle"))
        module = str(scope.get("hrm_module", "")).lower()
        local_block = _scope_int(scope.get("hrm_local_block"))
        if local_block is None:
            local_block = block
        if module in {"l", "h"}:
            return f"c{cycle} {module}{local_block + 1}"
        return f"c{cycle} b{block + 1}"
    loop = _scope_int(scope.get("loop"))
    return f"loop{loop} l{block + 1}" if loop is not None else f"l{block + 1}"


def _recurrent_route_records(inner: Any) -> list[dict[str, Any]]:
    selected = getattr(inner, "_all_mlp_selected_experts", None) or []
    scopes = getattr(inner, "_all_mlp_router_scopes", None) or []
    config = getattr(inner, "config", None)
    recurrent_layers = max(1, int(getattr(config, "recurrent_layers", 0) or 0))
    records: list[dict[str, Any]] = []
    recurrent_depth = 0
    for depth, selected_experts in enumerate(selected):
        if depth >= len(scopes):
            continue
        scope = scopes[depth] or {}
        if not isinstance(scope, dict) or scope.get("pool") != "mlp_recurrent":
            continue
        scope = dict(scope)
        if "hrm_cycle" not in scope and "loop" not in scope:
            scope["loop"] = recurrent_depth // recurrent_layers
        block = _logical_block(scope, config)
        cycle = _cycle_value(scope)
        cycle_key = (
            "hrm" if "hrm_cycle" in scope else "flat",
            -1 if cycle is None else int(cycle),
        )
        records.append({
            "computation_index": len(records),
            "block": int(block),
            "block_index": scope.get("block_index", ""),
            "cycle_key": cycle_key,
            "cycle_label": _cycle_label(scope),
            "label": _computation_label(scope, config),
            "axis_label": _axis_label(scope, config),
            "selected": selected_experts,
            "scope": scope,
        })
        recurrent_depth += 1

    previous_cycle_key = None
    for record in records:
        cycle_key = record["cycle_key"]
        record["cycle_start"] = (
            previous_cycle_key is not None and cycle_key != previous_cycle_key
        )
        previous_cycle_key = cycle_key
    return records


def _flat_residual_rows(inner: Any, *, step: int) -> list[dict[str, Any]]:
    entries = getattr(inner, "_recurrent_layer_diagnostics", None) or []
    config = getattr(inner, "config", None)
    recurrent_layers = max(1, int(getattr(config, "recurrent_layers", 0) or 0))
    rows: list[dict[str, Any]] = []
    previous_cycle_key = None
    previous_label = ""
    for index, entry in enumerate(entries):
        scope = dict(entry)
        if "hrm_cycle" not in scope and "loop" not in scope:
            scope["loop"] = index // recurrent_layers
        cycle = _cycle_value(scope)
        cycle_key = (
            "hrm" if "hrm_cycle" in scope else "flat",
            -1 if cycle is None else int(cycle),
        )
        to_label = _computation_label(scope, config)
        cycle_start = previous_cycle_key is not None and cycle_key != previous_cycle_key
        from_label = (
            f"{_cycle_label(scope)} input"
            if not previous_label or cycle_start
            else previous_label
        )
        row = {
            "step": int(step),
            "computation_index": int(entry.get("computation_index", index)),
            "block": _logical_block(scope, config),
            "block_index": scope.get("block_index", ""),
            "cycle_label": _cycle_label(scope),
            "from_label": from_label,
            "to_label": to_label,
            "axis_label": _axis_label(scope, config),
            "transition_label": f"{from_label} -> {to_label}",
            "cycle_start": int(cycle_start),
            "phase": str(entry.get("phase", "")),
        }
        for key in (
            "residual_rms",
            "relative_residual_rms",
            "cosine",
            "last_token_residual_rms",
            "last_token_relative_residual_rms",
            "last_token_cosine",
        ):
            value = _scalar(entry.get(key))
            if value is not None:
                row[key] = value
        rows.append(row)
        previous_cycle_key = cycle_key
        previous_label = to_label
    return rows


def _routing_similarity_rows(
    inner: Any,
    *,
    step: int,
    input_ids: torch.Tensor | None,
) -> list[dict[str, Any]]:
    records = _recurrent_route_records(inner)
    previous_by_block: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        block = int(record["block"])
        previous = previous_by_block.get(block)
        if previous is not None:
            prev_selected = previous["selected"]
            next_selected = record["selected"]
            all_metrics = _route_agreement(prev_selected, next_selected)
            if all_metrics is not None:
                total_tokens = int(prev_selected.shape[0])
                last_indices = _last_token_indices(
                    input_ids,
                    total_tokens,
                    device=prev_selected.device,
                )
                last_metrics = _route_agreement(
                    prev_selected,
                    next_selected,
                    indices=last_indices,
                )
                row = {
                    "step": int(step),
                    "computation_index": int(record["computation_index"]),
                    "previous_computation_index": int(previous["computation_index"]),
                    "block": block,
                    "block_index": record.get("block_index", ""),
                    "cycle_label": record["cycle_label"],
                    "from_label": previous["label"],
                    "to_label": record["label"],
                    "axis_label": record["axis_label"],
                    "transition_label": f"{previous['label']} -> {record['label']}",
                    "cycle_start": int(bool(record.get("cycle_start"))),
                    **all_metrics,
                }
                if last_metrics is not None:
                    row.update({
                        f"last_token_{key}": value
                        for key, value in last_metrics.items()
                    })
                rows.append(row)
        previous_by_block[block] = record
    return rows


def _summary(
    loop_rows: list[dict[str, Any]],
    routing_rows: list[dict[str, Any]],
    hrm_l_routing_rows: list[dict[str, Any]],
    hrm_h_routing_rows: list[dict[str, Any]],
    flat_residual_rows: list[dict[str, Any]],
    routing_similarity_rows: list[dict[str, Any]],
) -> dict[str, float]:
    summary: dict[str, float] = {}

    if loop_rows:
        summary["loop_count"] = float(len(loop_rows))
        for key in (
            "residual_rms",
            "relative_residual_rms",
            "cosine",
            "last_token_residual_rms",
            "last_token_relative_residual_rms",
            "last_token_cosine",
        ):
            mean_value = _mean(loop_rows, key)
            last_value = _last(loop_rows, key)
            if mean_value is not None:
                summary[f"{key}_mean"] = mean_value
            if last_value is not None:
                summary[f"{key}_last"] = last_value

    if routing_rows:
        for key in (
            "top1_agreement",
            "top1_switch_rate",
            "jaccard",
            "jaccard_distance",
            "last_token_top1_agreement",
            "last_token_top1_switch_rate",
            "last_token_jaccard",
            "last_token_jaccard_distance",
        ):
            mean_value = _mean(routing_rows, key)
            if mean_value is not None:
                summary[f"routing_{key}_mean"] = mean_value

    if hrm_l_routing_rows:
        for key in (
            "top1_agreement",
            "top1_switch_rate",
            "jaccard",
            "jaccard_distance",
            "last_token_top1_agreement",
            "last_token_top1_switch_rate",
            "last_token_jaccard",
            "last_token_jaccard_distance",
        ):
            mean_value = _mean(hrm_l_routing_rows, key)
            if mean_value is not None:
                summary[f"hrm_l_routing_{key}_mean"] = mean_value

    if hrm_h_routing_rows:
        for key in (
            "top1_agreement",
            "top1_switch_rate",
            "jaccard",
            "jaccard_distance",
            "last_token_top1_agreement",
            "last_token_top1_switch_rate",
            "last_token_jaccard",
            "last_token_jaccard_distance",
        ):
            mean_value = _mean(hrm_h_routing_rows, key)
            if mean_value is not None:
                summary[f"hrm_h_routing_{key}_mean"] = mean_value

    if flat_residual_rows:
        for key in (
            "relative_residual_rms",
            "last_token_relative_residual_rms",
            "cosine",
            "last_token_cosine",
        ):
            mean_value = _mean(flat_residual_rows, key)
            last_value = _last(flat_residual_rows, key)
            if mean_value is not None:
                summary[f"flat_layer_{key}_mean"] = mean_value
            if last_value is not None:
                summary[f"flat_layer_{key}_last"] = last_value

    if routing_similarity_rows:
        for key in (
            "top1_agreement",
            "jaccard",
            "last_token_top1_agreement",
            "last_token_jaccard",
        ):
            mean_value = _mean(routing_similarity_rows, key)
            if mean_value is not None:
                summary[f"routing_similarity_{key}_mean"] = mean_value

    return summary


def build_recurrent_diagnostics(
    model: Any,
    *,
    step: int,
    input_ids: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    """Build loop residual and route-stability diagnostics from last forward."""

    inner = _inner_model(model)
    if not hasattr(inner, "recurrent_blocks"):
        return None

    loops = _loop_rows(inner, step=step)
    routes = _routing_rows(inner, step=step, input_ids=input_ids)
    hrm_l_routes = _hrm_l_routing_rows(inner, step=step, input_ids=input_ids)
    hrm_h_routes = _hrm_h_routing_rows(inner, step=step, input_ids=input_ids)
    flat_residuals = _flat_residual_rows(inner, step=step)
    routing_similarity = _routing_similarity_rows(
        inner,
        step=step,
        input_ids=input_ids,
    )
    if (
        not loops
        and not routes
        and not hrm_l_routes
        and not hrm_h_routes
        and not flat_residuals
        and not routing_similarity
    ):
        return None

    return {
        "step": int(step),
        "kind": "recurrent_diagnostics",
        "summary": _summary(
            loops,
            routes,
            hrm_l_routes,
            hrm_h_routes,
            flat_residuals,
            routing_similarity,
        ),
        "loop_rows": loops,
        "routing_rows": routes,
        "hrm_l_routing_rows": hrm_l_routes,
        "hrm_h_routing_rows": hrm_h_routes,
        "flat_residual_rows": flat_residuals,
        "routing_similarity_rows": routing_similarity,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_loop_residuals(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["loop"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    if any("relative_residual_rms" in row for row in rows):
        ax.plot(
            x,
            [row.get("relative_residual_rms", float("nan")) for row in rows],
            marker="o",
            label="all tokens",
        )
    if any("last_token_relative_residual_rms" in row for row in rows):
        ax.plot(
            x,
            [row.get("last_token_relative_residual_rms", float("nan")) for row in rows],
            marker="o",
            label="last token",
        )
    ax.set_title("Loop Relative Residual RMS")
    ax.set_xlabel("loop")
    ax.set_ylabel("||h_next - h|| / ||h||")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_routing_changes(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_block: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        loop = int(row["loop"])
        block = int(row.get("block", -1))
        grouped[loop].append(row)
        by_block[block][loop].append(row)
    x = sorted(grouped)
    if not x:
        return
    blocks = sorted(block for block in by_block if block >= 0)

    def grouped_mean(key: str) -> list[float]:
        values = []
        for loop in x:
            rows_for_loop = grouped[loop]
            values.append(_mean(rows_for_loop, key) or float("nan"))
        return values

    def block_mean(block: int, key: str) -> list[float]:
        values = []
        rows_by_loop = by_block.get(block, {})
        for loop in x:
            values.append(_mean(rows_by_loop.get(loop, []), key) or float("nan"))
        return values

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_last_token = any("last_token_jaccard_distance" in row for row in rows)
    panels = [("jaccard_distance", "All-Token Top-k Distance")]
    if has_last_token:
        panels.append(("last_token_jaccard_distance", "Last-Token Top-k Distance"))

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10, 3.6 * len(panels)),
        squeeze=False,
        sharex=True,
    )
    cmap = plt.get_cmap("tab10")
    for ax, (metric_key, title) in zip(axes[:, 0], panels):
        for block in blocks:
            ax.plot(
                x,
                block_mean(block, metric_key),
                color=cmap(block % 10),
                marker=".",
                linewidth=1.0,
                alpha=0.45,
                label=f"block {block}",
            )
        ax.plot(
            x,
            grouped_mean(metric_key),
            color="black",
            marker="o",
            linewidth=2.5,
            label="avg",
        )
        ax.set_title(title)
        ax.set_ylabel("1 - top-k Jaccard")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("loop x compared with x+1")
    fig.suptitle("Routing Change Between Adjacent Loops", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_flat_residuals(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["computation_index"]) for row in rows]
    y = [row.get("relative_residual_rms", float("nan")) for row in rows]
    cycle_starts = sorted({
        int(row["computation_index"])
        for row in rows
        if int(row.get("cycle_start", 0)) == 1
    })

    fig, ax = plt.subplots(figsize=(max(10, min(24, 0.28 * len(x) + 8)), 4.8))
    for marker in cycle_starts:
        ax.axvline(
            marker,
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
    ax.plot(x, y, color="black", marker="o", linewidth=1.5)
    ax.set_title("Flattened Recurrent Layer Residual")
    ax.set_xlabel("recurrent computation")
    ax.set_ylabel("relative residual RMS")
    ax.grid(False)
    if len(x) <= 64:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                str(row.get("axis_label", row["computation_index"]))
                for row in rows
            ],
            rotation=60,
            ha="right",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_routing_similarity_by_block(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[int(row.get("block", -1))].append(row)
    blocks = sorted(block for block in by_block if block >= 0)
    all_x = sorted({int(row["computation_index"]) for row in rows})
    cycle_starts = sorted({
        int(row["computation_index"])
        for row in rows
        if int(row.get("cycle_start", 0)) == 1
    })
    labels_by_x: dict[int, str] = {}
    for row in rows:
        labels_by_x.setdefault(
            int(row["computation_index"]),
            str(row.get("axis_label", row["computation_index"])),
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, min(24, 0.28 * len(all_x) + 8)), 5.2))
    for marker in cycle_starts:
        ax.axvline(
            marker,
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
    cmap = plt.get_cmap("tab10")
    for block in blocks:
        block_rows = sorted(
            by_block[block],
            key=lambda row: int(row["computation_index"]),
        )
        ax.plot(
            [int(row["computation_index"]) for row in block_rows],
            [row.get("jaccard", float("nan")) for row in block_rows],
            color=cmap(block % 10),
            marker="o",
            linewidth=1.3,
            label=f"layer {block}",
        )
    ax.set_title("Routing Similarity vs Previous Same Layer")
    ax.set_xlabel("current recurrent computation")
    ax.set_ylabel("top-k Jaccard similarity")
    ax.grid(False)
    ax.legend(ncol=4, fontsize=8)
    if len(all_x) <= 64:
        ax.set_xticks(all_x)
        ax.set_xticklabels(
            [labels_by_x.get(value, str(value)) for value in all_x],
            rotation=60,
            ha="right",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_hrm_routing_transitions(
    path: Path,
    *,
    l_rows: list[dict[str, Any]],
    h_rows: list[dict[str, Any]],
) -> None:
    if not l_rows and not h_rows:
        return

    def plot_data(
        rows: list[dict[str, Any]],
        *,
        label_kind: str,
    ) -> dict[str, Any]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_block: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        labels: dict[int, str] = {}
        cycles: dict[int, int] = {}
        for row in rows:
            transition = int(row["transition_index"])
            block = int(row.get("block", -1))
            grouped[transition].append(row)
            by_block[block][transition].append(row)
            if label_kind == "L":
                label = (
                    f"c{int(row['cycle'])} "
                    f"L{int(row['l_index'])}->{int(row['next_l_index'])}"
                )
            else:
                label = f"c{int(row['cycle'])}->c{int(row['next_cycle'])}"
            labels.setdefault(transition, label)
            cycles.setdefault(transition, int(row["cycle"]))
        x = sorted(grouped)
        markers = [
            next_x
            for prev_x, next_x in zip(x, x[1:])
            if cycles.get(prev_x) != cycles.get(next_x)
        ]
        return {
            "grouped": grouped,
            "by_block": by_block,
            "labels": labels,
            "x": x,
            "blocks": sorted(block for block in by_block if block >= 0),
            "cycle_start_markers": markers,
        }

    def grouped_mean(data: dict[str, Any], key: str) -> list[float]:
        grouped = data["grouped"]
        return [
            _mean(grouped[transition], key) or float("nan")
            for transition in data["x"]
        ]

    def block_mean(data: dict[str, Any], block: int, key: str) -> list[float]:
        values = []
        rows_by_transition = data["by_block"].get(block, {})
        for transition in data["x"]:
            values.append(
                _mean(rows_by_transition.get(transition, []), key) or float("nan")
            )
        return values

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_specs: list[dict[str, Any]] = []
    if l_rows:
        l_data = plot_data(l_rows, label_kind="L")
        panel_specs.append({
            "data": l_data,
            "metric_key": "jaccard_distance",
            "title": "All-Token L Routing Distance",
            "block_prefix": "L block",
            "xlabel": "same-cycle L transition",
            "mark_cycle_starts": True,
        })
        if any("last_token_jaccard_distance" in row for row in l_rows):
            panel_specs.append({
                "data": l_data,
                "metric_key": "last_token_jaccard_distance",
                "title": "Last-Token L Routing Distance",
                "block_prefix": "L block",
                "xlabel": "same-cycle L transition",
                "mark_cycle_starts": True,
            })
    if h_rows:
        h_data = plot_data(h_rows, label_kind="H")
        panel_specs.append({
            "data": h_data,
            "metric_key": "jaccard_distance",
            "title": "All-Token H Routing Distance",
            "block_prefix": "H block",
            "xlabel": "H cycle transition",
            "mark_cycle_starts": False,
        })
        if any("last_token_jaccard_distance" in row for row in h_rows):
            panel_specs.append({
                "data": h_data,
                "metric_key": "last_token_jaccard_distance",
                "title": "Last-Token H Routing Distance",
                "block_prefix": "H block",
                "xlabel": "H cycle transition",
                "mark_cycle_starts": False,
            })
    if not panel_specs:
        return

    max_points = max(len(spec["data"]["x"]) for spec in panel_specs)

    fig, axes = plt.subplots(
        len(panel_specs),
        1,
        figsize=(max(10, min(18, 0.45 * max_points + 8)), 3.6 * len(panel_specs)),
        squeeze=False,
        sharex=False,
    )
    cmap = plt.get_cmap("tab10")
    for ax, spec in zip(axes[:, 0], panel_specs):
        data = spec["data"]
        x = data["x"]
        if spec["mark_cycle_starts"]:
            for boundary in data["cycle_start_markers"]:
                ax.axvline(
                    boundary,
                    color="0.45",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                )
        for block in data["blocks"]:
            ax.plot(
                x,
                block_mean(data, block, spec["metric_key"]),
                color=cmap(block % 10),
                marker=".",
                linewidth=1.2,
                alpha=0.65,
                label=f"{spec['block_prefix']} {block}",
            )
        ax.plot(
            x,
            grouped_mean(data, spec["metric_key"]),
            color="black",
            marker="o",
            linewidth=2.5,
            label="avg",
        )
        ax.set_title(spec["title"])
        ax.set_ylabel("1 - top-k Jaccard")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        ax.set_xlabel(spec["xlabel"])
        if len(x) <= 36:
            ax.set_xticks(x)
            ax.set_xticklabels(
                [data["labels"].get(value, str(value)) for value in x],
                rotation=45,
                ha="right",
            )
    fig.suptitle("HRM Routing Change", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_hrm_l_routing_transitions(path: Path, rows: list[dict[str, Any]]) -> None:
    _plot_hrm_routing_transitions(path, l_rows=rows, h_rows=[])


def save_recurrent_diagnostics_artifacts(
    model: Any,
    *,
    step_dir: str,
    step: int,
    input_ids: torch.Tensor | None = None,
) -> bool:
    payload = build_recurrent_diagnostics(model, step=step, input_ids=input_ids)
    if payload is None:
        return False

    out_dir = Path(step_dir) / "recurrent_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "step": payload["step"],
                "kind": payload["kind"],
                "summary": payload["summary"],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    loop_rows = payload.get("loop_rows", [])
    routing_rows = payload.get("routing_rows", [])
    hrm_l_routing_rows = payload.get("hrm_l_routing_rows", [])
    hrm_h_routing_rows = payload.get("hrm_h_routing_rows", [])
    flat_residual_rows = payload.get("flat_residual_rows", [])
    routing_similarity_rows = payload.get("routing_similarity_rows", [])
    _write_csv(out_dir / "loop_residuals.csv", loop_rows)
    _write_csv(out_dir / "routing_changes.csv", routing_rows)
    _write_csv(out_dir / "hrm_l_routing_transitions.csv", hrm_l_routing_rows)
    _write_csv(out_dir / "hrm_h_routing_transitions.csv", hrm_h_routing_rows)
    _write_csv(out_dir / "flattened_residuals.csv", flat_residual_rows)
    _write_csv(out_dir / "routing_similarity_by_block.csv", routing_similarity_rows)
    _plot_loop_residuals(out_dir / "loop_residuals.png", loop_rows)
    _plot_routing_changes(out_dir / "routing_changes.png", routing_rows)
    _plot_flat_residuals(out_dir / "flattened_residuals.png", flat_residual_rows)
    _plot_routing_similarity_by_block(
        out_dir / "routing_similarity_by_block.png",
        routing_similarity_rows,
    )

    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Recurrent Diagnostics\n\n")
        f.write(f"Step: {payload['step']}\n\n")
        for key, value in sorted(payload.get("summary", {}).items()):
            f.write(f"- `{key}`: {value:.8g}\n")

    return True
