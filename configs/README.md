# MoE config matrix

This directory holds the active configuration matrix the trainer
loads via `python scripts/train.py --config configs/<depth>/<file>.yaml`.
The matrix is locked by `tests/test_nested_configs_drift.py` and
`scripts/validate_configs.py`.

## Folder layout

The active matrix lives under three depth directories:

- `configs/4_layers/` — 13 yamls
- `configs/8_layers/` — 13 yamls
- `configs/16_layers/` — 13 yamls

Total: 39 yamls. Drift tests fail if this count is wrong.

## Depth ↔ `model.num_hidden_layers`

The folder name documents the **depth axis** of the matrix. The
mapping between folder and `model.num_hidden_layers` is:

| Folder | `model.num_hidden_layers` for `dense` / `standard_moe` / `global_moe` | `model.num_hidden_layers` for `moe_everything` |
|--------|----------------------------------------------------------------------|-----------------------------------------------|
| `configs/4_layers/` | 4 | 8 (`moe_everything` runs 2× depth iterations per "layer" because the branch router routes per depth call) |
| `configs/8_layers/` | 8 | 16 |
| `configs/16_layers/` | 16 | 32 (large; only viable on H200-class hardware) |

The non-`moe_everything` families consume one `num_hidden_layers`
value directly. `moe_everything` doubles it because each depth in the
shared backbone runs both an attention branch and an MLP branch under
the branch router; the configs follow that convention so the depth
axis lines up across families.

## Family × method matrix

Each depth directory contains one yaml per `(family, method)` pair,
plus one `dense.yaml` row:

| Family | Methods | Filename pattern |
|--------|---------|------------------|
| `dense` | (no balancing — n/a) | `dense.yaml` |
| `standard_moe` | `aux_loss`, `deepseek_bias`, `quantile` | `standard_moe_<method>.yaml` |
| `global_moe` | `aux_loss`, `deepseek_bias`, `quantile` | `global_moe_<method>.yaml` |
| `moe_everything` (`per_head_fully_independent`) | `aux_loss`, `deepseek_bias`, `quantile` | `moe_everything_per_head_fully_independent_<method>.yaml` |
| `moe_everything` (`per_head_precompute_kv`) | `aux_loss`, `deepseek_bias`, `quantile` | `moe_everything_per_head_precompute_kv_<method>.yaml` |

Total per depth: 1 dense + 4 MoE families × 3 methods = 13 yamls.

## Required active knobs per method

Every per-class block (`model.{mlp,attn,branch}_router`) whose
`balancing` is one of the matrix methods MUST carry the matching
active knob at the matrix value:

| `balancing` | Required active knob | Value |
|-------------|---------------------|-------|
| `aux_loss` | `router_aux_loss_coef` | `0.001` |
| `seq_aux_loss` | `seq_aux_loss_coef` | `0.0001` |
| `deepseek_bias` | `bias_update_rate` | `0.001` |
| `quantile` | `quantile_eta`, `quantile_target_q`, `quantile_global_state` | `0.005`, `0.5`, `True` |

Validators (`scripts/validate_configs.py` strict mode + the pytest
drift suite) fail if a matrix yaml is missing the active knob or
carries an off-axis knob.

## DEC-6 branch contract

Every yaml whose `model.attn_expert_mode == per_head_precompute_kv`
MUST set the branch router to the documented exploration_only
schedule:

```yaml
branch_router:
  balancing: exploration_only
  exploration_rate: 0.1
  exploration_decay: cosine
  exploration_min: 0.01
  exploration_warmup_steps: 1000
```

The drift test `test_precompute_kv_rows_set_dec6_branch_exploration`
and the CLI validator both enforce this.

## Top-level fields are forbidden in matrix yamls

After migration, active matrix yamls MUST NOT carry:

- `training.load_balancing_method` (per-class blocks are
  authoritative)
- `training.router_aux_loss_coef`
- `training.seq_aux_loss_coef`
- `training.bias_update_rate`
- `training.bias_update_zero_sum`
- `training.bias_warmup_start`
- `training.bias_warmup_steps`

The strict mode in `scripts/validate_configs.py` rejects any of these
on a yaml under `configs/{4,8,16}_layers/`. Yamls under
`configs/extras/` are exempt as legacy / non-matrix fixtures.

## Adding a new yaml to the matrix

1. Pick a `(depth, family, method)` triple. The matrix is locked at
   13 per depth × 3 depths = 39; adding a new triple requires
   updating `_MATRIX_FAMILIES` / `_MATRIX_METHODS` in
   `tests/test_nested_configs_drift.py` first.
2. Author the yaml in the right `configs/<depth>_layers/` directory.
3. Use the per-class blocks (`mlp_router`, `attn_router`,
   `branch_router`) — never top-level balancing fields.
4. For `per_head_precompute_kv` rows, follow the DEC-6 branch
   contract above.
5. Run `python scripts/validate_configs.py` (must pass) and
   `python -m pytest tests/test_nested_configs_drift.py` (must pass)
   before committing.

## `configs/extras/`

Non-matrix yamls (sanity / perlayer_prenorm / experimental seq_aux
variants / Round-26 nested examples that predate the 13×3 layout)
live under `configs/extras/`. These pass the validator's basic
checks but are exempt from the strict-mode active-matrix rules.
They are NOT part of the AC-18 matrix and should not be referenced
in benchmark sweeps.
