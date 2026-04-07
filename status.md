# Status

Updated: 2026-04-07 UTC

## Current State

- `PLAN.md` item 3 is now implemented in code and config:
  - branch-router aux loss is no longer applied in `moe_everything`
  - the shared-router precompute-KV path now uses routing ratios for `K/V` instead of straight-through `1`s
  - the main per-head configs now use `scale_branch_by_routing_weight: true`, `per_layer_mlp_router: true`, and nonzero router aux for attention + MLP routers
- The repo now has a runnable experiment matrix under `configs/plan/` for:
  - standard MoE on our current parquet corpus
  - standard MoE on a parquetized official FineWebEdu sample corpus
  - dense Qwen3-0.6B on our current parquet corpus
  - dense Qwen3-0.6B on the parquetized official FineWebEdu sample corpus
  - two retrofit per-head `moe_everything` configs
- A dense-model startup bug in `train.py` was fixed. Dense configs were crashing before training because the summary printer assumed MoE-only fields.

## Data Setup

- Existing corpus:
  - `data/parquet`
- Added comparison corpus:
  - `data/parquet_finewebedu_speedrun`
  - generated locally as `8` parquet shards x `10,000` rows each (`80,000` rows total)
  - command used:
    - `uv run python scripts/create_official_finewebedu_sample_parquet.py --out-dir data/parquet_finewebedu_speedrun --num-shards 8 --rows-per-shard 10000`

Important limitation:

- this comparison corpus is a repo-native parquet sample from official FineWebEdu
- it is the closest local analogue I could run directly through this repo's parquet pipeline
- it is not the exact modded-NanoGPT `cached_fineweb10B.py` token cache, which uses FineWeb rather than FineWebEdu

## Code Changes

- `src/models/mixture_of_everything.py`
  - removed branch-router aux-loss application from the loss path
  - changed the shared-router precompute-KV path so `K/V` use routing ratios when attention scaling is enabled
- `train.py`
  - fixed dense-config startup by making the architecture summary handle non-MoE models
- `tests/test_models.py`
  - updated regression coverage so branch aux stays disabled
  - extended the precompute-KV weighting test to verify `K_fresh` / `V_fresh` change with routing ratios
- `configs/moe_everything_per_head_*`
  - canonical per-head configs now explicitly use ratio scaling on the branch path
  - canonical per-head configs now use per-layer MLP routers
  - branch aux coefficients are set to `0.0`
- `configs/depth_matched/16_layers/*`
  - same policy updates applied to the depth-matched per-head configs
- `configs/scaling/*8gpu_100_nowandb.yaml`
  - enabled router aux for MLP + attention routers and enabled per-layer MLP routing for the per-head validation configs
- `configs/plan/*`
  - added the experiment matrix and retrofit configs for this plan
- `todo.md`
  - corrected the stale note that claimed branch-router aux was active

## Short-Horizon Matrix

All runs below used:

- `uv run python train.py ...`
- `1 x NVIDIA B200`
- `bf16`
- `10` optimizer steps
- eval every `10` steps over `4` batches

Results:

| Run | Step 1 loss | Step 10 loss | Step 10 eval CE |
| --- | ---: | ---: | ---: |
| standard MoE, our data | `12.1792` | `11.7667` | `11.6360` |
| standard MoE, FineWebEdu sample | `12.1687` | `11.7556` | `11.6344` |
| dense Qwen3-0.6B, our data | `12.1263` | `11.6046` | `11.4811` |
| dense Qwen3-0.6B, FineWebEdu sample | `12.1228` | `11.6192` | `11.4998` |

Short-horizon takeaway:

- at `10` steps, our current parquet corpus and the official FineWebEdu sample look extremely similar for both standard MoE and dense Qwen
- dense Qwen3-0.6B is improving slightly faster than the current 16-layer standard MoE baseline in this very short run
- none of these runs are remotely close to `3.28`; this matrix only answers early-optimization behavior, not the long-horizon target

Artifacts:

- `outputs/plan/standard_moe_our_data_step10`
- `outputs/plan/standard_moe_speedrun_style_finewebedu_step10`
- `outputs/plan/qwen3_0_6b_dense_our_data_step10`
- `outputs/plan/qwen3_0_6b_dense_speedrun_style_finewebedu_step10`

## Retrofit Smoke Runs

All runs below used:

- `uv run python train.py ...`
- `1 x NVIDIA B200`
- `bf16`
- `2` optimizer steps

Precompute-KV retrofit:

- config: `configs/plan/moe_everything_per_head_precompute_kv_retrofit.yaml`
- step `1`: total `13.1558`, CE `12.1316`, aux `1024.0377`, attention aux `512.0833`, branch aux `0.0000`
- step `2`: total `13.1443`, CE `12.1200`, aux `1024.0652`, attention aux `512.0885`, branch aux `0.0000`
- step times: `126.781s`, `114.322s`

Fully independent retrofit:

- config: `configs/plan/moe_everything_per_head_independent_retrofit.yaml`
- step `1`: total `16.2214`, CE `12.1237`, aux `4097.1470`, attention aux `3584.9429`, branch aux `0.0000`
- step `2`: total `16.2209`, CE `12.1237`, aux `4096.7598`, attention aux `3584.6232`, branch aux `0.0000`
- step times: `93.116s`, `73.306s`

Retrofit takeaway:

- both retrofitted per-head configs now train cleanly under the requested routing policy
- branch aux remains disabled in real training logs
- attention aux is active and dominates the auxiliary term, especially in the fully independent variant
- these were smoke runs only; they prove launch/trainability, not quality

Artifacts:

- `outputs/plan/moe_everything_per_head_precompute_kv_retrofit_step2`
- `outputs/plan/moe_everything_per_head_independent_retrofit_step2`

## Validation

- `uv run pytest -q tests/test_models.py -k 'branch_aux_loss_is_disabled or precompute_kv_scale_flag_controls_weighting or per_layer_mlp_router_creates_separate_routers or per_layer_attn_router_forward_and_grads'`
- result:
  - `6 passed, 96 deselected`
- GPU training validations:
  - standard MoE short run on our parquet corpus
  - standard MoE short run on the FineWebEdu sample corpus
  - dense Qwen3-0.6B short run on our parquet corpus
  - dense Qwen3-0.6B short run on the FineWebEdu sample corpus
  - per-head precompute-KV retrofit smoke run
  - per-head fully independent retrofit smoke run

## Open Items

- The exact modded-NanoGPT speedrun corpus is still not wired into this repo. The current comparison corpus is an official FineWebEdu parquet sample, not the speedrun token-cache pipeline.
- The `3.28` target is not settled by these runs. The matrix here is intentionally short so the repo has a reproducible local status update and a pushable code state.
- If we want to answer the original target rigorously, the next step is long-horizon training from the new `configs/plan/` entries, probably starting with:
  - dense Qwen3-0.6B vs standard MoE on the same corpus
  - then repeating on the exact NanoGPT/FineWeb token pipeline if we add ingestion support
