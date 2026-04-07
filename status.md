# Status

Updated: 2026-04-07 06:21 UTC

## Current State

- The routing-policy work from `PLAN.md` item 3 is still in place and already pushed:
  - branch-router aux loss is disabled in `moe_everything`
  - the shared precompute-KV path now uses routing ratios for `K/V`
  - the main per-head configs now use `scale_branch_by_routing_weight: true` and `per_layer_mlp_router: true`
- The repo now also has long-run benchmark configs for:
  - `configs/plan/gpt2_small_speedrun_style_long.yaml`
  - `configs/plan/qwen3_0_6b_gpt2_tokenizer_speedrun_long.yaml`
- Both long runs are active on `4 x NVIDIA B200` each.

## Benchmark Audit

I checked the local setup against the actual `modded-nanogpt` speedrun path (`README.md`, `train_gpt.py`, `data/cached_fineweb10B.py`).

Benchmark target facts:

- The target is still `3.28` validation cross-entropy on FineWeb.
- The reference pipeline uses cached GPT-2 token `.bin` shards from `kjj0/fineweb10B-gpt2`, not text parquet.
- The validation target is the fixed FineWeb validation bin (`10,485,760` tokens), not a small file-level holdout from the training directory.
- As of 2026-04-07 UTC, the current `modded-nanogpt` README no longer describes this as a "`10` minute" result; it reports a much faster current record on `8 x H100`.

Conclusion:

- there is no evidence of one catastrophic training bug in this repo
- but the current path is not benchmark-equivalent, so earlier comparisons to the `3.28` speedrun were not valid

## What Is Not Correct Yet

The main mismatches are structural:

- Data path mismatch:
  - this repo streams parquet text and tokenizes each document online in `StatefulParquetDataset`
  - the speedrun uses pretokenized GPT-2 token bins
- Dataloader mismatch:
  - `train.py` hard-codes `num_workers=0` for both train and eval because the iterable dataset is resume-stateful
  - that means config values like `data.num_workers: 4` are currently ignored
- Eval mismatch:
  - the local long-run configs evaluate only `20` batches from a parquet holdout
  - the speedrun evaluates on the fixed FineWeb validation bin, so the loss numbers are not directly comparable
- Corpus mismatch:
  - the local comparison corpus is a parquetized FineWebEdu sample
  - the speedrun target is FineWeb GPT-2 tokens
- Model mismatch:
  - local GPT-2 is a real GPT-2 small baseline
  - local Qwen is a dense Qwen3-0.6B-style baseline with GPT-2 tokenizer/vocab
  - the speedrun model stack is not a plain GPT-2 or plain Qwen baseline
- Optimizer / systems mismatch:
  - this repo uses a general `Accelerate` + `AdamW` training path
  - the speedrun uses a much more specialized trainer and kernel stack

There is also one concrete repo issue that blocks a clean `8 GPU` parquet-holdout run on the current sample:

- `data/parquet_finewebedu_speedrun` currently has `8` parquet shards
- with `world_size=8` and `holdout_fraction=0.05`, the eval split logic forces `7` validation files so every rank can get a val shard
- that leaves only `1` train file, which is not enough to shard training across `8` ranks
- so the current sample/holdout setup is structurally wrong for a single `8 GPU` train+eval job

## Long-Run Status

Live numbers below are from the active runs as of 2026-04-07 06:21 UTC.

GPT-2 small, `4 x B200`:

- config: `configs/plan/gpt2_small_speedrun_style_long.yaml`
- latest observed step: `2133`
- latest observed train loss: `4.9623`
- latest observed eval CE: `6.2536` at step `1750`
- latest checkpoint: step `2000`
- tokens seen at step `2000`: `522,977,280`
- steady throughput: about `350k tok/s` (`~0.74s/step`)

Qwen3-0.6B dense with GPT-2 tokenizer, `4 x B200`:

- config: `configs/plan/qwen3_0_6b_gpt2_tokenizer_speedrun_long.yaml`
- latest observed step: `1000`
- latest observed train loss: `5.1754`
- latest observed eval CE: `6.1202` at step `1000`
- latest checkpoint: step `1000`
- tokens seen at step `1000`: `261,357,568`
- steady throughput: about `164k tok/s` (`~1.60s/step`)

Takeaway:

- GPT-2 has already passed the classic `400M`-token scale by step `2000` and is still nowhere near `3.28`
- that does not prove a hidden math bug
- it does prove the current local benchmark path is not an apples-to-apples reproduction of the speedrun target

Throughput sanity check:

- GPT-2 on `4` GPUs is roughly `350k tok/s`
- if that scaled linearly to `8` GPUs, it would be about `700k tok/s`
- at that rate, `400M` tokens would take about `9.5` minutes
- that is much slower than the current `modded-nanogpt` record, but it is broadly consistent with a generic trainer rather than a broken idle-GPU run

## Earlier Local Validation

Previously completed and already pushed:

- routing-policy code/config updates
- dense-model startup fix in `train.py`
- short `10`-step matrix on our parquet corpus vs parquetized FineWebEdu sample
- `2`-step retrofit smoke runs for the per-head `moe_everything` configs
- targeted regression tests in `tests/test_models.py`

The short-horizon matrix and retrofit smokes were useful for launchability and policy validation, but they do not answer the `3.28` question.

## Bottom Line

- Something was indeed "not correct" in the earlier benchmark comparison, but it is mainly a setup-equivalence problem, not an obvious forward/backward bug.
- The largest benchmark-breaking issues are:
  - online parquet tokenization
  - mismatched eval protocol
  - FineWebEdu parquet sample instead of FineWeb GPT-2 bins
  - generic trainer/optimizer path instead of the specialized speedrun stack
  - current sample sharding that does not cleanly support one `8 GPU` train+eval run

## Next Steps

If the goal is a real answer to the `3.28` question, the next work should be:

1. Add a cached-token dataset path that reads GPT-2 token bins directly.
2. Use the exact FineWeb validation bin or an exact local copy of that protocol.
3. Regenerate the local sample with enough shards, or use separate train/eval dirs, so an `8 GPU` run is structurally valid.
4. Rerun GPT-2 on all `8` GPUs only after the data/eval path is benchmark-equivalent.
