# Status

Updated: 2026-04-07 07:06 UTC

## Current State

- The repo now has a real cached GPT-2 token-bin data path for FineWeb benchmarking.
- The benchmark path uses official cached FineWeb GPT-2 `.bin` shards plus fixed-token validation, instead of parquet text plus holdout files.
- I ran corrected benchmark-path experiments for:
  - GPT-2 small dense
  - Qwen3-0.6B-style dense with GPT-2 vocab/token ids
  - per-head `moe_everything` precompute-KV retrofit smoke

## What Was Fixed

The biggest issue was that earlier comparisons to the `3.28` NanoGPT speedrun were not benchmark-equivalent.

I fixed the local path by adding:

- `src/data/token_bin_dataset.py`
  - reads the cached GPT-2 token-bin format used by `modded-nanogpt` / `llm.c`
  - validates the bin header (`magic=20240520`, `version=1`)
  - supports deterministic DDP sharding and checkpoint resume state
  - supports fixed `max_tokens` for exact-style validation slices
- `train.py`
  - can now build either parquet datasets or token-bin datasets from config
  - skips tokenizer loading for token-bin runs
  - can run full finite validation datasets when `eval.max_batches <= 0`
- `scripts/download_fineweb10b_gpt2_bins.py`
  - downloads the official cached FineWeb GPT-2 bins from `kjj0/fineweb10B-gpt2`
- benchmark configs:
  - `configs/plan/gpt2_small_fineweb10b_gpt2_bins.yaml`
  - `configs/plan/qwen3_0_6b_fineweb10b_gpt2_bins.yaml`
  - `configs/plan/moe_everything_per_head_precompute_kv_fineweb10b_gpt2_bins.yaml`
- tests:
  - `tests/test_token_bin_dataset.py`

## Data Setup

Downloaded locally:

- `data/fineweb10B_gpt2/fineweb_val_000000.bin`
- `data/fineweb10B_gpt2/fineweb_train_000001.bin`
- `data/fineweb10B_gpt2/fineweb_train_000002.bin`
- `data/fineweb10B_gpt2/fineweb_train_000003.bin`
- `data/fineweb10B_gpt2/fineweb_train_000004.bin`
- `data/fineweb10B_gpt2/fineweb_train_000005.bin`

This gives:

- `500M` train tokens total
- exact validation taken from the first `10,485,760` val tokens

## Important Benchmark Finding

The current `modded-nanogpt` speedrun target is still `3.28` val cross-entropy on FineWeb, but the current record model is not vanilla GPT-2 small.

From the official `train_gpt.py` / `README.md`:

- the current reference model is custom (`11` layers, `6` heads, `128` head dim, `768` model dim)
- it also includes many benchmark-specific architecture and optimizer changes
- so a plain local GPT-2 small baseline should not be expected to match the speedrun result

That mismatch is real and explains the original confusion.

## Config Bug Found During Benchmarking

My first token-bin benchmark attempt copied `warmup_steps: 2000` into a `1000`-step budget run.

That was wrong for the short benchmark:

- the run spent effectively the whole budget in warmup
- the loss curve from that first attempt was not trustworthy

I corrected the benchmark configs to:

- `warmup_steps: 100`
- `max_steps: 800`
- exact eval / save every `200` steps

The results below are from the corrected path.

## Validation

Code validation:

- `./.venv/bin/python -m py_compile train.py src/data/token_bin_dataset.py scripts/download_fineweb10b_gpt2_bins.py`
- result: passed

Dataset tests:

- `./.venv/bin/python -m pytest -q tests/test_token_bin_dataset.py`
- result: `3 passed`

GPU smoke validations:

- GPT-2 token-bin smoke: `1` step on `1 x B200`
- Qwen token-bin smoke: `1` step on `1 x B200`
- per-head token-bin smoke: `2` steps on `8 x B200`

## Corrected Benchmark Results

### GPT-2 Small Dense, `8 x B200`

Config:

- `configs/plan/gpt2_small_fineweb10b_gpt2_bins.yaml`

Run:

- output dir: `outputs/plan/gpt2_small_fineweb10b_gpt2_bins_w100`
- exact checkpointed eval:
  - step `200`
  - tokens seen: `104,448,000`
  - val CE: `6.9820`
- later live training signal:
  - around step `490`, train CE was still about `6.25`
- steady throughput:
  - about `1.45M tok/s`
  - about `0.36s/step`

Takeaway:

- this is not remotely on a `3.28` trajectory
- at current speed, `400M` tokens would be about `763` steps and roughly `4.6` minutes of pure train time
- but the loss is still far too high, so the issue is not throughput anymore; it is model/algorithm mismatch versus the official speedrun

### Qwen3-0.6B-Style Dense, `8 x B200`

Config:

- `configs/plan/qwen3_0_6b_fineweb10b_gpt2_bins.yaml`

Run:

- output dir: `outputs/plan/qwen3_0_6b_fineweb10b_gpt2_bins_w100`
- exact checkpointed eval:
  - step `200`
  - tokens seen: `104,448,000`
  - val CE: `6.7938`
- steady throughput:
  - about `438k tok/s`
  - about `1.20s/step`

Takeaway:

- Qwen is slightly better than plain GPT-2 at the same token budget on this path
- but it is still nowhere near `3.28`

## Per-Head Retrofit Run

Config:

- `configs/plan/moe_everything_per_head_precompute_kv_fineweb10b_gpt2_bins.yaml`

Smoke run:

- output dir: `outputs/plan/moe_everything_per_head_precompute_kv_fineweb10b_gpt2_bins_smoke2`
- `8 x B200`
- `2` steps

Results:

- step `1`: total `12.0469`, CE `11.0225`, aux `1024.1090`, attention aux `511.8111`, branch aux `0.0000`
- step `2`: total `12.0409`, CE `11.0167`, aux `1024.0363`, attention aux `511.9798`, branch aux `0.0000`
- tokens seen at step `2`: `1,048,576`
- throughput:
  - about `4.0k tok/s` then `4.6k tok/s`
  - step times `131.539s` and `114.070s`

Takeaway:

- the per-head retrofit now launches and trains on the official token-bin path
- branch aux remains disabled as intended
- but this model is far too slow for a speculative long benchmark run until there is evidence that the dense baselines can get meaningfully closer to target

## Bottom Line

- The benchmark path itself is now fixed enough to answer the original question.
- On the corrected official-bin path:
  - GPT-2 small dense did not get close to `3.28`
  - Qwen3-0.6B-style dense also did not get close to `3.28`
- So the answer is:
  - no, these local dense baselines do not currently look capable of reaching `3.28` under this repo's trainer
  - the remaining gap is no longer “we used the wrong data/eval path”
  - the remaining gap is model/training-stack mismatch versus the actual `modded-nanogpt` speedrun

## Source Reference

Primary reference used for the benchmark target and dataset format:

- `https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/README.md`
- `https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/train_gpt.py`
- `https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/data/cached_fineweb10B.py`
