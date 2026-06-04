# Bench Results

This directory holds the outputs of `scripts/bench_step.py` for the
launch set under `configs/16_layers/`. The bench runs are
expected to be reproducible on Modal H200 (8 GPUs) and produce a
`results.json` payload that ranks the configurations by
`tokens_per_second` at the largest stable batch size.

## Running a single config

```bash
PYTHONPATH=. python scripts/bench_step.py \
    --config configs/16_layers/moe_everything_per_head_recompute_k_qk_v_o_deepseek_bias.yaml \
    --warmup 5 --measure 20 \
    --output bench/results.json
```

Each invocation appends one JSON record to `bench/results.json`.

## Searching the largest stable batch

```bash
PYTHONPATH=. python scripts/bench_step.py \
    --config configs/16_layers/moe_everything_per_head_recompute_kv_qk_v_o_ema_qk_v_deepseek_bias.yaml \
    --search --batch-min 1 --batch-max 64 \
    --output bench/results.json
```

`--search` binary-searches the largest batch size that completes one
full forward + backward + step without OOM, and reports the
`tokens_per_second` at that batch.

## Modal H200 sweep

The five-config launch-set sweep on 8xH200 lands in `bench/results.json`
under the benchmark deliverable. The CI-side bench (this directory's CPU
default) is structural only — it verifies the script runs end-to-end
and that the JSON schema is stable; the per-config tokens/sec figures
are not load-bearing on CPU.

## Output schema

Every record in `bench/results.json` has the shape:

```json
{
  "config": "<yaml path>",
  "model_type": "moe_everything",
  "device": "cuda",
  "warmup_iters": 5,
  "measure_iters": 20,
  "batch_size": 4,
  "seq_len": 2048,
  "tokens_per_step": 8192,
  "wall_seconds_mean": 0.142,
  "wall_seconds_std": 0.003,
  "tokens_per_second": 57700.0,
  "peak_gpu_memory_bytes": 12300000000,
  "loss_mean": 6.93,
  "loss_std": 0.04,
  "search_mode": false,
  "search_max_batch": null
}
```
