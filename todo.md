done

1. attention routers now weight selected experts by routing weights by default (`scale_attn_by_routing_weight: true`) instead of relying on the straight-through pure-selection path.

2. all router families now support Switch-style batch aux loss plus random exploration:
   - softmax routers gained tracked top-k assignments and exploration
   - DeepSeek routers gained exploration without losing biased routing
   - `moe_everything` now applies batch aux to MLP, attention, and branch routers
   - canonical configs were updated to turn on conservative aux/exploration settings

3. packed-data validation was added to `train.py`:
   - deterministic held-out eval stream from the parquet shards
   - `eval/ce_loss` and `eval/perplexity` logging for NanoGPT-style comparison
   - configs updated to enable eval on the main standard/global/per-head runs
   - GPU smoke tested with `uv run` on standard training/eval and on a per-head GPU forward/backward path
