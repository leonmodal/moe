# Load-balancing methods — TODO

Goal: have **four selectable load-balancing methods** wired into all three model families
(`standard_moe`, `global_moe`, `moe_everything`), gated by a single config knob:

```yaml
load_balancing_method: aux_loss | seq_aux_loss | deepseek_bias | quantile
```

References:
- Switch Transformer — https://arxiv.org/abs/2101.03961
- DeepSeek-V3 (bias + seq aux, §2.1.1–2.1.2) — https://arxiv.org/pdf/2412.19437
- Quantile balancing:
  - https://kexue.fm/archives/11619 (Jianlin Su, Chinese — original derivation)
  - https://jonathanc.net/blog/causal-routing-bias
  - https://openathena.ai/blog/quantile-balancing/

Reference repos (cross-check our impls and naming against these):
- `/Users/leon/Desktop/modal/moe/Megatron-LM` — has `aux_loss` (Switch), `seq_aux_loss`,
  `global_aux_loss`, `sinkhorn`, `none` as `moe_router_load_balancing_type`; expert-bias
  via `moe_router_enable_expert_bias`; z-loss via `moe_z_loss_coeff`. Source of truth
  for production-grade implementations.
- `/Users/leon/Desktop/modal/moe/nmoe` — sigmoid + DeepSeek-style bias update
  (`router_bias_update_rate`) with optional Switch aux (`aux_loss_alpha`). Compact
  reference for the bias-update path.
- `/Users/leon/Desktop/modal/moe/modal-nmoe` — our Modal-side fork; check that any
  trainer-side hooks (post-step bias update, all-reduce of counts) line up with what
  we're wiring here.

---

## 1. Verify Switch aux loss

- [ ] Confirm `load_balancing_loss_func` (`src/models/routing/load_balancing.py:39`) does not double-softmax (router already returns probs/sigmoid).
- [ ] Confirm `f_i` is computed from actual `selected_experts`, not recomputed top-k.
- [ ] Confirm `f_i` is rank-local (no all-reduce).
- [ ] Confirm uniform routing → loss ≈ 1.0; one-hot → loss ≈ N. Add unit test.
- [ ] Confirm consistent usage across `standard_moe.py:50`, `global_moe.py:184`, `moe_everything/model.py:529` and `:580` (attention experts).

## 2. Verify DeepSeek bias update

- [ ] Confirm `topk` uses biased scores, gather uses unbiased (`router.py:322` and `:341`).
- [ ] Confirm counts come from real forward only, not checkpoint recompute (`router.py:359`).
- [ ] Confirm `local_tokens_per_expert` is all-reduced before update in distributed mode (`training/routing.py:65`).
- [ ] Confirm zero-sum update `(s − mean(s)) * rate` and `±16` clamp.
- [ ] Confirm bias warmup schedule works (`get_bias_rate`, `training/routing.py:94`).
- [ ] Confirm bias buffer is persistent in checkpoint, count buffer is non-persistent and zeroed each step.
- [ ] Confirm bias update runs **post-optimizer-step**, not in `loss.backward()`.
- [ ] Add test: skewed routing for K steps → bias zero-sum invariant holds, overloaded experts get negative bias, loads converge to uniform.

## 3. Verify sequence-level aux loss

- [ ] Confirm `seq_load_balancing_loss_func` (`src/models/routing/load_balancing.py:166`) implements DeepSeek-V3 Eq. 17–20.
- [ ] Confirm Eq. 19 normalization is no-op for softmax router, converts sigmoid scores correctly for DeepSeek router.
- [ ] Confirm `f_i` uses actual `selected_experts` when provided.
- [ ] Confirm per-sequence then averaged over `B` and over MoE layers.
- [ ] Confirm token-mask path (MoE-Everything) divides by effective T per sequence.
- [ ] Confirm uniform routing → loss → 1.0. Add unit test.
- [ ] Confirm wired in all three families (`standard_moe.py:65`, `global_moe.py:199`, `moe_everything/model.py:542` and `:590`).

## 4. Implement quantile load balancing (new)

- [ ] Add `quantile_balancing_update(scores, state, target_q, eta)` in `src/models/routing/load_balancing.py` (pure function, no aux loss).
- [ ] Add `update_bias_from_quantile(...)` in `src/models/routing/bias.py`, mirroring `update_bias_from_counts`.
- [ ] In `src/models/router.py`, give `DeepSeekRouter` an optional path that updates `expert_bias = -EMA(quantile)` instead of the sign-update. Reuse the existing `expert_bias` buffer and the routing path (`topk(scores + bias)`, gather unbiased).
- [ ] Decide & implement distributed reduction: per-rank quantile averaged across ranks (`all_reduce(MEAN)`) for v0; t-digest later if needed.
- [ ] Add `_update_single_router_quantile_bias(...)` in `src/training/routing.py`. Dispatch in `update_expert_biases` based on `config.load_balancing_method`.
- [ ] Add config knobs: `load_balancing_method` (default `aux_loss`), `quantile_eta` (default 0.05), `quantile_target_q` (default `1 - top_k/num_experts`).
- [ ] When `load_balancing_method == "quantile"`, do **not** add Switch / seq aux loss to the total loss in any model `forward`.
- [ ] Add tests:
  - skewed input → after K steps loaded fraction approaches `K/N`.
  - quantile reaches uniform load in ≤½ the steps of `deepseek_bias` on the same input.
  - uniform input → bias stays near zero.

## 5. Wire `load_balancing_method` dispatch into all three families

- [ ] `standard_moe`: dispatch in `StandardMoEModel.forward` (loss inclusion) and `update_expert_biases` (bias path).
- [ ] `global_moe`: same. Decide whether quantile state is per-layer or global (default global = one shared `q_i` across all layers, since the pool is shared); expose `quantile_global_state: bool` flag.
- [ ] `moe_everything`: same, plus extend quantile to attention experts (Q/K/V/O) the same way the existing aux losses are extended in `moe_everything/model.py:580`.
- [ ] Make sure `none` is also a valid value for cleanly disabling all balancing.

## 6. Configs (DONE)

`configs/` is now flat with exactly three folders:
- `configs/4_layers/`, `configs/8_layers/`, `configs/16_layers/` — 8 yamls each
  (`dense`, `standard_moe`, `global_moe`, 5× `moe_everything_*`).

Each MoE config carries an inline `load_balancing_method` field with a comment
listing the valid values:
```yaml
load_balancing_method: aux_loss  # aux_loss | seq_aux_loss | deepseek_bias | quantile | none
```
To switch methods, edit the field in the config directly — no separate overlay folder.

All other configs (`scaling/`, `plan/`, `depth_matched/`, `depth_matched_fp32_no_liger/`,
and top-level standalone yamls) have been deleted.

## 7. 8×H200 benchmark protocol

Every config is benchmarked on the same distributed setup we already use for
real training: **8×H200 DDP**, bf16, `seq_len=1024`, the production trainer
loop (no special probe script). For each config, run **exactly 100 steps** and
record two numbers: median step time and token throughput.

- [ ] Write `scripts/bench_step.py` that re-uses `src/training/trainer.py` end-to-end
  (data loader, optimizer, DDP wrap, all-reduces, post-step bias update) and exits
  cleanly after 100 steps. No probe-only path — it must run the same code as
  training so the numbers are directly comparable.
- [ ] Drop the first 10 steps as warmup (CUDA caching allocator + cuDNN benchmark
  + first-iter compile/cudagraph). Take the median over the remaining 90.
- [ ] Metrics to report (one row per config):
  - **median step time** (s/step) — wall-clock between consecutive optimizer steps
  - **token throughput** (tokens/sec) = `effective_batch * seq_len / median_step_time`
    where `effective_batch = per_rank_batch * grad_accum * 8`
  - peak GPU memory (`torch.cuda.max_memory_allocated()` rank-max) — recorded but
    not used for sizing decisions
- [ ] Run across all configs in §9 (13 yamls × 3 depths = 39 runs). Persist results
  to `bench/results.json` with `{config_path, layers, depth_iters, per_rank_B,
  grad_accum, grad_ckpt, median_step_s, tokens_per_sec, peak_mem_gb}`.
- [ ] Each yaml ships with its own `batch_size` / `gradient_accumulation` /
  `gradient_checkpointing` already chosen — the benchmark just runs them as-is and
  reports what they actually deliver. No more "starting points" table.
- [ ] If a config OOMs in the first 10 steps, mark the row as `OOM` in the results
  and continue. The yaml owner adjusts the config and re-runs that single row.

What this replaces: the prior sizing table guessed per-config batch sizes from
back-of-envelope memory math. Real numbers from a 100-step run on 8×H200 are more
accurate, faster to update when the codebase changes (load-balancing rewrite, Liger
flip, etc.), and uniform across all configs. We'll just rerun the benchmark.

- [ ] Decide whether to flip `disable_liger: true → false` after the load-balancing
  changes land. Re-run §7 benchmark to quantify the throughput delta before
  committing the change.
- [ ] Make sure activations from `output_router_logits: true` are reflected in the
  numbers — they're on by default in every MoE config so this is the realistic
  training path.

## 8. Comparison run

- [ ] Run all four `load_balancing_method` values on `standard_moe` 8L, same data + seed, ~5K steps. Log per-step `f_i` distribution and final loss.
- [ ] Repeat on `global_moe` 8L and `moe_everything` 8L.
- [ ] Pin per-config defaults based on results.

## 9. New config matrix to author

Replace the current 8-yaml-per-depth set with this 13-yaml-per-depth matrix
(× 3 depths = 39 configs). Each non-dense row is one architecture × three
load-balancing variants (`deepseek_bias`, `aux_loss` Switch-style, `quantile`).

- [ ] **base** — `dense.yaml` (no MoE; one config per depth)
- [ ] **standard_moe** × {deepseek, switch, quantile}
  - per-layer expert pool (16 experts/layer), DeepSeek router class, vary balancing knob
- [ ] **global_moe** × {deepseek, switch, quantile}
  - single shared MLP pool across layers (16 × num_layers experts), DeepSeek router, vary balancing
- [ ] **recompute_kv** × {deepseek, switch, quantile}
  - `moe_everything`, `attn_expert_mode: per_head_recompute_kv`, shared expert banks across
    depth iters, **per-depth routers + per-depth prenorms** (the `_perlayer_prenorm` flavor:
    `per_layer_router=true`, `per_layer_attn_router=true`, `per_layer_norm=true`), vary balancing
- [ ] **recompute_kv_qkvo** × {deepseek, switch, quantile}
  - same recompute_kv attention bank, but **one single router reused at every depth iteration**
    (plain `recompute_kv` flavor: `per_layer_router=false`, `per_layer_attn_router=false`,
    `per_layer_mlp_router=false` → branch router, per-head attn routers, and MLP gate are all
    constructed once and the for-loop just calls them again every iter), vary balancing

Open questions to resolve before authoring:
- For each row, decide `branch_router_aux_loss_coef` and whether `branch_deepseek` should
  also flip with the balancing variant or stay fixed.
- Quantile balancing requires the implementation work in §4 to be done first.

The "vary balancing knob" means setting exactly one of the following per variant
(others zeroed out, so the comparison is clean):
- `deepseek` → `bias_update_rate=0.001`, `router_aux_loss_coef=0`, `seq_aux_loss_coef=0`
- `switch`   → `router_aux_loss_coef=0.001`, `bias_update_rate=0`, `seq_aux_loss_coef=0`
- `quantile` → `load_balancing_method=quantile` (post §4), all aux/bias coefs = 0

## 10. Per-router-class load-balancing (refactor)

The current configs are too coarse: a single global `bias_update_rate` /
`router_aux_loss_coef` / `seq_aux_loss_coef` applies (or doesn't) to every router
in the model. We want to be able to mix-and-match, e.g.:

```yaml
mlp_router:
  class: deepseek          # routing class: deepseek | softmax | sigmoid | sqrtsoftplus
  balancing: deepseek_bias # deepseek_bias | switch | seq_aux | quantile | none | exploration_only
  bias_update_rate: 0.001  # only used when balancing == deepseek_bias
  aux_loss_coef: 0         # only used when balancing == switch
attn_router:
  class: deepseek
  balancing: quantile
  quantile_eta: 0.05
branch_router:
  class: softmax
  balancing: exploration_only   # see §11
  exploration_rate: 0.1
```

What's required in code:

- [ ] **Config schema.** Replace the flat `use_deepseek_routing`, `branch_deepseek`,
  `router_aux_loss_coef`, `seq_aux_loss_coef`, `bias_update_rate`,
  `branch_router_aux_loss_coef`, `load_balancing_method`, `router_score_function`,
  `router_topk_ordering`, `router_exploration_rate`, `branch_router_exploration_rate`
  knobs with three nested groups: `mlp_router`, `attn_router`, `branch_router`. Keep
  a temporary back-compat shim in `model_factory.py` that maps the old flat fields
  onto the new nested ones, so existing yamls keep working during the transition.
- [ ] **Router construction.** `make_top1_router()` (`routing/helpers.py:28`) and
  `_make_gate()` (`mlp_bank.py:46`) should take an explicit per-class config dict
  instead of reading `config.use_deepseek_routing`. The `BranchRouter` constructor
  should likewise consume the `branch_router` group rather than `branch_deepseek`,
  `branch_sampling`, `branch_level`.
- [ ] **Loss accumulation.** In each model's `forward`, tag the source of each
  router-logits / selected-experts tuple with its class (`mlp` / `attn` / `branch`),
  and consult the per-class balancing config to decide which aux-loss term (if any)
  to add. `moe_everything/model.py:474-540` already separates body and attention
  aux losses; extend that pattern to fully decouple the three classes.
- [ ] **Bias-update dispatch.** Refactor `update_expert_biases` (`training/routing.py:11`)
  so it iterates routers and branches on the per-class config, not on a single global
  `bias_rate`. Also fix the singular-`branch_router` coverage gap noted in §5: the
  current walker only iterates `raw_model.branch_routers` (plural), so a shared
  branch router with `branch_deepseek=true` and `per_layer_router=false` never gets
  bias-updated.
- [ ] **Exploration rates.** Today `apply_router_exploration_rate` is a single-rate
  schedule across the whole model (with a special-case skip for branch when
  `branch_router_exploration_rate != router_exploration_rate`). Generalise to a
  per-class schedule so MLP, attn, and branch can each ramp their own exploration
  rate independently.
- [ ] **Validation.** Add a config-validator pass that: rejects `balancing=deepseek_bias`
  when the router `class` doesn't support it (e.g., a softmax router has no
  `expert_bias` buffer); requires `per_layer_router=true` whenever the branch router
  uses `deepseek_bias`; warns when two classes select conflicting balancing flavors
  with the same z-loss / seq-aux coefficients still set.

Migration plan:
1. Land §10 schema + back-compat shim with all existing yamls untouched (CI green).
2. Land §4 quantile so it's available as a per-class option.
3. Land §11 exploration-only branch so it's available as a per-class option.
4. Rewrite the §9 yaml matrix using the new nested schema; delete the back-compat shim.

## 11. Exploration-only branch router

Motivation: the branch router's job is to decide ATTN vs MLP per token at each
depth. We do **not** actually want a uniform 50/50 split — we want the model to
discover whatever attn/mlp allocation is best for the data. Forcing load
balancing on the branch (Switch aux, DeepSeek bias toward 50/50, etc.) actively
hurts that, because it pushes the model away from the compute graph it would
otherwise prefer.

But removing all balancing risks **router collapse**: the branch router latches
onto one branch (say all-MLP) early because gradients flow only through the
chosen branch (`BranchRouter.forward` zeros out the unchosen branch via
`attn_mask`/`mlp_mask` — `routing/routers.py:124–131`), and the unused branch
parameters never receive gradient again. The router has no incentive to
re-explore.

The fix: keep load-balancing **off** for the branch but force occasional
exploration so both branches stay alive long enough to learn. Concretely, this
is a new value for the per-class `balancing` field:

```yaml
branch_router:
  class: softmax
  balancing: exploration_only
  exploration_rate: 0.1            # P(force a uniformly random branch) per token, training only
  exploration_decay: cosine        # constant | cosine | linear
  exploration_min: 0.01            # floor at end of decay
  exploration_warmup_steps: 1000
```

Implementation sketch:

- [ ] In `BranchRouter.forward` (`routing/routers.py:67`), when the per-class
  config selects `exploration_only`:
  - Skip *all* aux-loss accumulation for this router (`branch_router_aux_loss_coef`
    is forced to 0; no DeepSeek bias is registered or updated).
  - With probability `p_explore(step)`, replace the argmax `choice` with a
    uniform-random 0/1 draw per token (or per-sequence if `branch_level=seq`).
    The replaced tokens contribute gradient to the branch they were forced into.
  - Keep the existing softmax probability scaling on `w_attn` / `w_mlp` so the
    forward stays differentiable; only the *selection* is overridden.
- [ ] Add `apply_router_exploration_rate` support for a *separate* branch
  schedule (today it forces branch and body to share or skips branch entirely).
- [ ] Optional softer alternative: instead of hard random override, add a small
  entropy bonus on the branch-router probabilities, `H(p_attn, p_mlp) * coef`,
  added (positive) to the loss to keep the distribution from collapsing.
  Default off; gate behind `branch_router.entropy_bonus_coef > 0`.
- [ ] Telemetry: log per-step branch fraction (`% tokens choosing ATTN`),
  per-depth if `per_layer_router=true`, and current `p_explore`. Without this
  we can't tell collapse from a healthy emergent imbalance.
- [ ] Tests:
  - With `exploration_only`, branch fractions can drift to non-50/50 and stay
    there (no aux-loss pulling them back).
  - Forced-collapse synthetic input (e.g., feed the same token repeatedly with
    a teacher signal that always wants MLP) → after K steps both branches still
    receive nonzero gradient because exploration keeps firing.
  - With `exploration_rate=0`, model can collapse to one branch (this is the
    failure case the feature exists to prevent).

Open questions:
- Does exploration-only on the **body** routers (MLP / attn experts) make sense
  too? Probably yes — same logic applies — but starts to look identical to the
  existing `router_exploration_rate` knob with all aux/bias terms zeroed. Worth
  confirming the framing is consistent across all three router classes once §10
  lands.
- For seq-level branch (`branch_level=seq`), exploration probability applies
  per-sequence; double-check this doesn't make the early-step gradient signal
  too noisy.
