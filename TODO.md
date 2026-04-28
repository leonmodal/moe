# Outstanding TODO

Items that have been explicitly flagged as future work but are NOT in the current implementation plan (`docs/plan.md`). Each entry names the originating context and what would be needed to bring it into scope.

## Load-Balancing Methods

### Per-family tuning of `quantile_eta` / `quantile_target_q` (deferred from `docs/plan.md` DEC-11)

**Current defaults (DEC-11 RESOLVED):**
- `quantile_eta: 0.05` (per draft §4)
- `quantile_target_q: 1 - effective_top_k / effective_num_experts` per router class:
  - plain top-k MLP routers: `1 - top_k / num_experts`
  - group-limited routing (`num_groups`/`group_topk`): same formula — the relevant pool is still `num_experts`
  - per-head top-1 attention routers in `moe_everything`: `1 - 1/n_attn_experts`

**Open question — tune per family:**
- Sweep `quantile_eta ∈ {0.01, 0.02, 0.05, 0.1, 0.2}` × {`standard_moe`, `global_moe`, `moe_everything`} × multiple seeds, training to convergence (~5K+ steps each).
- Same kind of sweep for `quantile_target_q` deviations.
- Compare final loss + balance metrics + convergence speed; pick per-family defaults if the data justifies a deviation from the analytic formula.

**Why deferred:**
- Tuning sweeps require training-to-convergence to be reliable, and convergence-runs are expensive (each ~hours on 8×H200). Bundling them into the correctness/throughput plan would balloon scope.
- The analytic defaults are reasonable starting points; AC-10's "skewed input → uniform load within K steps where K ≤ ½ deepseek_bias" sanity check would catch order-of-magnitude wrong defaults.
- Spiritually identical to the §8 comparison sweep, which is also deferred (DEC-9).

**Bring into scope when:**
- The §8 comparison sweep is run as a follow-up plan (DEC-9 lift), and the per-family tuning sweep can ride along with it (same hardware budget, same logging path).
- Or: quantile balancing produces a clearly-suboptimal balance metric on `bench/comparison_summary.md` (when that exists) and we suspect the defaults are the cause.

**Pointers:**
- Implementation: `src/models/routing/load_balancing.py:quantile_balancing_update` (post task15) and `src/models/routing/bias.py:update_bias_from_quantile` (post task16).
- Documentation: `docs/routing.md` "Quantile Balancing" section (added by task15-17 doc obligations), which carries the same open-question note as this entry.

## Speed Borrowing (also deferred — see `docs/plan.md` "Out of Scope")

### RDEP-style expert parallelism port from `modal-nmoe`

**What it is:** `modal-nmoe`'s custom expert-parallel pattern that avoids NCCL all-to-all on the MoE critical path. Touches `modal-nmoe/nmoe/csrc/` (custom CUDA kernels) and the dispatch pipeline.

**Why deferred:**
- Multi-week project; much bigger surface than other Priority 3 speed-borrowing items.
- We're DDP-only today (no expert parallelism); RDEP requires introducing EP infra first.
- nmoe targets B200 specifically; H200 hardware adaptation likely needed.
- AC-23 sanity equivalence is unlikely to survive an EP rewrite without dedicated work.

**Bring into scope when:**
- The non-EP speed-borrowing wins from `docs/plan.md` Milestones J/K leave significant throughput on the table (i.e., the matrix is meaningfully bottlenecked on dispatch / collectives).
- A dedicated RDEP-port plan is authored as a follow-up to the current correctness/throughput plan.

### FP8 / NVFP4 expert weights

**Status in current plan:** Milestone K is an analyze-only feasibility study (`docs/research/fp8_nvfp4_port_feasibility.md`). Implementation deferred.

**Bring into scope when:**
- The feasibility analysis from Milestone K reports favorable risk/reward.
- A dedicated FP8/NVFP4 port plan is authored as a follow-up.

### `disable_liger: true → false` matrix flip

**Current state:** the 39-config matrix keeps `disable_liger: true` (DEC-16). Milestone J's "Liger re-enablement for `moe_everything`" runs as a 3-config sanity sweep, but the matrix-wide flip waits.

**Bring into scope when:**
- Milestone J reports that Liger `swiglu` + `fused_linear_cross_entropy` can be re-enabled for `moe_everything` without breaking AC-23 sanity equivalence AND with a measurable throughput win.
- A follow-up plan flips the matrix and re-baselines `bench/results.json`.

## Architecture Refactors

### Megatron-style autograd-hook aux-loss refactor (deferred from `docs/plan.md` DEC-15)

**Current state in plan:** DEC-15 sets `output_router_logits=True` for aux methods only (`aux_loss`, `seq_aux_loss`); DETACH-ONLY for non-aux methods. The aux-loss path still consumes `output.router_logits` from the model's forward output.

**Cleaner design (matches Megatron exactly):** apply aux-loss INSIDE the router's forward via an autograd-hook trick (Megatron uses a custom autograd Function called `MoEAuxLossAutoScaler` — `Megatron-LM/megatron/core/transformer/moe/router.py` `_apply_aux_loss` and friends). The aux-loss tensor is stitched into the main loss via the hook; `output_router_logits` is no longer needed at all.

**Why deferred:** meaningful surgery to `MoEverythingForCausalLM.forward`, `StandardMoEModel.forward`, and `GlobalMoEForCausalLM.forward`. Not gated on Priority 1 correctness — the DETACH-ONLY shim from DEC-15 already gets us most of the activation-memory win.

**Bring into scope when:**
- The §8 comparison sweep is run as a follow-up plan (DEC-9 lift), OR
- The Priority 2 throughput characterization (AC-24) reveals that aux methods are leaving meaningful activation-memory on the table that this refactor would recover.

**Pointers:**
- `Megatron-LM/megatron/core/transformer/moe/router.py:285` (`_apply_aux_loss`) — reference implementation.
- `Megatron-LM/megatron/core/transformer/moe/moe_utils.py:save_to_aux_losses_tracker` — the autograd-hook entry point.
- Affected files in this repo: `src/models/standard_moe.py`, `src/models/global_moe.py`, `src/models/moe_everything/model.py`, `src/training/model_factory.py` (`output_router_logits` dispatch).

## Comparison & Defaults

### §8 comparison sweep (DEC-9 deferred)

**What it is:** 4 methods × 3 families × ~5K training steps each = 12 training runs to characterize `(family, method) → (final_loss, mean_f_i_KL, max_f_i)`.

**Why deferred:** the current plan is correctness-first + throughput-characterization; running training-to-convergence comparisons is a follow-up.

**Bring into scope when:**
- The current plan's correctness work (Priority 1) has landed and AC-23 sanity equivalence holds.
- The §7 throughput-characterization work (Priority 2) has produced `bench/results.json` so we know the per-config compute cost of running a 5K-step sweep.

### Pin per-config `load_balancing_method` defaults from §8 results

**Status:** depends on §8 sweep above. Deferred together.

## How to update this file

When adding a new TODO:
- Name the originating context (`docs/plan.md` DEC-N, AC-N, or the conversation that generated it).
- Document the current state, the deferred work, why it was deferred, and the trigger condition for bringing it into scope.
- Cite specific file paths and symbols so a future reader can find the affected code without re-deriving the context.
