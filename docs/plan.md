# MoE Load-Balancing Methods — Correctness, Throughput, and Speed Borrowing

> **Scope pivot 2026-04-28**: this plan is reframed around three priorities (in order):
> 1. **Correctness** of every load-balancing method + every model family, validated by an "insane amount" of unit tests. Headline correctness test: `moe_everything` with `sanity_check_mode='alternating_global_moe'` produces identical forward / loss / gradient to `global_moe` (proving the precompute-kv pipeline is wired correctly).
> 2. **Throughput characterization**: for each model in the matrix, find the maximum stable batch (using `gradient_accumulation`, `gradient_checkpointing`, and chunked / fused CE loss as the levers), then measure the fastest end-to-end token throughput at that batch.
> 3. **Speed borrowing** from `nmoe/`, `modal-nmoe/`, and `Megatron-LM/` — port grouped GEMM patterns, kernel fusion, FP8/NVFP4 paths, RDEP, and any other measurable speed wins (with each port gated by a measured throughput delta).
>
> The §8 comparison sweep (4 methods × 3 families × ~5K-step training runs) is **DEFERRED**. We will run it later, after the three priorities above are done. The §9 13-yaml × 3-depth matrix is partially in scope: configs MUST exist to support the §7 / throughput-characterization sweep, but we do NOT pin per-config defaults from sweep results yet (DEC-9 = deferred).

## Goal Description

The plan delivers a correctness-first MoE load-balancing rewrite that ships **four selectable load-balancing methods** (`aux_loss`, `seq_aux_loss`, `deepseek_bias`, `quantile`, plus `none`) into all three model families (`standard_moe`, `global_moe`, `moe_everything`), gated by a single config knob `load_balancing_method`. The deliverables, in priority order:

**Priority 1 — Correctness.** Verify the existing aux-loss and bias-update implementations against their reference papers (`Switch Transformer`, `DeepSeek-V3`); fix the latent correctness gaps the audit surfaces (the `update_expert_biases` walker is dead code today; attention aux loss in `moe_everything` is gradient-free; `BranchRouter.forward` is gradient-checkpointing-unsafe; load-balancing state is split per-router instead of bank-level for shared-expert architectures); implement the new `quantile` method; refactor the flat router-config schema into per-router-class groups (`mlp_router`, `attn_router`, `branch_router`) with a `balancing` field per group; add an `exploration_only` mode for the branch router; and ship a comprehensive test suite (every AC has positive + negative + DDP + gradient-checkpointing + sanity-equivalence coverage).

**Priority 2 — Throughput characterization.** For each model in the 13-yaml × 3-depth matrix (39 configs), use `scripts/bench_step.py` to **search** for the maximum stable batch (using `gradient_accumulation`, `gradient_checkpointing`, and a chunked / fused CE loss path as the levers — chunked CE does not exist as a custom implementation today; Liger's `fused_linear_cross_entropy` is disabled for `moe_everything` and would either need to be enabled there or replaced by a port from `nmoe`/`modal-nmoe`), then measure the fastest end-to-end token throughput at that batch on 8×H200 DDP, with `seq_len=1024`. The benchmark runs the production trainer with W&B / checkpoint / eval / heatmaps disabled (DEC-8). Output: `bench/results.json` with `(config_path, max_batch, grad_accum, grad_ckpt, chunked_ce_on, median_step_s, tokens_per_sec, peak_mem_gb)` per config.

**Priority 3 — Speed borrowing.** Each speed-up is gated by a measured throughput delta on at least three configs from the matrix (one per family), at the max stable batch from Priority 2. Candidates to port from `nmoe/`, `modal-nmoe/`, and `Megatron-LM/` include: grouped GEMM (`grouped_mm` already wired in our `set_experts_implementation`; verify it's actually faster than the eager path on H200), FP8 / NVFP4 expert weights (`modal-nmoe` has these, our codebase doesn't yet), kernel fusion (Liger's full kernel set works for `standard_moe` / `global_moe` already; for `moe_everything` we need to check why `swiglu` and `fused_linear_cross_entropy` are disabled and whether they can be re-enabled), RDEP-style expert parallelism (a much bigger port — flagged for a later plan), and any per-step CUDA graph / cudagraph optimizations the bench reveals as bottlenecks.

All GPU work runs on Modal H200 sandboxes (max 8 nodes concurrent, idle-off) using the project's `.claude/skills`.

> **Out of scope for this plan**: §8 comparison sweep (12 training runs to pick winners — deferred); pinning per-config `load_balancing_method` defaults from §8 results (DEC-9 = deferred); cross-arch tuning of `quantile_eta` / `target_q` per family (deferred — defaults from DEC-11 stand for v0).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive tests (expected to PASS) and negative tests (expected to FAIL when behavior is wrong).

- AC-1: `load_balancing_method` is the dispatch knob authoritative for the *flat* schema; once the per-router-class schema (AC-13) lands, per-class `mlp_router.balancing` / `attn_router.balancing` / `branch_router.balancing` become canonical and the top-level `load_balancing_method` is treated as a legacy/global shorthand that the back-compat shim (DEC-7) expands into the three per-class fields. `model_factory.py:build_model` and every model family's `forward` / `update_expert_biases` path consult the per-class field after expansion.
  - Positive Tests:
    - For each value in `{aux_loss, seq_aux_loss, deepseek_bias, quantile, none}`, a unit test instantiates each model family from a yaml stub, runs one forward+backward+optimizer-step+post-step-bias-update on a fixed seed, and asserts (a) only the expected loss term is added to the total, and (b) only the expected bias-update path runs.
    - With `load_balancing_method: aux_loss` set in the yaml, the trainer's effective `seq_aux_loss_coef` and `bias_update_rate` resolve to zero regardless of any legacy field values present.
    - Post-AC-13: a yaml that sets BOTH a top-level `load_balancing_method` AND per-class `*.balancing` fields is either rejected (preferred) or the per-class fields take precedence with a deprecation warning; `model_factory.py` documents the rule and a regression test locks it in.
  - Negative Tests:
    - A yaml with `load_balancing_method: aux_loss` AND non-zero `bias_update_rate` is either rejected by the validator or `bias_update_rate` is auto-zeroed (per DEC-3a); silent-permit must fail this test.
    - With `load_balancing_method: deepseek_bias` and `router_aux_loss_coef = 0.001`, Switch aux must NOT be added to the total loss.

- AC-2: The `update_expert_biases` walker reaches every `DeepSeekRouter` instance in `standard_moe`, `global_moe`, and `moe_everything` (MLP and attention banks), every `BranchRouter` (singular *and* plural attribute name), and zero softmax routers. The walker must work after `unwrap_model(...)` returns the *outer* `…ForCausalLM` wrapper (not the inner `…Model`).
  - Positive Tests:
    - A test instantiates each family with `load_balancing_method: deepseek_bias`, calls `update_expert_biases`, and asserts `expert_bias` of every `DeepSeekRouter` and `branch_bias` of every `BranchRouter(use_deepseek_style=True)` shifts by the expected zero-sum delta on synthetic skewed input.
    - A coverage-counter test asserts the walker visited the documented number of routers for each family (e.g. `standard_moe(8L)` → 8 MLP routers; `moe_everything(8L, per_head_precompute_kv)` → 1 MLP router + 4 attention routers + 1 or 8 branch routers depending on `per_layer_router`).
    - A DDP smoke test (2 ranks) calls `update_expert_biases(..., distributed=True)` and asserts the per-rank `local_tokens_per_expert` are all-reduced (sum) before the bias delta is computed; per-rank biases are identical post-update.
    - An FSDP smoke test (2 ranks, FSDP auto-wrap) confirms the walker traversal still finds every router after FSDP module-tree wrapping (or the test is skipped with an explicit "FSDP out of scope per DEC-14" message — see DEC-14).
  - Negative Tests:
    - A model whose layers contain only `ExplorationTopKRouter` (softmax) must produce zero `expert_bias` writes.
    - With `load_balancing_method ≠ deepseek_bias`, `update_expert_biases` must be a no-op (exit early, no bias buffer mutation).
    - A test that calls `update_expert_biases` on a model whose `unwrap_model(...)` returns the `…ForCausalLM` wrapper while the walker's `get_all_routers()` is defined only on the inner `…Model` (the regression Codex flagged in round 1) must fail by reporting zero routers visited.

- AC-3: `bias_update_rate`, `bias_warmup_start`, `bias_warmup_steps`, `seq_aux_loss_coef`, `router_aux_loss_coef`, `load_balancing_method`, and per-projection bias rates (`bias_rate_q/k/v/o/mlp/branch`) are all read from a single canonical block per the schema decision (DEC-3b).
  - Positive Tests:
    - A test that places each field under both `model:` and `training:` blocks and confirms the loader resolves them deterministically per the documented rule.
  - Negative Tests:
    - A yaml with `bias_update_rate: 0.001` placed only under the *wrong* block produces a validator error or warning (not a silent zero, which is today's behavior).

- AC-4: Switch aux loss documented uniform-routing baseline matches its implementation, `f_i` stays rank-local (no all-reduce, per draft §1), and call sites pass *actual* `selected_experts` (not recomputed top-k).
  - Positive Tests:
    - `tests/test_aux_loss_fix.py::test_uniform_routing` passes with the documented baseline (DEC-1: either `top_k` or `1.0`); a skewed input produces a strictly larger value.
    - A no-collective assertion test wraps `load_balancing_loss_func` and asserts no `dist.all_reduce`/`all_gather`/`reduce_scatter` is invoked during its body (uses a stubbed `dist` module that fails on any collective).
    - A divergence test injects `selected_experts` that deliberately differ from `torch.topk(scores, k)` (e.g. exploration-perturbed) and asserts `load_balancing_loss_func(..., selected_experts=...)` produces a numerically different result than the internal-topk-recompute path — locking the "actual hard assignment" call-site contract.
    - Token-mask path produces the same per-active-token loss as a dense sequence with the same active tokens.
  - Negative Tests:
    - A test mocking double-softmax (applying softmax twice on already-softmaxed scores) must produce a smaller-than-correct loss for the same skewed input — locking the existing "no double softmax" fix in place.
    - Replacing the call-site `selected_experts=self._collect_selected_experts()` with `selected_experts=None` (forcing internal recompute) and running with non-trivial `router_exploration_rate` must change the asserted loss value, demonstrating the actual-vs-recomputed gap the §1 bullet warns about.

- AC-5: Sequence-level aux loss reproduces DeepSeek V3 Eqs. 17–20 with uniform routing → `1.0`, runs per-sequence then averaged over batch and active MoE layers, and respects `token_masks` by dividing by effective T per sequence.
  - Positive Tests:
    - `tests/test_seq_loss_vs_paper.py` (extend if needed) asserts uniform routing → `1.0` for both softmax and sigmoid (post Eq. 19 normalization) routers.
    - Skewed input produces strictly larger loss; an entire layer skipped via `token_masks` does not contribute to the average.
    - `selected_experts=None` path and explicit-`selected_experts` path agree to fp32 round-off.
  - Negative Tests:
    - A version of `seq_load_balancing_loss_func` that omits Eq. 19 normalization for sigmoid routers must produce a strictly larger uniform-routing loss (locks the normalize step in).

- AC-6: DeepSeek bias update parity test verifies zero-sum semantics (per DEC-2), `±16` clamp, no-op on zero counts, post-optimizer-step ordering, bias warmup schedule, and DDP all-reduce of counts.
  - Positive Tests:
    - After K=200 steps of synthetically skewed routing, the bias buffer's running sum stays within fp32 round-off of zero (when DEC-2 selects nmoe zero-sum); overloaded experts have negative bias; under-loaded have positive; load distribution converges to within 5% of uniform within 500 steps.
    - Bias buffer is persistent across `save_checkpoint`/`load_checkpoint`; `local_tokens_per_expert` is non-persistent and zero-after-update.
    - Bias-warmup test: with `bias_warmup_start=0.0`, `bias_update_rate=0.001`, `bias_warmup_steps=100`, the effective rate at step 0 is 0, at step 50 is 0.0005, at step 100 is 0.001, and stays at 0.001 for steps > 100. Reproduces `get_bias_rate(...)` semantics from `src/training/routing.py:94`.
    - DDP all-reduce test (2 ranks): each rank sees a different skewed routing on the same step; after `update_expert_biases(..., distributed=True)`, both ranks' `local_tokens_per_expert` (pre-zero) sum to the same global count and both ranks' post-update `expert_bias` are identical.
  - Negative Tests:
    - A test that calls `update_expert_biases` from inside `loss.backward()` (or before `optimizer.step()`) must fail; the trainer's call-site test asserts the call happens strictly after `optimizer.step()`.
    - With zero counts (no tokens routed to any expert), the bias buffer must NOT change.
    - Disabling the DDP `all_reduce` (single-rank semantics on the multi-rank test) makes per-rank `expert_bias` diverge between ranks — locks the "must all-reduce before update" invariant.

- AC-7: DeepSeek router forward-pass recipe (gather → normalize → scale) is bit-equivalent (to bf16 round-off) with `nmoe.Router` on a fixed seed; `topk_scaling_factor` defaults are documented per-config.
  - Positive Tests:
    - A parity test loads `nmoe/nmoe/model.py:Router` and our `DeepSeekRouter` with shared weights, runs the same input, and asserts identical `(weights, indices)` to bf16 round-off; this includes the unbiased gather, the `top_k > 1` sum-to-1 normalize, the `top_k == 1` skip, and the `topk_scaling_factor` post-norm rescale.
    - An autocast-enabled test confirms the scoring path runs in fp32 sigmoid on fp32 logits regardless of the outer autocast scope.
    - A config-comment audit test checks that every `configs/*/` yaml that sets `topk_scaling_factor` also carries a comment naming the source value (DeepSeek-V3 default `2.5`, nmoe NVFP4 default `1.0`); a yaml without the comment fails the audit.
  - Negative Tests:
    - Replacing `scores.gather(...)` with `biased_scores.gather(...)` causes the parity test to fail.
    - Removing the `top_k > 1` guard on normalization causes the parity test to fail at `top_k == 1`.
    - Swapping the order of normalize and `topk_scaling_factor` causes the parity test to fail.

- AC-8: Attention aux loss in `moe_everything` is gradient-bearing — adding `router_aux_loss_coef * attention_aux_loss` to the total loss produces a non-zero gradient on the attention router weights.
  - Positive Tests:
    - A test runs one forward+backward on `moe_everything` with `attn_expert_mode: per_head_fully_independent`, deliberately-skewed routing, and `router_aux_loss_coef=0.001`, and asserts every attention router's `weight.grad.norm() > 0`.
    - The same test with `seq_aux_loss_coef=0.0001` asserts the attention seq-aux contribution also produces non-zero gradients.
  - Negative Tests:
    - With `router_aux_loss_coef=0.0`, the attention router weights' grad must be unchanged from the no-aux baseline (i.e., the aux loss contributes nothing when its coef is zero).

- AC-9: Under gradient checkpointing, `BranchRouter` and `DeepSeekRouter` count buffers and exploration sampling are checkpoint-safe — recompute does NOT mutate `local_counts`/`local_tokens_per_expert` and re-runs use the same exploration mask.
  - Positive Tests:
    - A test runs one forward+backward with `gradient_checkpointing=True` on `moe_everything` with `branch_router(use_deepseek_style=True)` and asserts `local_counts` matches the checkpointing-disabled run on the same seed.
    - The `last_exploration_mask` recorded on first pass equals the mask used during recompute (not freshly resampled).
  - Negative Tests:
    - Disabling the `is_checkpoint_recompute()` guard in `BranchRouter.forward` makes the test fail with double-counted `local_counts`.
    - Replacing the cached exploration mask with a fresh `torch.rand(...)` on recompute changes branch choice on the recompute pass and breaks gradient consistency.

- AC-10: The new `quantile` load-balancing method is implemented as a pure function and integrated into the `update_expert_biases` dispatch, with explicit fp32 mixed-precision boundaries and explicit accumulation semantics across gradient-accumulation micro-batches and branch-masked paths.
  - Positive Tests:
    - `quantile_balancing_update(scores, state, target_q, eta)` exists in `src/models/routing/load_balancing.py`; `update_bias_from_quantile(...)` exists in `src/models/routing/bias.py`; `_update_single_router_quantile_bias(...)` exists in `src/training/routing.py` and is reachable via `update_expert_biases` when `load_balancing_method == "quantile"`.
    - Score accumulation: each `DeepSeekRouter.forward` in training mode appends its (detached) raw scores into a non-persistent `local_quantile_scores` buffer (or equivalent rolling accumulator); the post-step quantile is computed across **all** accumulated micro-batches in the optimizer step and the buffer is cleared after the bias update. A grad-accumulation test (`gradient_accumulation_steps=4`) asserts the post-step quantile equals the quantile computed over the concatenated 4-microbatch scores.
    - Branch-masked path: in `moe_everything`, when an MLP router's `token_mask` is empty for a given depth iteration (no tokens routed to MLP), the accumulator records nothing and the post-step quantile is computed from non-empty micro-batches only.
    - Zero-active-token case: an entire optimizer step with zero active tokens for a router produces `update_bias_from_quantile` no-op (no EMA update, no clamp, no buffer mutation) — covered by a dedicated test.
    - Mixed-precision boundary: the score accumulation, quantile computation, EMA update, bias buffer, and clamp all run in fp32 even when the surrounding `forward` is under `torch.autocast(dtype=bf16)`. A test under bf16 autocast asserts every intermediate dtype is `torch.float32`.
    - A skewed-input test reaches uniform load (within 5% of `1/N`) within K steps where K ≤ ½ the steps `deepseek_bias` needs on the same input/seed.
    - Uniform input produces a near-zero bias trajectory (drift bounded by `eta * rms(noise)` per step).
    - Default `target_q = 1 - top_k / num_experts` is documented; the per-router-class adaptation rule (DEC-11) is documented for group-limited and per-head-top-1 attention routers.
  - Negative Tests:
    - With `load_balancing_method == "quantile"`, neither Switch aux nor seq aux is added to the total loss in any model family's `forward`.
    - With `world_size == 1`, all `dist.*` calls are guarded by `dist.is_initialized()` and the test passes outside DDP.
    - Replacing fp32 quantile/EMA with bf16 produces a measurable bias-buffer drift on a fixed seed (locks the fp32 boundary in).
    - Replacing the cross-microbatch accumulator with a "use only the last forward" cache makes the grad-accumulation test fail.

- AC-11: The DDP reduction for the quantile statistic is documented and tested for accuracy.
  - Positive Tests:
    - A DDP test (4 ranks, intentionally skewed per-rank score distributions) compares the implemented reduction (per DEC-4) against the exact global quantile (`all_gather` baseline) and asserts the bias is bounded within the documented tolerance.
    - With identically-distributed per-rank inputs, every reduction strategy agrees with the `all_gather` baseline to fp32 round-off.
  - Negative Tests:
    - An adversarial input where per-rank means agree but per-rank distributions are bimodal makes the per-rank-mean reduction diverge from the exact global quantile by > tolerance, demonstrating the bias bound is real.

- AC-12: Quantile EMA state lifecycle matches the user's checkpoint policy (DEC-5: persistent or non-persistent).
  - Positive Tests:
    - A resume-parity test trains for K steps, checkpoints, resumes, trains for K more steps, and asserts the bias buffer at step 2K matches an uninterrupted 2K run to fp32 round-off (when EMA is persistent).
    - If non-persistent, the resume-parity test asserts EMA state recomputes deterministically over a documented warmup window after resume.
  - Negative Tests:
    - Toggling persistence flips which test passes — both cannot pass simultaneously without a code change.

- AC-13: Per-router-class config schema (`mlp_router`, `attn_router`, `branch_router`) replaces the flat schema. Migration is **one-time, on-disk** (DEC-7 RESOLVED): a migrator script rewrites every yaml under `configs/` from flat → nested in a single PR, no runtime shim is shipped, and the validator (AC-17) rejects any yaml that still uses the flat schema afterwards. Migration parity covers both forward/loss outputs AND post-step bias-update behavior, and is asserted **before** the migrator PR merges (after merge, the flat-shape code path no longer exists). The per-class `balancing` enum is canonical: **`aux_loss | seq_aux_loss | deepseek_bias | quantile | none`** for `mlp_router` and `attn_router`, **`aux_loss | seq_aux_loss | deepseek_bias | quantile | none | exploration_only`** for `branch_router` (the `exploration_only` value exists only on the branch class). The original draft §10's `switch` and `seq_aux` tokens are rewritten by the migrator (`switch` → `aux_loss`, `seq_aux` → `seq_aux_loss`); the runtime does NOT accept those tokens post-migration.
  - Positive Tests:
    - A forward-parity migration test instantiates the same model from (a) an old flat yaml and (b) the migrated nested yaml, runs the same input on the same seed, and asserts identical `loss`, `aux_loss`, and router outputs to fp32 round-off, for every yaml currently under `configs/{4,8,16}_layers/`.
    - A bias-update parity migration test runs one full optimizer step (forward + backward + `optimizer.step()` + `update_expert_biases`) on flat-yaml and migrated-yaml models with the same seed and asserts identical `expert_bias` (or quantile EMA state) on every router; this covers `deepseek_bias` and `quantile` methods explicitly so the migration cannot silently break the update dispatch while preserving forward outputs.
    - A unit test confirms decoupling — setting `attn_router.balancing: quantile` while `mlp_router.balancing: deepseek_bias` produces exactly the expected per-class behavior in `forward` and `update_expert_biases`.
  - Negative Tests:
    - A yaml with `branch_router.balancing: deepseek_bias` and `branch_router.class: softmax` is rejected by the validator (softmax has no `expert_bias` buffer).
    - A yaml with `mlp_router.bias_update_rate: 0.001` and `mlp_router.balancing: aux_loss` is rejected by the validator.

- AC-14: `BranchRouter` supports `balancing: exploration_only` mode that disables all aux-loss accumulation and forces uniform-random branch choice with probability `p_explore(step)` per token (or per sequence when `branch_level: seq`). Tests use deterministic / windowed assertions, not flaky per-step probabilistic assertions.
  - Positive Tests:
    - **RNG-injected branch test**: with `branch_router.balancing: exploration_only` and a `torch.Generator` seeded so the first 4 of 8 tokens trip the exploration mask, the recorded `last_exploration_mask` exactly matches the expected pattern; the chosen branch for the masked tokens equals the injected uniform random value, not the argmax.
    - **Windowed gradient-flow test**: over a 100-step window with `exploration_rate=0.5` (deterministic seed), the cumulative gradient norm on each branch's parameters is strictly positive AND the per-branch contribution count is within 10% of the analytic expectation (`expected_attn = 0.5 * 100 * batch * seq` give-or-take exploration assignments) — a windowed assertion, not a per-step "always non-zero" assertion.
    - **Schedule shape test (parameterized over all three schedules)**: for each `exploration_decay ∈ {constant, cosine, linear}` with `exploration_min: 0.01`, `exploration_warmup_steps: 1000`, the recorded `p_explore` trajectory matches the analytic schedule pointwise to fp32 round-off (each schedule is deterministic given step + config). All three are required to pass.
    - **Telemetry test**: per-step branch fraction (`% tokens choosing ATTN`) and current `p_explore` are logged via the trainer's logging path on every step.
  - Negative Tests:
    - **Deterministic-collapse test**: with `exploration_rate=0.0` and `balancing: exploration_only`, a forced-collapse synthetic input (constant teacher signal that always wants MLP, fixed seed, K=500 steps) makes the cumulative ATTN-branch gradient drop below `1e-6` of the MLP-branch gradient — the failure mode this feature exists to prevent. The "K steps" and the threshold are documented in the test comment so the bound is explicit, not flaky.
    - With `balancing: exploration_only`, neither `branch_router_aux_loss_coef` nor any DeepSeek bias update affects the branch router state (call-site check + buffer-shape check).

- AC-15: `apply_router_exploration_rate` walker generalizes to per-class exploration schedules — `mlp_router`, `attn_router`, `branch_router` each have independent `exploration_rate`, `exploration_decay`, `exploration_min`, and `exploration_warmup_steps`.
  - Positive Tests:
    - A test sets distinct schedules per class and asserts the walker pushes each class's rate independently at each step; coverage assert verifies every router was updated.
  - Negative Tests:
    - When all three classes share the same schedule (back-compat case), the walker still produces consistent per-step rates.

- AC-16: `none` is a valid value for every per-class `balancing` field and cleanly disables all balancing for that class (no aux loss term, no bias update, no exploration override).
  - Positive Tests:
    - With every class set to `balancing: none`, one training step produces zero bias deltas, zero aux-loss contribution, and the unchanged forward output matches the corresponding baseline up to routing.
  - Negative Tests:
    - With one class set to `none` and another set to `aux_loss`, only the latter contributes — separability test.

- AC-17: Configuration validator (`scripts/validate_configs.py` or equivalent) covers the new schema and rejects illegal combinations.
  - Positive Tests:
    - Every yaml in the new 13-yaml × 3-depth matrix passes the validator.
  - Negative Tests:
    - Each illegal combination has a dedicated test asserting validator rejection with a specific error: (a) `balancing: deepseek_bias` on a softmax router; (b) shared branch router with `balancing: deepseek_bias` (rejected unless `per_layer_router=true` AND the walker explicitly handles the singular-attribute case — covered by task3 which fixes the singular-`branch_router` discovery); (c) z-loss/seq-aux coefficients still set when classes disagree on flavor.

- AC-18: New 13-yaml × 3-depth (4/8/16) config matrix exists at `configs/{4,8,16}_layers/`, with per-yaml `batch_size`, `gradient_accumulation`, `gradient_checkpointing` already chosen.
  - Positive Tests:
    - 39 yamls total, all parse and pass the validator (covering AC-17 positive).
    - Each non-dense yaml is one architecture × one balancing variant from `{deepseek_bias, aux_loss, quantile}`; for `aux_loss` rows `router_aux_loss_coef = 0.001`, others zero; for `deepseek_bias` rows `bias_update_rate = 0.001`, aux coefs zero; for `quantile` rows `load_balancing_method = quantile` and all aux/bias coefs zero.
  - Negative Tests:
    - A drift-detection test diffs the yamls against the matrix specification and fails if any row is missing or carries an off-axis coefficient.

- AC-19: `scripts/bench_step.py` exists, re-uses `src/training/trainer.py` end-to-end (data loader, optimizer, DDP wrap, all-reduces, post-step bias update), runs exactly 100 steps, drops the first 10 steps as warmup, takes the median over the remaining 90, and writes one row per config to `bench/results.json` with `{config_path, layers, depth_iters, per_rank_B, grad_accum, grad_ckpt, median_step_s, tokens_per_sec, peak_mem_gb}`.
  - Positive Tests:
    - The script runs on 8×H200 DDP with `seq_len=1024` and produces `bench/results.json` with one row.
    - An OOM in the first 10 steps marks the row as `OOM` and continues; the script does not crash the rest of the matrix.
    - Step-time reported by `bench_step.py` agrees with single-step measurement from the production trainer on a small config to within 5% (locks "uses the same code as training" invariant).
    - **Structural warmup test**: a unit test instruments the script with a counter and asserts the recorded step-time list has length 100, the first 10 entries are tagged `warmup=True`, the median is taken over exactly the remaining 90, and the JSON row carries the same counts.
  - Negative Tests:
    - A test that mutates the script to drop only the first 5 (instead of 10) warmup steps fails the structural test — locks the warmup count, not its statistical effect.
    - A test that mutates the script to take the *mean* (instead of median) over the 90 steps fails the structural test — locks the median estimator.

- AC-20: 39-config benchmark run on 8×H200 produces a populated `bench/results.json`, persisted to the repo (or referenced from a Modal volume per the project's checkpoint convention). Reproducibility is a manual / nightly expectation, not an automated CI check.
  - Positive Tests (manual / nightly):
    - All 39 rows are present; OOM rows are flagged; `bench/README.md` documents how to re-run.
    - Re-running on the same 8×H200 hardware reproduces medians within ~10% (loose bound; H200 thermals, network jitter, and Modal scheduling all contribute to variance — narrower bounds are not a reliable acceptance gate).
  - Negative Tests (manual):
    - A re-run with a deliberately changed model (e.g., `disable_liger: true → false` flip if DEC-16 selects "measured rerun") produces a visibly different median, proving the bench is sensitive to the changes it's supposed to measure.

- ~~AC-21~~ (DEFERRED 2026-04-28 per DEC-9): Comparison run (§8) — 4 methods × 3 families × ~5K-step training runs to characterize `(family, method) → (final_loss, mean_f_i_KL, max_f_i)`. Removed from this plan's hard ACs; will be re-introduced in a follow-up plan after Priority 1 / 2 / 3 land. The 39-config matrix from AC-18 still exists (it's needed for AC-24), but its yamls don't need to be tuned for any particular `load_balancing_method` winner. Tracked in `TODO.md` under "§8 comparison sweep".

- AC-23 (added 2026-04-28, Priority 1 headline test): `moe_everything` configured with `sanity_check_mode='alternating_global_moe'` produces forward / loss / gradient identical (to bf16 round-off) to a `global_moe` reference model with matching weights, on the same input + seed.
  - Positive Tests:
    - `tests/test_sanity_equivalence_precompute_kv.py` (new): instantiate `global_moe` (8L, deepseek routing) with seed=42, save its state-dict; instantiate `moe_everything` with `attn_expert_mode: per_head_precompute_kv`, `sanity_check_mode: alternating_global_moe`, `num_hidden_layers: 16` (so the alternating pattern reduces to 8 logical layers), use `init_mapping.copy_global_to_alternating_sanity` to copy the `global_moe` weights in; run the same input through both; assert (a) `output.logits` agrees to bf16 round-off; (b) `output.loss` agrees to fp32 round-off after CE; (c) `loss.backward()` produces the same gradient norms on every shared parameter to bf16 round-off.
    - The same test runs with `gradient_checkpointing=True` on the `moe_everything` side and asserts the same equivalences (locks AC-9's checkpoint-safety in the equivalence path).
    - The same test runs under DDP with 2 ranks and asserts per-rank-identical post-step `expert_bias` (the bank-level bias from DEC-19 makes this a stronger property than per-router; both ranks must reach the same bank-level bias after the all-reduce).
    - A long-running variant: train both models for 50 steps on the same data + seed, assert `output.loss` agrees to fp32 round-off at every step (not just step 0). Catches drift from any non-deterministic op in the `moe_everything` path that doesn't exist in `global_moe`.
  - Negative Tests:
    - Removing `sanity_check_mode='alternating_global_moe'` (and switching `moe_everything` to its real branch routing) breaks the equivalence — confirms the test is sensitive to the sanity-mode short-circuit.
    - Mutating one weight in the `moe_everything` MLP bank (e.g. `model.mlp_bank.experts.weight[0, 0, 0] += 1e-4`) breaks the gradient parity — confirms the gradient-comparison tolerance is real, not vacuous.
    - Initializing `moe_everything` with `init_strategy='random'` instead of `copy_global_to_alternating_sanity` breaks the forward parity — confirms the init-mapping is what makes the equivalence hold.

- AC-24 (added 2026-04-28, Priority 2): `scripts/bench_step.py` searches for the max stable batch per config and reports the fastest throughput at that batch.
  - Positive Tests:
    - The bench script exposes a `--search` mode that, for a given config, runs binary search over `(per_rank_batch, gradient_accumulation, gradient_checkpointing, chunked_ce)` to find the largest batch that doesn't OOM in the first 10 steps (a single-config run takes O(log batch_max) bench iterations of 10 warmup steps each — fast).
    - For each config, `bench/results.json` records `{config_path, max_per_rank_batch, max_global_batch, grad_accum_at_max, grad_ckpt_at_max, chunked_ce_at_max, median_step_s_at_max, tokens_per_sec_at_max, peak_mem_gb_at_max}`.
    - Reproducibility: re-running `--search` on the same hardware produces the same `max_per_rank_batch` (the search is deterministic given the OOM-check is seed-stable; H200 thermals can shift step time but not OOM thresholds).
    - The search respects a configured "headroom" (e.g. `--memory-headroom-gb 5`) so we don't push to the bleeding-edge OOM boundary.
  - Negative Tests:
    - A config that OOMs even at `per_rank_batch=1` and `grad_accum=1` and `grad_ckpt=true` and `chunked_ce=true` is recorded as `OOM` and the script continues to the next config — same robustness as AC-19.
    - Removing the binary-search loop (e.g. always running at the yaml's hard-coded batch) makes the test fail to detect headroom on configs whose yaml-specified batch is below max — proves the search is doing real work.

- AC-25 (added 2026-04-28, Priority 3): each speed-up borrowed from `nmoe/` / `modal-nmoe/` / `Megatron-LM/` is gated by a measured throughput delta on three configs (one per family) at the max stable batch from AC-24, with the delta and the touched files documented.
  - Positive Tests (one per borrowed speed-up):
    - **Grouped GEMM verification**: `experts_implementation: "grouped_mm"` in `model_factory.py:240` already exists. AC test: bench three matrix configs with `grouped_mm` vs `eager`, record the delta. If `grouped_mm` is faster on H200, document the win; if not, document why and consider porting `nmoe/csrc/` or `modal-nmoe/nmoe/csrc/` kernels.
    - **Liger fused_linear_cross_entropy for `moe_everything`**: `model_factory.py:75` sets `fused_linear_cross_entropy=False` for `moe_everything`. AC test: investigate why (probably a shape-mismatch with the per-depth-router branch), enable it (or port a custom chunked CE), bench three configs, record the delta. Touched files: `model_factory.py`, possibly a new `src/utils/chunked_ce.py`.
    - **Liger swiglu for `moe_everything`**: `model_factory.py:74` sets `swiglu=False`. Same procedure as fused CE.
    - **FP8 / NVFP4 expert weights**: `modal-nmoe/nmoe/csrc/` has FP8 paths. AC test: a feasibility analysis (analyze task) reports whether porting is in-scope for THIS plan or deferred to a follow-up — H200 supports FP8 natively, so this could be a real win.
    - Each enabled port has a corresponding `bench/results.json` row tagged with the optimization name (e.g. `optimization: liger_fused_ce`).
  - Negative Tests:
    - A port that produces NEGATIVE throughput delta on the matrix (slower than the baseline) is reverted; the bench row records the delta and the revert decision.
    - A port that produces a throughput win but breaks AC-23 sanity equivalence is reverted (correctness > speed).

- AC-22: Implementation work runs on Modal H200 sandboxes only (max 8 nodes concurrent, idle-off when not actively training/benchmarking) using the project skills.
  - Positive Tests:
    - All training/benchmark commits reference the Modal app/volume by name; `wandb_run_id` is checkpointed.
    - The launching agent's command-history shows `--max-gpus=8` (or equivalent) for every training/benchmark invocation.
  - Negative Tests:
    - Any fall-back to local GPU during automated execution is logged with an explicit reason; the test asserts no silent fall-back happens.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation includes: bank-level load balancing (DEC-19) for every shared-expert architecture; per-router-class refactor (AC-13) via the one-time YAML migrator (DEC-7); the `quantile` method with **exact global quantile** via `all_gather` (AC-10, AC-11); the configurable `bias_update_zero_sum` flag (DEC-2); the Megatron-style `softmax_position` rename + top-1 guard (DEC-17); unified `BranchRouter` bias path (DEC-18); `exploration_only` with `constant`/`cosine`/`linear` decay schedules and an optional negative-entropy bonus (AC-14); a config validator covering every illegal combination listed in §10 (AC-17); `scripts/bench_step.py` with the `--search` mode that finds max stable batch + fastest throughput per config (AC-24); the full 39-config benchmark on 8×H200 with OOM-row-handled `results.json` (AC-20); the **sanity-equivalence** test (AC-23) covering forward + loss + gradient + DDP + checkpoint-resume + 50-step-drift between `moe_everything` (sanity mode) and `global_moe`; speed-borrowing wins from `nmoe/` / `modal-nmoe/` / `Megatron-LM/` (AC-25) including grouped GEMM verification, Liger `swiglu` and `fused_linear_cross_entropy` re-enablement for `moe_everything`, and an FP8 feasibility analysis; full numerical-parity tests against `nmoe.Router` (AC-7); and a clean `docs/` update covering every new option (Documentation Discipline rule).

### Lower Bound (Minimum Acceptable Scope)

The implementation includes: **Priority 1 correctness landing first** — a flat-schema repair (Milestone A) that fixes the dead `update_expert_biases` walker (AC-2), the gradient-free attention aux loss (AC-8), the gradient-checkpointing-unsafe `BranchRouter` (AC-9), and the `bias_update_rate` field-placement ambiguity (AC-3); verification tests for §1, §2, §2a, §3 (AC-4, AC-5, AC-6, AC-7); the **sanity-equivalence test (AC-23) is non-negotiable** — `moe_everything` (sanity mode) must match `global_moe` to bf16 round-off on forward + loss + gradient before anything else lands; the `quantile` method on the flat schema (AC-10) with `all_gather` (DEC-4); the per-router-class refactor (AC-13) via the migrator (DEC-7); bank-level load balancing (DEC-19) for `global_moe` and `moe_everything` only; `exploration_only` with at least `constant` decay (the cosine + linear schedules can be deferred if AC-14 over-runs); `load_balancing_method` as the dispatch knob (AC-1); the **39-config matrix existing** (AC-18) so AC-24 has configs to bench against — but **none of the matrix yamls are tuned for any `load_balancing_method` winner**, the comparison sweep is out of scope; **Priority 2 throughput characterization (AC-24) lands** — `bench_step.py --search` finds max stable batch + fastest throughput per config, output to `bench/results.json`; **Priority 3 speed borrowing (AC-25) is selective** — only the speed-ups that produce a measurable positive delta on three matrix configs ship; the FP8 / NVFP4 port is deferred to a follow-up plan.

### Out of Scope (Deferred)

- §8 comparison sweep (12 training runs) — DEC-9 = deferred. Will be authored as a follow-up plan after Priority 1-3 land.
- Pinning per-config `load_balancing_method` defaults from §8 results — same.
- Cross-arch tuning of `quantile_eta` / `quantile_target_q` per family — defaults from DEC-11 stand for v0.
- RDEP-style expert parallelism port from `modal-nmoe` — flagged in AC-25 but deferred to a follow-up (it's a much bigger surface than the other speed-borrowing items).
- §10 `disable_liger: true → false` flip in the matrix — DEC-16's "sanity sweep" still runs but the matrix flip waits until AC-25 reports whether Liger's `swiglu` and `fused_linear_cross_entropy` can be safely re-enabled for `moe_everything`.

### Allowed Choices

- **Implementation order between §10 (per-router-class schema) and §4 (quantile)**: free choice. If §4 lands first on the flat schema, the §10 migration must include a re-port of `quantile` onto the new schema (covered by AC-13's migration-parity test).
- **DDP quantile reduction**: per-rank-mean (cheap, biased), exact global via `all_gather` (correct, slower), histogram/sketch (memory-bounded, approximate), or t-digest (most expensive). v0 must pick one; the choice is recorded in DEC-4. The lower bound permits per-rank-mean only.
- **Back-compat shim shape (DEC-7)**: runtime shim in `model_factory.py` (one source of truth, removable as a single PR), one-time YAML migrator script (cleaner long-term, requires a CI gate that no remaining yaml uses the old schema), or DUAL (shim + migrator + CI gate). Either is acceptable.
- **Switch aux uniform baseline (DEC-1)**: keep the current `top_k` baseline (matches Switch paper, requires the §1 draft bullet "uniform → 1.0" to be edited to "uniform → top_k") OR normalize to `1.0` (more interpretable, requires updating `tests/test_aux_loss_fix.py::test_uniform_routing` from 4.0 → 1.0).
- **Exploration-only branch alternative**: hard random override (default), entropy bonus, or a KL-style objective. v0 must pick one; the lower bound is hard random override only. (The `constant`/`cosine`/`linear` decay choice is *within* the hard-random-override family and is covered by AC-14, not by this allowed-choice axis.)
- **`quantile_eta` and `quantile_target_q` defaults**: `eta=0.05`, `target_q=1 - top_k/num_experts` per the draft, with the per-router-class adaptation rule from DEC-11 for group-limited routing and per-head-top-1 attention routers.
- **Cannot use**: rolling-mean replacements for the EMA quantile estimator that don't have a documented bias bound; ad-hoc per-yaml field placement (every field has exactly one canonical block); raw `dist.all_reduce` calls without `dist.is_initialized()` guards; modifying read-only reference repos (`Megatron-LM/`, `nmoe/`, `modal-nmoe/`).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

```text
# Milestone A — flat-schema repair (no behavior change to working paths)

src/models/standard_moe.py:
    add `def get_all_routers(self): yield each layer.mlp.gate`

src/models/global_moe.py:
    add `def get_all_routers(self): yield self.shared_router`  # adapt to actual structure

src/models/moe_everything/model.py:
    add `def get_all_routers(self): yield from MLP routers + ATTN q/k/v/o routers`
    decide: do we need `update_global_bias` as a real path, or remove the dead branch
    in update_expert_biases?
    fix attention_bank `_store_router_info`: split detached `for_logging` view from
    gradient-bearing `for_aux_loss` view; only the logging view is detached.

src/training/routing.py:
    add singular `branch_router` discovery alongside plural `branch_routers`
    guard every dist call with `if dist.is_initialized()`

src/models/routing/routers.py:
    BranchRouter.forward — wrap local_counts increment in
        `if torch.is_grad_enabled() and not is_checkpoint_recompute()`
    exploration sampling — cache mask on first pass, reuse on recompute

src/training/config.py:
    decide canonical block for bias_update_rate (DEC-3b outcome) and emit a
    deprecation warning when the field is found in the wrong block

# Milestone B — verification tests (lock invariants before changing them)

tests/test_aux_loss_fix.py — extend with selected_experts parity, token-mask path
tests/test_seq_loss_vs_paper.py — uniform → 1.0 (sigmoid normalize), token-mask path
tests/test_bias_update_detailed.py — convergence test, persistence test, DDP test
tests/test_router_options.py (or new test_router_parity_nmoe.py) — DeepSeek↔nmoe parity
tests/test_attention_aux_gradient.py (new) — AC-8 gradient flow
tests/test_branch_router_recompute.py (new) — AC-9
tests/test_quantile_load_balance.py (new) — AC-10, AC-11

# Milestone C — flat-schema dispatch + quantile

src/training/model_factory.py:
    read `load_balancing_method` from mcfg
    apply legacy-coef conflict policy (DEC-3a)

src/models/routing/load_balancing.py:
    def quantile_balancing_update(scores, state, target_q, eta):
        # update EMA: state['ema_q'] = (1 - eta) * state['ema_q'] + eta * quantile(scores, q=target_q)
        # return new bias contribution: -state['ema_q']

src/models/routing/bias.py:
    def update_bias_from_quantile(router, eta, target_q, distributed):
        with torch.no_grad():
            # router.local_quantile_scores is a non-persistent buffer (a list
            # or rolling tensor) that DeepSeekRouter.forward appends to on
            # every training-mode call: each micro-batch, each branch-masked
            # depth in moe_everything. Empty accumulator → no-op (zero-active-
            # tokens case).
            if not router.local_quantile_scores:
                return
            all_scores = torch.cat(router.local_quantile_scores, dim=0).float()
            q_local = torch.quantile(all_scores, target_q, dim=0)  # fp32
            if distributed and dist.is_initialized():
                # DEC-4 picks reduction; per-rank-mean shown:
                dist.all_reduce(q_local, op=dist.ReduceOp.SUM)
                q_local /= dist.get_world_size()
            # EMA, bias write, and clamp all in fp32 (under any outer autocast).
            state['ema_q'] = (1 - eta) * state['ema_q'] + eta * q_local
            router.expert_bias[:] = -state['ema_q'].to(torch.float32)
            router.expert_bias.clamp_(-16.0, 16.0)
            # Clear after update so the next optimizer step starts fresh.
            router.local_quantile_scores.clear()

src/training/routing.py:
    def update_expert_biases(model, *, load_balancing_method, ...):
        if load_balancing_method == "deepseek_bias":  for r in routers: _update_single_router_bias(r)
        elif load_balancing_method == "quantile":     for r in routers: _update_single_router_quantile_bias(r)
        else: return  # no-op for aux_loss/seq_aux_loss/none

src/models/{standard_moe,global_moe,moe_everything}/...:
    forward gates aux-loss accumulation by `load_balancing_method`

# Milestone D — exploration_only branch

src/models/routing/routers.py:
    BranchRouter.__init__ accepts balancing='exploration_only'
    BranchRouter.forward:
        if balancing == 'exploration_only' and self.training:
            with prob p_explore(step): choice = uniform_random({0,1}) per token
            (cache the mask via the same cache-on-first-pass discipline as AC-9)

# Milestone E — per-router-class schema (deferred for lower bound)

src/training/model_factory.py:
    READ ONLY THE NESTED SHAPE: mcfg must have 'mlp_router'/'attn_router'/'branch_router' keys.
    A flat-shape yaml is rejected at config load by the validator (no in-memory remap path).

scripts/migrate_configs.py (new, one-shot):
    walk every yaml under configs/, rewrite flat → nested (and rewrite alias tokens
    `switch` → `aux_loss`, `seq_aux` → `seq_aux_loss`) in a single PR.

src/models/router.py / routers.py / mlp_bank.py:
    router constructors take per-class config dict, not flat config

scripts/validate_configs.py:
    add per-class validator pass with the illegal-combination rejections from AC-17;
    becomes a CI gate that rejects any yaml under configs/ still using the flat schema.

# Milestone F — config matrix

For each depth in {4, 8, 16}:
    For each row in {dense, standard_moe, global_moe, precompute_kv, precompute_kv_global_router}:
        For each balancing in {deepseek_bias, aux_loss, quantile} (skip dense):
            author yaml under configs/<depth>_layers/<row>_<balancing>.yaml

# Milestone G — bench

scripts/bench_step.py:
    parse --config, --steps=100, --warmup=10
    call run_training with a custom hook that records median step time + tokens/s + peak mem
    exit cleanly after 100 steps; write one row to bench/results.json

# Milestone H — comparison

For each (family, method) in {standard_moe, global_moe, moe_everything} × 4 methods:
    run_training for 5K steps on standard data + seed
    log per-step f_i to wandb
    collect final loss
    emit summary row to bench/comparison_summary.md
```

### Relevant References

- `src/models/routing/load_balancing.py` — current `load_balancing_loss_func`, `seq_load_balancing_loss_func`, `normalize_router_scores`.
- `src/models/routing/bias.py` — `update_bias_from_counts` (mirror for `update_bias_from_quantile`).
- `src/models/router.py` — `DeepSeekRouter`, `ExplorationTopKRouter`, `is_checkpoint_recompute()`, `sample_router_exploration_mask`, `apply_router_exploration`.
- `src/models/routing/routers.py` — `BranchRouter`, `BranchRouterRecorder`.
- `src/training/routing.py` — `update_expert_biases`, `_update_single_router_bias`, `_update_branch_router_bias`, `apply_router_exploration_rate`, `get_bias_rate`, `exploration_rate_schedule`, `collect_router_z_loss`.
- `src/training/model_factory.py` — `build_model`, `_set_router_params`, `_set_deepseek_router_params` (note: the draft's "model_factory.py" path is at `src/training/model_factory.py`, not `src/models/model_factory.py`).
- `src/training/trainer.py` — post-optimizer-step bias-update call site (currently at lines 429–446).
- `src/training/config.py` — `TrainingConfig`, the field-block decision (DEC-3b).
- `src/models/moe_everything/attention_bank.py` — `_store_router_info` (the detach point that drops attention aux gradient flow; AC-8 fix lives here).
- `src/models/moe_everything/model.py` — `MoEverythingForCausalLM.forward` aux-loss accumulation; both `branch_router` (singular) and `branch_routers` (plural) attributes depending on `per_layer_router`.
- `Megatron-LM/megatron/core/transformer/moe/moe_utils.py` — `get_updated_expert_bias` (alternative reference for DEC-2).
- `nmoe/nmoe/model.py` — `Router` forward recipe (AC-7 parity reference) and `Router.update_bias` (zero-sum reference).
- `tests/test_aux_loss_fix.py`, `tests/test_seq_loss_vs_paper.py`, `tests/test_bias_update_detailed.py`, `tests/test_router_options.py`, `tests/test_router_exploration_warmup.py`, `tests/test_routing_stats.py`, `tests/test_megatron_comparison.py`.
- `.claude/skills/modal-experiment` and `.claude/skills/gpu-debugging` — Modal H200 sandbox launch.

## Dependencies and Sequence

### Priority 1 — Correctness

1. **Milestone A — Flat-schema repair**: Fix dead bias walker, gradient-free attention aux, BranchRouter recompute safety, `bias_update_rate` field-placement ambiguity. No behavioral change to working paths.
   - Phase A1: Fix `update_expert_biases` walker — covers AC-2 (singular-`branch_router` discovery is folded into DEC-18's unified path).
   - Phase A2: Fix attention aux loss gradient flow — covers AC-8.
   - Phase A3: Make `BranchRouter` checkpoint-safe — covers AC-9.
   - Phase A4: Resolve `bias_update_rate` field-placement (DEC-3b) — covers AC-3.

2. **Milestone B — Verification tests**: Lock current invariants for §1, §2, §2a, §3 before touching them.
   - Phase B1: Switch aux baseline test (`top_k` per DEC-1; flip rank-local → `all_reduce(SUM)` per DEC-4) — covers AC-4.
   - Phase B2: Sequence aux uniform-1.0 test, token-mask path test, selected_experts parity test — covers AC-5.
   - Phase B3: DeepSeek↔nmoe parity test + DEC-2 (`bias_update_zero_sum` configurable) — covers AC-6, AC-7.

3. **Milestone C — Bank-level load-balancing refactor (DEC-19)**: Move `expert_bias`, `local_tokens_per_expert`, and (later) the quantile EMA from per-router buffers to per-bank buffers for `global_moe` and `moe_everything`. Per-layer routers become stateless consumers.
   - Phase C1: Refactor `MlpExpertBank` / `AttentionExpertBank` / `GlobalMoEModel` to own the buffers. Update `DeepSeekRouter.__init__` to take a back-pointer to the owning bank.
   - Phase C2: Update `update_expert_biases` walker to traverse banks, not routers (modulo `standard_moe`'s per-layer pools).
   - Phase C3: Update aux-loss accumulation in `forward` to aggregate across per-layer routers feeding the same bank, then compute one aux loss per bank.
   - Phase C4: Bias-buffer count regression test — `global_moe(8L)` → 1 bias; `moe_everything(per_layer_mlp_router=true)` → 1 MLP bias + 4 attn biases. Covers DEC-19's AC implications on AC-2.

4. **Milestone D — `load_balancing_method` dispatch + `quantile` method**: Single-knob dispatch becomes authoritative; `quantile` is the new method (with bank-level state per DEC-19).
   - Phase D1: `model_factory.py` reads `load_balancing_method`; legacy-coef conflict policy applied (DEC-3a) — covers AC-1.
   - Phase D2: `quantile_balancing_update`, `update_bias_from_quantile`, dispatch in `update_expert_biases` — covers AC-10. State on the BANK per DEC-19.
   - Phase D3: DDP `all_gather` reduction (DEC-4) + bank-level accumulator + AC-11 parity test.
   - Phase D4: Quantile EMA persistence (DEC-5) + resume-parity test on the BANK's `quantile_ema` — covers AC-12.

5. **Milestone E — `exploration_only` branch + `softmax_position` rename**: New branch-router balancing mode + Megatron-style softmax-position knob with top-1 guard.
   - Phase E1: Add `balancing` + `exploration_*` knobs to `BranchRouter`; implement `constant`, `cosine`, `linear` decay schedules — covers AC-14.
   - Phase E2: Per-class exploration schedules in `apply_router_exploration_rate` — covers AC-15.
   - Phase E3: Rename `router_topk_ordering` → `softmax_position`; add top-1 guard validator (DEC-17). Old name accepted as deprecated alias for one release.

6. **Milestone F — Per-router-class schema (§10) via one-time migrator (DEC-7)**: Replace flat schema with nested `mlp_router`/`attn_router`/`branch_router` in a single PR (no runtime shim).
   - Phase F1: Schema definition + canonical `balancing` enum (per AC-13).
   - Phase F2: One-shot `scripts/migrate_configs.py` that rewrites every yaml flat → nested + `switch`/`seq_aux` aliases + `router_topk_ordering` → `softmax_position` + per-router-bias-state-dict → per-bank-bias-state-dict (DEC-19 checkpoint converter). Same PR.
   - Phase F3: Validator pass + CI gate that rejects flat-schema yamls — covers AC-17.
   - Phase F4: Migration-parity test runs **before** the migrator PR merges; deleted from the suite after merge.

7. **Milestone G — Sanity equivalence (AC-23, Priority 1 headline)**: `moe_everything(sanity_check_mode='alternating_global_moe')` ≡ `global_moe` on forward + loss + gradient + DDP + 50-step-drift.
   - Phase G1: Author `tests/test_sanity_equivalence_precompute_kv.py` covering all sub-cases of AC-23.
   - Phase G2: Run the test under DDP + gradient checkpointing variations.
   - Phase G3: Doc the equivalence in `docs/architecture.md` so future readers know the precompute-kv pipeline is verified against `global_moe` as ground truth.

### Priority 2 — Throughput characterization

8. **Milestone H — 39-config matrix authoring**: Author the 13-yaml × 3-depth configs on the new schema (no per-method tuning since DEC-9 is deferred — each yaml just exists with reasonable defaults).
   - Phase H1: Generator or hand-written 39 yamls — covers AC-18.
   - Phase H2: Validator passes on all 39 — covers AC-17 positive path.

9. **Milestone I — `bench_step.py --search` for max-batch + throughput (AC-24, AC-19, AC-20, AC-22)**: For each of the 39 configs, find the max stable batch with `(per_rank_batch, gradient_accumulation, gradient_checkpointing, chunked_ce)` levers, then measure fastest throughput at that batch on 8×H200 DDP.
   - Phase I1: Implement `--search` mode in `bench_step.py` (binary search on per_rank_batch with OOM detection in first 10 steps).
   - Phase I2: Implement chunked / fused CE loss path — for `standard_moe` / `global_moe` this is Liger's `apply_liger_kernel_to_qwen3_moe()` already on; for `moe_everything` either re-enable Liger's `fused_linear_cross_entropy=True` (today disabled at `model_factory.py:75`) or port a custom `chunked_ce` from `nmoe`/`modal-nmoe`. Document the choice and the resulting peak-memory delta.
   - Phase I3: Modal launch + 39-config sweep, OOM-tolerant — covers AC-20.
   - Phase I4: Compute-envelope discipline (DEC-12) + Documentation (`bench/README.md` describing the search procedure and how to reproduce).

### Priority 3 — Speed borrowing

10. **Milestone J — Grouped GEMM + Liger re-enablement (AC-25)**: Verify `experts_implementation: "grouped_mm"` is faster than `eager` on H200 across the matrix (a measurable result either way); investigate why `swiglu=False` and `fused_linear_cross_entropy=False` are set for `moe_everything` and re-enable each one if it produces a measurable throughput win without breaking AC-23 sanity equivalence.
    - Phase J1: Bench three matrix configs (one per family) with `grouped_mm` vs `eager`; record delta to `bench/results.json`.
    - Phase J2: Re-enable Liger `swiglu` for `moe_everything`; bench three configs; record delta and revert if AC-23 fails.
    - Phase J3: Re-enable Liger `fused_linear_cross_entropy` for `moe_everything`; bench three configs; record delta and revert if AC-23 fails.

11. **Milestone K — FP8 / NVFP4 feasibility (AC-25, analyze-only)**: Document what porting `modal-nmoe`'s FP8/NVFP4 expert-weight paths would entail. NOT implemented in this plan — the implementation is a follow-up.
    - Phase K1: Read `modal-nmoe/nmoe/csrc/` and `Megatron-LM/megatron/core/transformer/moe/` for the FP8/NVFP4 expert-weight patterns; produce a one-page feasibility analysis at `docs/research/fp8_nvfp4_port_feasibility.md`.

### Out of Scope (Deferred to Follow-up Plans)

- **§8 comparison sweep** (12 training runs to pick winners) — DEC-9 = deferred. Authored as a follow-up plan after Priority 1-3 land.
- **Pinning per-config `load_balancing_method` defaults** from §8 results — same.
- **RDEP-style expert parallelism port** from `modal-nmoe` — flagged in AC-25 but deferred (much bigger surface than the other speed-borrowing items).
- **DEC-16 `disable_liger` flip** — the sanity sweep still runs but the matrix flip waits until AC-25 reports whether Liger's `swiglu` and `fused_linear_cross_entropy` can be safely re-enabled for `moe_everything`. The matrix-wide flip is therefore part of a Priority 3 follow-up, not this plan.

### Dependency Notes

- Milestone B depends on Milestone A: don't lock invariants on broken code.
- Milestone C depends on Milestones A and B: bank-level state migration touches every router; flat schema must be correct + tests must exist first.
- Milestone D depends on Milestones A, B, and C: dispatch + quantile lands on the bank-level state from C.
- Milestone E can land on the flat schema in parallel with C/D (it doesn't touch bank state); E3 (`softmax_position` rename) is independent of bank-level work.
- Milestone F (schema migration) depends on Milestones C, D, and E: the migrator rewrites yamls onto a schema that already includes per-class balancing, the bank-state-dict converter, and the `softmax_position` rename in one shot.
- Milestone G (sanity equivalence) depends on Milestones A through F — it's the headline correctness test that proves the whole correctness work is right. **Cannot be skipped.**
- Milestone H (matrix authoring) depends on Milestone F: yamls written on the new schema.
- Milestones I, J, K (Priority 2 + 3) depend on Milestone H.
- Priority 3 milestones depend on Priority 2 (need max-batch + throughput baseline before measuring speed-up deltas).
- §10 (Milestone F) BLOCKS §11 only at the per-class config schema level. Milestone E lands on the flat schema first and migrates alongside Milestone F.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Investigate `get_all_routers` strategy across 3 model families; document target shape and the `unwrap_model(...)` boundary (walker sees the `…ForCausalLM` wrapper) | AC-2 | analyze | - |
| task2 | Implement `get_all_routers` on `StandardMoEModel`, `DeepSeekStandardMoEModel`, `GlobalMoEForCausalLM`, `DeepSeekGlobalMoEForCausalLM`, AND `MoEverythingForCausalLM` (NOT only the inner `MoEverythingModel`); add walker coverage test | AC-2 | coding | task1 |
| task3 | Add singular-`branch_router` discovery to `update_expert_biases` walker | AC-2 | coding | task2 |
| task4 | Audit `attention_bank.py:_store_router_info`; design split between detached-logging and gradient-bearing tensors | AC-8 | analyze | - |
| task5 | Implement attention aux loss gradient fix; add gradient-flow test | AC-8 | coding | task4 |
| task6 | Add `is_checkpoint_recompute()` guard and exploration-mask cache to `BranchRouter.forward` | AC-9 | coding | - |
| task7 | Resolve canonical block for `bias_update_rate`/`seq_aux_loss_coef`/`load_balancing_method` (DEC-3b outcome); update `TrainingConfig` and emit deprecation warning when fields are found in the wrong block | AC-3 | coding | - |
| task8 | Decide and document Switch aux uniform baseline (DEC-1); update `tests/test_aux_loss_fix.py` accordingly | AC-4 | analyze | - |
| task9 | Extend `tests/test_aux_loss_fix.py` per task8 outcome; add selected_experts-parity and token-mask path tests | AC-4 | coding | task8 |
| task10 | Extend `tests/test_seq_loss_vs_paper.py` for uniform-1.0 (sigmoid normalize), token-mask path, selected_experts parity | AC-5 | coding | - |
| task11 | Decide DeepSeek bias parity reference (DEC-2); document choice in `bias.py` | AC-6 | analyze | - |
| task12 | New `tests/test_router_parity_nmoe.py`: DeepSeekRouter↔nmoe.Router parity (forward recipe + bias update) | AC-6, AC-7 | coding | task11 |
| task13 | Wire `load_balancing_method` into `build_model`; apply legacy-coef conflict policy per DEC-3a | AC-1 | coding | task7 |
| task14 | Gate aux-loss accumulation in `forward` of all 3 families by `load_balancing_method` | AC-1 | coding | task13 |
| task15 | Implement `quantile_balancing_update` in `load_balancing.py` (pure function); doc `target_q` rule per DEC-11 | AC-10 | coding | - |
| task16 | Implement `update_bias_from_quantile` in `bias.py`; reuse `expert_bias` buffer; add a non-persistent `local_quantile_scores` accumulator on `DeepSeekRouter` that appends *every* training-mode forward's detached fp32 scores (covering all gradient-accumulation micro-batches and all branch-masked depths in `moe_everything`); compute the post-step quantile across the concatenated accumulator in fp32, update the fp32 EMA, write to the fp32 `expert_bias`, clamp `±16`, and clear the accumulator after the bias update; no-op the entire path when the accumulator is empty | AC-10 | coding | task15 |
| task17 | Add `_update_single_router_quantile_bias` and dispatch from `update_expert_biases` | AC-10 | coding | task16, task2 |
| task18 | DDP quantile reduction implementation per DEC-4; add `all_gather`-baseline parity test | AC-11 | coding | task16 |
| task19 | Quantile EMA state persistence per DEC-5; add resume-parity test | AC-12 | coding | task16 |
| task20 | Add `balancing: exploration_only` to `BranchRouter`; implement `constant`, `cosine`, and `linear` decay schedules with `exploration_min` and `exploration_warmup_steps` as documented in draft §11 (each schedule is deterministic given step + config) | AC-14 | coding | task6 |
| task21 | Generalize `apply_router_exploration_rate` to per-class schedules (using flat fields pre-§10) | AC-15 | coding | task7 |
| task22 | (§10) Define nested `mlp_router`/`attn_router`/`branch_router` config schema. Schema must accommodate quantile-specific fields (`quantile_eta`, `quantile_target_q`, `quantile_global_state`) and use the canonical `balancing` enum from AC-13 (`aux_loss | seq_aux_loss | deepseek_bias | quantile | none`, plus `exploration_only` only on the branch class). Per DEC-7 (RESOLVED): produce a one-shot migrator script that rewrites every yaml under `configs/` flat → nested, including the alias rewrite `switch` → `aux_loss` and `seq_aux` → `seq_aux_loss`. NO runtime shim is shipped | AC-13 | analyze | task13, task17, task18, task19, task20 |
| task23 | Implement nested schema in `src/training/model_factory.py`; refactor router constructors to take per-class dict. The factory reads ONLY the nested shape — no flat-to-nested in-memory remapping. Run the migrator from task22 against every `configs/` yaml in the same PR | AC-13 | coding | task22 |
| task24 | Add config validator pass for new schema (illegal combinations from §10); the validator becomes a CI gate that rejects any yaml under `configs/` still using the flat schema or the alias tokens | AC-17 | coding | task23 |
| task25 | Migration-parity test: every existing yaml instantiates equivalent model on flat-vs-migrated schema, with both forward/loss parity AND post-step bias-update parity (one full optimizer step including `update_expert_biases`) for `deepseek_bias` and `quantile` methods. This test runs **before** task23 merges (when the flat-shape code path still exists); after task23 merges the flat-shape path is gone and this test is removed from the suite | AC-13 | coding | task23 |
| task26 | Author 39 yamls in `configs/{4,8,16}_layers/` on the new schema (or generator script if cheaper) | AC-18 | coding | task24 |
| task27 | Implement `scripts/bench_step.py` reusing `run_training`; emit one row to `bench/results.json` | AC-19 | coding | task26 |
| task28 | (Modal) Launch 39-config benchmark sweep on 8×H200; collect `bench/results.json` | AC-20, AC-22 | coding | task17, task19, task27, task35, task36 |
| task29 | (Modal) Run 12-run comparison sweep (3 families × 4 methods, ~5K steps); emit `bench/comparison_summary.md` | AC-21, AC-22 | coding | task28 |
| task30 | Pin per-config defaults from comparison results (per DEC-9 outcome) OR document tradeoffs only | AC-21 | coding | task29 |
| task31 | Author `bench/README.md` describing how to re-run the bench and comparison | AC-20 | coding | task28 |
| task32 | Compute-envelope discipline: confirm Modal max-8-H200 + idle-off invariant in launching agents (DEC-12) | AC-22 | analyze | - |
| ~~task33~~ | ~~`global_moe` quantile EMA shared-vs-per-layer~~ — **DELETED** (superseded by DEC-19; bank-level state is the only path for shared-pool architectures) | — | — | — |
| task34 | FSDP support level (DEC-14): add FSDP smoke variant on AC-2/AC-10/AC-11 OR explicitly skip with a documented "FSDP not exercised" message | AC-2, AC-10, AC-11 | coding | task17, task18 |
| task35 | `output_router_logits` policy for non-aux methods (DEC-15): keep-on / detach-only / disable; apply chosen policy in every model family's `forward` and document it in `bench_step.py` and per-method yamls — must land BEFORE the 39-config bench so all rows use a consistent policy | AC-19, AC-20 | analyze | task14, task17 |
| task36 | `disable_liger` flip plan (DEC-16): run a 3-config sanity sweep with `disable_liger: false` after §1-§5 land; pin matrix setting based on result. The sanity sweep itself is a small Modal job; it must complete BEFORE task28 so the 39-config bench knows the matrix `disable_liger` value | AC-20 | coding | task14, task17 |
| task37 | (DEC-17) Analyze `ExplorationTopKRouter` softmax-position rename: design the `softmax_position: pre_topk \| post_topk` field (Megatron-compatible), the deprecation alias from `router_topk_ordering: post \| pre`, and the validator rule that rejects `post_topk + top_k=1` with Megatron's error message | DEC-17 | analyze | - |
| task38 | (DEC-17) Implement softmax-position rename + alias + top-1 guard in `src/models/router.py:ExplorationTopKRouter`; absorb the yaml-side field rewrite into the DEC-7 migrator (task22); update `docs/routing.md` and `docs/configuration.md` to document the new knob, the Megatron equivalence, and the top-1 error | DEC-17 | coding | task37, task22 |

## Claude-Codex Deliberation

### Agreements

- The single dispatch knob `load_balancing_method` is the right shape for §5; today it is a no-op string and must become authoritative.
- The DeepSeek bias-update path is currently broken in `standard_moe`/`global_moe` and the singular-`branch_router` case in `moe_everything` (the walker calls `raw_model.get_all_routers()` and `raw_model.update_global_bias()`, neither of which exists in this repo) — fixing this is a hard prerequisite for any new method work.
- The §10 nested per-router-class schema is the correct architecture; the only debate is **when** to land it relative to §4 and §11.
- The §11 `exploration_only` branch is justified by the existing `BranchRouter.forward` masking (gradients flow only to the chosen branch); without exploration the branch can latch.
- `scripts/bench_step.py` should re-use `run_training` end-to-end rather than ship a probe-only path.
- Reference repos (`Megatron-LM/`, `nmoe/`, `modal-nmoe/`) are read-only; when citing them in code/comments, include a hash or "as of plan authoring" note so future readers know whether the reference has drifted.

### Resolved Disagreements

- **Walker target after `unwrap_model`**: Codex r1 caught that `unwrap_model(...)` returns `MoEverythingForCausalLM` (not the inner `MoEverythingModel`); `get_all_routers` must live on the wrapper or descend into `raw_model.model`. Resolution: implement on every `…ForCausalLM`/`…Model` class explicitly; task2 updated.
- **AC-4 `f_i` rank-locality**: Codex r1 noted the original AC-4 didn't prove `f_i` stays rank-local. Resolution: AC-4 now includes a no-collective-invocation assertion test.
- **AC-6 bias warmup + DDP all-reduce**: Codex r1 noted both were missing despite being explicit in draft §2. Resolution: AC-6 now has dedicated warmup-schedule and DDP-all-reduce tests.
- **AC-10 mixed-precision and accumulation**: Codex r1 caught that "cache router scores on forward" is wrong under gradient accumulation, branch masking, and zero-active-tokens. Resolution: AC-10 now specifies cross-microbatch accumulation, branch-mask handling, zero-active-tokens no-op, and explicit fp32 boundaries.
- **AC-13 migration-parity scope**: Codex r1 caught that forward/loss parity alone can mask broken bias-update dispatch after schema migration. Resolution: AC-13 now requires post-step bias-update parity (one full optimizer step including `update_expert_biases`) on every yaml.
- **AC-14 deterministic vs probabilistic asserts**: Codex r1 flagged the original "non-zero gradient at every step" and "collapses within K steps" as flaky. Resolution: AC-14 now uses RNG-injected branch tests, windowed cumulative-gradient tests, and a deterministic forced-collapse test with documented thresholds.
- **AC-19 structural test**: Codex r1 flagged the variance-widening negative test as unreliable. Resolution: AC-19 now has a structural test on warmup-count and median-estimator choice.
- **AC-17 cross-reference fix**: Codex r1 caught the wrong task-ID reference for shared `branch_router` + `deepseek_bias`. Resolution: AC-17 now references task3 (singular-`branch_router` discovery) instead of task20.
- **Pre-split DEC-3 split into DEC-3a / DEC-3b**: Codex r1 noted the original DEC-3 conflated conflict-policy and config-block-placement (separate decisions, separate migration consequences). Resolution: split into DEC-3a (conflict policy) and DEC-3b (canonical block).
- **AC-7 `topk_scaling_factor` documentation**: Codex r1 OPTIONAL improvement — added a config-comment audit test to AC-7 positive tests.
- **AC-20 reproducibility threshold**: Codex r1 flagged the 3% threshold as unreliable across 8×H200 thermals/jitter. Resolution: relaxed to ~10% and labeled "manual / nightly", not automated CI.
- **New DECs from Codex r1 UNRESOLVED**: DEC-13 (`global_moe` quantile EMA shared vs per-layer), DEC-14 (FSDP support level), DEC-15 (`output_router_logits` policy for non-aux methods), DEC-16 (`disable_liger` flip plan) added to Pending User Decisions.
- **Round 2: AC-1 ↔ AC-13 authority conflict**: Codex r2 caught that the plan called `load_balancing_method` "single authoritative" while AC-13 made per-class `*.balancing` independently authoritative. Resolution: AC-1 now reads as flat-schema authoritative, with the back-compat shim (DEC-7) expanding the top-level field into per-class fields after AC-13 lands. Per-class is canonical post-migration.
- **Round 2: stale `DEC-3` references**: Codex r2 found references to plain `DEC-3` after the split. Resolution: replaced with `DEC-3a` (conflict policy) or `DEC-3b` (canonical block) at every site.
- **Round 2: AC-10 vs task16 / pseudocode drift**: Codex r2 caught the conceptual pseudocode and task16 still saying "cache router scores on forward / `_last_scores`" — stale under the AC-10 cross-microbatch accumulator spec. Resolution: rewrote both task16 and the pseudocode to require the cross-microbatch accumulator, fp32 boundaries, branch-mask handling, zero-active-tokens no-op, and post-update clearing.
- **Round 2: AC-14 cosine vs task20 constant-only**: Codex r2 noticed AC-14 required `cosine` schedule testing while task20 said "constant decay v0". Resolution: task20 now requires `constant`, `cosine`, and `linear` schedules; lower-bound text updated to drop the "constant only" carve-out.
- **Round 2: task dependency graph**: Codex r2 noted task22 (schema) didn't depend on task17/18/19 (quantile dispatch); task28 (39-config bench) didn't depend on quantile dispatch; task34 (FSDP smoke) needed task18 (DDP quantile); task35/36 (policy decisions) needed to land BEFORE task28 not after. Resolution: dependency graph rewired accordingly.
- **Round 2: DEC-13/14/15/16 Codex Position placeholders**: Codex r2 rewrote each "decide explicitly" placeholder with concrete recommendations (DEC-13 SHARED with `quantile_global_state: true`; DEC-14 OPTIONAL v0 with explicit FSDP smoke/skip; DEC-15 DETACH-ONLY for non-aux methods; DEC-16 KEEP `disable_liger: true` for v0 + separate sanity sweep).
- **Round 2: original draft non-normative**: Codex r2 OPTIONAL — added a non-normative-appendix note above the draft section so future readers know the structured plan wins where they conflict.
- **Round 2: task25 wording**: Codex r2 OPTIONAL — task25 now explicitly mentions post-step bias-update parity to match AC-13's stronger scope.
- **Round 3: task28 missing DEC-15/DEC-16 dependencies**: Codex r3 caught that task35/task36 said "must land before task28" but task28 didn't depend on them. Resolution: added `task35, task36` to task28's `Depends On`.
- **Round 3: per-class `balancing` enum naming**: Codex r3 caught inconsistency between the structured plan's `aux_loss | seq_aux_loss` and the original draft §10's `switch | seq_aux`. Resolution: AC-13 and task22 now name `aux_loss | seq_aux_loss | deepseek_bias | quantile | none` (plus `exploration_only` only on the branch class) as the canonical enum, with the DEC-7 shim accepting `switch` / `seq_aux` as aliases that emit a deprecation warning.
- **Round 3: AC-14 parameterized schedule test**: Codex r3 OPTIONAL — AC-14's schedule-shape test now parameterizes over `constant`, `cosine`, and `linear` (matching task20's expanded scope).

### Convergence Status

- After Round 1: `partially_converged` (REQUIRED_CHANGES applied; UNRESOLVED items captured as DEC-13..DEC-16).
- After Round 2: `partially_converged` (substantive REQUIRED_CHANGES around AC-1↔AC-13 authority, AC-10 accumulator pseudocode, task20 schedule scope, task dependency graph, and DEC-13..DEC-16 Codex Position placeholders — all applied).
- After Round 3: `converged` (Codex r3 returned only two terminal REQUIRED_CHANGES — task28's `Depends On` was missing task35/task36, and the per-class `balancing` enum needed canonical naming + alias documentation. Both applied. Codex r3 OPTIONAL nits — historical "DEC-3" mentions in narrative text and AC-14 schedule-shape test parameterization — also addressed).
- Final Status: `converged`. Remaining work is human decisions on DEC-1..DEC-16; no further Claude/Codex disagreement is open.

## Pending User Decisions

- DEC-1: Switch aux uniform-routing baseline — keep `top_k` (matches Switch paper, current implementation, and the existing `tests/test_aux_loss_fix.py::test_uniform_routing` assertion of `4.0` for top-4) OR normalize to `1.0` (more interpretable across different `top_k` values; the draft currently asserts "uniform → 1.0" which conflicts with the implementation).
  - Claude Position: Keep `top_k`; rewrite the §1 draft bullet "uniform → 1.0" to "uniform → top_k". Reason: changing the loss scale invalidates every existing `router_aux_loss_coef` calibration across all 39 configs.
  - Codex Position: Decide explicitly — current test asserts `4.0` for top-4, draft asserts `1.0`. Conflict must be resolved before §1 can be marked verified.
  - Tradeoff Summary: Keeping `top_k` requires zero coefficient retuning across 39 yamls and matches the original Switch paper's formulation. Switching to `1.0` requires retuning every `router_aux_loss_coef` by `1/top_k` and updating the §1 doc plus the test assertion.
  - Decision Status: **RESOLVED 2026-04-27** — keep `top_k` baseline. The draft's "uniform → 1.0" claim is wrong; uniform → `top_k` is correct (top-4 → 4.0). The existing `tests/test_aux_loss_fix.py::test_uniform_routing` assertion of `4.0` stands. Task8 (analyze) reduces to: rewrite the §1 draft bullet from "uniform → 1.0" to "uniform → top_k" in any documentation that mirrors the draft text.

- DEC-2: DeepSeek bias-update reference — `nmoe` zero-sum `(s − mean(s)) * rate` (matches `_update_single_router_bias` in `src/training/routing.py:53` and `nmoe/nmoe/model.py:93`, `modal-nmoe/nmoe/model.py:93`) OR Megatron-LM `sign(avg − counts) * rate` without mean subtraction (`Megatron-LM/megatron/core/transformer/moe/moe_utils.py:1160`).
  - Claude Position: Stick with `nmoe` zero-sum — already implemented, matches `_update_single_router_bias`, and the zero-sum invariant guarantees `expert_bias.sum() ≈ 0` post-clamp. Document the choice and the reference hash in `bias.py`.
  - Codex Position: "Matches reference" needs to name which reference wins. Both are defensible.
  - Tradeoff Summary: `nmoe` zero-sum guarantees the bias mean stays at zero; Megatron's plain `sign` allows the bias mean to drift. The `±16` clamp makes drift bounded either way.
  - Decision Status: **RESOLVED 2026-04-27** — make it CONFIGURABLE. New config knob `bias_update_zero_sum: bool` (default `true` = nmoe-style `(s − mean(s)) * rate` zero-sum, current behavior; `false` = Megatron-style plain `sign(avg − counts) * rate` with no mean subtraction). Implementation:
    - `_update_single_router_bias` in `src/training/routing.py` reads the flag from the per-class config (post-AC-13: `mlp_router.bias_update_zero_sum` / `attn_router.bias_update_zero_sum` / `branch_router.bias_update_zero_sum`); pre-AC-13 reads it from `cfg["training"]["bias_update_zero_sum"]`.
    - AC-6 expands to test BOTH paths: `bias_update_zero_sum=true` asserts `expert_bias.sum() ≈ 0` post-clamp; `bias_update_zero_sum=false` asserts the bias-buffer mean is allowed to drift but stays bounded by the `±16` clamp.
    - `bias.py` documents both references (nmoe `nmoe/nmoe/model.py:93`, modal-nmoe `modal-nmoe/nmoe/model.py:93`, Megatron `Megatron-LM/megatron/core/transformer/moe/moe_utils.py:1160`) so a future reader can map our knob to either reference.
    - The 39-config matrix yamls (task26) all use the default `true` (nmoe behavior); the `false` path is a flag for users porting Megatron-trained checkpoints.

- DEC-3a: Legacy-coef conflict policy when `load_balancing_method` is set — REJECT (raise during config load) OR AUTO-ZERO (silently zero conflicting coefs and warn) OR PERMIT (keep both active, document the additive behavior).
  - Claude Position: REJECT. Catching it at config load saves N hours of "why isn't my deepseek bias working" debugging. AUTO-ZERO is the second choice.
  - Codex Position: Conflict resolution must be specified explicitly; PERMIT is the most surprising and should be ruled out.
  - Tradeoff Summary: REJECT requires every legacy yaml to be cleaned up before it passes validation; AUTO-ZERO keeps backward compatibility but hides the cleanup; PERMIT preserves current (broken) behavior.
  - Decision Status: **RESOLVED 2026-04-27** — AUTO-ZERO + warn. The conflict path zeros the conflicting legacy coef silently and emits a clear deprecation warning to stderr / W&B. **Constraint on our own configs**: the 39-config matrix yamls (task26) MUST be authored clean — `aux_loss` rows have zero `bias_update_rate` and `seq_aux_loss_coef`; `deepseek_bias` rows have zero aux coefs; `quantile` rows have all aux/bias coefs zero. Auto-zero is a safety net for external / legacy yamls only, NOT a license to write sloppy configs in this repo. The validator from AC-17 should additionally reject conflict in any yaml under `configs/` (stricter local rule than the project-wide AUTO-ZERO default).

- DEC-3b: Canonical config block for `bias_update_rate`, `bias_warmup_start`, `bias_warmup_steps`, `seq_aux_loss_coef`, `router_aux_loss_coef`, `load_balancing_method`, and per-projection bias rates — UNDER `model:` (matches every yaml today, but `TrainingConfig` would need to start reading from `cfg["model"]`) OR UNDER `training:` (matches `TrainingConfig` today, but every yaml needs migration) OR PER-FIELD (some live under `model:`, others under `training:`, with a documented rule).
  - Claude Position: UNDER `training:` — matches `TrainingConfig`, smaller code change; emit a deprecation warning when these fields are found under `model:` and migrate every existing yaml in a single sweep.
  - Codex Position: Decide explicitly; today's "field is in `model:` but `TrainingConfig` reads `training:`" is the actual cause of `bias_update_rate` being silently zero in every yaml.
  - Tradeoff Summary: `training:` is closer to current code; `model:` is closer to current yamls. PER-FIELD risks confusion. Either monolithic choice requires migration, but `training:` migration is yaml-only.
  - Decision Status: **RESOLVED 2026-04-27** — UNDER `training:`. Task7 migrates `bias_update_rate`, `bias_warmup_start`, `bias_warmup_steps`, `seq_aux_loss_coef`, `router_aux_loss_coef`, `load_balancing_method`, and `bias_rate_q/k/v/o/mlp/branch` from `cfg["model"]` to `cfg["training"]` in every yaml under `configs/`. `TrainingConfig.from_dict` continues to read from `cfg["training"]` (no code change there) and additionally emits a deprecation warning when any of these fields is found under `cfg["model"]`. `model_factory.py` no longer reads these fields from `mcfg`.

- DEC-4: DDP reduction across ALL load-balancing methods — pick a consistent "all-rank" policy that matches Megatron's choices wherever they apply.
  - Claude Position (initial): Per-rank-mean for quantile only; rank-local for Switch aux.
  - Codex Position: Approximate per-rank-mean must be labeled approximate; the choice must be tested against the exact global quantile.
  - Decision Status: **RESOLVED 2026-04-27** — global "all-rank" semantics across the board, with the right collective per statistic:
    - **DeepSeek bias update**: `all_reduce(SUM)` of `local_tokens_per_expert` before the update. Already implemented in `src/training/routing.py:65`. Matches Megatron-LM `moe_utils.py:1155` and nmoe `model.py:337`. **No code change**.
    - **Branch router bias update (DeepSeek-style)**: `all_reduce(SUM)` of `local_counts` before the update. Already implemented in `src/training/routing.py:83`. **No code change** (modulo DEC-18 unification below).
    - **Switch aux loss**: `all_reduce(SUM)` of `tokens_per_expert` AND of `router_prob_per_expert` before computing the loss. **Code change required** in `src/models/routing/load_balancing.py:108-111` — replace the rank-local computation with `all_reduce(SUM)` (guarded by `dist.is_initialized()`). Matches Megatron's `global_tokens_per_expert` path. The `f_i is kept local per rank — no all_reduce` comment is replaced with a comment explaining the global-aggregate semantics. **Knock-on**: any `router_aux_loss_coef` previously calibrated against rank-local will need re-validation; the 39-config matrix is not yet calibrated so this absorbs cleanly here.
    - **Sequence aux loss**: stays per-sequence, no DP cross-rank reduction. Matches Megatron's `tp_cp_group`-only reduce (we have neither TP nor CP, so this reduces to no-op). nmoe does not implement seq_aux.
    - **Quantile (new)**: `all_gather` of raw scores → concatenate → exact global quantile. Per-rank-mean of per-rank quantiles is biased and is NOT used. Cost: `O(B * T) = O(32 * 1024) = 32K fp32 floats / rank / layer / step` ≈ 128 KB/rank/layer/step (the per-token scores live on the score axis of length 1, but we collect across the token axis). With 16 layers × 8 ranks this is ~16 MB total per step — well within H200 NVLink budget. Implementation in `update_bias_from_quantile` (`src/models/routing/bias.py`): `gathered = [torch.empty_like(scores) for _ in range(world_size)]; dist.all_gather(gathered, scores); global_scores = torch.cat(gathered, dim=0); q_global = torch.quantile(global_scores, target_q, dim=0)`.
    - AC-11 narrows to: assert `all_gather` quantile equals `torch.quantile` over the manually-concatenated rank tensors to fp32 round-off; per-rank-mean parity test is removed (we don't ship that path).
    - Task18 updated: implement `all_gather` reduction; no per-rank-mean fast path.
    - All `dist.*` calls remain guarded by `dist.is_initialized()` so `world_size==1` paths are unchanged.

- DEC-5: Quantile EMA state checkpoint persistence — PERSIST (resume bit-equivalence) OR NON-PERSISTENT (recompute deterministically over a documented warmup window after resume).
  - Claude Position: Persist. Resume bit-equivalence is a cheap property to keep and tests for it are already standard in `tests/test_checkpoint_resume.py`.
  - Codex Position: Decide explicitly — affects resume equivalence.
  - Tradeoff Summary: Persistent state adds `N * 4` bytes per router to the checkpoint (negligible). Non-persistent forces a warmup-window contract that must be documented and tested.
  - Decision Status: **RESOLVED 2026-04-27** — PERSIST. The quantile EMA tensor (length-N, fp32) is a `register_buffer(..., persistent=True)` on each router that runs `balancing: quantile`. AC-12 narrows to the bit-equivalent resume path only; the non-persistent warmup-window path is removed. `tests/test_checkpoint_resume.py` is extended to cover the quantile resume case.

- DEC-6: §9 branch-router treatment in the 13-yaml matrix — FIXED `none` (clean ablation, branch behavior is constant across rows; risks branch collapse) OR FIXED `exploration_only` (more realistic, isolates branch-balancing from MLP-balancing variable) OR VARIES with the balancing column (3× more configs but isolates branch-balancing effects).
  - Claude Position: Fixed `exploration_only` for `precompute_kv*` rows (which have a branch router) and N/A for `standard_moe`/`global_moe` rows (no branch router). This isolates the MLP balancing variable cleanly without exposing the matrix to branch collapse.
  - Codex Position: Decide explicitly — without a decision, branch behavior becomes a confounded variable across the matrix.
  - Tradeoff Summary: Fixed `none` risks branch collapse mid-run (especially for `quantile` rows that take longer to converge); fixed `exploration_only` has no branch-balancing confound but adds `exploration_rate` as a hyperparameter; VARIES triples the row count and complicates analysis.
  - Decision Status: **RESOLVED 2026-04-27** — Fixed `exploration_only` for every `precompute_kv*` row in the §9 matrix. The 6 `precompute_kv*` rows (3 balancing variants × 2 architecture flavors `precompute_kv` and `precompute_kv_global_router`) all set `branch_router.balancing: exploration_only` with the documented default `exploration_rate: 0.1`, `exploration_decay: cosine`, `exploration_min: 0.01`, `exploration_warmup_steps: 1000`. The `standard_moe` and `global_moe` rows have no branch router so this knob is N/A for them. Task26 (yaml authoring) reflects this.

- DEC-7: §10 back-compat shim shape — RUNTIME shim in `model_factory.py` (kept until end of migration, removable as a single PR) OR ONE-TIME yaml MIGRATOR script (rewrites every yaml in-place, no shim) OR DUAL (shim + migrator + CI gate).
  - Claude Position: Runtime shim for the transition (Milestone E lands without breaking any existing yaml), followed by a one-shot migrator at the start of Milestone F (rewrites yamls onto nested schema, removes the shim).
  - Codex Position: Indefinite dual semantics hide stale configs; prefer one-time YAML migrator + validator + delete shim.
  - Tradeoff Summary: Runtime shim trades correctness-by-migration (yamls remain in the old shape) for ease-of-rollback (revert one PR to undo). One-time migrator forces every yaml to migrate cleanly, at the cost of a bigger single PR. DUAL is the safest but most complex.
  - Decision Status: **RESOLVED 2026-04-27** — ONE-TIME YAML MIGRATOR, no runtime shim. The configs are not set in stone (the user can always change them), so backward compatibility for old-shape yamls is not required. **Implications**:
    - Task22 (schema definition) drops the shim-design subtask; replaced by a one-shot migrator script that rewrites every yaml under `configs/` from flat → nested in a single PR.
    - Task23 (`model_factory.py` refactor) reads the new nested shape only — there is no flat-to-nested in-memory remapping path.
    - Task24 (validator) rejects any yaml under `configs/` that still uses the flat schema; this becomes a CI gate so future yamls cannot drift back.
    - Task25 (migration-parity test) runs once, BEFORE the migrator PR merges, to prove flat-vs-nested instantiation produces identical models on every yaml. After the migrator merges, the parity test is removed (the flat-shape path no longer exists).
    - The DEC-7 alias rule from AC-13 / task22 (the original draft §10's `switch` and `seq_aux` tokens) becomes part of the migrator: the migrator rewrites those tokens to `aux_loss` / `seq_aux_loss`. The runtime no longer accepts the alias tokens after migration.
    - Codex's preferred approach (matches Codex r1 OPTIONAL_IMPROVEMENTS).

- DEC-8: Benchmark logging path (§7) — PRODUCTION (W&B + checkpoint saves on, measures exactly what training experiences) OR STRIPPED (W&B + checkpoints off, isolates compute throughput).
  - Claude Position: PRODUCTION path. The whole point of "uses the same code as training" is to measure what training actually experiences.
  - Codex Position: Decide — production path may include W&B/checkpoint overhead that varies by config.
  - Tradeoff Summary: PRODUCTION is more realistic but noisier (W&B network jitter, checkpoint I/O if the bench window crosses `save_every`). STRIPPED is cleaner but no longer "what training experiences"; bench numbers may not predict real training step time.
  - Decision Status: **RESOLVED 2026-04-27** — PRODUCTION trainer body with **W&B explicitly DISABLED** for bench runs. `scripts/bench_step.py` calls `run_training` (the same code path as training) with overrides:
    - `wandb` config block forced off (no W&B init, no per-step log calls).
    - Checkpoint saves disabled within the bench window: `save_every` set higher than the bench-step count (e.g. `save_every=10_000` while bench runs 100 steps), so no `save_checkpoint` call fires during the measurement window.
    - Eval disabled within the bench window: `eval.every` set to 0 (or higher than bench-step count) so `run_validation` doesn't fire.
    - Everything else stays at production: data loader, optimizer, DDP wrap, all-reduces, post-step bias update, log_training_step (stdout only with `wandb_run=None`), routing-heatmap saves disabled by setting `heatmap_every=0`.
    - This keeps "uses the same code as training" intact while removing the two known sources of measurement jitter (W&B network calls and disk I/O).
    - The `disable_liger` setting comes from the yaml itself, not from the bench script — DEC-16 governs that flag.
    - Task27 reflects these overrides; `scripts/bench_step.py` documents them inline.
    - `bench/README.md` makes it explicit that bench numbers are "production trainer minus W&B/checkpoint/eval/heatmaps", so anyone reading the numbers knows what's measured.

- DEC-9: §8 comparison purpose — PICK A WINNER (pin one method as the default per family) OR CHARACTERIZE TRADEOFFS (no default change, just document `(family × method) → (final_loss, mean_f_i_KL, max_f_i)` table).
  - Claude Position: CHARACTERIZE for v0; pinning a default is a downstream decision once we have data.
  - Codex Position: Decide explicitly — affects whether AC-21 narrows to logging-only.
  - Tradeoff Summary: PICK A WINNER lets us simplify defaults per family; CHARACTERIZE keeps optionality but leaves users to guess which method to enable.
  - Decision Status: **DEFERRED 2026-04-28** — neither A nor B; the §8 comparison sweep is out of scope for this plan entirely. The 12 training runs (4 methods × 3 families × ~5K steps) will be authored as a follow-up plan once Priority 1 (correctness) and Priority 2 (throughput characterization) are landed and the speed-borrowing wins from Priority 3 are committed. AC-21 is removed from this plan's hard ACs and downgraded to a "follow-up plan owns this" footnote. The 39-config matrix (AC-18) still exists — it's needed for AC-24 (max-batch search + throughput) — but its yamls don't have to be tuned for any particular `load_balancing_method` winner; each yaml just exists with whatever default is reasonable for its row's purpose.

- DEC-10: "8L MoE-Everything" terminology — config-folder depth (8L = `configs/8_layers/`) OR `num_hidden_layers` field OR depth-iter count (`actual_depths` in `MoEverythingModel`).
  - Claude Position: Config-folder depth. The folder is the user-facing organizing principle and matches the §9 matrix axis. Document this in `configs/README.md` so future readers don't conflate it with `num_hidden_layers`.
  - Codex Position: Current `configs/8_layers/moe_everything_per_head_precompute_kv.yaml` uses `num_hidden_layers: 16`, so the terminology is already overloaded — pick one.
  - Tradeoff Summary: Config-folder depth is more stable across yaml-author choices; `num_hidden_layers` is more precise but requires every reference to be qualified.
  - Decision Status: **RESOLVED 2026-04-28** — config-folder depth ("8L" = `configs/8_layers/`). The `num_hidden_layers: 16` setting in `moe_everything` configs is intentional: each "logical layer" in `moe_everything` is split into TWO depth iterations (one ATTN-branch step + one MLP-branch step under `sanity_check_mode='alternating_global_moe'`, or one branch-routed mixed step in normal operation). So the for-loop runs 16 times to deliver 8 logical attention+MLP layers — the `8L` naming reflects the logical layer count, not the for-loop iteration count. **Doc obligation**: `configs/README.md` (NEW — to be authored alongside the matrix authoring in Milestone H) MUST explain this naming convention clearly so future yaml authors don't accidentally set `num_hidden_layers: 8` for a moe_everything config in `configs/8_layers/` (which would actually produce a 4-logical-layer model).

- DEC-11: Quantile `target_q` default per router class — `1 - top_k / num_experts` (draft default; valid for plain top-k MLP routers, naive for group-limited and per-head top-1 attention routers) OR per-router-class adaptive default using each class's effective `top_k` and `num_experts`.
  - Claude Position: Use the same `1 - effective_top_k / effective_num_experts` formula per class — for group-limited routing the relevant pool is still `num_experts` so the formula is unchanged; for per-head top-1 attention routers (in `moe_everything`) use `1 - 1/n_attn_experts`. Document the rule in `bias.py` and in the per-class config docstring.
  - Codex Position: Default ignores group-limited routing and per-head top-1; needs an explicit per-class rule.
  - Tradeoff Summary: One-formula-with-class-effective-K is cheap and mostly right. A more principled per-class tuning would require an empirical sweep before pinning defaults.
  - Decision Status: **RESOLVED 2026-04-28** — per-class `target_q = 1 - effective_top_k / effective_num_experts`. Concrete defaults:
    - Plain top-k MLP routers: `1 - top_k / num_experts` (e.g. `num_experts=16, top_k=4` → `target_q = 0.75`).
    - Group-limited routers (`num_groups`/`group_topk` set): same formula — the relevant pool is still `num_experts`. The group-limited mechanism narrows which experts are *eligible* but doesn't change the target balance fraction.
    - Per-head top-1 attention routers in `moe_everything` (q/k/v/o): `1 - 1 / n_attn_experts` (e.g. `n_attn_experts=4` → `target_q = 0.75`).
    - Per-class adaptation rule documented in `docs/routing.md` §5.1 (the Quantile Balancing implementation reference, added when task15-17 land) and in the docstring of `update_bias_from_quantile` (`src/models/routing/bias.py`).
    - Per-family TUNING is a separate question (deferred — `TODO.md`); the formula above is the v0 default for all three families.

- DEC-12: Compute envelope discipline — confirm: max 8 H200 nodes concurrent, idle-off when not actively training/benchmarking, all GPU work on Modal via `.claude/skills/`.
  - Claude Position: Yes — mirror the existing project memory `feedback_modal_conda.md`. Include a "GPU envelope check" gate in any agent that launches training: assert `--max-gpus=8` and post-run shutdown.
  - Codex Position: N/A (resource discipline, not architectural).
  - Tradeoff Summary: This is a constraint, not a tradeoff — confirming explicitly to lock the agent behavior.
  - Decision Status: **RESOLVED 2026-04-28** — Modal-only, with skill selection by job shape:
    - **Interactive debugging / profiling / quick prototyping** → `.claude/skills/gpu-debugging` (SSH-able sandboxes; tear down when done). Use this for: debugging a flaky test on real H200, running `nsys` / `ncu` profiles, sanity-checking a config before launching the matrix bench, exploring whether a Liger flag breaks AC-23 on a single config.
    - **Longer runs (training jobs, benchmark sweeps, ~hours-long workloads)** → `.claude/skills/modal-experiment` (Modal apps with `modal run` / `modal deploy`, attached volumes for data + checkpoints, secrets for W&B / HuggingFace, retries + auto-resume from checkpoints). Use this for: AC-24 max-batch search across 39 configs, AC-25 Liger re-enablement throughput delta, AC-23 50-step drift test on multiple configs, the eventual §8 follow-up sweeps.
    - **Parallel work across multiple Modal jobs** → `.claude/skills/sub-agents` (spawn N parallel Claude Code agents, each owning its own Modal app/sandbox, with structured progress reports). Use this for: running the AC-24 search across 39 configs in parallel batches (each batch = ≤8 H200), or parallelizing the AC-23 sanity-equivalence variants (DDP, gradient-checkpointed, 50-step-drift) across separate H200 sandboxes.
    - **General Modal SDK usage** → `.claude/skills/modal` (referenced when writing custom Modal apps that aren't covered by the specialized skills above).
    - **Hard constraints across all skills**:
      - Max 8 H200 nodes concurrent across the entire workflow (sub-agents tracker enforces this so parallel sub-agents don't accidentally over-allocate).
      - Idle-off: every Modal sandbox shuts down on `exit` / `__exit__`; every `modal run` job exits cleanly when the script returns; no long-lived idle reservations.
      - Modal CLI requires `conda activate modal` before use (per `feedback_modal_conda.md`).
      - Volumes / checkpoints / data are NEVER deleted without explicit user instruction (per `feedback_no_delete_without_asking.md`).
      - Each launching agent verifies post-run shutdown before reporting completion.
    - **AC-22 expanded**: tests assert each skill is used for its intended job shape (e.g. AC-24 max-batch sweep uses `modal-experiment`, AC-23 single-config sanity-equivalence test uses `gpu-debugging`). The assertion is mostly documentation in `bench/README.md` and per-task notes; not a runtime CI check.

- DEC-13: `global_moe` quantile EMA state placement — SHARED (one `state['ema_q']` reused across every layer router; matches the shared MLP pool semantics) OR PER-LAYER (each layer router carries its own EMA state; diagnoses per-layer score-distribution drift).
  - Claude Position: SHARED — `global_moe`'s defining feature is one shared expert pool, so a single EMA tracking the global quantile is the conceptually-clean default. Expose `quantile_global_state: bool` (default `true`) for the per-layer alternative.
  - Codex Position: SHARED with `quantile_global_state: true` as the default. Matches `global_moe`'s shared-pool semantics; the per-layer flag exists for diagnosing calibration drift but should not be the default.
  - Tradeoff Summary: SHARED is one buffer of size `N`. PER-LAYER is `num_layers * N`. SHARED collapses cleanly under quantile balancing because `global_moe`'s router output is supposed to converge to the same calibration anyway. PER-LAYER protects against early-training calibration drift.
  - Decision Status: **SUPERSEDED 2026-04-27 by DEC-19** — DEC-19 makes ALL load-balancing state (bias, count buffer, quantile EMA) bank-level for shared-pool architectures. The `quantile_global_state` flag is no longer needed because there is no per-layer alternative — every per-layer router that routes into a shared bank reads/writes the bank's single quantile EMA. The "per-layer alternative" only exists in `standard_moe` where the expert pool is itself per-layer (so per-layer EMAs match per-layer pools). The corresponding task33 is deleted.

- DEC-14: FSDP support level for the rewrite — REQUIRED (every test that runs on DDP also runs on FSDP) OR OPTIONAL (DDP is the supported target; FSDP is "best-effort, not gated") OR OUT-OF-SCOPE (FSDP-specific code paths skipped, AC-2/AC-10/AC-11 explicitly say "DDP only").
  - Claude Position: OPTIONAL for v0 — every AC test runs on DDP; an FSDP smoke test on AC-2 exists but is skipped if FSDP env is unavailable. Promote to REQUIRED after the rewrite lands and FSDP is exercised for a real training run.
  - Codex Position: OPTIONAL for v0 with an explicit FSDP smoke/skip policy: AC-2/AC-10/AC-11 each ship an FSDP smoke variant that skips with a documented "FSDP not exercised in this run" message rather than silently passing. Promote to REQUIRED only after FSDP is exercised in a real training run on this codebase.
  - Tradeoff Summary: REQUIRED triples test surface; OPTIONAL covers the common case but leaves FSDP regressions discovered late; OUT-OF-SCOPE preserves DDP velocity but explicitly tells FSDP users to re-test before merging.
  - Decision Status: **RESOLVED 2026-04-28** — OPTIONAL v0 with explicit skip-with-reason policy. Concretely: AC-2, AC-10, and AC-11 each ship an FSDP smoke variant that runs only when an `FSDP_AVAILABLE` env-or-fixture flag is set; otherwise the test calls `pytest.skip("FSDP not exercised in this run; re-run with FSDP_AVAILABLE=1 on a multi-rank FSDP environment")` instead of silently passing. This avoids false-greens (the failure mode where the test "passes" because it never actually exercised FSDP). Promote to REQUIRED in a follow-up plan once the codebase has at least one real FSDP training run on the books with these skill paths exercised. Until then, the trainer's FSDP code path is documented as "DDP-tested, FSDP-best-effort" in `docs/distributed.md`.

- DEC-15: `output_router_logits` policy for non-aux methods — KEEP `True` ALWAYS (current default; logging cost stays consistent across methods) OR DETACH-ONLY (router logits remain available for telemetry but are detached so they cost no autograd memory) OR FULLY DISABLE for `none`/`deepseek_bias`/`quantile` methods (saves activation memory, but loses telemetry until re-enabled).
  - Claude Position: KEEP `True` ALWAYS — the §7 benchmark protocol requires the realistic training path, which has `output_router_logits=True` on by default. Switching it off for some methods would make per-method bench numbers incommensurable.
  - Codex Position: DETACH-ONLY for non-aux methods (`deepseek_bias`, `quantile`, `none`). Telemetry stays available so the §8 comparison still produces `f_i` plots, but the autograd graph for non-aux methods doesn't carry the router-logit tensors. Bench numbers under DETACH-ONLY are still comparable across methods so long as the bench script applies the same policy.
  - Tradeoff Summary: KEEP-ON makes bench numbers comparable across methods. DETACH-ONLY trims the autograd graph for `none`/non-aux methods. DISABLE is the most aggressive memory-saver but loses every telemetry plot (`f_i` distribution, branch fraction, etc.) we depend on for the §8 comparison.
  - Decision Status: **RESOLVED 2026-04-28** — match the nmoe/Megatron architecture instead of the HF-Qwen3MoE convention we inherited. Concretely:
    - **Non-aux methods (`deepseek_bias`, `quantile`, `none`): DETACH-ONLY.** Router scores are stored as router-internal state (`_last_top_k_idx`, `local_tokens_per_expert`, plus a new `_last_router_scores_detached` attribute for telemetry). The model's `forward` does NOT include them in the autograd graph for these methods. Matches nmoe's `last_loads`/`last_importance` side-effect-attribute pattern and Megatron's no-`output_router_logits`-flag design exactly.
    - **Aux methods (`aux_loss`, `seq_aux_loss`): KEEP `output_router_logits=True` for THIS plan.** The minimal change keeps the existing aux-loss path in `forward` working without a deeper refactor.
    - **Follow-up (tracked in `TODO.md`)**: refactor the aux-loss path to apply aux loss INSIDE the router's forward via Megatron's `MoEAuxLossAutoScaler`-style autograd hook, so even aux methods stop needing `output_router_logits` in the model output. This is a meaningful surgery to `MoEverythingForCausalLM.forward` / `StandardMoEModel.forward` / `GlobalMoEForCausalLM.forward` and is deferred — not gated on Priority 1.
    - **Implementation**: `model_factory.py` sets `output_router_logits` based on the resolved `load_balancing_method`: `True` only for `aux_loss` / `seq_aux_loss`, `False` for `deepseek_bias` / `quantile` / `none`. Each model family's `forward` handles the `None` case by reading from the router-internal `_last_router_scores_detached` attribute when telemetry is needed (e.g. for `RoutingStats` plots). Tests cover both branches. `bench_step.py` applies the same policy across the matrix automatically (since each yaml has a `load_balancing_method`); bench numbers ARE comparable per the row's method, just not strictly across rows that use different methods — which is fine since the bench is a max-batch + throughput characterization, not a method-vs-method comparison.

- DEC-16: 39-config matrix `disable_liger` setting — KEEP `true` (current default; CI-safe but slower) OR FLIP TO `false` AFTER §1-§5 LAND (measured rerun to quantify Liger throughput delta) OR PER-METHOD (Liger is on for `aux_loss`/`seq_aux_loss`, off for `quantile` if Liger interacts badly with custom router state).
  - Claude Position: KEEP `true` for the v0 39-config benchmark (apples-to-apples). Run a separate 3-config sanity sweep after §1-§5 land with `disable_liger: false` to quantify the throughput delta; if it's > 5% and there are no correctness regressions, flip the matrix in a follow-up.
  - Codex Position: KEEP `disable_liger: true` for v0 + run a separate 3-config sanity sweep with `disable_liger: false` AFTER §1-§5 land. Decide the matrix flip in a follow-up plan once the sanity sweep has data — flipping the matrix mid-effort would invalidate the comparison numbers.
  - Tradeoff Summary: KEEP `true` keeps the bench reproducible. FLIP could materially improve throughput but risks Liger-vs-DeepSeek-router interactions we haven't tested. PER-METHOD breaks comparability.
  - Decision Status: **RESOLVED 2026-04-28** — KEEP `disable_liger: true` for the 39-config matrix in this plan. Milestone J (AC-25 speed-borrowing) runs the 3-config sanity sweep with `swiglu` and `fused_linear_cross_entropy` re-enabled for `moe_everything` to measure the Liger throughput delta. The MATRIX-WIDE flip is **deferred to a follow-up plan** (tracked in `TODO.md` "`disable_liger: true → false` matrix flip"); flipping mid-effort would invalidate the AC-24 throughput characterization numbers. Per-method Liger toggling is rejected — breaks comparability across rows in the matrix.

- DEC-19: Bank-level load balancing (added 2026-04-27, supersedes any per-layer-router-bias assumption baked into earlier ACs). When the EXPERT WEIGHTS are shared across multiple per-layer routers, the load-balancing STATE belongs on the bank, not the router. The per-layer routers are stateless consumers of bank-level state.
  - Decision Status: **RESOLVED 2026-04-27** — bank-level load balancing for every architecture that has a shared expert pool. Implementation:
    - **State location**: `expert_bias` (length-N, persistent, fp32), `local_tokens_per_expert` (length-N, non-persistent, fp32), and the quantile EMA buffer (`quantile_ema`, length-N, persistent, fp32 — DEC-5) live on the BANK module, not on the router. Banks affected:
      - `src/models/global_moe.py:GlobalMoEModel` — owns ONE `expert_bias` over the global expert pool. The per-layer routers (`GlobalSparseMoeBlock.gate`) read from it.
      - `src/models/moe_everything/mlp_bank.py:MlpExpertBank` — owns ONE `expert_bias` over the shared MLP expert pool (regardless of `per_layer_mlp_router`).
      - `src/models/moe_everything/attention_bank.py:AttentionExpertBank` — owns FOUR `expert_bias` buffers, one per attention expert class (`q`, `k`, `v`, `o`), each over the corresponding shared attention expert pool (regardless of `per_layer_attn_router`).
    - **Routers become stateless consumers**: `DeepSeekRouter.forward` no longer registers `expert_bias` or `local_tokens_per_expert` as `self.*` buffers when constructed inside a bank that owns them. Instead, the router takes a reference to the bank's buffers at construction time and reads/writes through that reference. The bank passes `expert_bias` and `local_tokens_per_expert` into the router as constructor arguments (or the router queries `self.bank.expert_bias` via a back-pointer).
    - **Standalone routers keep their own state**: `standard_moe` has per-LAYER expert pools (each layer owns its own MLP expert weights), so each layer's `DeepSeekRouter` still owns its own `expert_bias` — there's no shared bank to attach to. The `DeepSeekRouter` class therefore remains capable of owning its own buffers; the bank-level path is an additional construction option, not a replacement.
    - **Bias update walker** (AC-2 expansion): `update_expert_biases` walks BOTH (a) standalone `DeepSeekRouter` instances (in `standard_moe`) AND (b) shared expert banks (in `global_moe` and `moe_everything`). Each bank exposes the same buffer interface `expert_bias`, `local_tokens_per_expert`, optional `quantile_ema` so the unified `_update_single_router_bias` walker handles them with no special-case.
    - **Aux loss aggregation at bank level**: `MoEverythingForCausalLM.forward` and `GlobalMoEForCausalLM.forward` aggregate `tokens_per_expert` and `router_prob_per_expert` across **all per-layer routers that feed the same bank** and compute ONE aux-loss tensor per bank. This is mathematically equivalent to summing per-layer aux losses (up to a factor) but is cheaper and matches the bank-level interpretation. Same for `seq_load_balancing_loss_func`: the per-sequence f_i and P_i are aggregated across per-layer routers feeding the same bank before the per-sequence loss is computed.
    - **Quantile**: the per-step accumulator and the EMA buffer (DEC-5) live on the bank. Every per-layer router that routes into the bank appends its detached scores to `bank.local_quantile_scores` (the cross-microbatch + cross-per-layer-router accumulator). Post-step `update_bias_from_quantile` runs once per bank (not once per router), `all_gather`s the bank's accumulator (per DEC-4), computes the global quantile, updates `bank.quantile_ema`, writes `bank.expert_bias`. Bank's `local_quantile_scores` is cleared after the update.
    - **Bias buffer count goes DOWN**: `global_moe(num_layers=16)` collapses from 16 `expert_bias` buffers (one per layer) to 1 (on the model). `moe_everything(per_layer_mlp_router=true, num_depths=16)` collapses from 16 MLP biases to 1, plus 16 q/k/v/o-router biases collapse to 4 (one per attention class) instead of 64.
    - **Architectures NOT affected**: `standard_moe` has per-layer expert pools (each `Qwen3MoeSparseMoeBlock` owns its own `experts`), so per-layer biases stay. `branch_router` has no expert bank (the "pool" is just `[ATTN, MLP]`), so DEC-19 does not change branch-router state — it stays per-router, unified per DEC-18.
    - **Migration**: existing checkpoints have per-router `expert_bias` buffers. The DEC-7 migrator script (task22) rewrites checkpoint state-dict keys: `model.layers.<i>.mlp.gate.expert_bias` → `model.<bank-attr>.expert_bias` with appropriate aggregation (mean of per-layer biases is a sensible initialization for resume-from-old-ckpt, since they were independently learning offsets toward the same uniform target). The migrator emits a one-time warning on resume describing the conversion.
    - **AC implications**:
      - AC-2 expanded: walker covers banks AND standalone routers; coverage-counter test asserts the right number of ownership entities per architecture (e.g., `global_moe(8L)` → 1 bank-level bias; `standard_moe(8L)` → 8 per-layer biases; `moe_everything(8L, per_layer_mlp_router=true)` → 1 MLP bias + 4 attention biases (q/k/v/o)).
      - AC-6 bias-update parity test runs at bank level for shared-bank architectures.
      - AC-10/AC-11 score-accumulation, quantile, and DDP `all_gather` paths all run at bank level for shared-bank architectures; one `all_gather` per bank per step, not per router per step (this also reduces the DDP collective count — net improvement).
      - AC-12 EMA-persistence test asserts the bank's `quantile_ema` is in the checkpoint, NOT the router's.
      - AC-13 migration-parity test must include the bank-vs-router state-dict shape change.
    - **Task implications**:
      - task1, task2 broadened: walk banks AND routers via a uniform `get_all_balancing_owners()` interface (or rename `get_all_routers` to `get_all_balancing_owners` since "router" no longer captures the shape).
      - task15-task19 (quantile): everything moves to bank level for shared-bank architectures. AC-10 pseudocode updated accordingly.
      - task22 (migrator): also performs the per-router-bias → bank-bias state-dict rewrite for old checkpoints.
      - task33 (DEC-13 `quantile_global_state` flag): becomes obsolete — DEC-19 makes shared-bank quantile state the only path. Remove task33; mark DEC-13 as superseded by DEC-19.
    - Codex Position: Agreed in retrospect; this is the cleaner architecture. The "per-layer router with own bias over a shared pool" is double-bookkeeping — DEC-19 removes it.
    - Doc: `docs/routing.md` and `docs/architecture.md` get a "Bank vs Router state ownership" subsection explaining the rule (shared pool → bank owns balancing state; per-layer pool → router owns balancing state).

- DEC-18: Unified router-bias path — kill the legacy `_update_branch_router_bias` and the `branch_bias` / `local_counts` buffer names; route every router (MLP / attention / branch) through the single `_update_single_router_bias` walker. **Decision Status: RESOLVED 2026-04-27** — YES, unify. Implementation:
  - `BranchRouter.__init__` (`src/models/routing/routers.py:50-65`) renames `register_buffer("branch_bias", ...)` → `register_buffer("expert_bias", torch.zeros(2, dtype=torch.float32))` and `register_buffer("local_counts", ..., persistent=False)` → `register_buffer("local_tokens_per_expert", torch.zeros(2, dtype=torch.float32), persistent=False)`. Length stays 2 (the math handles `N=2` as a special case of `N=any` for free).
  - `BranchRouter.forward` (`src/models/routing/routers.py:115-119`) updates references: `self.local_counts += counts` → `self.local_tokens_per_expert += counts`.
  - `update_expert_biases` (`src/training/routing.py:11-50`) drops the separate branch-router code path. The `get_all_routers()` walker discovers `BranchRouter` instances the same way it discovers `DeepSeekRouter` instances (both expose `expert_bias` + `local_tokens_per_expert`), and `_update_single_router_bias` runs on both without specialcase. Delete `_update_branch_router_bias` entirely.
  - The per-class `balancing` enum from AC-13 applies to `branch_router` exactly as to `mlp_router` and `attn_router`: `aux_loss | seq_aux_loss | deepseek_bias | quantile | none | exploration_only` (the `exploration_only` value is branch-only). All five non-`exploration_only` values reuse the *same* code path as MLP/attention routers.
  - `MoEverythingForCausalLM.forward` accumulates `branch_router`'s aux-loss / seq-aux contribution into the same total as MLP/attn aux losses when `branch_router.balancing ∈ {aux_loss, seq_aux_loss}` — the dispatch is uniform, not branch-specific.
  - **Migration**: existing checkpoints using `branch_bias` / `local_counts` are renamed by a one-shot checkpoint converter (added to the DEC-7 migrator script — same PR). The converter rewrites old state-dict keys `*.branch_bias` → `*.expert_bias` and old `*.local_counts` references are dropped (they're non-persistent so they aren't in the checkpoint anyway). The runtime no longer accepts the old buffer names.
  - **AC implications**: AC-2's positive coverage test now asserts the SINGLE walker visits every router (MLP + attention + branch) and produces correct length-N (or length-2 for branch) bias deltas. The "singular `branch_router` discovery" line in AC-2 no longer needs a special case — it's just "the walker finds every router, including the singular and plural branch attributes".
  - **Tests**: `tests/test_bias_update_detailed.py` extended to drive the unified path with all three router classes (MLP / attn / branch); the legacy `_update_branch_router_bias` test is deleted.
  - **Doc**: `docs/routing.md` rewrites the BranchRouter section to use the canonical buffer names; the "DeepSeek-style branch routing" subsection is folded into the unified "DeepSeek bias balancing" section since they're the same thing now.
  - Codex Position: Agreed (this aligns with Codex's r1 finding that `_update_branch_router_bias` was a coverage-gap hack).

- DEC-17: Softmax-router score-vs-top-k ordering (added 2026-04-27 from user research on Megatron). Megatron-LM exposes `--moe-router-pre-softmax` to choose between **pre-softmax** (softmax over all experts → top-k → gather) and **post-softmax** (top-k on raw logits → softmax over the k selected only). With `top_k=1`, post-softmax of a single value collapses to `1.0` and kills router-weight gradient — Megatron raises an explicit error in that case (`Megatron-LM/megatron/core/transformer/transformer_config.py:1875`: "Please use --moe-router-pre-softmax when topk is 1"). Our existing `ExplorationTopKRouter` already has a `router_topk_ordering: post | pre` knob (`src/models/router.py:163-263`), but the naming is INVERTED relative to Megatron and there's no top-1 guard.
  - Decision options: (A) KEEP existing `topk_ordering` naming + add the top-1 guard; (B) RENAME to Megatron's terminology (`softmax_position: pre_topk | post_topk`) + add the top-1 guard + accept the old `topk_ordering` token as a deprecated alias; (C) ADD MEGATRON-COMPATIBLE NAME alongside the existing one (no rename, both work, document the equivalence).
  - Claude Position: **(B) RENAME with deprecation alias.** `softmax_position: pre_topk` (default — gradient-safe at any `top_k`, matches Megatron's `--moe-router-pre-softmax`) or `post_topk` (rejected by validator when `top_k == 1` with the same error message Megatron uses). Old `router_topk_ordering: post` maps to `softmax_position: pre_topk`; old `router_topk_ordering: pre` maps to `softmax_position: post_topk`. The DEC-7 migrator rewrites the field name at the same time the schema migrates flat → nested.
  - Codex Position: This is an open question — pick one. Either pick (B) for clarity (Megatron's terminology is the field's lingua franca) or (A) for minimal-churn. Do NOT pick (C); two parallel knobs invite drift.
  - Tradeoff Summary: (A) preserves the existing field name (good for any external configs that already set it) but keeps the inverted semantics that confused this round; (B) aligns with Megatron and is unambiguous to anyone porting from there, but every `configs/` yaml needs a one-token rewrite (which the DEC-7 migrator can absorb at zero extra cost); (C) is the lowest-friction but two-knobs-for-one-thing always rots.
  - Tests: AC for this decision — a softmax router with `softmax_position: pre_topk` and `top_k=1` produces a non-zero gradient on the router weight on a fixed input/seed; a softmax router with `softmax_position: post_topk` and `top_k=1` is rejected by the validator with the documented error message. Both tested under the explanatory unit-test pair. Sigmoid routers (`DeepSeekRouter`, the `nmoe`/`modal-nmoe` style) are NOT affected — they have their own `top_k > 1` normalization-skip already (`src/models/router.py:346`) and the `softmax_position` knob does not apply to them.
  - Implementation tasks: new task37 (analyze the `ExplorationTopKRouter` rename + write the validator rule) and task38 (coding: implement the rename, the deprecation alias, the validator error, and the top-1 guard); both must update `docs/routing.md` and `docs/configuration.md` per the Documentation Discipline rule. The DEC-7 migrator absorbs the yaml-side field rewrite.
  - Decision Status: **RESOLVED 2026-04-27** — option B. Rename to `softmax_position: pre_topk | post_topk` (matching Megatron). Old `router_topk_ordering: post` maps to `softmax_position: pre_topk`; old `router_topk_ordering: pre` maps to `softmax_position: post_topk`. The DEC-7 migrator rewrites every yaml at the same time it migrates flat → nested. Validator rejects `softmax_position: post_topk` + `top_k: 1` with the Megatron error message ("Please use --moe-router-pre-softmax when topk is 1" — adapted to "Please use `softmax_position: pre_topk` when `top_k` is 1"). `ExplorationTopKRouter` honors the new field name; the old field name still parses for one release with a deprecation warning then is removed in a follow-up. Default is `pre_topk` (gradient-safe at any `top_k`).

## Implementation Notes

### Modal H200 Compute Envelope

All training and benchmarking work runs on Modal H200 sandboxes via the project's `.claude/skills/modal-experiment` and `.claude/skills/gpu-debugging` skills. Hard limits enforced by the implementing agent:

- Max 8 H200 nodes concurrent (any single launch).
- Idle-off when not actively training/benchmarking (Modal containers stop after the job completes; no long-lived idle reservations).
- Modal CLI requires `conda activate modal` first (per project memory `feedback_modal_conda.md`).
- Volumes/checkpoints/data are not deleted without explicit user instruction (per project memory `feedback_no_delete_without_asking.md`).

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers. These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead — e.g., `dispatch_load_balancing_method()` rather than `ac1_load_balancing_dispatch()`.

### Migration Safety

- Every yaml change must round-trip through the configuration validator.
- The §10 schema migration must include a flat-vs-nested instantiation parity test that runs as part of CI before the back-compat shim is removed.
- Existing checkpoints from old yamls must continue to load through the transition; if a structural change forces incompatibility, document it explicitly and add a checkpoint-converter pass.

### Reference Repo Discipline

- `Megatron-LM/`, `nmoe/`, `modal-nmoe/` are read-only references for cross-checking implementations and naming. Do NOT modify them.
- When citing them in code/comments, include a short note on which version is referenced (a hash, a date, or "as of plan authoring") so future readers know whether the reference has drifted.

### Documentation Discipline

Every new config option, knob, balancing method, schedule, or behavior introduced by this plan MUST be reflected in `/Users/leon/Desktop/modal/moe/docs/` in the same PR. The doc files that exist today (as of plan authoring 2026-04-27):

- `docs/configuration.md` — every new config field lands here with its type, default, allowed values, and a one-line description.
- `docs/routing.md` — router classes, balancing methods, exploration schedules, and the `pre`/`post` softmax-position semantics.
- `docs/training.md` — trainer loop wiring, post-step bias-update path, and per-class exploration schedules.
- `docs/architecture.md` — high-level architecture (touched only when a section's overall shape changes).
- `docs/distributed.md` — DDP/FSDP semantics for any new distributed paths (e.g. quantile reduction).
- `docs/research/` — research notes; cite when implementing a paper/recipe so future readers find the source.

Concretely, the per-task doc obligations from this plan are:

- task7 (DEC-3b: yaml-block migration) → update `docs/configuration.md` to document the new canonical `training:` block and the `model:` deprecation warning.
- task11 (DEC-2: bias-update normalization toggle) → document the new `bias_update_zero_sum` flag in `docs/configuration.md` and `docs/routing.md`.
- task13 (DEC-3a: legacy-coef conflict policy) → document the AUTO-ZERO + warn behavior and the `configs/` strict-rejection rule in `docs/configuration.md`.
- task15-17 (quantile method) → expand the `docs/routing.md` "Quantile Balancing" stub at §5.1 (added 2026-04-28) into a full implementation reference covering `quantile_eta`, `quantile_target_q`, the EMA semantics, the bank-level state ownership (DEC-19), the cross-microbatch accumulator (AC-10), and the `all_gather` DDP reduction (DEC-4). The §5.2 open-question note about per-family tuning STAYS — task15-17 must NOT delete it; it remains live until the deferred sweep is run (per `TODO.md`).
- task20 (exploration_only branch) → document the new `balancing: exploration_only` value, the three decay schedules, and the per-step telemetry in `docs/routing.md`.
- task22-24 (DEC-7: schema migration) → rewrite the per-router-class section of `docs/configuration.md` to describe the nested `mlp_router` / `attn_router` / `branch_router` groups and the canonical `balancing` enum.
- task27 (bench script) → add `bench/README.md` (and a pointer from `docs/training.md`) describing the bench protocol, the `disable_liger` policy (DEC-16), and the `output_router_logits` policy (DEC-15).
- Pre/post-softmax option (DEC-17): document the `softmax_position` knob and the top-1 guard in `docs/routing.md`.

This is a hard rule, not a nice-to-have. A PR that introduces a new option without the matching doc update is incomplete and the validator from task24 should call it out as a CI failure when the option-vs-doc-coverage check is implemented.

--- Original Design Draft Start ---

> **Non-normative appendix.** The text below is the verbatim original draft as authored before this plan was generated. It is preserved as a historical reference, not as a normative spec. Where the draft conflicts with the structured plan above (e.g. "uniform Switch loss → 1.0" in §1, "per-rank-mean" in §4, "8 yamls per depth" in §6, "Decide whether to flip `disable_liger: true → false`" in §7), the structured plan above wins. The DECs (DEC-1..DEC-16) record where the draft was ambiguous and which way the plan resolves each ambiguity.

# Load-balancing methods — TODO

when running gpu codes, also use modal computes here. see the skills here /Users/leon/Desktop/modal/moe/.claude/skills. use a maximum of 8 nodes of h200 and turn it off whenever possible.

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

### 2a. Verify the gather → normalize → scale recipe

The DeepSeek-V3 router contract has *two* halves; the bias-update bullets above
cover the selection half. The forward-pass weight half is the post-gather
normalization + post-norm scaling. Reference implementation is
`nmoe/nmoe/model.py:78-90` (sigmoid → bias-aware topk → unbiased gather →
sum-to-1 normalize → `routed_scaling_factor` rescale). Our equivalent is
`src/models/router.py:317-351`. The two must stay byte-equivalent on the
hot path; if either drifts, a downstream config that copies a `topk_scaling_factor`
from a DeepSeek paper will silently mean something different.

- [ ] Confirm the scoring path is **fp32 sigmoid on fp32 logits** under
  `torch.autocast(enabled=False)`, not bf16 (`router.py:305-318`,
  `nmoe/model.py:79-82`). bf16 sigmoid loses resolution near the saturation
  ends and changes which expert wins on near-ties.
- [ ] Confirm the gathered weights are **unbiased** (`scores.gather`, not
  `biased_scores.gather`). The bias is a selection-time signal only; letting
  it leak into the gradient-bearing weight changes the gradient direction.
- [ ] Confirm the **sum-to-1 normalization** runs whenever `norm_topk_prob=true`
  and `top_k > 1`, and is **skipped for `top_k == 1`** (where `w/w == 1.0` kills
  gradient flow through the router weight). nmoe's Router targets `topk > 1`
  only and so doesn't carry that guard — our version (`router.py:346`) must.
- [ ] Confirm the **post-norm rescale** by `topk_scaling_factor` is applied
  *after* normalization (`router.py:350-351`, `nmoe/model.py:88-89`). Order
  matters: scaling pre-normalize gets divided back out.
- [ ] Confirm `topk_scaling_factor` defaults: DeepSeek-V3 uses `2.5`, nmoe
  defaults to `1.0` (NVFP4 setting), our configs set `2.5`. Document the
  reasoning in the config comment so a future reader knows what the number
  means.
- [ ] Add a numerical parity test: same input + weights → our DeepSeekRouter
  and nmoe's Router produce identical `(weights, indices)` to bf16 round-off.
  Run it as part of CI so the recipe doesn't silently drift.
- [ ] When §10 lands (per-router-class refactor), make sure the new
  `attn_router` / `mlp_router` / `branch_router` paths all preserve this exact
  forward-pass recipe whenever `class: deepseek` is selected — including the
  `top_k == 1` normalization skip for per-head attention routers, which today
  are top-1.

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
- [ ] **precompute_kv** × {deepseek, switch, quantile}
  - `moe_everything`, `attn_expert_mode: per_head_precompute_kv`, shared expert banks across
    depth iters, **per-depth routers + per-depth prenorms** (the `_perlayer_prenorm` flavor:
    `per_layer_router=true`, `per_layer_attn_router=true`, `per_layer_norm=true`), vary balancing
- [ ] **precompute_kv_global_router** × {deepseek, switch, quantile}
  - same precompute_kv attention bank, but **one single router reused at every depth iteration**
    (plain `precompute_kv` flavor: `per_layer_router=false`, `per_layer_attn_router=false`,
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

--- Original Design Draft End ---
