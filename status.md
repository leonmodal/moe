# Status

Updated: 2026-04-08 01:53 UTC

## Torch-Native Rewrite Checkpoint

I started the in-repo replacement for the Accelerate trainer.

New files / entrypoints:

- `/tmp/moe/train_torch.py`
- `TRAIN_ENTRYPOINT=train_torch.py ./scripts/train.sh ...`

What is working now:

- raw `torch.distributed` launch, no Accelerate dependency
- `--dist-strategy none|ddp|fsdp`
- dense + GPT-2 dense + standard/global/moe-everything model builders
- parquet and GPT-2 token-bin dataset paths
- optimizer/scheduler reuse from existing utils, including Muon
- gradient accumulation, checkpoint save/load, auto-resume, eval, wandb
- mapped init path for `global_to_alternating_sanity`

Smoke tests completed on this machine:

- single-process GPT-2 bin run: `1` step completed
- 2-rank DDP GPT-2 bin run: `1` step completed
- 2-rank FSDP GPT-2 bin run: `1` step completed
- 2-rank DDP DeepSeek standard MoE parquet run: `1` train step + eval completed

Representative logs:

- `/tmp/moe/outputs/train_torch_smoke.log`
- `/tmp/moe/outputs/train_torch_smoke_ddp.log`
- `/tmp/moe/outputs/train_torch_smoke_fsdp3.log`
- `/tmp/moe/outputs/train_torch_moe_smoke.log`
- `/tmp/moe/outputs/train_torch_script_smoke.log`

Notes:

- FSDP checkpointing currently uses `FSDP.state_dict_type(...)`, which works but emits deprecation warnings in this torch version.
- This is a runtime rewrite checkpoint, not the final speedrun benchmark replacement yet. The next step is to move the actual benchmark recipe onto `train_torch.py` and compare against the modded-nanogpt reference again.

## Torch-Native Benchmark Update

I then ran the actual 8x B200 speedrun-style dense benchmark through `train_torch.py`.

Key runtime change:

- `train_torch.py` no longer uses `FSDP(..., use_orig_params=True)`.

Optimizers available in the Torch-native trainer:

- `optimizer: adamw`
- `optimizer: muon`
  - supports `muon_lr`
  - supports `muon_weight_decay`
  - supports `adam_lr`
  - keeps the Muon momentum warmup path

### Main 1150-step Torch-native run

Run:

- config: `/tmp/moe/configs/plan/speedrun_dense_fineweb_gpt2_bins.yaml`
- launcher: `train_torch.py --dist-strategy ddp`
- log: `/tmp/moe/outputs/plan/speedrun_dense_fineweb_gpt2_bins_torch.log`
- checkpoint dir: `/tmp/moe/outputs/plan/speedrun_dense_fineweb_gpt2_bins_torch`

Observed results:

- completed all `1150` steps
- final validation CE: `3.5600`
- average throughput from step `50+`: about `1.234M tok/s`
- median throughput from step `50+`: about `1.234M tok/s`

Validation curve:

- `50`: `6.2333`
- `100`: `5.4135`
- `150`: `4.9889`
- `200`: `4.7188`
- `250`: `4.4139`
- `400`: `4.0413`
- `700`: `3.7192`
- `950`: `3.5874`
- `1000`: `3.5735`
- `1050`: `3.5649`
- `1100`: `3.5607`
- `1150`: `3.5600`

Interpretation:

- rewriting the runtime in raw torch did **not** produce a major throughput improvement
- it **did** materially improve convergence versus the older trainer on the same recipe
- this is still not enough to hit the `3.28` target

For comparison:

- old trainer at roughly the same benchmark point: `eval@1000 = 3.7176`, throughput about `1.16M-1.19M tok/s`
- new Torch-native trainer: `eval@1000 = 3.5735`, throughput about `1.234M tok/s`

So the rewrite bought:

- roughly `3-6%` more throughput
- roughly `0.14` lower validation CE at step `1000`

### Failed extension attempt: resume to 1700 with non-zero floor

Run:

- config: `/tmp/moe/configs/plan/speedrun_dense_fineweb_gpt2_bins_torch_resume1700.yaml`
- log: `/tmp/moe/outputs/plan/speedrun_dense_fineweb_gpt2_bins_torch_resume1700.log`

Result:

- resumed from step `1150` with `max_steps: 1700` and `min_lr_ratio: 0.1`
- this reintroduced LR too aggressively
- `eval@1250 = 3.7106`, clearly worse than the `3.5600` floor from the completed 1150-step run
- I terminated this branch early

### Failed fresh attempt: 1695-step run from scratch with non-zero floor

Run:

- config: `/tmp/moe/configs/plan/speedrun_dense_fineweb_gpt2_bins_torch_fresh1695.yaml`
- log: `/tmp/moe/outputs/plan/speedrun_dense_fineweb_gpt2_bins_torch_fresh1695.log`

Observed partial curve before termination:

- `100`: `5.4275`
- `150`: `5.0017`
- `200`: `4.7166`
- `250`: `4.4086`
- `300`: `4.2545`
- `650`: `3.7917`
- `700`: `3.7583`
- `750`: `3.7330`
- `800`: `3.7037`
- `850`: `3.6773`

Interpretation:

- this was consistently behind the better zero-floor Torch-native run at the same steps
- I terminated it rather than spend the rest of the budget on a losing schedule

Current conclusion:

- the Torch-native rewrite is the right direction and is measurably better than the old trainer
- but the speedrun target is still not met
- the remaining gap to the modded-nanogpt reference is now mostly about model/training recipe quality, not just Accelerate overhead

## Side-by-Side Code Diagnosis vs modded-nanogpt

I compared the working pre-FA3 reference in `/tmp/modded-nanogpt-prefa3/train_gpt.py`
against this repo's Torch-native dense path.

### 1. We are not training the same model class

Reference model:

- `/tmp/modded-nanogpt-prefa3/train_gpt.py`
- custom GPT block with:
  - merged QKVO parameter tensor and zero-init output projection
  - ReLU^2 MLP with zero-init projection
  - token value embeddings
  - U-net / skip-connection scalar routing
  - one skipped attention layer
  - fixed attention scale `0.12`
  - gated attention output
  - logit soft-cap before CE

Relevant lines:

- attention block: `578-617`
- MLP: `619-637`
- custom GPT body: `659-764`

Current repo dense model:

- `train_torch.py` builds `transformers.Qwen3ForCausalLM`
- standard Qwen3 decoder:
  - separate Q/K/V/O projections
  - standard residual decoder blocks
  - SwiGLU MLP
  - no value-embedding path
  - no U-net skip structure
  - no logit soft-cap
  - standard HF init

Relevant lines:

- Qwen3 MLP: `/tmp/moe/.venv/lib64/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:70-83`
- Qwen3 attention: `/tmp/moe/.venv/lib64/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:221-294`
- Qwen3 decoder/model/LM head: `/tmp/moe/.venv/lib64/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:297-524`
- HF generic init: `/tmp/moe/.venv/lib64/python3.12/site-packages/transformers/modeling_utils.py:2251-2291`

Conclusion:

- the current "speedrun" benchmark is not a trainer-only comparison
- it is a different model architecture with different init behavior

### 2. The optimizer and LR schedule are materially different

Reference:

- Adam branch: `lr=0.008`, `betas=(0.8, 0.95)`, `eps=1e-10`, `weight_decay=0.0`
- Muon branch: `lr=0.05`, `weight_decay=0.0`
- per-parameter `lr_mul` on embeddings and scalar params
- constant LR for the first `55%` of training, then linear decay to `0.1x`
- no grad clipping

Relevant lines:

- optimizer implementations: `/tmp/modded-nanogpt-prefa3/train_gpt.py:386-537`
- optimizer setup + schedule: `/tmp/modded-nanogpt-prefa3/train_gpt.py:892-916`
- train step: `/tmp/modded-nanogpt-prefa3/train_gpt.py:998-1013`

Current repo benchmark config:

- Muon `0.02`, Muon WD `0.02`
- Adam WD `0.1`
- `beta1=0.9`
- cosine decay
- `warmup_steps=200`
- grad clipping at `1.0`
- no parameter-specific `lr_mul` / `wd_mul`

Relevant lines:

- benchmark config: `/tmp/moe/configs/plan/speedrun_dense_fineweb_gpt2_bins.yaml:20-35`
- training loop / clipping / scheduler step: `/tmp/moe/train_torch.py:816-980`
- current Muon implementation: `/tmp/moe/src/utils/muon.py:61-248`

Conclusion:

- even if the model were identical, the optimization recipe still is not
- the current benchmark is using a substantially different optimizer regime

### 3. Loss scaling is different

Reference:

- training CE uses `reduction="sum"` during training and `"mean"` during eval

Relevant line:

- `/tmp/modded-nanogpt-prefa3/train_gpt.py:763`

Current repo:

- Qwen3 uses HF `ForCausalLMLoss`
- HF defaults to mean reduction when `num_items_in_batch` is not provided

Relevant lines:

- Qwen3 loss call: `/tmp/moe/.venv/lib64/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:522-524`
- HF loss dispatch: `/tmp/moe/.venv/lib64/python3.12/site-packages/transformers/modeling_utils.py:4361-4373`
- HF causal LM loss implementation: `/tmp/moe/.venv/lib64/python3.12/site-packages/transformers/loss/loss_utils.py:25-61`

Conclusion:

- this changes the effective gradient scale substantially
- reference optimizer hyperparameters cannot be ported directly into the current HF dense path

### 4. The token pipeline is not equivalent

Reference:

- trains with `train_seq_len = 48 * 1024`
- each rank gets one long sequence
- training samples are aligned to BOS token `50256`
- document boundaries are enforced inside attention masks

Relevant lines:

- BOS-aligned batch start search: `/tmp/modded-nanogpt-prefa3/train_gpt.py:781-820`
- doc-boundary masking: `/tmp/modded-nanogpt-prefa3/train_gpt.py:690-728`
- training hyperparameters: `/tmp/modded-nanogpt-prefa3/train_gpt.py:825-835`

Current repo:

- benchmark config uses `seq_len: 1024`
- token-bin dataset walks fixed strided windows through the shard
- no BOS alignment
- no document-boundary masking for dense runs

Relevant lines:

- token-bin benchmark config: `/tmp/moe/configs/plan/speedrun_dense_fineweb_gpt2_bins.yaml:47-64`
- token-bin dataset iteration: `/tmp/moe/src/data/token_bin_dataset.py:67-208`

Conclusion:

- we are feeding a different context structure to the model
- this is both a quality difference and a major throughput difference versus the FlexAttention reference

### Bottom line

The current repo does not miss `3.28` because of one hidden bug in DDP or FSDP.
It misses because the current "speedrun dense" path is not actually the modded-nanogpt recipe:

- different model
- different init
- different optimizer math
- different LR schedule
- different loss scaling
- different sequence packing and masking

If the goal is truly to match the reference, the next step is not more tuning on top of the HF Qwen dense path.
The next step is to build an in-repo reference-style dense path that copies the modded-nanogpt model/data/optimizer semantics much more directly.

## Goal

Match the FineWeb GPT-2 speedrun target on the local 8x B200 box:

- target loss: `<= 3.28` validation cross-entropy
- dataset: GPT-2 tokenized `.bin` shards
- compare directly against modded-nanogpt on the same machine

## What I Actually Ran

This is not based on 3-step smoke tests. I ran long 8-GPU jobs on the local machine:

1. `modded-nanogpt` reference on the same FineWeb GPT-2 bins until completion
2. this repo on the same bins for `1150` steps
3. this repo resumed/extended to `1700` steps to check whether it catches up

## Reference Benchmark: modded-nanogpt

Repo:

- cloned to `/tmp/modded-nanogpt`

Important compatibility note:

- current `HEAD` does **not** run on B200 here because its FlashAttention-3 path fails with:
  - `CUDA error: no kernel image is available for execution on the device`
- I therefore used a pre-FA3 worktree at commit `12dfed77678c6ef53ec0540def0156ca852aeb5b`
  in `/tmp/modded-nanogpt-prefa3`

Run:

- command: `DATA_PATH=/tmp/modded-nanogpt ./run.sh`
- log: `/tmp/modded-nanogpt-prefa3/reference_prefa3_run.log`

Observed results:

- final step: `1695`
- final validation loss: `3.2762`
- step time at convergence: about `65.48 ms`
- aggregate throughput: about `6.01M tok/s`
  - computed from `393,216 tokens/step / 0.06548 s`
- estimated MFU: about `55.2%`
  - estimate only, not directly logged
  - assumes ~`275.7M` trainable params for this custom model and ~`2.25 PFLOP/s` BF16 peak per B200

Key validation points from the log:

- step `125`: `4.6065`
- step `250`: `4.0721`
- step `1250`: `3.3930`
- step `1500`: `3.3180`
- step `1625`: `3.2888`
- step `1695`: `3.2762`

Conclusion:

- the reference setup does reach the target on this hardware

## This Repo: Same Dataset, Same 8x B200 Machine

Dataset wiring:

- symlinked `./data/fineweb10B_gpt2` to the same bins used by modded-nanogpt

Benchmark config used:

- `/tmp/moe/configs/plan/speedrun_dense_fineweb_gpt2_bins_benchmark.yaml`

Model in this repo:

- dense Qwen-style decoder
- `0.152B` total params
- `12` layers, `768` hidden, `12` heads, `seq_len=1024`
- Muon optimizer

### 1150-step run

Run:

- log: `/tmp/moe/outputs/run_speedrun_dense_bins_benchmark.log`

Observed results:

- aggregate throughput: about `1.16M` to `1.19M tok/s`
- estimated MFU: about `5.9%`
  - estimate only, not directly logged
  - assumes `152M` params and ~`2.25 PFLOP/s` BF16 peak per B200

Validation:

- step `200`: `4.6106`
- step `400`: `4.0024`
- step `600`: `3.7931`
- step `800`: `3.7224`
- step `1000`: `3.7176`

### 1700-step extension

Resume config:

- `/tmp/moe/configs/plan/speedrun_dense_fineweb_gpt2_bins_benchmark_resume1700.yaml`

Run:

- resumed from checkpoint `1000`
- log: `/tmp/moe/outputs/run_speedrun_dense_bins_benchmark_resume1700.log`

Observed results:

- final step: `1700`
- final validation loss: `3.8780`
- throughput stayed around `1.14M` to `1.18M tok/s`

Important note:

- this was a resume-extension from the 1150-step schedule, not a fresh 1700-step run from scratch
- even so, it is enough to show this setup is not on a trajectory to match the reference

Conclusion:

- this repo is still about `5x` slower than the reference on the same machine
- it also fails to approach the reference loss target

### Additional tries after that

I ran three more experiments after the first comparison to make sure the gap was real and not just a bad schedule choice.

#### A. GPT-2-shaped "referenceish" short run

Config:

- `/tmp/moe/configs/plan/gpt2_referenceish_fineweb_gpt2_bins_short200.yaml`

Changes:

- switched to `gpt2_dense`
- matched the reference head geometry more closely: `6` heads, `128` head_dim
- used a more reference-like optimizer: Muon `0.05`, Adam `0.008`, no weight decay, constant LR

Result:

- throughput improved to about `1.41M tok/s`
- validation got dramatically worse: `eval@200 = 7.2911`

Interpretation:

- this geometry can run faster in this stack
- but it is not learning the task well enough to matter

#### B. Qwen-shaped "referenceish" short run

Config:

- `/tmp/moe/configs/plan/qwen_referenceish_fineweb_gpt2_bins_short200.yaml`

Changes:

- kept the Qwen-style dense model
- used the same more aggressive Muon/Adam settings as above

Result:

- throughput stayed around `1.18M tok/s`
- validation also got worse: `eval@200 = 5.2031`

Interpretation:

- pulling the optimizer much closer to the reference recipe did not help in this HF/Qwen-style stack

#### C. Fresh 1695-step run from scratch

Config:

- `/tmp/moe/configs/plan/speedrun_dense_fineweb_gpt2_bins_fresh1695.yaml`

Why this run mattered:

- this was the best-faith attempt to rescue the earlier benchmark
- same healthier Qwen/Muon recipe as before
- no resume
- longer horizon, `1695` steps, closer to the reference endpoint
- cosine floor set to `0.1` instead of decaying all the way to zero

Observed validation curve:

- step `125`: `5.1359`
- step `250`: `4.3459`
- step `375`: `4.0455`
- step `500`: `3.8865`
- step `625`: `3.8032`
- step `750`: `3.7617`
- step `875`: `3.7477`
- step `1000`: `3.7457`
- step `1125`: `3.7572`
- step `1250`: `3.7807`
- step `1375`: `3.8113`
- step `1500`: `3.8425`
- step `1625`: `3.8834`

Result:

- best observed validation was `3.7457` at step `1000`
- after that it got worse, so I terminated the run instead of spending more time on a clearly losing curve

Interpretation:

- the earlier conclusion holds even after a fresh full-horizon run
- this recipe does not reach `3.28`

## Bugs Found And Fixed

### 1. Qwen attention backend was falling back to eager

File:

- `/tmp/moe/train.py`

Issue:

- Qwen-family configs were being built without an explicit attention backend
- that leaves them on eager attention instead of `sdpa`

Fix:

- defaulted Qwen-family construction to `attn_implementation=sdpa`

### 2. FineWebEdu plan configs pointed at a missing path

Files:

- `/tmp/moe/configs/plan/qwen3_0_6b_dense_speedrun_style_finewebedu.yaml`
- `/tmp/moe/configs/plan/standard_moe_speedrun_style_finewebedu.yaml`

Issue:

- both referenced `./data/parquet_finewebedu_speedrun`, which does not exist locally

Fix:

- switched them to `./data/parquet`

### 3. Token-bin eval inherited `repeat: true`

File:

- `/tmp/moe/train.py`

Issue:

- eval config was merged from train config and accidentally inherited dataset repetition

Fix:

- force token-bin eval datasets to default to `repeat: false` unless explicitly requested

### 4. Token-bin repeat logic used recursive iteration

File:

- `/tmp/moe/src/data/token_bin_dataset.py`

Issue:

- validation on token-bin data eventually hit a `RecursionError` because repeat mode used recursive `yield from self.__iter__()`

Fix:

- rewrote repeat handling as an iterative loop

## What Is Still Wrong

The current repo is now more correct, but it is still not competitive with the speedrun baseline. The remaining gap looks structural, not just a single broken line.

Main reasons:

1. The model/training stack is materially different from the reference.
   - modded-nanogpt is a custom speedrun model with FlexAttention, custom fused kernels, FP8 paths, ReLU squared, custom residual structure, extra value embeddings, soft-capped logits, and a highly tuned training script
   - this repo is a HuggingFace/Qwen-style dense model with a much heavier software stack

2. Throughput is the biggest immediate gap.
   - reference: about `6.01M tok/s`
   - this repo: about `1.18M tok/s`
   - that alone destroys wall-clock competitiveness

3. Loss trajectory still misses the target badly.
   - reference reaches `3.2762`
   - this repo bottoms out around `3.7176` in the long benchmark and the 1700-step extension ends worse at `3.8780`

4. Extending the run does not rescue it.
   - the training loss keeps falling
   - validation does not follow
   - that points to the current recipe/model being the wrong one for this target, not just “needs a few more steps”

5. The faster GPT-2-shaped variant is also not the answer.
   - it gains some throughput
   - but its validation loss is catastrophically worse than the Qwen-based baseline

## SpeedrunGPT Port — Target Reached

Updated: 2026-04-08

I ported the full modded-nanogpt reference model and training recipe into this repo as a new model type `speedrun_gpt`.

### What was ported

New files:

- `src/models/speedrun_gpt.py` — full GPT model: merged QKVO, ReLU², value embeddings, U-net skips, gated attention, logit softcapping, FlexAttention with doc masking + sliding window, FP8 lm_head
- `src/utils/triton_newton_schulz.py` — Triton kernels for Newton-Schulz orthogonalization
- `src/utils/dist_optimizers.py` — DistAdam + DistMuon (distributed optimizers with built-in gradient sync, no DDP wrapper)
- `configs/plan/speedrun_gpt_fineweb_gpt2_bins.yaml` — benchmark config

Modified files:

- `train_torch.py` — added `speedrun_gpt` model type in `build_model()`, added `run_speedrun_training()` function with BOS-aligned data generator, sliding window schedule, kernel warmup, and reference validation loop
- `src/utils/training.py` — added `stable_decay` LR schedule (constant then linear cooldown)

### Benchmark result: 8x B200

Run:

- command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run torchrun --nproc_per_node=8 train_torch.py --config configs/plan/speedrun_gpt_fineweb_gpt2_bins.yaml --dist-strategy none`

Validation curve:

- step `0`: `10.8258`
- step `125`: `4.6008`
- step `250`: `4.0727`
- step `500`: `3.7298`
- step `750`: `3.5825`
- step `1000`: `3.4885`
- step `1250`: `3.3913`
- step `1500`: `3.3165`
- step `1625`: `3.2877`
- step `1695`: **`3.2748`**

Performance:

- step avg: `65.5 ms`
- throughput: `~6.0M tok/s` (matching reference)
- peak memory: `35,210 MiB` allocated, `50,900 MiB` reserved
- total training time: `~111 seconds`

### Comparison with reference

| Step | Reference | This Repo | Delta |
|------|-----------|-----------|-------|
| 125  | 4.6065    | 4.6008    | -0.006 |
| 250  | 4.0721    | 4.0727    | +0.001 |
| 1250 | 3.3930    | 3.3913    | -0.002 |
| 1500 | 3.3180    | 3.3165    | -0.002 |
| 1625 | 3.2888    | 3.2877    | -0.001 |
| 1695 | 3.2762    | 3.2748    | -0.001 |

The loss curve matches the reference to within `0.006` at every checkpoint. Final loss `3.2748` is slightly better than the reference `3.2762`.

### Bottom line

- **Target met.** `3.2748 <= 3.28`.
- Throughput matches reference (`~6.0M tok/s`).
- The previous HF Qwen dense path (`3.56` final loss) was an apples-to-oranges comparison — different model, optimizer, loss scaling, schedule, and data pipeline.
- The new `speedrun_gpt` path faithfully replicates the reference recipe and achieves equivalent results.

## SpeedrunMoEGPT — Per-Head Routed Attention + Shared MLP

Updated: 2026-04-08

Built two MoE variants of the speedrun model that replace per-layer fixed attention/MLP weights with shared expert banks + per-head-slot routing.

### Architecture

Both variants keep the speedrun recipe (ReLU² MLP, value embeddings, U-net skips, logit softcapping, gated attention, sum loss, DistAdam+DistMuon).

**Routing design**: Each head slot has its **own dedicated router** doing **top-1** from the expert pool. NOT one router picking top-K — K separate routers each picking top-1.

**Expert counts** (parameter-matched to baseline):
- Attention experts: `66` (matching 11 attn layers × 6 heads = 66 unique head weight sets)
- MLP experts: `12` (matching 12 MLP layers)
- Both banks shared across all depths

**`speedrun_moe_fully_independent`**:
- 4H routers per depth: 6 Q-routers + 6 K-routers + 6 V-routers + 6 O-routers = 24 per depth
- Q/K/V route independently on input hidden state
- O routes independently on attention output
- ONE attention call per depth (standard SDPA)
- Total: 276.7M params

**`speedrun_moe_precompute_kv`**:
- H routers per depth: 6 QKVO-routers = 6 per depth (bundled — one decision picks Q+K+V+O)
- Per-expert KV tables ensure Q-K subspace alignment
- E_active attention calls per depth
- Total: 276.1M params

### New files

- `src/models/speedrun_moe_gpt.py` — SpeedrunMoEGPT model with AttentionExpertBank, MLPExpertBank, per-head-slot routing
- `src/utils/routing_loss.py` — Switch-style aux load balancing loss
- `configs/plan/speedrun_moe_fully_independent.yaml`
- `configs/plan/speedrun_moe_precompute_kv.yaml`

### Features

- **Triton grouped GEMM**: Uses existing `src/models/triton_grouped_gemm.py` for efficient routed projections
- **Switch aux loss**: `router_aux_loss_coef` in config, applied to all routers (attention + MLP)
- **Routing stats**: Per-router load balance ratio, expert utilization, routing entropy logged to wandb
- **Checkpoint saving**: Model + optimizer state saved at `save_every` steps
- **Modal multinode**: `modal_train.py` updated to dispatch speedrun model types to `train_torch.py`

### Verification

- Single-GPU forward/backward verified for both modes
- Aux loss working (init ~1.0, expected for uniform routing)
- Routing stats: balance 0.7+, utilization 1.0, entropy 0.99+ at init
- 8-GPU distributed run requires Triton kernel warmup (autotuning on first launch)

### Known issues

- First 8-GPU launch is slow (Triton autotuning for grouped GEMM + Newton-Schulz across many parameter shapes)
- `precompute_kv` is memory-heavy due to per-expert attention loops — needs smaller batch than `fully_independent`
- Batch size limited by distributed optimizer NCCL buffer overhead (~46GB non-PyTorch per GPU)

## Modal Training Runs (2026-04-14)

### Active Runs (v2 — zero-sum bias, per-projection rates)

All runs: single-node 8x H200, grad_ckpt (use_reentrant=True), W&B project: `speedrun-moe`

| Experiment | Branch | Attn Routing | BS | Grad Accum | W&B Name |
|---|---|---|---|---|---|
| fi_ds_branch_token_attn_v2 | seq DeepSeek | token Q/K/V/O | 64 | 1 | `speedrun_moe_fi_ds_branch_token_attn_v2` |
| pkv_ds_branch_v2 | seq DeepSeek | bundled QKVO precompute KV | 32 | 2 | `speedrun_moe_pkv_ds_branch_v2` |

### Stopped (v1 — old bias with drift, checkpoints preserved on volume)

| Experiment | W&B Name |
|---|---|
| ds_branch_token_attn | `speedrun_moe_fi_ds_branch_token_attn` |
| ds_branch_seq_qkvo | `speedrun_moe_fi_ds_branch_seq_qkvo` |
| ds_branch_seq_qk | `speedrun_moe_fi_ds_branch_seq_qk` |
| pkv_ds_branch | `speedrun_moe_pkv_ds_branch` |

### Completed

| Experiment | Val Loss | W&B Name |
|---|---|---|
| baseline speedrun_gpt (dense) | 3.3799 (10K steps, 8x H200) | `speedrun_gpt_fineweb_gpt2_bins` |

### Previous Runs (checkpoints deleted, W&B logs preserved)

- `speedrun_moe_fully_independent` — token branch, token attn, per-router LB
- `speedrun_moe_precompute_kv` — token branch, bundled QKVO routing
- `speedrun_moe_fi_branch_sampling` — token branch (sampling), token attn
- `speedrun_moe_fi_global_lb` — token branch, token attn, global LB (single bias — bug)
- `speedrun_moe_fi_seq_branch` — seq branch (softmax), token attn, global LB
- `speedrun_moe_fi_seq_branch_seq_qkvo` — seq branch (softmax), seq attn, global LB
- `speedrun_moe_fi_seq_branch_seq_qk` — seq branch (softmax), seq QK, global LB
- `speedrun_moe_fi_alternating_seq_qk` — alternating attn/mlp, seq QK
- Earlier ds_branch runs with wrong single `_global_attn_bias`

### Architecture Details (Current Runs)

- 12 layers → 24 depth steps (each depth: attn OR mlp via branch router)
- 66 attention experts, 12 MLP experts (shared projection banks across depths)
- Per-depth routers (separate weights per depth), per-depth branch router
- DeepSeek-style branch router: sigmoid + per-depth bias, seq-level (mean pool)
- DeepSeek-style expert routers: sigmoid + bias, top-1, NO normalization (preserves gradient flow to router)
- Global load balancing: per-projection-type bias (Q/K/V/O/MLP each separate), all-reduced across ranks
- Branch bias: per-depth (not global), updated from per-depth counts (also all-reduced)
- Constant bias update rate: 0.001 (no warmup schedule)
- Gradient checkpointing: use_reentrant=True (required for shared param banks)
- Auto-resume from checkpoints on Modal volume
- Kaiming uniform router weight init

### Key Fixes Applied

1. Per-projection global bias (Q/K/V/O/MLP separate, not one shared attn bias)
2. All-reduce token counts across ranks before bias update (keeps biases synchronized)
3. Skip top-1 normalization to preserve router gradient flow (norm only for top-K, K>1)
4. None-grad handling in DistMuon/DistAdam for branch routing edge cases
5. Gradient checkpointing stats dedup (prevent double-counting during recompute)
6. Per-depth routers (not shared across depths)
7. Router kaiming init (not zero init)
8. Seq-level branch routing skips sparse query path (needs full B*T for correct mean pool)
9. Zero-sum bias update: `s - s.mean()` prevents bias drift (from nmoe reference)
10. Per-projection bias rates: Q/K=0.005, V/O/MLP=0.003, branch=0.001
11. Bias clamp ±16 safety bound
12. Branch ratio plots use hard routing decisions (sums to 1.0), not soft sigmoid scores
13. Aux loss correctly accumulated across gradient accumulation steps
14. Bias plots: global expert biases (Q/K/V/O/MLP) + branch bias by depth
15. 4-row branch_routing.png: token fraction, attn/mlp ratio, sigmoid weights, branch bias
