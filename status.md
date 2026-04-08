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

## Bottom Line

The current state is:

- the reference speedrun works on this machine only if you back off from the current FA3 `HEAD`
- the reference reaches the target
- this repo does not
- I fixed the concrete correctness bugs I found, but the repo is still not close enough in either throughput or validation loss
- additional ablations did not uncover an easy optimizer or schedule fix

If the objective is truly to match the speedrun, the next step is not more blind training on the current stack. The next step is to either:

1. port the reference architecture/training recipe much more faithfully, or
2. stop comparing this HF/Qwen-style stack to the speedrun target as if they were equivalent
