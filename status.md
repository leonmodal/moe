# Status

Updated: 2026-04-08 00:03 UTC

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
