# External MoE Optimization Research

Research findings from Megatron-LM, modal-nmoe, and nmoe codebases, focused on training stability and throughput improvements applicable to our MoE training stack.

## Megatron-LM Findings

### Grouped GEMM / Expert Compute
- Megatron uses CUTLASS-backed `grouped_gemm` or TransformerEngine GroupedLinear for expert dispatch
- Key transferable idea: fuse dispatch/permutation/kernel around the GEMM, not just swap GEMM backends
- Router fusion (`--moe-router-fusion`) and permute/unpermute fusion (`--moe-permute-fusion`) reduce overhead
- **Applicability**: High — we already have Triton grouped GEMM; adding fusion around it is the main win
- **Complexity**: Medium

### Load Balancing
- Supports `aux_loss`, `seq_aux_loss`, `global_aux_loss`, `sinkhorn`, and aux-loss-free expert bias
- Group-limited routing: device-limited (groups = EP size) or node-limited (groups = nodes in EP group)
- Expert bias default rate: `1e-3`, maintained in float32 even in bf16 training
- Early-training stability: reduce EP or increase expert TP for ~200 steps to avoid router collapse
- **Applicability**: Very high — seq_aux_loss and group-limited routing are immediate wins
- **Complexity**: Low for aux-loss variants; Medium for group-limited routing

### Training Stability
- Router dtype in fp32/fp64 recommended when expert count is large
- Z-loss with recommended starting value `1e-3`
- Aux-loss coefficient recommended at `1e-2`
- Input jitter (`moe_input_jitter_eps`) for additional stability
- Float32 maintenance of expert bias — critical in mixed-precision training
- **Applicability**: Very high — highest ROI items for our scale
- **Complexity**: Low

### Communication Overlap
- AllToAll path: preprocess → permute → A2A over EP → AG over TP → expert compute → reverse
- DeepEP backend: fused dispatch/combine kernels merging permutation and all-to-all
- `--overlap-moe-expert-parallel-comm` overlaps EP A2A with compute
- **Applicability**: Medium for single-node; High for multi-node EP
- **Complexity**: Medium for overlap scheduling; High for fused comm kernels

## modal-nmoe / nmoe Findings

### RDEP (Route-Dependent Expert Parallelism)
- Transport unit is a "route row": one (token, top-k slot) with activation and metadata
- Replaces NCCL AllToAll with direct P2P writes via CUDA IPC handles
- 2-phase dispatch: count per-destination rows, exchange counts, compute offsets, write deterministically
- Removes remote atomics from hot forward dispatch path
- Performance: 2x throughput over TP+NCCL baseline on 8xB200 (24,871 vs 12,447 tok/s)
- **Applicability**: High on single-node NVLink; Less applicable for standard DDP/FSDP
- **Complexity**: High

### Expert Compute Organization
- BF16 uses `torch._grouped_mm`; blockscaled uses CUTLASS/CuTe grouped kernel
- Fused W1/W3 interleaving: SwiGLU stage 1 is one grouped GEMM instead of two
- `ExpertAdamW`: updates expert weights and emits FP8/NVFP4 caches in same CUDA step
- Dense params on ZeRO-2; expert params sharded by ownership and stepped locally
- **Applicability**: Very high for throughput; fused W1/W3 is a clean win
- **Complexity**: Medium-high

### Training Stability
- Loss-free balancing: global expert loads all-reduced, mean-centered sign updates, clipped to ±16
- Small aux loss (`~1e-4`) as backup improves both loss and throughput
- Expert LR should be ~equal to dense LR (not 15x multiplier)
- Dropless by contract: capacity sized to worst-case, no silent truncation
- **Applicability**: Very high — validates our existing approach
- **Complexity**: Low

## Priority Recommendations

### Highest ROI (implement now)
1. **Router math in float32**: Keep router weights, expert bias, and gating in fp32
2. **Z-loss**: Add `z_loss_coef` parameter (start at `1e-3`)
3. **Global bias in float32**: Already done via buffer; verify in mixed-precision
4. **Small aux backup**: Combine loss-free bias with `aux_loss_coef ~1e-4`
5. **Group-limited routing**: Already implemented; ensure it's used in all DeepSeek configs

### Medium Priority (benchmark first)
1. **Fused W1/W3 interleaving**: Merge gate_proj and up_proj into single grouped GEMM
2. **Permute/unpermute fusion**: Fuse around Triton grouped GEMM
3. **Seq-level aux loss**: Add as loss option alongside batch aux loss
4. **Early-step capacity warmup**: Reduce router exploration for first ~200 steps

### Lower Priority (larger investment)
1. **RDEP-style IPC dispatch**: High value on single-node but requires CUDA kernel work
2. **Expert optimizer fusion**: Combine weight update + quantized cache emission
3. **MoE Parallel Folding**: Separate expert sharding from dense model sharding
4. **DeepEP-class fused comm kernels**: Only if EP traffic is a bottleneck

## Integration Status

Per AC-10.1, this table distinguishes **observed externally (not integrated here)** from **implemented here with benchmark evidence**.

### Observed externally (not integrated)

| Technique | Source | Why not integrated yet |
|-----------|--------|------------------------|
| Z-loss | Megatron-LM | Benchmark-first follow-up (coefficient tuning) |
| Fused W1/W3 GEMM | Megatron-LM + custom kernel work | Benchmark-first follow-up |
| RDEP dispatch | nmoe | Benchmark-first follow-up; requires CUDA kernel work |
| Permute/unpermute fusion around Triton GEMM | Megatron-LM | Benchmark-first follow-up |
| DeepEP-class fused comm kernels | DeepEP | Only relevant if EP traffic is a bottleneck |
| Expert optimizer fusion | modal-nmoe | Larger investment |
| MoE Parallel Folding | Megatron-LM | Larger investment |

### Implemented here

| Technique | Source of truth in repo | Notes |
|-----------|-------------------------|-------|
| FP32 router math | `src/models/routing/fp32_ops.py`, `src/models/router.py` | Pre-existing |
| Expert bias (DeepSeek) | `src/models/router.py`, `src/models/routing/bias.py`, `src/training/routing.py::update_expert_biases` | Pre-existing |
| Group-limited routing | `src/models/router.py::group_limited_topk` | Pre-existing |
| Seq aux loss (partial) | `src/models/routing/load_balancing.py::seq_load_balancing_loss_func` | Pre-existing; coefficient tuning still benchmark-first |
| **Early-step router-exploration warmup** | `src/training/routing.py::exploration_rate_schedule` + `apply_router_exploration_rate`, `src/training/config.py::router_exploration_warmup_start` / `router_exploration_warmup_steps`, trainer loop in `src/training/trainer.py` | **Round 4 integration (AC-10). Benchmark evidence below.** |

### Implemented here with benchmark evidence

**Early-step router-exploration warmup.** The trainer linearly ramps the effective router-exploration rate from `router_exploration_warmup_start` to the model-configured `router_exploration_rate` over `router_exploration_warmup_steps` steps, then holds at the target. Default (`warmup_steps=0`) is a no-op and preserves existing configs unchanged. Reproduce with:

```bash
uv run python scripts/benchmark_router_exploration_warmup.py \
    --steps 50 --warmup-start 0.5 --warmup-steps 15 --target-rate 0.02
```

Representative result on 1x H200 (50 steps, batch 2, seq_len 32, target_rate 0.02, warmup_start 0.5 for 15 steps):

```
standard_moe
  no_warmup  (rate==target)  final_loss=1.5637
  warmup     (0.5 → 0.02 / 15 steps)  final_loss=1.5816   delta +1.15%

moe_everything
  no_warmup  (rate==target)  final_loss=1.6278
  warmup     (0.5 → 0.02 / 15 steps)  final_loss=1.6058   delta -1.35%
```

Both deltas are within the noise band of a 50-step micro-benchmark and are well inside the AC-11 loss-sanity envelope (no loss stuck at ~4 or above). The feature is therefore kept as an opt-in config knob (default disabled via `router_exploration_warmup_steps=0`) rather than enabled by default — it does not regress loss sanity and provides operators with a stability lever for the first few hundred steps of a real training run when desired.
