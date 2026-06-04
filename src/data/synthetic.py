"""Synthetic attention-pattern datasets from Zhao et al. NeurIPS 2026.

Two tasks, both designed to study how transformers learn sparse attention
patterns against a known ground truth:

  * Linear map: sample a sparse binary matrix `A in {0,1}^{S x S}` once per
    dataset instance. Each example is `[x_0, x_1]` flattened, where
    `x_0 in {0,1}^S` is uniformly random and `x_1 = A x_0 mod 2`. The model
    must learn that token `S + i` depends on the `s` positions of `x_0`
    indexed by row `A[i]` — that mask is the ground-truth attention pattern.

  * Cellular automata: sample `N` lookup tables `R: {0,...,C-1}^3 -> {0,...,C-1}`
    once per dataset instance. Each example picks one rule, samples
    `x_0 in {0,...,C-1}^S`, and applies the rule `T-1` times to produce
    `[x_0, x_1, ..., x_{T-1}]`. The ground-truth attention pattern is the
    local 3-window mask shifted by S between consecutive states.

Both datasets are infinite streams of fresh examples. We don't shard across
ranks because every rank generates independent examples (no overlap risk).
The `get_state` / `set_state` interface is implemented for trainer checkpoint
compatibility, but state is just the rng counter — examples are i.i.d.

The ground-truth matrix `A` (linear map) or local-window mask (CA) is stored
on the dataset object as `ground_truth_A` so the eval loop can pick it up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass
class SyntheticConfig:
    task: str  # 'linear_map' or 'cellular_automata'
    state_size: int  # S
    trajectory_length: int  # T
    num_colors: int  # C (vocab size)
    sparsity: int  # s (linear_map only)
    num_rules: int  # N (cellular_automata only)
    recursive_depth: int  # k (cellular_automata only)
    task_seed: int  # seed for ground-truth A / rules (must match across train+eval)
    stream_seed: int  # seed for per-example sampling (distinct for train vs eval)


class _SyntheticBase(IterableDataset):
    """Common state / iteration scaffolding for synthetic datasets."""

    def __init__(self, config: SyntheticConfig, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        # Per-rank rng so each rank generates independent examples. The stream
        # seed mixes in the rank so two ranks with the same seed config don't
        # generate identical examples. `task_seed` is held separately and used
        # by subclasses for sampling the shared ground-truth A / rules.
        self._rank_seed = config.stream_seed * 1_000_003 + rank
        self._example_idx = 0
        self._rng = np.random.default_rng(self._rank_seed)
        # `ground_truth_A` is sampled lazily by subclasses in `__init__`.
        self.ground_truth_A: np.ndarray | None = None
        self.seq_len: int = 0

    def get_state(self) -> dict:
        return {"example_idx": int(self._example_idx)}

    def set_state(self, state: dict) -> None:
        self._example_idx = int(state.get("example_idx", 0))
        # Re-seed and fast-forward by drawing the same number of bytes as the
        # generation path would have consumed. Since examples are i.i.d., we
        # only need the rng to be deterministic from this point forward; we
        # re-seed with (rank_seed + example_idx) so resume produces a fresh,
        # reproducible stream from the resume point.
        self._rng = np.random.default_rng(self._rank_seed + self._example_idx)

    def save_state(self, path) -> None:
        import json

        with open(path, "w") as f:
            json.dump(self.get_state(), f)

    def load_state(self, path) -> None:
        import json
        from pathlib import Path

        if Path(path).exists():
            with open(path) as f:
                self.set_state(json.load(f))


class LinearMapDataset(_SyntheticBase):
    """Sparse linear map task: predict A @ x mod 2 from concatenated state.

    Per the paper (S=16, s=3, T=2, C=2): seq_len = S * T = 32.
    Ground-truth attention pattern: row i of A specifies which input positions
    must be attended to in order to predict output position i. We store the
    full matrix on the dataset; eval code derives per-position attention
    targets from it.
    """

    def __init__(self, config: SyntheticConfig, rank: int = 0, world_size: int = 1):
        super().__init__(config, rank=rank, world_size=world_size)
        if config.num_colors != 2:
            raise ValueError(
                f"linear_map task requires num_colors=2, got {config.num_colors}"
            )
        if config.trajectory_length != 2:
            raise ValueError(
                f"linear_map task requires trajectory_length=2, got {config.trajectory_length}"
            )
        S = config.state_size
        s = config.sparsity
        if not 1 <= s <= S:
            raise ValueError(f"sparsity must be in [1, S], got s={s} S={S}")
        # Sample A once per dataset instance with the rank-shared task seed so
        # every rank — and train + eval — sees the SAME A (one pattern to
        # learn per experiment).
        shared_rng = np.random.default_rng(config.task_seed)
        A = np.zeros((S, S), dtype=np.int64)
        for i in range(S):
            nz = shared_rng.choice(S, size=s, replace=False)
            A[i, nz] = 1
        self.ground_truth_A = A
        self.seq_len = S * config.trajectory_length

    def _sample_example(self) -> np.ndarray:
        S = self.config.state_size
        x0 = self._rng.integers(0, 2, size=S, dtype=np.int64)
        x1 = (self.ground_truth_A @ x0) % 2
        return np.concatenate([x0, x1], axis=0)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        S = self.config.state_size
        seq_len = self.seq_len
        while True:
            seq = self._sample_example()
            self._example_idx += 1
            # Standard next-token labels; mask the last position with -100 so
            # the CE loss ignores it (there is no next token to predict).
            input_ids = torch.from_numpy(seq.astype(np.int64))
            labels = torch.full((seq_len,), -100, dtype=torch.long)
            labels[: seq_len - 1] = input_ids[1:]
            # Also mask the first S-1 positions: their targets are the
            # uniform-random tail of x_0 which is genuinely unpredictable, so
            # including them in the loss just adds a constant noise floor of
            # (S-1)/(ST) ln C (paper notation). Masking them sharpens the
            # signal on the actually-learnable predictions x_0[-1] -> x_1[0]
            # and x_1[i-1] -> x_1[i].
            labels[: S - 1] = -100
            yield {"input_ids": input_ids, "labels": labels}


class CellularAutomataDataset(_SyntheticBase):
    """Cellular automata task: predict next state from local 3-window rule.

    Per the paper (S=16, T=16, C=4, N=256, k=1): seq_len = S * T = 256.
    The ground-truth attention pattern is a local 3-window on the previous
    state's positions, shifted by S between states. We store the local
    window offsets as a 3-element array on the dataset; eval code expands
    it to per-position attention targets at runtime.
    """

    def __init__(self, config: SyntheticConfig, rank: int = 0, world_size: int = 1):
        super().__init__(config, rank=rank, world_size=world_size)
        C = config.num_colors
        if C < 2:
            raise ValueError(f"num_colors must be >= 2, got {C}")
        S = config.state_size
        N = config.num_rules
        k = config.recursive_depth
        # Sample N lookup tables, each composed k times. Each lookup table is
        # a length-C^3 array indexed by (left, center, right) -> next color.
        # Composition: rule_R(x)_i = R(x_{i-1}, x_i, x_{i+1}); apply k times to
        # define a single dataset rule. We pre-compute each rule's flattened
        # lookup table (length C^3) for fast per-example sampling. Uses the
        # rank-shared `task_seed` so train and eval see identical rules.
        shared_rng = np.random.default_rng(config.task_seed)
        rules: list[np.ndarray] = []
        for _ in range(N):
            # Sample k independent base rules.
            base_rules = [
                shared_rng.integers(0, C, size=C ** 3, dtype=np.int64) for _ in range(k)
            ]
            if k == 1:
                composed = base_rules[0]
            else:
                # Compose k rules: for every (l, c, r), apply base rules in
                # sequence. We materialize the composed rule's lookup by
                # iterating over all C^3 windows and walking k steps.
                composed = np.zeros(C ** 3, dtype=np.int64)
                for idx in range(C ** 3):
                    left = (idx // (C * C)) % C
                    center = (idx // C) % C
                    right = idx % C
                    window = (left, center, right)
                    for r in base_rules:
                        out = r[window[0] * C * C + window[1] * C + window[2]]
                        window = (window[1], out, out)  # propagate locally
                    composed[idx] = window[1]
            rules.append(composed)
        self._rules = np.stack(rules, axis=0)  # [N, C^3]
        # Local window offsets relative to the current position in the
        # PREVIOUS state. Eval code interprets these as ground-truth attention
        # targets shifted by S between consecutive states.
        self.ground_truth_A = np.array([-1, 0, 1], dtype=np.int64)
        self.seq_len = S * config.trajectory_length

    def _apply_rule(self, rule: np.ndarray, x: np.ndarray) -> np.ndarray:
        C = self.config.num_colors
        # Circular boundary: left neighbor of position 0 wraps to position S-1.
        left = np.roll(x, 1)
        right = np.roll(x, -1)
        idx = left * (C * C) + x * C + right
        return rule[idx]

    def _sample_example(self) -> np.ndarray:
        S = self.config.state_size
        T = self.config.trajectory_length
        C = self.config.num_colors
        rule_idx = self._rng.integers(0, self.config.num_rules)
        rule = self._rules[rule_idx]
        x = self._rng.integers(0, C, size=S, dtype=np.int64)
        states = [x]
        for _ in range(T - 1):
            x = self._apply_rule(rule, x)
            states.append(x)
        return np.concatenate(states, axis=0)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        S = self.config.state_size
        seq_len = self.seq_len
        while True:
            seq = self._sample_example()
            self._example_idx += 1
            input_ids = torch.from_numpy(seq.astype(np.int64))
            labels = torch.full((seq_len,), -100, dtype=torch.long)
            labels[: seq_len - 1] = input_ids[1:]
            # Mask the first S-1 positions for the same reason as linear map:
            # predicting positions within the initial random state x_0 is
            # impossible by construction.
            labels[: S - 1] = -100
            yield {"input_ids": input_ids, "labels": labels}


def build_synthetic_dataset(
    cfg_dict: dict,
    *,
    rank: int,
    world_size: int,
) -> _SyntheticBase:
    """Construct a synthetic dataset from a `data:` config dict.

    Recognized `format` values:
      * `synthetic_linear_map`: LinearMapDataset
      * `synthetic_cellular_automata`: CellularAutomataDataset

    Required fields:
      * `state_size` (S) - default 16
      * `num_colors` (C) - default 2 for linear_map, 4 for cellular_automata
      * `trajectory_length` (T) - default 2 for linear_map, 16 for cellular_automata
      * `sparsity` (s) - linear_map only, default 3
      * `num_rules` (N) - cellular_automata only, default 256
      * `recursive_depth` (k) - cellular_automata only, default 1
      * `task_seed` - default 0; controls ground-truth A / rules. Train and
        eval datasets MUST use the same task_seed so they learn/evaluate the
        same pattern.
      * `stream_seed` - default 0; controls the per-example rng. Train and
        eval typically use different stream_seeds.
    """
    fmt = cfg_dict.get("format", "")
    if fmt == "synthetic_linear_map":
        task = "linear_map"
        default_C = 2
        default_T = 2
    elif fmt == "synthetic_cellular_automata":
        task = "cellular_automata"
        default_C = 4
        default_T = 16
    else:
        raise ValueError(f"Unknown synthetic format: {fmt!r}")

    config = SyntheticConfig(
        task=task,
        state_size=int(cfg_dict.get("state_size", 16)),
        trajectory_length=int(cfg_dict.get("trajectory_length", default_T)),
        num_colors=int(cfg_dict.get("num_colors", default_C)),
        sparsity=int(cfg_dict.get("sparsity", 3)),
        num_rules=int(cfg_dict.get("num_rules", 256)),
        recursive_depth=int(cfg_dict.get("recursive_depth", 1)),
        task_seed=int(cfg_dict.get("task_seed", 0)),
        stream_seed=int(cfg_dict.get("stream_seed", 0)),
    )
    if task == "linear_map":
        return LinearMapDataset(config, rank=rank, world_size=world_size)
    return CellularAutomataDataset(config, rank=rank, world_size=world_size)
