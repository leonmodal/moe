"""
DeepSeek V3 aux-loss-free router with expert bias.

Replaces softmax routing with sigmoid + non-gradient expert bias:
  1. FP32 gating linear — logits = fp32(W) @ fp32(h)
  2. Sigmoid scoring — scores = sigmoid(logits)
  3. Group-limited top-k — select group_topk groups, then topk experts within
  4. Biased selection — topk(scores + expert_bias) with unbiased weight gather
  5. Normalization + scaling — normalize to sum-to-1, then multiply by scaling_factor
  6. Bias update after each step — bias += sign(avg - tokens_per_expert) * rate

The bias is updated externally by update_expert_biases() in train.py,
not through gradient descent — this prevents the router from gaming the loss.
"""
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from contextvars import ContextVar

from .modeling_qwen3_moe import Qwen3MoeTopKRouter


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_groups: int,
    group_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform top-k routing on a subset of expert groups.

    Matches Megatron-LM implementation:
      1. Divide experts into num_groups equal groups
      2. Score each group by sum of top-(topk // group_topk) expert scores within it
      3. Select group_topk groups per token
      4. Pick topk experts from those groups only

    Args:
        scores: (T, E) routing scores.
        topk: Number of experts to select per token.
        num_groups: Number of expert groups.
        group_topk: Number of groups to select per token.

    Returns:
        (top_scores, top_indices) each of shape (T, topk).
    """
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // num_groups

    # Score each group by sum of top-(topk // group_topk) within that group
    group_scores = (
        scores.view(num_tokens, num_groups, experts_per_group)
        .topk(topk // group_topk, dim=-1)[0]
        .sum(dim=-1)
    )  # (T, num_groups)

    # Select top groups
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]  # (T, group_topk)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)  # (T, num_groups)

    # Expand group mask to expert mask
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, experts_per_group)
        .reshape(num_tokens, -1)
    )  # (T, E)

    # Mask out experts in non-selected groups, then topk
    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    top_scores, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
    return top_scores, top_indices


_CHECKPOINT_RECOMPUTE = ContextVar("checkpoint_recompute", default=False)


@contextmanager
def checkpoint_recompute_context(enabled: bool):
    token = _CHECKPOINT_RECOMPUTE.set(enabled)
    try:
        yield
    finally:
        _CHECKPOINT_RECOMPUTE.reset(token)


def is_checkpoint_recompute() -> bool:
    return _CHECKPOINT_RECOMPUTE.get()


def sample_router_exploration_mask(
    scores: torch.Tensor,
    exploration_rate: float,
) -> torch.Tensor | None:
    """Sample a per-token exploration mask for router top-k selection."""
    if exploration_rate <= 0.0 or scores.shape[0] == 0:
        return None
    return torch.rand(scores.shape[0], device=scores.device) < exploration_rate


def apply_router_exploration(
    selection_scores: torch.Tensor,
    exploration_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Replace selected tokens' top-k scores with random scores during training."""
    if exploration_mask is None or not exploration_mask.any():
        return selection_scores

    explored_scores = selection_scores.clone()
    explored_scores[exploration_mask] = torch.rand_like(explored_scores[exploration_mask])
    return explored_scores


def collect_router_topk_indices(routers: Iterable[nn.Module]) -> tuple[torch.Tensor, ...] | None:
    """Collect the most recent top-k expert assignments from a list of routers."""
    selected = []
    for router in routers:
        idx = getattr(router, "_last_top_k_idx", None)
        if idx is None:
            return None
        selected.append(idx)
    return tuple(selected) if selected else None


def collect_router_z_losses(routers: Iterable[nn.Module]) -> torch.Tensor | None:
    """Sum the most recent per-router z-losses across routers with `z_loss_coef > 0`.

    Routers cache their per-call contribution on `_last_z_loss` (a scalar tensor,
    or `None` when the coefficient is zero). The returned tensor preserves
    autograd so the caller can add it directly to the total loss.
    """
    acc = None
    for router in routers:
        z = getattr(router, "_last_z_loss", None)
        if z is None:
            continue
        acc = z if acc is None else acc + z
    return acc


_SCORE_FUNCTIONS = {"softmax", "sigmoid", "sqrtsoftplus"}
_TOPK_ORDERINGS = {"post", "pre"}


def _apply_score_function(logits: torch.Tensor, score_function: str) -> torch.Tensor:
    """Compute per-expert scores from raw router logits.

    - ``softmax``: competing probabilities summing to 1 across experts (Switch / Qwen3 default).
    - ``sigmoid``: independent (0, 1) scores per expert (DeepSeek-style, non-competing).
    - ``sqrtsoftplus``: sqrt(softplus(x)); non-competing, smoother floor than sigmoid for large negatives.
    """
    if score_function == "softmax":
        return F.softmax(logits, dim=-1, dtype=torch.float)
    if score_function == "sigmoid":
        return torch.sigmoid(logits)
    if score_function == "sqrtsoftplus":
        return torch.sqrt(F.softplus(logits))
    raise ValueError(f"Unknown router_score_function: {score_function!r} (expected one of {sorted(_SCORE_FUNCTIONS)})")


class ExplorationTopKRouter(Qwen3MoeTopKRouter):
    """Top-k router with configurable scoring function and top-k ordering.

    Config knobs (all optional, sensible defaults preserve the pre-existing softmax
    post-softmax-topk behaviour):

    - ``router_score_function`` ∈ {``softmax``, ``sigmoid``, ``sqrtsoftplus``}
      (default ``softmax``). Maps raw logits to per-expert scores.
    - ``router_topk_ordering`` ∈ {``post``, ``pre``} (default ``post``).
      * ``post``: apply score function to all experts → top-k on scored values → gather.
      * ``pre``: top-k on raw logits → apply score function only to the k selected logits.
      Indices are the same either way for softmax (monotonic), but weight values (and
      gradients) differ, and for sigmoid / sqrtsoftplus the two orderings are distinct.
    - ``num_groups`` / ``group_topk``: optional group-limited top-k. When set, picks the
      top ``group_topk`` expert groups first (groups scored by sum of their top-``(topk // group_topk)``
      expert scores), then top-k among experts in those groups. Matches the DeepSeek path.
    - ``router_exploration_rate``: per-token probability of replacing the selection
      scores with uniform noise during training. Used by the warmup schedule.
    - ``router_z_loss_coef``: magnitude regularizer on the raw logits; the per-call
      z-loss is cached on ``_last_z_loss`` and the training loop accumulates it into
      the auxiliary-loss term.
    """

    def __init__(self, config):
        super().__init__(config)
        self.exploration_rate = float(getattr(config, "router_exploration_rate", 0.0) or 0.0)
        self.score_function = getattr(config, "router_score_function", "softmax")
        if self.score_function not in _SCORE_FUNCTIONS:
            raise ValueError(
                f"router_score_function must be one of {sorted(_SCORE_FUNCTIONS)}, "
                f"got {self.score_function!r}"
            )
        self.topk_ordering = getattr(config, "router_topk_ordering", "post")
        if self.topk_ordering not in _TOPK_ORDERINGS:
            raise ValueError(
                f"router_topk_ordering must be one of {sorted(_TOPK_ORDERINGS)}, "
                f"got {self.topk_ordering!r}"
            )
        self.num_groups = getattr(config, "num_groups", None)
        self.group_topk = getattr(config, "group_topk", None)
        self.z_loss_coef = float(getattr(config, "router_z_loss_coef", 0.0) or 0.0)
        self._last_top_k_idx = None
        self._last_exploration_mask = None
        self._last_z_loss = None

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            raw_logits = F.linear(hidden_states.float(), self.weight.float())  # (T, E) fp32
            probs = _apply_score_function(raw_logits, self.score_function)      # (T, E) fp32

            if self.topk_ordering == "pre":
                # Top-k directly on raw logits; score function only applies to the
                # k selected logits afterwards.
                exploration_mask = None
                selection_scores = raw_logits
                if self.training and self.exploration_rate > 0.0:
                    exploration_mask = sample_router_exploration_mask(raw_logits, self.exploration_rate)
                    selection_scores = apply_router_exploration(raw_logits, exploration_mask)
                if self.num_groups is not None and self.group_topk is not None:
                    _, router_indices = group_limited_topk(
                        selection_scores, self.top_k, self.num_groups, self.group_topk,
                    )
                else:
                    _, router_indices = torch.topk(selection_scores, self.top_k, dim=-1)
                selected_logits = raw_logits.gather(1, router_indices)
                router_top_value = _apply_score_function(selected_logits, self.score_function)
            else:
                # Default: score function on all experts, then top-k on the scored values.
                exploration_mask = None
                selection_scores = probs
                if self.training and self.exploration_rate > 0.0:
                    exploration_mask = sample_router_exploration_mask(probs, self.exploration_rate)
                    selection_scores = apply_router_exploration(probs, exploration_mask)
                if self.num_groups is not None and self.group_topk is not None:
                    _, router_indices = group_limited_topk(
                        selection_scores, self.top_k, self.num_groups, self.group_topk,
                    )
                else:
                    _, router_indices = torch.topk(selection_scores, self.top_k, dim=-1)
                router_top_value = probs.gather(1, router_indices)

            if self.z_loss_coef > 0.0:
                # Router z-loss: penalize large logit magnitudes to keep the router
                # numerics stable. `logsumexp(logits)^2` averaged over tokens,
                # then scaled by the configured coefficient.
                z = torch.logsumexp(raw_logits, dim=-1)
                self._last_z_loss = (z * z).mean() * self.z_loss_coef
            else:
                self._last_z_loss = None

        if self.norm_topk_prob:
            router_top_value = router_top_value / (router_top_value.sum(dim=-1, keepdim=True) + 1e-20)
        router_scores = router_top_value.to(raw_logits.dtype)
        self._last_top_k_idx = router_indices.detach()
        self._last_exploration_mask = None if exploration_mask is None else exploration_mask.detach()
        # First return value keeps the prior contract: the per-expert scored probs
        # (post-softmax / post-sigmoid / post-sqrtsoftplus) are what downstream aux
        # loss and routing stats read.
        return probs, router_scores, router_indices


class DeepSeekRouter(Qwen3MoeTopKRouter):
    """
    Sigmoid + expert-bias router (DeepSeek V3 style).

    Subclasses Qwen3MoeTopKRouter so that OutputRecorder hooks
    (used by HF for collecting router_logits) still fire.
    """

    def __init__(self, config):
        super().__init__(config)
        # DeepSeek V3 routing parameters
        self.scaling_factor = getattr(config, "topk_scaling_factor", None)
        self.num_groups = getattr(config, "num_groups", None)
        self.group_topk = getattr(config, "group_topk", None)
        self.exploration_rate = float(getattr(config, "router_exploration_rate", 0.0) or 0.0)
        # Optional z-loss — regularises raw-logit magnitudes. Independent of the
        # DeepSeek bias-update balancing signal; both can be on simultaneously.
        self.z_loss_coef = float(getattr(config, "router_z_loss_coef", 0.0) or 0.0)

        # Persistent buffer: survives checkpointing
        self.register_buffer(
            "expert_bias",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )
        # Non-persistent buffer: reset each step, not saved in state_dict
        self.register_buffer(
            "local_tokens_per_expert",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        # Last forward's selected experts (T, K). Used by seq aux loss
        # so f_i matches actual biased routing assignments.
        self._last_top_k_idx = None
        self._last_exploration_mask = None
        self._last_z_loss = None
        # AC-9: snapshot of `top_k_idx` written ONLY on the real forward.
        # On gradient-checkpoint recompute, the router uses this snapshot
        # to produce the same biased top-k selection without re-sampling
        # `apply_router_exploration`'s `torch.rand_like`. This is a defensive
        # safety net against checkpoint configurations where
        # `preserve_rng_state=False` is set (PyTorch's default `True` already
        # makes recompute deterministic, but this cache survives even when
        # RNG preservation is disabled, e.g. by some custom checkpoint
        # wrappers).
        self._cached_top_k_idx_for_recompute: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            # 1. FP32 gating linear — match Megatron-LM moe_router_dtype=fp32
            router_logits = F.linear(
                hidden_states.float(), self.weight.float()
            )  # (T, E) in fp32

            if self.z_loss_coef > 0.0:
                z = torch.logsumexp(router_logits, dim=-1)
                self._last_z_loss = (z * z).mean() * self.z_loss_coef
            else:
                self._last_z_loss = None

            # 2. Sigmoid scoring in fp32
            scores = torch.sigmoid(router_logits)  # (T, E) in (0, 1), fp32

            # 3. Biased top-k selection (with optional group-limited routing)
            bias = self.expert_bias.float()
            biased_scores = scores + bias.unsqueeze(0)  # (T, E)
            exploration_mask = None
            selection_scores = biased_scores
            if self.training and self.exploration_rate > 0.0:
                # AC-9: under gradient-checkpoint recompute, reuse the cached
                # exploration mask from the real forward so the recompute
                # branch decision is identical and gradient consistency
                # holds. Belt-and-suspenders on top of `torch.utils.checkpoint`'s
                # default `preserve_rng_state=True`.
                cached = getattr(self, "_last_exploration_mask", None)
                if (
                    is_checkpoint_recompute()
                    and cached is not None
                    and cached.shape == biased_scores.shape[:1]
                    and cached.device == biased_scores.device
                ):
                    exploration_mask = cached
                else:
                    exploration_mask = sample_router_exploration_mask(biased_scores, self.exploration_rate)
                selection_scores = apply_router_exploration(biased_scores, exploration_mask)

            # AC-9: under gradient-checkpoint recompute, reuse the cached
            # top_k_idx from the real forward. This bypasses the
            # exploration + top-k computation entirely and guarantees the
            # recompute's biased selection matches the real forward — even
            # if `apply_router_exploration`'s internal `torch.rand_like`
            # would otherwise produce different values.
            cached_idx = getattr(self, "_cached_top_k_idx_for_recompute", None)
            if (
                is_checkpoint_recompute()
                and cached_idx is not None
                and cached_idx.shape[0] == selection_scores.shape[0]
                and cached_idx.device == selection_scores.device
            ):
                top_k_idx = cached_idx
            elif self.num_groups is not None and self.group_topk is not None:
                # Group-limited top-k: select from top groups only
                _, top_k_idx = group_limited_topk(
                    selection_scores,
                    topk=self.top_k,
                    num_groups=self.num_groups,
                    group_topk=self.group_topk,
                )
            else:
                _, top_k_idx = torch.topk(selection_scores, self.top_k, dim=-1)  # (T, K)

            # 4. Gather unbiased scores for selected experts
            router_top_value = scores.gather(1, top_k_idx)  # (T, K)

            # 5. Normalize + scaling factor
            # Skip normalization for top-1: w/w=1.0 kills gradient.
            # For top-K (K>1), normalize so weights sum to 1.
            if self.norm_topk_prob and self.top_k > 1:
                router_top_value = router_top_value / (
                    router_top_value.sum(dim=-1, keepdim=True) + 1e-20
                )
            if self.scaling_factor is not None:
                router_top_value = router_top_value * self.scaling_factor

        router_top_value = router_top_value.to(hidden_states.dtype)
        self._last_exploration_mask = None if exploration_mask is None else exploration_mask.detach()

        # 6. Accumulate token counts only on the real forward pass.
        # With activation checkpointing, the backward recompute runs the router
        # a second time; those passes must not update bias statistics.
        if torch.is_grad_enabled() and not is_checkpoint_recompute():
            with torch.no_grad():
                counts = torch.bincount(
                    top_k_idx.reshape(-1),
                    minlength=self.num_experts,
                ).float()
                self.local_tokens_per_expert += counts
                self._last_top_k_idx = top_k_idx.detach()
                # AC-9: stash the real-forward top_k_idx for recompute reuse.
                self._cached_top_k_idx_for_recompute = top_k_idx.detach()
        else:
            self._last_top_k_idx = top_k_idx.detach()

        # Return sigmoid probs as router_logits (values in (0,1), don't sum to 1)
        return scores, router_top_value, top_k_idx
