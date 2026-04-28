"""Unified router implementations.

This module is the source of truth for all router classes:
- DeepSeekRouter: Sigmoid + expert-bias router (DeepSeek V3 style)
- ExplorationTopKRouter: Softmax top-k with random exploration
- BranchRouter: Binary attention/MLP branch selection per token
- BranchRouterRecorder: Parameterless recorder for deterministic routing

Re-exports from src.models.router for DeepSeek/ExplorationTopK routers
(which are tightly coupled to the Qwen3MoeTopKRouter base class).
BranchRouter and BranchRouterRecorder are defined here as the source of truth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.router import (
    DeepSeekRouter,
    ExplorationTopKRouter,
    group_limited_topk,
    sample_router_exploration_mask,
    apply_router_exploration,
    collect_router_topk_indices,
    checkpoint_recompute_context,
    is_checkpoint_recompute,
)


def _straight_through_ones(probs: torch.Tensor) -> torch.Tensor:
    """Forward: returns ones. Backward: gradient flows through probs."""
    return probs + (1.0 - probs).detach()


class BranchRouter(nn.Module):
    """Binary router: ATTN (0) or MLP (1) per token.

    Hard routing: each token picks one branch via argmax.
    The selected branch output is scaled by its softmax probability
    for gradient flow (same pattern as MoE expert routing).

    NOTE: BranchRouter has NO load-balancing loss by design.

    Modes (from speedrun extraction):
    - use_sampling: Sample from distribution instead of argmax during training
    - use_seq_level: Mean-pool tokens and make one decision per sequence
    - use_deepseek_style: Sigmoid + persistent branch bias instead of softmax
    """

    def __init__(self, hidden_size: int, exploration_rate: float = 0.0,
                 scale_by_routing_weight: bool = True,
                 use_sampling: bool = False, use_seq_level: bool = False,
                 use_deepseek_style: bool = False,
                 exploration_only_rate: float = 0.0,
                 exploration_only: bool | None = None,
                 balancing: str = "none"):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 2, bias=False)
        self.exploration_rate = exploration_rate
        self.scale_by_routing_weight = scale_by_routing_weight
        self.use_sampling = use_sampling
        self.use_seq_level = use_seq_level
        # `balancing`: branch-router balancing mode. Allowed values:
        #   "none"             - default; no exploration override.
        #   "exploration_only" - every step draws a per-token explore
        #                        mask at probability
        #                        `exploration_only_rate` (the
        #                        scheduled `p_explore(step)` rate).
        #                        For masked tokens, the routing
        #                        decision is replaced by a uniform
        #                        Bernoulli draw. Branch aux/bias
        #                        paths are disabled by construction
        #                        in this mode.
        # Backwards-compat aliases:
        #   `exploration_only=True`  -> balancing="exploration_only",
        #                                exploration_only_rate=1.0
        #   `exploration_only_rate > 0` (without explicit balancing)
        #     auto-promotes balancing to "exploration_only".
        # `aux_loss` / `seq_aux_loss`: the model's forward computes
        # the loss term from `last_probs` / `last_selected_experts`
        # via the same helpers used for MLP and attention routers.
        # `deepseek_bias`: BranchRouter already owns `expert_bias`
        # and `local_tokens_per_expert` for the binary ATTN/MLP
        # pool via the canonical balancing-owner interface. The
        # post-step walker dispatches per-owner so this update
        # fires regardless of what method MLP / attention pick.
        # `quantile` still needs owner-state plumbing (persistent
        # `quantile_ema` + accumulator) before it can route
        # through the post-step walker; rejected at construction
        # time until that lands.
        _allowed_balancing = (
            "none", "exploration_only",
            "aux_loss", "seq_aux_loss",
            "deepseek_bias",
        )
        if balancing not in _allowed_balancing:
            raise ValueError(
                f"BranchRouter balancing must be one of {_allowed_balancing}, "
                f"got {balancing!r}"
            )
        if exploration_only is True:
            balancing = "exploration_only"
            if exploration_only_rate == 0.0:
                exploration_only_rate = 1.0
        elif exploration_only is False:
            balancing = "none"
            exploration_only_rate = 0.0
        # Auto-promote: a positive rate without explicit balancing
        # implies the user wants exploration-only mode.
        if balancing == "none" and exploration_only_rate > 0.0:
            balancing = "exploration_only"
        if not (0.0 <= exploration_only_rate <= 1.0):
            raise ValueError(
                f"exploration_only_rate must be in [0, 1], got {exploration_only_rate}"
            )
        self.balancing = balancing
        # When `balancing == "deepseek_bias"`, the router MUST consume
        # `expert_bias` during scoring — otherwise the post-step
        # walker would mutate a buffer that the runtime ignores. Fold
        # the per-class method choice into the runtime style flag
        # instead of leaving it gated on a separate legacy
        # `use_deepseek_style` config knob: a single attribute drives
        # both buffer creation, count tracking, and biased argmax.
        self.use_deepseek_style = use_deepseek_style or balancing == "deepseek_bias"
        self.exploration_only_rate = exploration_only_rate
        # Optional `torch.Generator` for deterministic exploration-only
        # draws. Tests inject a seeded generator to assert the
        # exact mask + branch-choice pattern. Production leaves it
        # `None` and relies on the global RNG state (which
        # `torch.utils.checkpoint`'s `preserve_rng_state=True`
        # restores correctly on recompute).
        self.exploration_generator: torch.Generator | None = None
        self.last_probs = None
        self.last_selected_experts = None
        # Last-step exploration-only mask (per-token bool) and the
        # rate that produced it. Reset to None / 0.0 at the start of
        # every forward; populated only when exploration-only is
        # actually active.
        self.last_exploration_only_mask = None
        self.last_exploration_only_rate = 0.0
        # Rely on `torch.utils.checkpoint`'s default
        # `preserve_rng_state=True` for recompute determinism. An
        # earlier design had explicit caches for
        # `_last_exploration_mask` / `_last_exploration_random` /
        # `_last_sampling_choice`, but those caused saved-tensor count
        # mismatches under `torch.utils.checkpoint(use_reentrant=False)`
        # (real forward sampled → 2 saved tensors; recompute used
        # cache → 0 saved tensors). The count-buffer guard in
        # `forward` plus PyTorch's RNG preservation are sufficient.
        # The canonical balancing-owner buffers (`expert_bias` and
        # `local_tokens_per_expert`) are needed whenever the runtime
        # uses biased argmax — `self.use_deepseek_style` already
        # subsumes both the explicit flag and the per-class
        # `balancing == "deepseek_bias"` upgrade above.
        if self.use_deepseek_style:
            # Every load-balancing owner — standalone DeepSeekRouter,
            # BranchRouter, or shared expert bank — exposes the same
            # `expert_bias` (persistent fp32) and
            # `local_tokens_per_expert` (non-persistent fp32) buffers so
            # the unified `update_expert_biases` walker can update them
            # without special-casing the branch path.
            self.register_buffer(
                "expert_bias",
                torch.zeros(2, dtype=torch.float32),
            )
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(2, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, hidden_states: torch.Tensor):
        # Reset exploration-only telemetry on every forward so a later
        # rate=0 / eval-mode call cannot read a stale mask/rate left
        # over from a prior nonzero-rate forward.
        self.last_exploration_only_mask = None
        self.last_exploration_only_rate = 0.0
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            # Route in the gate's weight dtype. Under standard DDP/fp32 training
            # the weight is fp32 so this preserves the prior "route in fp32"
            # behavior; under FSDP with `MixedPrecision(param_dtype=bf16)` the
            # weight is bf16 and casting hidden_states to float() would mix
            # dtypes in the matmul. Matching the weight dtype keeps both paths
            # valid.
            gate_dtype = self.gate.weight.dtype
            # Seq-level routing: mean-pool then broadcast
            B, T = None, None
            if self.use_seq_level and hidden_states.ndim == 3:
                B, T, D = hidden_states.shape
                seq_repr = hidden_states.to(gate_dtype).mean(dim=1)
                logits = self.gate(seq_repr)
            else:
                logits = self.gate(hidden_states.to(gate_dtype))

            if (
                self.balancing == "exploration_only"
                and self.exploration_only_rate > 0.0
                and self.training
            ):
                # Exploration-only routing: with probability
                # `exploration_only_rate` per token (the explore mask),
                # the token's selection is replaced by a uniform
                # Bernoulli draw. Otherwise it follows the regular
                # argmax path. Probs are still populated via the
                # gate's softmax/sigmoid for downstream telemetry and
                # weight-gradient flow.
                if self.use_deepseek_style:
                    probs = torch.sigmoid(logits)
                else:
                    probs = F.softmax(logits, dim=-1)
                argmax_choice = probs.argmax(dim=-1)
                gen = self.exploration_generator
                rand_kwargs = {"device": probs.device}
                if gen is not None:
                    rand_kwargs["generator"] = gen
                rand_mask = torch.rand(probs.shape[:-1], **rand_kwargs)
                explore_mask = rand_mask < self.exploration_only_rate
                random_choice = (
                    torch.rand(probs.shape[:-1], **rand_kwargs) < 0.5
                ).long()
                choice = torch.where(explore_mask, random_choice, argmax_choice)
                # Record for tests / telemetry. `.detach()` on the
                # mask so storing it doesn't retain gradient history.
                # When `use_seq_level=True` the per-seq mask is broadcast
                # to (B, T) so telemetry consumers see the same shape as
                # the post-broadcast `choice` tensor, mirroring how
                # `branch_explore_fraction` and the per-token mask probes
                # in tests treat the mask as token-shaped.
                stored_mask = explore_mask.detach()
                if self.use_seq_level and B is not None:
                    stored_mask = stored_mask.unsqueeze(1).expand(B, T)
                self.last_exploration_only_mask = stored_mask
                self.last_exploration_only_rate = float(self.exploration_only_rate)
            elif self.use_deepseek_style:
                scores = torch.sigmoid(logits)
                biased = scores + self.expert_bias.to(scores.dtype)
                if self.training and self.use_sampling:
                    # Sample from a bias-aware distribution so the
                    # learned `expert_bias` actually shifts the
                    # sampling categorical — otherwise the post-step
                    # walker mutates a buffer that the sampling path
                    # ignores, and a heavily-loaded expert never
                    # gets routed away from.
                    # Renormalize via softmax over `biased` (in fp32
                    # to keep negative-bias clamps stable) to produce
                    # a valid `multinomial` distribution.
                    # Recompute determinism: rely on
                    # `torch.utils.checkpoint`'s default
                    # `preserve_rng_state=True` so `torch.multinomial`
                    # produces identical draws on real forward and
                    # recompute.
                    sample_dist = F.softmax(biased.float(), dim=-1)
                    flat = sample_dist.view(-1, 2)
                    choice = torch.multinomial(flat, 1).view(scores.shape[:-1])
                else:
                    choice = biased.argmax(dim=-1)
                probs = scores
            else:
                probs = F.softmax(logits, dim=-1)
                if self.training and self.use_sampling:
                    # Recompute determinism: same RNG-preservation rationale as above.
                    flat = probs.view(-1, 2)
                    choice = torch.multinomial(flat, 1).view(probs.shape[:-1])
                else:
                    choice_scores = probs.float()
                    if self.training and self.exploration_rate > 0.0:
                        # Recompute determinism: rely on `torch.utils.checkpoint`'s default
                        # `preserve_rng_state=True` for recompute determinism.
                        # Both `torch.rand` for the binary mask and
                        # `torch.rand_like` for the override values produce
                        # identical outputs across the real forward and
                        # the recompute when RNG state is preserved (which it
                        # is by default). An earlier design cached the mask
                        # explicitly, but that introduced a saved-tensor
                        # count mismatch under
                        # `torch.utils.checkpoint(use_reentrant=False)`.
                        explore_mask = torch.rand(
                            choice_scores.shape[:-1], device=choice_scores.device
                        ) < self.exploration_rate
                        if explore_mask.any():
                            choice_scores = choice_scores.clone()
                            choice_scores[explore_mask] = torch.rand_like(
                                choice_scores[explore_mask]
                            )
                    choice = choice_scores.argmax(dim=-1)

            # Broadcast seq-level decision to all tokens
            if self.use_seq_level and B is not None:
                choice = choice.unsqueeze(1).expand(B, T)
                probs = probs.unsqueeze(1).expand(B, T, 2)

        # Track counts for bias update (DeepSeek style). Skip on
        # gradient-checkpoint recompute so backward replays do not
        # double-count. Also skip when balancing is "exploration_only":
        # the mode is by definition not driven by the bias-update
        # signal, so counting random branch picks would inject noise
        # into a buffer the trainer is asked to never read for this
        # mode.
        if (
            self.use_deepseek_style
            and self.balancing != "exploration_only"
            and self.training
            and torch.is_grad_enabled()
            and not is_checkpoint_recompute()
        ):
            with torch.no_grad():
                flat_choice = choice.reshape(-1)
                counts = torch.bincount(flat_choice, minlength=2).float()
                self.local_tokens_per_expert += counts

        # Rely on `torch.utils.checkpoint`'s default
        # `preserve_rng_state=True` for recompute determinism: an
        # earlier explicit cache caused a saved-tensor count mismatch
        # (real forward sampled, recompute used cache) which
        # `torch.utils.checkpoint(use_reentrant=False)` rejects. The
        # count-buffer guard above is the only checkpoint-specific
        # state we still need to manage.

        probs = probs.to(hidden_states.dtype)
        self.last_probs = probs
        self.last_selected_experts = choice.unsqueeze(-1).detach()
        attn_mask = (choice == 0).unsqueeze(-1)
        mlp_mask = (choice == 1).unsqueeze(-1)
        if self.scale_by_routing_weight:
            w_attn = probs[..., 0:1] * attn_mask
            w_mlp = probs[..., 1:2] * mlp_mask
        else:
            w_attn = _straight_through_ones(probs[..., 0:1]) * attn_mask
            w_mlp = _straight_through_ones(probs[..., 1:2]) * mlp_mask
        return w_attn, w_mlp, attn_mask, mlp_mask


class BranchRouterRecorder(nn.Module):
    """Parameterless recorder used when branch routing is deterministic."""

    def __init__(self):
        super().__init__()
        self.last_probs = None
        self.last_selected_experts = None


__all__ = [
    "DeepSeekRouter",
    "ExplorationTopKRouter",
    "BranchRouter",
    "BranchRouterRecorder",
    "group_limited_topk",
    "sample_router_exploration_mask",
    "apply_router_exploration",
    "collect_router_topk_indices",
    "checkpoint_recompute_context",
    "is_checkpoint_recompute",
]
