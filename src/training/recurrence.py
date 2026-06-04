"""Recurrent-training schedules and deterministic step sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


RECURRENT_MODEL_TYPES = frozenset({
    "recurrent_standard_moe",
    "recurrent_global_moe",
    "recurrent_moe_everything",
    "hrm_recurrent_standard_moe",
})


@dataclass(frozen=True)
class RecurrenceStep:
    """No-grad/grad recurrence split for one forward pass."""

    num_steps_no_grad: int
    num_steps_with_grad: int
    mean_recurrence: int
    mean_backprop_depth: int
    extra_steps: tuple[int, ...] = ()

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.num_steps_no_grad, self.num_steps_with_grad, *self.extra_steps],
            device=device,
            dtype=torch.long,
        )


def is_recurrent_model_type(model_type: str | None) -> bool:
    return str(model_type or "") in RECURRENT_MODEL_TYPES


def recurrence_cfg(cfg: dict) -> dict:
    """Return the top-level recurrence config block, if present."""

    block = cfg.get("recurrence", {})
    return block if isinstance(block, dict) else {}


def mean_recurrence_for_step(cfg: dict, *, global_step: int, max_steps: int) -> int:
    """Match retrofitting-recurrence's integer mean-recurrence scheduler.

    The original code used a dummy optimizer + ``warmup_stable_decay`` scheduler
    and then took ``ceil(get_last_lr()[0])`` before each data step. With no
    decay and a zero-start warmup, this is equivalent to the closed form below:
    step 0 clamps to 1, warmup reaches ``max_mean_rec``, and the value stays
    there after warmup.
    """

    rec = recurrence_cfg(cfg)
    sched = rec.get("mean_recurrence_schedule", {})
    if not isinstance(sched, dict) or not bool(sched.get("turn_on", False)):
        return max(1, int(rec.get("mean_recurrence", rec.get("eval_recurrence", 1))))

    max_mean_rec = max(1, int(sched.get("max_mean_rec", 32)))
    warmup_value = float(sched.get("warmup", 0.1))
    warmup_steps = (
        math.ceil(warmup_value * max_steps)
        if warmup_value <= 1.0
        else math.ceil(warmup_value)
    )
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return max_mean_rec

    progress = max(0.0, float(global_step)) / float(warmup_steps)
    warmup_type = str(sched.get("warmup_type", "linear"))
    if warmup_type == "linear":
        fraction = progress
    elif warmup_type == "1-sqrt":
        fraction = 1.0 - math.sqrt(max(0.0, 1.0 - progress))
    else:
        raise ValueError(
            f"Unsupported recurrence warmup_type={warmup_type!r}; "
            "expected 'linear' or '1-sqrt'."
        )

    return max(1, math.ceil(max_mean_rec * fraction))


def mean_backprop_depth_for_step(cfg: dict, *, global_step: int, max_steps: int) -> int:
    """Resolve the scheduled mean backprop depth.

    The current configs use a fixed ``mean_backprop_depth=8``. This also
    supports the retrofitting-recurrence warmup block for compatibility.
    """

    rec = recurrence_cfg(cfg)
    sched = rec.get("mean_backprop_depth_schedule", {})
    if not isinstance(sched, dict) or not bool(sched.get("turn_on", False)):
        return max(0, int(rec.get("mean_backprop_depth", 8)))

    max_backprop = max(1, int(sched.get("max_backprop", 8)))
    start = max(1.0, float(sched.get("start", 1)) - 1.0)
    warmup_value = float(sched.get("warmup", 0.1))
    warmup_steps = (
        math.ceil(warmup_value * max_steps)
        if warmup_value <= 1.0
        else math.ceil(warmup_value)
    )
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return max_backprop
    min_ratio = max(0.0, min(1.0, start / float(max_backprop)))
    progress = max(0.0, float(global_step)) / float(warmup_steps)
    value = max_backprop * (min_ratio + (1.0 - min_ratio) * progress)
    return max(1, math.ceil(value))


def _sample_lognormal_poisson_plus_one(
    *,
    generator: torch.Generator,
    mean: float,
    sigma: float = 0.5,
) -> int:
    mean = max(1.0, float(mean))
    mu = math.log(mean) - (sigma**2 / 2)
    rate = torch.zeros((1,), dtype=torch.float32).log_normal_(
        mean=mu,
        std=sigma,
        generator=generator,
    )
    return int((torch.poisson(rate, generator=generator) + 1).item())


def sample_recurrence_step(
    *,
    data_step: int,
    mean_recurrence: int,
    mean_backprop_depth: int,
) -> RecurrenceStep:
    """Sample ``(n_no_grad, n_with_grad)`` exactly like retrofitting-recurrence.

    The original sampler uses a checkpoint-stable CPU generator seeded from the
    data step, samples a lognormal rate with ``sigma=0.5``, then samples
    ``Poisson(rate) + 1`` total recurrent passes. The first ``p-s`` passes run
    without gradients and the final ``min(s, p)`` passes backpropagate.
    """

    mean_recurrence = max(1, int(mean_recurrence))
    mean_backprop_depth = max(0, int(mean_backprop_depth))
    if mean_recurrence - mean_backprop_depth < 0:
        mean_backprop_depth = mean_recurrence

    t = max(mean_recurrence - mean_backprop_depth, 0)
    s = mean_backprop_depth

    generator = torch.Generator(device="cpu")
    generator.manual_seed((514229 + int(data_step)) % (2**31 - 1))

    p = _sample_lognormal_poisson_plus_one(
        generator=generator,
        mean=t + s,
    )
    n = max(0, p - s)
    k = min(s, p)

    return RecurrenceStep(
        num_steps_no_grad=int(n),
        num_steps_with_grad=int(k),
        mean_recurrence=mean_recurrence,
        mean_backprop_depth=mean_backprop_depth,
    )


def _positive_partition(
    *,
    total: int,
    parts: int,
    generator: torch.Generator,
) -> tuple[int, ...]:
    """Random positive integer partition with deterministic CPU sampling."""

    parts = max(1, int(parts))
    total = max(parts, int(total))
    counts = torch.ones(parts, dtype=torch.long)
    extra = total - parts
    if extra <= 0:
        return tuple(int(x) for x in counts.tolist())

    weights = torch.empty(parts, dtype=torch.float32).exponential_(
        1.0,
        generator=generator,
    )
    picks = torch.multinomial(
        weights,
        num_samples=extra,
        replacement=True,
        generator=generator,
    )
    counts += torch.bincount(picks, minlength=parts).to(dtype=torch.long)
    return tuple(int(x) for x in counts.tolist())


def sample_hrm_recurrence_step(
    *,
    data_step: int,
    mean_recurrence: int,
    mean_backprop_depth: int,
) -> RecurrenceStep:
    """Sample HRM H/L cycles with roughly flat-loop compute parity.

    The flat recurrent baseline applies an 8-layer core ``R`` times.  HRM uses
    two 4-layer modules, so the comparable budget is ``2 * R`` module calls.
    We first sample the old flat recurrence count, then sample the number of
    H cycles from the square-root scale of that module budget and allocate the
    remaining module calls as positive L counts across H cycles.
    """

    base = sample_recurrence_step(
        data_step=data_step,
        mean_recurrence=mean_recurrence,
        mean_backprop_depth=mean_backprop_depth,
    )
    total_flat_steps = max(1, base.num_steps_no_grad + base.num_steps_with_grad)
    module_budget = max(2, 2 * total_flat_steps)

    generator = torch.Generator(device="cpu")
    generator.manual_seed((2178309 + int(data_step)) % (2**31 - 1))

    max_h_cycles = max(1, module_budget // 2)
    h_mean = math.sqrt(float(module_budget))
    h_cycles = _sample_lognormal_poisson_plus_one(
        generator=generator,
        mean=h_mean,
    )
    h_cycles = max(1, min(int(h_cycles), max_h_cycles))

    l_total = max(h_cycles, module_budget - h_cycles)
    l_counts = _positive_partition(
        total=l_total,
        parts=h_cycles,
        generator=generator,
    )

    grad_module_budget = max(0, 2 * int(base.num_steps_with_grad))
    h_with_grad = 0
    used_modules = 0
    if grad_module_budget > 0:
        for l_count in reversed(l_counts):
            cycle_modules = 1 + int(l_count)
            if h_with_grad == 0 or used_modules + cycle_modules <= grad_module_budget:
                used_modules += cycle_modules
                h_with_grad += 1
            else:
                break
    h_no_grad = max(0, h_cycles - h_with_grad)

    return RecurrenceStep(
        num_steps_no_grad=h_no_grad,
        num_steps_with_grad=h_with_grad,
        mean_recurrence=mean_recurrence,
        mean_backprop_depth=mean_backprop_depth,
        extra_steps=l_counts,
    )


def recurrence_step_for_data_step(
    cfg: dict,
    *,
    data_step: int,
    global_step: int,
    max_steps: int,
) -> RecurrenceStep:
    mean_rec = mean_recurrence_for_step(
        cfg,
        global_step=global_step,
        max_steps=max_steps,
    )
    backprop_depth = mean_backprop_depth_for_step(
        cfg,
        global_step=global_step,
        max_steps=max_steps,
    )
    if str(cfg.get("model", {}).get("type", "")) == "hrm_recurrent_standard_moe":
        return sample_hrm_recurrence_step(
            data_step=data_step,
            mean_recurrence=mean_rec,
            mean_backprop_depth=backprop_depth,
        )
    return sample_recurrence_step(
        data_step=data_step,
        mean_recurrence=mean_rec,
        mean_backprop_depth=backprop_depth,
    )


def eval_recurrence_num_steps(cfg: dict, *, device: torch.device) -> torch.Tensor:
    rec = recurrence_cfg(cfg)
    n = max(1, int(rec.get("eval_recurrence", 32)))
    return torch.tensor([n, 0], device=device, dtype=torch.long)
