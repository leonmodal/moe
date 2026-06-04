"""Compatibility helpers for Transformers attention mask factories."""

from __future__ import annotations

import inspect
from typing import Any, Callable


def call_mask_function(mask_function: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a Transformers mask factory across minor signature changes."""

    try:
        signature = inspect.signature(mask_function)
    except (TypeError, ValueError):
        return mask_function(**kwargs)

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return mask_function(**kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in params}
    return mask_function(**filtered)
