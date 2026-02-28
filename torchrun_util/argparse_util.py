#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from argparse import Action


class env(Action):
    """Get argument values from ``PET_{dest}`` before defaulting to the given ``default`` value."""

    def __init__(self, dest, default=None, required=False, **kwargs) -> None:
        env_name = f"PET_{dest.upper()}"
        default = os.environ.get(env_name, default)
        if default:
            required = False
        super().__init__(dest=dest, default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class check_env(Action):
    """Check whether the env var ``PET_{dest}`` exists before defaulting to the given ``default`` value."""

    def __init__(self, dest, default=False, **kwargs) -> None:
        env_name = f"PET_{dest.upper()}"
        default = bool(int(os.environ.get(env_name, "1" if default else "0")))
        super().__init__(dest=dest, const=True, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)
