import argparse
from argparse import REMAINDER, ArgumentParser, Namespace

import pydantic

from .argparse_util import check_env, env


class PydanticArgparseWrapper(pydantic.BaseModel):
    parser: ArgumentParser = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.parser = self.__class__.parser

    def to_argparse(self) -> ArgumentParser:
        args_list = []
        positional_args = []
        for action in self.parser._actions:
            if action.dest == "parser":
                continue
            value = getattr(self, action.dest, None)
            if value is None:
                continue
            if action.option_strings:
                if isinstance(value, bool):
                    if value:
                        args_list.append(action.option_strings[0])
                else:
                    args_list.append(action.option_strings[0])
                    args_list.append(str(value))
            elif action.nargs == REMAINDER:
                positional_args.extend(value)
            else:
                positional_args.append(str(value))
        return self.parser.parse_args(args_list + positional_args)


def convert_to_pydantic(argparse_parser: argparse.ArgumentParser) -> type:
    fields = {}
    for action in argparse_parser._actions:
        if action.dest == "help":
            continue
        if action.dest != argparse.SUPPRESS:
            field_type = (
                list[str] if action.nargs else (action.type if action.type else str)
            )
            default_value = (
                action.default if action.default is not argparse.SUPPRESS else ...
            )
            if action.required:
                fields[action.dest] = (field_type, ...)
            else:
                fields[action.dest] = (field_type, default_value)

    model_klass = pydantic.create_model(
        argparse_parser.description, **fields, __base__=PydanticArgparseWrapper
    )
    model_klass.parser = argparse_parser
    return model_klass
