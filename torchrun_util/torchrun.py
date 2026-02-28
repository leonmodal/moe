from .argparse_to_pydantic import convert_to_pydantic


def run(**kwargs):
    """
    Utility function which allows for launching torchrun's argparse-based CLI interface via
    a regular Python function call with pydantic-based validation.
    """
    from torch.distributed.run import run, get_args_parser

    argparse_parser = get_args_parser()
    pydantic_model = convert_to_pydantic(argparse_parser)
    pydantic_args = pydantic_model(**kwargs)
    run(pydantic_args.to_argparse())
