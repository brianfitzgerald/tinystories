import os
import tempfile
import fire
from typing import Optional

import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import init_logger
from vllm.utils import (
    FlexibleArgumentParser,
)

os.environ["HF_HOME"] = "/weka/home-brianf/huggingface"


TIMEOUT_KEEP_ALIVE = 5  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

logger = init_logger("vllm.entrypoints.openai.api_server")


def main(run: str = "", steps: Optional[int] = None):
    """
    Load the run specified by the run_name.
    If steps is provided, use that ckpt, otherwise use latest.
    """

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    # don't actually parse cli args, just return the object
    args = parser.parse_args([])

    run_directory = f"/weka/home-brianf/runs/{run}"
    if not os.path.exists(run_directory):
        raise ValueError(f"Run directory {run_directory} not found")
    checkpoints = os.listdir(run_directory)
    checkpoints = [x for x in checkpoints if x.startswith("checkpoint-")]
    sorted_checkpoints = list(sorted(checkpoints, key=lambda x: int(x.split("-")[-1])))
    sorted_checkpoints = [int(x.split("-")[-1]) for x in sorted_checkpoints]
    if steps in sorted_checkpoints:
        checkpoint_dir = f"{run_directory}/checkpoint-{steps}"
    elif steps is not None and steps not in sorted_checkpoints:
        raise ValueError(f"Checkpoint {steps} not found in {run_directory}")
    else:
        latest_ckpt = sorted_checkpoints[-1]
        checkpoint_dir = f"{run_directory}/checkpoint-{latest_ckpt}"

    logger.info(f"Using checkpoint: {checkpoint_dir}")
    args.model = checkpoint_dir
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))


if __name__ == "__main__":
    fire.Fire(main)
