import uvloop
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import (
    FlexibleArgumentParser,
)
from vllm.entrypoints.openai.api_server import run_server
import os
import tempfile
from typing import Set
from vllm.logger import init_logger
import asyncio

os.environ["HF_HOME"] = "/weka/home-brianf/huggingface"


TIMEOUT_KEEP_ALIVE = 5  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()
def main():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
