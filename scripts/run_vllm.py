import os
import tempfile

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
