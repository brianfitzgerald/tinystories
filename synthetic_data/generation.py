import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, cast

import google.genai as genai
import google.genai.types
from anthropic import AnthropicError, AsyncAnthropic
from anthropic.types.message import Message
from anthropic.types.message_param import MessageParam
from datasets import Dataset, concatenate_datasets
from dotenv import dotenv_values
from huggingface_hub import login
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from wrapt_timeout_decorator import timeout

from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    gather_with_concurrency_limit,
)

SHAREGPT_TO_OPENAI_ROLE = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


def save_output_dataset(
    hf_dataset: Dataset,
    dataset_name: str,
    new_dataset_rows: List[Dict],
    format: DatasetFormat,
):
    dataset_new_rows = Dataset.from_list(new_dataset_rows)
    concatted_dataset = concatenate_datasets([hf_dataset, dataset_new_rows])

    if format == DatasetFormat.HF_DATASET:
        logger.info(f"Uploading {len(new_dataset_rows)} new rows to the Hub...")
        concatted_dataset.push_to_hub(dataset_name)
    elif format == DatasetFormat.PARQUET:
        logger.info(f"Saving {len(new_dataset_rows)} new rows to parquet...")
        concatted_dataset.to_parquet(f"{dataset_name}.parquet")
    else:
        raise ValueError(f"Unsupported output format: {format}")


@dataclass
class GenWrapperArgs:
    model_id: Optional[str] = None
    lora_name: Optional[str] = None
    dotenv: dict[str, str] = field(default_factory=dict)
    max_concurrent: int = 8


class GenerationWrapper(ABC):
    """
    Abstract method for various ways of generating data.
    """

    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__()
        self.args = args

    @abstractmethod
    async def generate(self, conversations: List[Conversation]) -> List[str]:
        pass


MOCK_SNIPPET = """
def solution(problem_input):
    return []
"""


class MockGenerator(GenerationWrapper):
    def __init__(self, _: GenWrapperArgs) -> None:
        self.mock_completions = []

    def set_mock_completions(self, completions: List[str]) -> None:
        self.mock_completions = completions

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        if self.mock_completions:
            return self.mock_completions
        return [MOCK_SNIPPET] * len(conversations)


MAX_RETRIES = 3


class OpenAIGenerationWrapper(GenerationWrapper):
    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__(args)
        api_key = args.dotenv.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAIGenerationWrapper")
        self.oai_client = AsyncOpenAI(api_key=api_key)
        self.model_name = args.model_id or "gpt-4o-mini"
        self.n_retries = MAX_RETRIES
        self.temperature = 0.2
        self.max_tokens = 4096

    @timeout(30)
    async def generate(self, conversations: List[Conversation]) -> List[str]:
        self.n_retries = MAX_RETRIES
        while True:
            completion_requests = []
            for conversation in conversations:
                stop_tokens = [128001, 128008, 128009]
                request = self.oai_client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    temperature=self.temperature,
                    max_completion_tokens=2048,
                    stop=["</solution>"],
                    extra_body={
                        "stop_token_ids": stop_tokens,
                        "skip_special_tokens": False,
                    },
                )
                completion_requests.append(request)
            try:
                logger.info(
                    f"Generating {len(completion_requests)} requests with {self.model_name}, max concurrent: {self.args.max_concurrent}"
                )
                results: List[ChatCompletion] = await gather_with_concurrency_limit(
                    self.args.max_concurrent, *completion_requests
                )
                if not results:
                    logger.error(results)
                    raise ValueError("No completions returned")
                assert all(
                    len(result.choices) > 0 for result in results
                ), "No completions returned"
                completions = [
                    result.choices[0].message.content
                    for result in results
                    if result.choices[0].message.content is not None
                ]
                return completions
            except Exception as e:
                logger.error(
                    f"Error while generating: {e}, retries left: {self.n_retries}"
                )
                self.n_retries -= 1
                if self.n_retries <= 0:
                    raise e


def _get_model_id(base_url: str):
    logger.info("Fetching models from OpenAI client")
    sync_client = OpenAI(base_url=base_url)
    all_models = sync_client.models.list()

    logger.info(f"Models: {[x.id for x in all_models.data]}")
    return all_models.data[0].id


class VLLMWrapper(OpenAIGenerationWrapper):
    def __init__(self, args: GenWrapperArgs) -> None:
        base_url = "http://localhost:8000/v1"
        self.oai_client = AsyncOpenAI(
            base_url=base_url,
        )
        self.temperature = 0.4
        self.model_name = args.model_id
        logger.info(f"Using model: {self.model_name}")
        # If true, list models and call the first one listed
        self.n_retries = MAX_RETRIES
        self.temperature = 0.2
        self.max_tokens = 4096
        self.args = args
        if args.model_id is None:
            logger.info(f"Using model: {self.model_name}")
            self.model_name = _get_model_id(base_url)

    async def _list_models(self):
        models_list = await self.oai_client.models.list()
        model_ids = [m.id for m in models_list.data]
        logger.info(f"Models: {model_ids}")
        self.model_name = model_ids[0]
        logger.info(f"Using model ID: {self.model_name}")


class OpenRouterGenerationWrapper(OpenAIGenerationWrapper):
    def __init__(self, args: GenWrapperArgs) -> None:
        super().__init__(args)
        api_key = args.dotenv.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouterGenerationWrapper"
            )
        self.oai_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.temperature = 0.4


class AnthropicGenerationWrapper(GenerationWrapper):
    def __init__(self, args: GenWrapperArgs) -> None:
        api_key = args.dotenv.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for AnthropicGenerationWrapper"
            )
        self.client = AsyncAnthropic(
            api_key=api_key,
        )

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        try:
            completion_requests = []
            for conversation in conversations:
                conversation = cast(List[MessageParam], conversation)
                request = self.client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    messages=conversation,
                    temperature=0,
                    max_tokens=4096,
                )
                completion_requests.append(request)
            results: List[Message] = await gather_with_concurrency_limit(
                4, *completion_requests
            )
            completions = [
                result.content[0].text  # type: ignore
                for result in results
                if result.content is not None
            ]
            return completions
        except AnthropicError as e:
            logger.error(f"Error while generating: {e}")
            return []


GEMINI_ROLE_MAP = {
    "system": "model",
    "user": "user",
    "assistant": "model",
}


def _openai_conversation_to_gemini(conversation: Conversation):
    return [
        google.genai.types.Content(
            role=GEMINI_ROLE_MAP[message["role"]],
            parts=[google.genai.types.Part(text=message["content"])],  # type: ignore
        )
        for message in conversation
    ]


class GeminiWrapper(GenerationWrapper):
    def __init__(self, args: GenWrapperArgs) -> None:
        api_key = args.dotenv.get("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY is required for GeminiWrapper")
        self.client = genai.Client(api_key=api_key)

    @timeout(30)
    async def generate(self, conversations: List[Conversation]):
        reqs = []
        for conv in [_openai_conversation_to_gemini(c) for c in conversations]:
            reqs.append(
                self.client.aio.models.generate_content(
                    model="gemini-1.5-flash-8b",
                    contents=conv,  # type: ignore
                )
            )
        try:
            results: List[
                google.genai.types.GenerateContentResponse
            ] = await gather_with_concurrency_limit(4, *reqs)
        except Exception as e:
            logger.error(f"Error while generating: {e}")
            return []
        for r in results:
            if r.text is None:
                logger.error(f"Null text for: {r}")
        return [result.text for result in results]


class RemoteModel(str, Enum):
    CLAUDE_3_5 = "claude-3-5"
    QWEN_QWQ = "qwen-qwq"
    DEEPSEEK_V3 = "deepseek-v3"
    GPT_4O_MINI = "gpt-4o-mini"
    MOCK = "mock"
    VLLM = "vllm"


@dataclass
class RemoteModelChoice:
    model: type[GenerationWrapper]
    args: Optional[GenWrapperArgs] = None


MODEL_CONFIGS: dict[str, RemoteModelChoice] = {
    RemoteModel.QWEN_QWQ: RemoteModelChoice(
        OpenRouterGenerationWrapper,
        GenWrapperArgs(model_id="qwen/qwq-32b-preview"),
    ),
    RemoteModel.DEEPSEEK_V3: RemoteModelChoice(
        OpenRouterGenerationWrapper,
        GenWrapperArgs(model_id="deepseek/deepseek-chat", max_concurrent=16),
    ),
    RemoteModel.CLAUDE_3_5: RemoteModelChoice(AnthropicGenerationWrapper),
    RemoteModel.GPT_4O_MINI: RemoteModelChoice(
        OpenAIGenerationWrapper,
        GenWrapperArgs(model_id="gpt-4o-mini", max_concurrent=32),
    ),
    RemoteModel.MOCK: RemoteModelChoice(MockGenerator),
    RemoteModel.VLLM: RemoteModelChoice(
        VLLMWrapper,
        GenWrapperArgs(max_concurrent=8),
    ),
}


def get_generation_wrapper(model_name: str) -> GenerationWrapper:
    model_name_enum = RemoteModel(model_name)
    current_dir = Path(__file__).resolve().parent.parent
    dotenv: Dict[str, str] = dotenv_values(os.path.join(current_dir, ".env"))  # type: ignore
    hf_token = dotenv["HF_TOKEN"]
    logger.info(f"Logging in with token: {hf_token}")
    login(token=hf_token, add_to_git_credential=True)
    config = MODEL_CONFIGS[model_name_enum]
    if config.args is None:
        config.args = GenWrapperArgs(dotenv=dotenv)
    else:
        config.args.dotenv = dotenv
    return config.model(config.args)
