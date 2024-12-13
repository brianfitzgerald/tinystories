from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from typing import Coroutine, List, Dict, Mapping, cast
from datasets import Dataset, concatenate_datasets
from anthropic.types.message_param import MessageParam
from anthropic.types.message import Message
from anthropic import AsyncAnthropic, AnthropicError
from loguru import logger
from enum import Enum

from wrapt_timeout_decorator import timeout
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    gather_with_concurrency_limit,
)
import aiohttp
import google.genai as genai
import google.genai.types


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


class GenerationWrapper(ABC):
    """
    Abstract method for various ways of generating data.
    """

    @abstractmethod
    async def generate(self, conversations: List[Conversation]) -> List[str]:
        pass


MOCK_SNIPPET = """
def solution(problem_input):
    return []
"""


class MockGenerator(GenerationWrapper):
    def __init__(self, _: Dict[str, str]):
        self.mock_completions = []

    def set_mock_completions(self, completions: List[str]) -> None:
        self.mock_completions = completions

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        if self.mock_completions:
            return self.mock_completions
        return [MOCK_SNIPPET] * len(conversations)


class LocalGenerator(GenerationWrapper):
    def __init__(self, _: Dict[str, str]):
        pass

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        url = "http://0.0.0.0:8080/generate"
        payload = {"conversations": conversations, "max_length": 4096}
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                logger.info(f"Generate request status: {response.status}")
                if response.status != 200:
                    raise ValueError(f"Failed to generate: {response.status}")
                response_body = await response.json()
                completions = response_body["completions"]
                completions = [completions]
                return completions


class VLLMWrapper(GenerationWrapper):
    def __init__(self, dotenv: Dict[str, str]):
        from vllm import LLM, SamplingParams  # type: ignore

        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=128
        )
        logger.info("Loading local pipeline...")
        self.model = LLM(model="HuggingFaceH4/zephyr-7b-beta", dtype="auto")
        logger.info("Pipeline loaded.")
        self.tokenizer = self.model.get_tokenizer()

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        convs_formatted: str = [self.tokenizer.apply_chat_template(c, tokenize=False) for c in conversations]  # type: ignore
        responses = self.model.generate(convs_formatted, self.sampling_params)
        responses_text = [r.outputs[0].text for r in responses]
        responses_text = [r.replace("<|assistant|>", " ") for r in responses_text]
        return responses_text


MAX_RETRIES = 3


class OpenAIGenerationWrapper(GenerationWrapper):
    def __init__(self, dotenv: Mapping[str, str]):
        api_key = dotenv.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAIGenerationWrapper")
        self.oai_client = AsyncOpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini"
        self.max_concurrent = 8
        self.n_retries = MAX_RETRIES
        self.temperature = 0.2
        self.max_tokens = 4096

    @timeout(30)
    async def generate(self, conversations: List[Conversation]) -> List[str]:
        self.n_retries = MAX_RETRIES
        while True:
            completion_requests = []
            for conversation in conversations:
                request = self.oai_client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    temperature=self.temperature,
                    # max_completion_tokens=self.max_tokens,
                )
                completion_requests.append(request)
            try:
                logger.info(
                    f"Generating {len(completion_requests)} requests with {self.model_name}, max concurrent: {self.max_concurrent}"
                )
                results: List[ChatCompletion] = await gather_with_concurrency_limit(
                    self.max_concurrent, *completion_requests
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


class OpenRouterGenerationWrapper(OpenAIGenerationWrapper):
    def __init__(self, dotenv: Dict[str, str]):
        api_key = dotenv.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouterGenerationWrapper"
            )
        self.oai_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = "meta-llama/llama-3.1-70b-instruct"
        self.max_concurrent = 4
        self.temperature = 0.4


class AnthropicGenerationWrapper(GenerationWrapper):
    def __init__(self, dotenv: Dict[str, str]):
        api_key = dotenv.get("ANTHROPIC_API_KEY")
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
    def __init__(self, dotenv) -> None:
        api_key = dotenv.get("GOOGLE_API_KEY")
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
            results: List[google.genai.types.GenerateContentResponse] = (
                await gather_with_concurrency_limit(4, *reqs)
            )
        except Exception as e:
            logger.error(f"Error while generating: {e}")
            return []
        for r in results:
            if r.text is None:
                logger.error(f"Null text for: {r}")
        return [result.text for result in results]


class GenerationSource(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MOCK = "mock"
    LOCAL = "local"


MODEL_WRAPPER_CLASSES = {
    GenerationSource.OPENAI: OpenAIGenerationWrapper,
    GenerationSource.VLLM: VLLMWrapper,
    GenerationSource.OPENROUTER: OpenRouterGenerationWrapper,
    GenerationSource.ANTHROPIC: AnthropicGenerationWrapper,
    GenerationSource.GEMINI: GeminiWrapper,
    GenerationSource.MOCK: MockGenerator,
    GenerationSource.LOCAL: LocalGenerator,
}
