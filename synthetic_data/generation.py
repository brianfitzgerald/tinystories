from abc import ABC, abstractmethod
import openai
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Dict, Mapping, cast
from datasets import Dataset, concatenate_datasets
from anthropic.types.message_param import MessageParam
from anthropic.types.message import Message
from anthropic import AsyncAnthropic, AnthropicError
from loguru import logger
import google.generativeai as genai
from google.generativeai.types import ContentDict
from enum import Enum

from synthetic_data.utils import Conversation, gather_with_concurrency_limit


SHAREGPT_TO_OPENAI_ROLE = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


def upload_dataset(
    hf_dataset: Dataset, dataset_name: str, new_dataset_rows: List[Dict]
):
    dataset_new_rows = Dataset.from_list(new_dataset_rows)
    dataset_new_rows.to_csv(f"dataset_samples/{dataset_name}.csv")

    concat_dataset = concatenate_datasets([hf_dataset, dataset_new_rows])

    logger.info(f"Uploading {len(new_dataset_rows)} new rows to the Hub...")
    concat_dataset.push_to_hub(dataset_name)


class GenerationWrapper(ABC):
    """
    Abstract method for various ways of generating data.
    """

    @abstractmethod
    async def generate(self, conversations: List[Conversation]) -> List[str]:
        pass


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
        self.oai_client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = "gpt-3.5-turbo"
        self.max_concurrent = 16
        self.n_retries = MAX_RETRIES

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        while True:
            completion_requests = []
            for conversation in conversations:
                request = self.oai_client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    temperature=0.2,
                    max_tokens=512,
                )
                completion_requests.append(request)
            try:
                results: List[ChatCompletion] = await gather_with_concurrency_limit(
                    self.max_concurrent, *completion_requests
                )
                self.n_retries = MAX_RETRIES
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
                    return []


class OpenRouterGenerationWrapper(OpenAIGenerationWrapper):
    def __init__(self, dotenv: Dict[str, str]):
        api_key = dotenv.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouterGenerationWrapper"
            )
        self.oai_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = "meta-llama/llama-3.1-405b-instruct"
        self.max_concurrent = 32


class GroqGenerationWrapper(OpenAIGenerationWrapper):
    def __init__(self, dotenv: Dict[str, str]):
        api_key = dotenv.get("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("GROQ_API_KEY is required for OpenRouterGenerationWrapper")
        self.oai_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model_name = "mixtral-8x7b-32768"


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


def _chatgpt_to_gemini(conversation: Conversation) -> List[ContentDict]:

    out: List[ContentDict] = []
    for message in conversation:
        assert "content" in message, f"Message {message} does not have 'content' key"
        out.append(
            {
                "role": message["role"],
                "parts": [
                    {
                        "type": "text",
                        "content": message["content"],
                    }  # type: ignore
                ],
            }
        )

    return out


class GeminiWrapper(GenerationWrapper):

    def __init__(self) -> None:
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        completions = []
        for conversation in conversations:
            conv = _chatgpt_to_gemini(conversation)
            completion = self.model.generate_content_async(conv)
            completions.append(completion)

        return completions


class GenerationSource(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


MODEL_WRAPPER_CLASSES = {
    GenerationSource.OPENAI: OpenAIGenerationWrapper,
    GenerationSource.VLLM: VLLMWrapper,
    GenerationSource.OPENROUTER: OpenRouterGenerationWrapper,
    GenerationSource.GROQ: GroqGenerationWrapper,
    GenerationSource.ANTHROPIC: AnthropicGenerationWrapper,
    GenerationSource.GEMINI: GeminiWrapper,
}
