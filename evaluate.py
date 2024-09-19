from fire import Fire
from dataclasses import dataclass
from typing import Type
from synthetic_data.generation import (
    GenerationWrapper,
    GeminiWrapper,
    OpenAIGenerationWrapper,
)

from datasets import load_dataset, Dataset
from loguru import logger


@dataclass
class ModelConfig:
    name: str
    wrapper: Type[GenerationWrapper]


MODEL_CONFIGS = [
    ModelConfig(name="Gemini", wrapper=GeminiWrapper),
    ModelConfig(name="Gemini", wrapper=OpenAIGenerationWrapper),
]


def main():
    dataset = []


Fire(main)
