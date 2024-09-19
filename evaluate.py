from fire import Fire
from dataclasses import dataclass
from typing import Type, Dict
from synthetic_data.generation import (
    GenerationWrapper,
    GeminiWrapper,
    OpenAIGenerationWrapper,
)
import os
from dotenv import dotenv_values


from datasets import load_dataset, Dataset
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ModelConfig:
    name: str
    wrapper: GenerationWrapper


current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv: Dict[str, str] = dotenv_values(os.path.join(current_dir, ".env"))  # type: ignore

MODEL_CONFIGS = [
    ModelConfig(name="Gemini", wrapper=GeminiWrapper()),
    ModelConfig(name="GPT-4o", wrapper=OpenAIGenerationWrapper(dotenv)),
]


def evaluate_sample(model_config: ModelConfig, sample: dict):
    pass


def main(max_concurrent: int = 4):
    dataset = load_dataset("roborovski/dolly-entity-extraction")

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:

        all_futures = []

        for sample in dataset:
            for model_config in MODEL_CONFIGS:
                future = executor.submit(evaluate_sample, model_config, sample)
                all_futures.append(future)

        for future in as_completed(all_futures):
            print(future.result())


Fire(main)
