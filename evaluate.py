from fire import Fire
from dataclasses import dataclass
from typing import Dict, List, cast
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

from synthetic_data.tasks import DollyEntityExtraction, HumanEval, BaseTask
from synthetic_data.utils import Conversation


@dataclass
class ModelConfig:
    name: str
    wrapper: GenerationWrapper


@dataclass
class EvalTask:
    name: str
    dataset_uri: str
    task_class: type[BaseTask]


current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv: Dict[str, str] = dotenv_values(os.path.join(current_dir, ".env"))  # type: ignore


MODEL_CONFIGS = [
    ModelConfig(name="Gemini", wrapper=GeminiWrapper()),
    ModelConfig(name="GPT-4o", wrapper=OpenAIGenerationWrapper(dotenv)),
]


def evaluate_sample(model_config: ModelConfig, prompt: List[Conversation]):
    out = model_config.wrapper.generate(prompt)
    return out


def _to_list_of_samples(dict_of_lists):
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


TASKS = [
    EvalTask(
        "dolly-entity-extraction",
        "roborovski/dolly-entity-extraction",
        DollyEntityExtraction,
    ),
    EvalTask(
        "humaneval",
        "openai/openai_humaneval",
        HumanEval
    )
]


def main(max_concurrent: int = 4, task_name: str = "dolly-entity-extraction"):
    dataset = cast(Dataset, load_dataset("roborovski/dolly-entity-extraction"))["train"]

    eval_task = next(t for t in TASKS if t.name == task_name)
    task = eval_task.task_class()

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:

        all_futures = []

        for batch in dataset.iter(batch_size=128):  # type: ignore
            prompts_batch = [
                task.format_inference_conversation(sample)
                for sample in _to_list_of_samples(batch)
            ]
            for model_config in MODEL_CONFIGS:
                for prompt in prompts_batch:
                    future = executor.submit(evaluate_sample, model_config, prompt)
                    all_futures.append(future)

        for future in as_completed(all_futures):
            print(future.result())


Fire(main)
