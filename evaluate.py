import asyncio
import os
from typing import Dict, List, cast

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import dotenv_values
from fire import Fire
from rich.console import Console
from rich.progress import Progress

from evaluation.code_execution import EvalResult, evaluate_codecontests
from synthetic_data.generation import (
    MODEL_WRAPPER_CLASSES,
    GenerationSource,
    GenerationWrapper,
)
from synthetic_data.tasks import ALL_TASKS
from synthetic_data.utils import Conversation, dictl

current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv: Dict[str, str] = dotenv_values(os.path.join(current_dir, ".env"))  # type: ignore


async def sample_worker(
    model_wrapper: GenerationWrapper, prompt: Conversation, sample: Dict
):
    out = await model_wrapper.generate([prompt])
    return out, sample


async def main(
    batch_size: int = 4,
    task_name: str = "codecontests",
    gen_source: str = GenerationSource.GEMINI.value,
):
    console = Console()
    task = ALL_TASKS[task_name](console)

    console = Console()
    gen_source_enum = GenerationSource(gen_source)
    model_wrapper: GenerationWrapper = MODEL_WRAPPER_CLASSES[gen_source_enum](dotenv)

    eval_results: List[EvalResult] = []
    with Progress() as progress:
        for eval_task in task.eval_tasks:
            dataset = cast(Dataset, load_dataset(eval_task.dataset_uri))[
                eval_task.eval_split
            ]
            progress_bar_title = f"Eval task: {eval_task.dataset_uri}"
            prog_task = progress.add_task(progress_bar_title, total=len(dataset))
            for batch in dataset.iter(batch_size=batch_size):  # type: ignore
                all_futures = []
                samples_batch = dictl(batch)
                eval_prompts_batch = [
                    task.format_inference_conversation(sample, eval_task)
                    for sample in samples_batch
                ]
                for prompt, sample in zip(eval_prompts_batch, samples_batch):
                    all_futures.append(
                        asyncio.create_task(
                            sample_worker(model_wrapper, prompt, sample)
                        )
                    )
                results: List[tuple[str, dict]] = await asyncio.gather(*all_futures)
                eval_results.extend(evaluate_codecontests(console, results, eval_task))
                progress.advance(prog_task, 1)

    n_all_tests_passed = sum(
        sum(res.tests_pass) == len(res.tests_pass)
        for res in eval_results
    )
    n_tests_passed = sum(sum(res.tests_pass) for res in eval_results)
    total_n_tests = sum(len(res.tests_pass) for res in eval_results)
    console.print(
        f"Samples where all tests passed: {n_all_tests_passed}/{len(eval_results)}"
    )
    console.print(f"Total tests passed: {n_tests_passed}/{total_n_tests}")

    test_results_dicts = []
    for res in eval_results:
        test_results_dicts.append(
            {
                "prompt": res.prompt,
                "generated": res.generated_code,
                "test": res.test,
                "entry_point": res.entry_point,
                "err": res.err,
                "evaluation_results": res.tests_pass,
            }
        )

    test_results_pd = pd.DataFrame(test_results_dicts)
    test_results_pd.to_csv("test_results.csv")


Fire(main)
