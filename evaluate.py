import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, cast

from datasets import Dataset, load_dataset
from dotenv import dotenv_values
from fire import Fire
from rich.console import Console
from rich.progress import Progress

from evaluation.code_execution import evaluate_sample_humaneval, print_code_snippet
from synthetic_data.generation import (
    MODEL_WRAPPER_CLASSES,
    GenerationSource,
    GenerationWrapper,
)
from synthetic_data.tasks import ALL_TASKS
from synthetic_data.utils import Conversation, ldictl

current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv: Dict[str, str] = dotenv_values(os.path.join(current_dir, ".env"))  # type: ignore


async def sample_worker(
    model_wrapper: GenerationWrapper, prompt: List[Conversation], sample: Dict
):
    out = await model_wrapper.generate([prompt])  # type: ignore
    return out, sample


def _print_test_results(err: Optional[str], results: List[bool], console: Console):
    result_str = "Results: "
    if err:
        result_str += f"[red]Execution error: {err}[/red]"
    else:
        passed = sum(results)
        total = len(results)
        result_str += f"{passed}/{total} tests passed"
    console.print(result_str)


@dataclass
class EvalResult:
    prompt: str
    generated: str
    test: str
    entry_point: str
    err: Optional[str]
    evaluation_results: List[bool]


async def main(
    max_concurrent: int = 16,
    task_name: str = "humaneval",
    generation_source: GenerationSource = GenerationSource.OPENAI,
):
    console = Console()
    task = ALL_TASKS[task_name](console)

    dataset = cast(Dataset, load_dataset(task.seed_data_location))["test"]

    console = Console()
    model_wrapper: GenerationWrapper = MODEL_WRAPPER_CLASSES[generation_source](dotenv)

    eval_results: List[EvalResult] = []

    with Progress() as progress:
        prog_task = progress.add_task("Evaluating", total=len(dataset))
        for batch in dataset.iter(batch_size=max_concurrent):  # type: ignore
            all_futures = []
            samples_batch = ldictl(batch)
            prompts_batch = [
                task.format_inference_conversation(sample) for sample in samples_batch
            ]
            for prompt, sample in zip(prompts_batch, samples_batch):
                all_futures.append(
                    asyncio.create_task(sample_worker(model_wrapper, prompt, sample))
                )

            results = await asyncio.gather(*all_futures)
            for result, sample in results:
                for generated in result:
                    console.print(f"Function: {sample['entry_point']}")
                    console.print(f"Canonical solution:")
                    print_code_snippet(sample["canonical_solution"], console)
                    generated_code = generated.replace("```", "").replace("python", "")
                    err, evaluation_results = evaluate_sample_humaneval(
                        sample["prompt"],
                        generated_code,
                        sample["test"],
                        sample["entry_point"],
                    )
                    console.print(f"Generated solution:")
                    print_code_snippet(generated_code, console)
                    console.print(f"Test code:")
                    print_code_snippet(sample["test"], console)
                    _print_test_results(err, evaluation_results, console)
                    console.print("=" * console.size.width)
                    progress.advance(prog_task, 1)
                    eval_results.append(
                        EvalResult(
                            sample["prompt"],
                            generated_code,
                            sample["test"],
                            sample["entry_point"],
                            err,
                            evaluation_results,
                        )
                    )

    n_all_tests_passed = sum(
        sum(res.evaluation_results) == len(res.evaluation_results)
        for res in eval_results
    )
    n_tests_passed = sum(sum(res.evaluation_results) for res in eval_results)
    total_n_tests = sum(len(res.evaluation_results) for res in eval_results)
    console.print(
        f"Samples where all tests passed: {n_all_tests_passed}/{len(eval_results)}"
    )
    console.print(f"Total tests passed: {n_tests_passed}/{total_n_tests}")


Fire(main)
