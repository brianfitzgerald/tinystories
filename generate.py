import asyncio
import os
from typing import Dict, List, Optional, cast
import pandas as pd
from loguru import logger
from rich.console import Console

import fire
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from dotenv import dotenv_values
from huggingface_hub import login
from tqdm import tqdm
from synthetic_data.tasks import (
    CodeContests,
    DPOTask,
    Goody2,
    PromptUpsample,
    BaseTask,
    Toolformer,
    SyntheticToolCalls,
    SquadExtractiveQA,
    DollyEntityExtraction,
    HumanEval,
)

from synthetic_data.generation import (
    MODEL_WRAPPER_CLASSES,
    GenerationWrapper,
    GenerationSource,
    MockGenerator,
    upload_dataset,
)
from synthetic_data.utils import (
    SeedDataFormat,
    print_result_dicts,
)

DATA_TASKS: Dict[str, type[BaseTask]] = {
    "toolformer": Toolformer,
    "prompt_upsample": PromptUpsample,
    "synthetic_tool_calls": SyntheticToolCalls,
    "squad_extractive_qa": SquadExtractiveQA,
    "dolly_entity_extraction": DollyEntityExtraction,
    "goody": Goody2,
    "humaneval": HumanEval,
    "codecontests": CodeContests,
}


def main(
    # n batches
    upload_every: int = 10,
    batch_size: int = 2,
    restart: bool = False,
    resume_input_position: bool = True,
    generation_source: GenerationSource = GenerationSource.OPENAI,
    task_name: str = "codecontests",
    n_epochs: int = 1,
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.

    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"

    console = Console()
    task = DATA_TASKS[task_name](console)
    split = task.seed_data_split
    logger.info("Logging into the Hub...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv = dotenv_values(os.path.join(current_dir, ".env"))
    hf_token = dotenv["HF_TOKEN"]
    logger.info(f"Logging in with token: {hf_token}")
    login(token=hf_token, add_to_git_credential=True)

    model_wrapper: GenerationWrapper = MODEL_WRAPPER_CLASSES[generation_source](dotenv)

    logger.info("Loading output dataset...")
    if restart:
        output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})
    else:
        try:
            output_dataset = cast(
                Dataset,
                load_dataset(
                    f"{task.output_dataset_org}/{task.output_dataset_name}",
                    split="train",
                ),
            )
            # TODO filter for rows that don't need completion
        except (EmptyDatasetError, ValueError):
            logger.info("No existing dataset found, starting from scratch...")
            output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})

    input_dataset: Dataset
    input_dataset_location: Optional[str] = None
    if task.seed_data_format in (
        SeedDataFormat.HF_DATASET,
        SeedDataFormat.PARQUET,
        SeedDataFormat.TSV,
    ):
        input_dataset_location = task.seed_data_location

    logger.info(
        f"Loading input dataset: {input_dataset_location}, format: {task.seed_data_format.value}"
    )
    assert input_dataset_location
    if task.seed_data_format in (SeedDataFormat.HF_DATASET, SeedDataFormat.SYNTHETIC):
        if len(output_dataset) > 0 and resume_input_position:
            logger.info(f"Resuming from position {len(output_dataset)}")
            split = f"{split}[{len(output_dataset)}:]"
        input_dataset = cast(Dataset, load_dataset(input_dataset_location, split=split))
    elif task.seed_data_format == SeedDataFormat.TSV:
        seed_data = pd.read_csv(input_dataset_location, on_bad_lines="skip")
        input_dataset = Dataset.from_pandas(seed_data)
    elif task.seed_data_format == SeedDataFormat.PARQUET:
        input_dataset = Dataset.from_parquet(input_dataset_location)  # type: ignore
    else:
        raise ValueError(f"Unrecognized seed_data_format: {task.seed_data_format}")

    input_dataset = task.preprocess_dataset(input_dataset)

    logger.info(
        f"Input dataset length: {len(input_dataset)} output: {len(output_dataset)}"
    )
    new_dataset_rows: List[Dict] = []
    logger.info(f"Generating with model {generation_source.value}")

    for _ in range(n_epochs):
        for batch_idx, batch in enumerate(
            tqdm(input_dataset.iter(batch_size=batch_size))
        ):
            batch = cast(Dict, batch)
            conversations_batch = task.format_input_conversation(batch)

            if isinstance(task, CodeContests) and isinstance(model_wrapper, MockGenerator):
                model_wrapper.set_mock_completions(
                    [
                        f"def solution(problem_input):\n    return []"
                        for _ in range(len(conversations_batch))
                    ]
                )

            logger.info(
                f"Generating batch of {len(conversations_batch)} completions..."
            )
            completions = asyncio.run(model_wrapper.generate(conversations_batch))

            output_rows_batch = task.format_output_rows(completions)
            print_result_dicts(output_rows_batch)
            new_dataset_rows.extend(output_rows_batch)
            if batch_idx % upload_every == 0 and batch_idx > 0:
                upload_dataset(
                    output_dataset, task.output_dataset_name, new_dataset_rows
                )


if __name__ == "__main__":
    fire.Fire(main)
