import asyncio
from typing import Dict, List, Optional, cast

import fire
import pandas as pd
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from loguru import logger
from rich.console import Console
from tqdm import tqdm

from synthetic_data.generation import (
    get_generation_wrapper,
    MockGenerator,
    RemoteModel,
    save_output_dataset,
)
from synthetic_data.tasks import ALL_TASKS
from synthetic_data.utils import DatasetFormat


def main(
    upload_every_n_batches: int = 10,
    batch_size: int = 8,
    restart: bool = False,
    resume_input_position: bool = True,
    model: str = RemoteModel.DEEPSEEK_V3.value,
    task_name: str = "codecontests_cot_sft",
    n_epochs: int = 5,
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.
    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"

    console = Console()
    task = ALL_TASKS[task_name](console)
    split = task.seed_data_split

    generation_wrapper = get_generation_wrapper(model)
    output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})

    logger.info("Loading output dataset...")
    if not restart:
        if task.output_dataset_format == DatasetFormat.HF_DATASET:
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
        elif task.output_dataset_format == DatasetFormat.PARQUET:
            try:
                output_dataset = cast(
                    Dataset, Dataset.from_parquet(f"{task.output_dataset_name}.parquet")
                )
            except FileNotFoundError:
                logger.info("No existing dataset found, starting from scratch...")
        else:
            output_dataset = Dataset.from_dict({k: [] for k in task.dataset_columns})

    input_dataset: Dataset
    input_dataset_location: Optional[str] = None
    if task.seed_data_format in (
        DatasetFormat.HF_DATASET,
        DatasetFormat.PARQUET,
        DatasetFormat.TSV,
    ):
        input_dataset_location = task.seed_data_location

    logger.info(
        f"Loading input dataset: {input_dataset_location}, format: {task.seed_data_format.value}"
    )
    assert input_dataset_location
    split = None
    if len(output_dataset) > 0 and resume_input_position:
        logger.info(f"Resuming from position {len(output_dataset)}")
        split = f"{split}[{len(output_dataset)}:]"
    if task.seed_data_format == DatasetFormat.HF_DATASET:
        input_dataset = cast(Dataset, load_dataset(input_dataset_location, split=split))
    elif task.seed_data_format == DatasetFormat.TSV:
        seed_data = pd.read_csv(input_dataset_location, on_bad_lines="skip")
        input_dataset = Dataset.from_pandas(seed_data)
    elif task.seed_data_format == DatasetFormat.PARQUET:
        input_dataset_pd = pd.read_parquet(input_dataset_location)
        if len(output_dataset) > 0 and resume_input_position:
            input_dataset = Dataset.from_pandas(
                input_dataset_pd.iloc[len(output_dataset) :]
            )
        else:
            input_dataset = Dataset.from_pandas(input_dataset_pd)
    else:
        raise ValueError(f"Unrecognized seed_data_format: {task.seed_data_format}")

    input_dataset = task.preprocess_dataset(input_dataset)

    logger.info(
        f"Input dataset length: {len(input_dataset)} Output dataset: {len(output_dataset)}"
    )
    new_dataset_rows: List[Dict] = []
    logger.info(f"Generating with model {model} for task {task_name}")

    n_batches = len(input_dataset) // batch_size

    for _ in range(n_epochs):
        for batch_idx, batch in enumerate(
            tqdm(input_dataset.iter(batch_size=batch_size), total=n_batches)
        ):
            batch = cast(Dict, batch)
            conversations_batch = task.format_input_conversation(batch)

            if isinstance(generation_wrapper, MockGenerator):
                generation_wrapper.set_mock_completions(
                    [
                        f"def solution(problem_input):\n    return []"
                        for _ in range(len(conversations_batch))
                    ]
                )

            logger.info(
                f"Generating batch of {len(conversations_batch)} completions..."
            )
            try:
                completions = asyncio.run(
                    generation_wrapper.generate(conversations_batch)
                )
            except TimeoutError:
                logger.error(f"Timeout error on batch {batch_idx}")
                continue

            output_rows_batch = task.format_output_rows(completions)
            new_dataset_rows.extend(output_rows_batch)
            if batch_idx % upload_every_n_batches == 0 and batch_idx > 0:
                save_output_dataset(
                    output_dataset,
                    task.output_dataset_name,
                    new_dataset_rows,
                    task.output_dataset_format,
                )


if __name__ == "__main__":
    fire.Fire(main)
