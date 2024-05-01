import asyncio
import os
from typing import Dict, List, Optional, cast
import pandas as pd

import fire
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from dotenv import dotenv_values
from huggingface_hub import login
from tqdm import tqdm
from synthetic_data.tasks import (
    DPODataTask,
    PromptUpsample,
    SFTDataTask,
    Toolformer,
    SyntheticToolCalls,
    SquadExtractiveQA
)

from synthetic_data.generation import (
    GenerationWrapper,
    GroqGenerationWrapper,
    AnthropicGenerationWrapper,
    OpenAIGenerationWrapper,
    OpenRouterGenerationWrapper,
    VLLMWrapper,
    upload_dataset,
)
from synthetic_data.utils import (
    GenerationSource,
    SeedDataFormat,
    print_result_dicts,
)

DATA_TASKS: Dict[str, type[SFTDataTask]] = {
    "toolformer": Toolformer,
    "prompt_upsample": PromptUpsample,
    "synthetic_tool_calls": SyntheticToolCalls,
    "squad_extractive_qa": SquadExtractiveQA,
}

MODEL_WRAPPER_CLASSES = {
    GenerationSource.OPENAI: OpenAIGenerationWrapper,
    GenerationSource.VLLM: VLLMWrapper,
    GenerationSource.OPENROUTER: OpenRouterGenerationWrapper,
    GenerationSource.GROQ: GroqGenerationWrapper,
    GenerationSource.ANTHROPIC: AnthropicGenerationWrapper,
}


def main(
    # n batches
    upload_every: int = 10,
    batch_size: int = 4,
    restart: bool = False,
    pairs: bool = False,
    resume_input_position: bool = True,
    generation_source: GenerationSource = GenerationSource.ANTHROPIC,
    task_name: str = "squad_extractive_qa",
    n_epochs: int = 1,
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.

    pairs - generate multiple completions and score them. Use the 2 widest scores
    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"

    task = DATA_TASKS[task_name]()
    is_dpo_task = isinstance(task, DPODataTask)

    if pairs and not isinstance(task, DPODataTask):
        raise ValueError("generate_pairs is only supported for DPO tasks.")

    print("Logging into the Hub...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv = dotenv_values(os.path.join(current_dir, ".env"))
    hf_token = dotenv["HF_TOKEN"]
    print(f"Logging in with token: {hf_token}")
    login(token=hf_token, add_to_git_credential=True)

    model_wrapper: GenerationWrapper = MODEL_WRAPPER_CLASSES[generation_source](dotenv)

    empty_dataset_format = (
        task.empty_dpo_dataset_format if pairs and is_dpo_task else task.empty_dataset_format
    )

    print("Loading output dataset...")
    if restart:
        output_dataset = Dataset.from_dict(empty_dataset_format)
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
            print("No existing dataset found, starting from scratch...")
            output_dataset = Dataset.from_dict(empty_dataset_format)

    input_dataset: Dataset
    input_dataset_location: Optional[str] = None
    if is_dpo_task and task.dpo_seed_cache_dataset_name and pairs:
        input_dataset_location = (
            f"{task.output_dataset_org}/{task.dpo_seed_cache_dataset_name}"
        )
    elif task.seed_data_format == SeedDataFormat.HF_DATASET:
        input_dataset_location = task.seed_data_location
    elif task.seed_data_format == SeedDataFormat.TSV:
        input_dataset_location = task.seed_data_location

    print(
        f"Loading input dataset: {input_dataset_location}, format: {task.seed_data_format.value}"
    )
    split = "train"
    assert input_dataset_location
    if (
        task.seed_data_format in (SeedDataFormat.HF_DATASET, SeedDataFormat.SYNTHETIC)
        or pairs
    ):
        if len(output_dataset) > 0 and resume_input_position:
            print(f"Resuming from position {len(output_dataset)}")
            split = f"train[{len(output_dataset)}:]"
        input_dataset = cast(Dataset, load_dataset(input_dataset_location, split=split))
    elif task.seed_data_format == SeedDataFormat.TSV:
        seed_data = pd.read_csv(input_dataset_location, on_bad_lines="skip")
        input_dataset = Dataset.from_pandas(seed_data)
    else:
        raise ValueError(f"Unrecognized seed_data_format: {task.seed_data_format}")

    print(f"Input dataset length: {len(input_dataset)} output: {len(output_dataset)}")
    new_dataset_rows: List[Dict] = []
    print("Running...")

    for i in range(n_epochs):

        # If we are generating the completion pair, i.e. the second step for DPO
        if is_dpo_task and pairs:
            for batch_idx, batch in enumerate(
                input_dataset.iter(batch_size=batch_size)
            ):
                batch = cast(Dict, batch)
                full_conversations_batch = task.format_dpo_input_conversations(batch)

                completions = asyncio.run(
                    model_wrapper.generate(full_conversations_batch)
                )

                output_rows_batch = task.get_dpo_dataset_output_batch(completions)
                print_result_dicts(output_rows_batch)
                new_dataset_rows.extend(output_rows_batch)
                if batch_idx % upload_every == 0 and batch_idx > 0:
                    upload_dataset(
                        output_dataset, task.output_dataset_name, new_dataset_rows
                    )
        else:
            for batch_idx, batch in enumerate(
                tqdm(input_dataset.iter(batch_size=batch_size))
            ):
                batch = cast(Dict, batch)
                full_conversations_batch = task.format_input_conversation(batch)

                print(f"Generating {len(full_conversations_batch)} completions...")
                completions = asyncio.run(
                    model_wrapper.generate(full_conversations_batch)
                )

                output_rows_batch = task.format_output_rows(completions)
                print_result_dicts(output_rows_batch)
                new_dataset_rows.extend(output_rows_batch)
                if batch_idx % upload_every == 0 and batch_idx > 0:
                    upload_dataset(
                        output_dataset, task.output_dataset_name, new_dataset_rows
                    )


if __name__ == "__main__":
    fire.Fire(main)
