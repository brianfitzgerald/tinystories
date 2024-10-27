from abc import ABC
import sys
import random
import traceback
from typing import Dict, List, Optional
from evaluation.code_execution import evaluate_sample, print_code_snippet
from synthetic_data.conversion import chatml_to_conversation
from synthetic_data.generation import SHAREGPT_TO_OPENAI_ROLE
from synthetic_data.prompts import (
    ENTITY_EXTRACTION_TUNING_INSTRUCTION,
    format_dalle_prompt_template,
    format_code_generation_prompt,
    format_entity_extraction_conversation_template,
    get_tool_usage_prompt,
    get_toolformer_dpo_negative_completion_prompt,
    format_goody_prompt_template,
)
from synthetic_data.tools import (
    DROPOUT_TYPES_JSON,
    DROPOUT_TYPES_TOOLFORMER,
    get_tool_description_json,
    get_tool_descriptions,
)
import json
from pydantic import ValidationError
from datasets import Dataset
from loguru import logger
from rich.console import Console

from synthetic_data.utils import (
    Conversation,
    SeedDataFormat,
    SyntheticToolCallRow,
    SyntheticToolCallDPORow,
    ToolFormerDPORow,
    ToolFormerRow,
    ExtractiveQARow,
    chunk_list,
    dictl,
    extract_json_code_blocks,
    is_valid_python,
    clean_message,
    get_matches,
    ldictl,
)


class BaseTask(ABC):
    seed_data_format: SeedDataFormat = SeedDataFormat.SYNTHETIC
    seed_data_split = "train"

    seed_data_location: str
    output_dataset_name: str
    output_dataset_org: str = "roborovski"

    dataset_columns: List[str] = []

    def __init__(self, console: Console) -> None:
        super().__init__()
        self.console = console

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        """
        Prompt template to use for generating initial seed data.
        """
        raise NotImplementedError

    def format_output_rows(self, completions: List[str]) -> List[Dict]:
        """
        Take the completed conversation and format it into the final dataset format.
        """
        raise NotImplementedError

    def format_inference_conversation(self, batch: Dict) -> List[Conversation]:
        """
        Prompt template to use for generating initial seed data.
        """
        raise NotImplementedError

    def evaluate_completion(self, prompt: List[Conversation]):
        raise NotImplementedError


class DPOTask(BaseTask):

    n_dpo_completions: int = 2


class PromptUpsample(BaseTask):
    seed_data_format = SeedDataFormat.TSV
    seed_data_location = "gs://openai-datasets/prompt-upsample/seed-data.tsv"
    output_dataset_name = "prompt-upsample"
    output_dataset_org = "openai"

    dataset_columns = ["Prompt", "Category", "Upsampled"]

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        prompts = batch["Prompt"]
        return [format_dalle_prompt_template(prompt) for prompt in prompts]


class Toolformer(DPOTask):
    seed_data_format = SeedDataFormat.SYNTHETIC
    seed_data_location = "seed_data_files/domain_specific_tasks.csv"

    output_dataset_name = "synthetic-toolformer-dpo"
    dataset_org = "roborovski"

    dataset_columns = ["system", "question", "chosen", "rejected"]

    original_rows_batch: List[ToolFormerRow] = []

    def format_output_rows(self, completions_batch: List[str]) -> Dict:
        for completion in completions_batch:
            question, tool_call, call_result, agent_output = get_matches(completion)
            row = ToolFormerRow(question, tool_call, call_result, agent_output)

            assert is_valid_python(row.call_result)
            assert is_valid_python(row.tool_call)

        return {
            "call_result": row.call_result,
            "tool_call": row.tool_call,
            "agent_output": row.agent_output,
        }

    # TODO re add the original row to the input conv
    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        conversations = batch["conversations"]

        dropout_types_batch = random.choices(
            DROPOUT_TYPES_TOOLFORMER, k=len(conversations)
        )

        conversations_batch: List[Conversation] = []
        original_rows_batch: List[ToolFormerRow] = []
        for i, conversation in enumerate(conversations):
            messages = [message["content"] for message in conversation]
            for _ in range(self.n_dpo_completions):
                question, tool_call, call_result, agent_output = messages
                original_row = ToolFormerRow(
                    question=question,
                    call_result=call_result,
                    tool_call=tool_call,
                    agent_output=agent_output,
                )
                tool_descriptions = get_tool_descriptions(dropout_types_batch[i])
                conversation = get_toolformer_dpo_negative_completion_prompt(
                    question, tool_descriptions
                )
                conversations_batch.append(conversation)
                original_rows_batch.append(original_row)
        self.original_rows_batch = original_rows_batch
        return conversations_batch

    def get_dpo_dataset_output_batch(self, completions_batch: List[str]) -> List[Dict]:
        # Get completions for each prompt, rank, and choos the 2 highest
        new_rows_batch = []
        dpo_rows_batch: List[ToolFormerRow] = []
        for completion, original_row in zip(
            completions_batch, self.original_rows_batch
        ):
            try:
                tool_call, call_result, agent_output = get_matches(completion)

                dpo_row = ToolFormerRow(
                    original_row.question,
                    call_result,
                    tool_call,
                    agent_output,
                )

                dpo_rows_batch.append(dpo_row)

                if dpo_row.tool_call == original_row.tool_call:
                    continue

                output_row = ToolFormerDPORow(
                    question=original_row.question,
                    call_result_accepted=original_row.call_result,
                    tool_call_accepted=original_row.tool_call,
                    agent_output_accepted=original_row.agent_output,
                    call_result_rejected=call_result,
                    tool_call_rejected=tool_call,
                    agent_output_rejected=agent_output,
                )

                row_dict = output_row.__dict__
                new_rows_batch.append(row_dict)
            except Exception as e:
                logger.error(f"Error in parsing completion: {e}")
                continue

        return new_rows_batch


class SyntheticToolCalls(DPOTask):
    seed_data_format = SeedDataFormat.TSV
    seed_data_location = "seed_data_files/domain_specific_tasks.csv"

    dataset_org = "roborovski"

    output_dataset_name = "synthetic-tool-calls-v2-dpo-pairs"

    dataset_columns = ["tool", "question", "tool_call", "call_result"]

    original_rows_batch: List[SyntheticToolCallRow] = []

    # TODO fix
    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        conversations = []
        for category, task in zip(batch["Category"], batch["Task"]):
            conversations.append(get_tool_usage_prompt(category, task))

        n_samples = len(batch["tool"])
        dropout_types_batch = random.choices(DROPOUT_TYPES_JSON, k=n_samples)

        conversations_batch: List[Conversation] = []
        original_rows_batch: List[SyntheticToolCallRow] = []
        for i in range(n_samples):
            for _ in range(self.n_dpo_completions):
                try:
                    question = batch["question"][i]
                    tool = batch["tool"][i]
                    original_row = SyntheticToolCallRow(
                        tool=tool,
                        question=question,
                        tool_call=batch["tool_call"][i],
                        call_result=batch["call_result"][i],
                        agent_output=batch["agent_output"][i],
                    )
                    tool_descriptions = get_tool_description_json(
                        tool, dropout_types_batch[i]
                    )
                    conversation = get_toolformer_dpo_negative_completion_prompt(
                        question, tool_descriptions, True
                    )
                    conversations_batch.append(conversation)
                    original_rows_batch.append(original_row)
                except Exception as e:
                    logger.error(f"Error in formatting DPO input: {e}")
                    traceback.print_exc()
                    continue
        self.original_rows_batch = original_rows_batch
        return conversations_batch

    # TODO fix
    def format_output_rows(self, completions_batch: List[str]) -> List[Dict]:
        # Get completions for each prompt, rank, and choos the 2 highest
        new_rows_batch = []
        for completion, original_row in zip(
            completions_batch, self.original_rows_batch
        ):
            try:
                tool_call, call_result, agent_output = get_matches(completion)

                if (
                    tool_call == original_row.tool_call
                    and agent_output == original_row.agent_output
                ):
                    continue

                dpo_row = SyntheticToolCallDPORow(
                    original_row.tool,
                    original_row.question,
                    original_row.tool_call,
                    original_row.call_result,
                    original_row.agent_output,
                    tool_call,
                    call_result,
                    agent_output,
                )
                row_dict = dpo_row.__dict__
                new_rows_batch.append(row_dict)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in parsing completion: {e}")
                continue

        return new_rows_batch


class GlaiveDPO(DPOTask):
    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        glaive_conversations = [
            chatml_to_conversation(chat, system)
            for chat, system in zip(batch["chat"], batch["system"])
        ]
        prompt_conversations: List[Conversation] = []
        for conversation in glaive_conversations:
            completion_conv = []
            for msg in conversation:
                if msg["from"] == "gpt":
                    break
                completion_conv.append(
                    {
                        "role": SHAREGPT_TO_OPENAI_ROLE[msg["from"]],
                        "content": msg["value"],
                    }
                )
            prompt_conversations.append(completion_conv)
        return prompt_conversations

    def _clean_glaive_conversation(
        self, completions: List[str], glaive_conversations: List[Dict]
    ) -> List[Dict]:
        new_rows_batch = []
        for completion, glaive_conversation in zip(completions, glaive_conversations):
            system_msg, user_msg, accepted_msg, rejected_msg = (
                "",
                "",
                "",
                "",
            )
            for msg in glaive_conversation:
                role, content = msg["from"], msg["value"]
                if role == "system":
                    system_msg = clean_message(content)
                if role == "human":
                    user_msg = clean_message(content)
                if role == "gpt":
                    accepted_msg = clean_message(content)
                    rejected_msg = completion
                    break
            new_rows_batch.append(
                {
                    "system": system_msg,
                    "question": user_msg,
                    "chosen": accepted_msg,
                    "rejected": rejected_msg,
                }
            )

        # TODO fix
        return [
            {"role": SHAREGPT_TO_OPENAI_ROLE[msg["from"]], "content": msg["value"]}
            for msg in new_rows_batch
        ]


class SquadExtractiveQA(BaseTask):
    """
    Performs the following steps:
    - Generate JSON struct from the prompt.
    - Dropout certain fields from the JSON struct.
    - Save the original JSON struct and the dropout JSON struct.
    """

    seed_data_format = SeedDataFormat.HF_DATASET
    seed_data_location = "rajpurkar/squad_v2"
    output_dataset_name = "squad-extractive-qa"
    output_dataset_org = "roborovski"

    dataset_columns = ["id", "context", "json_schema", "fields"]

    def __init__(self) -> None:
        super().__init__()
        self.ids = []
        self.contexts = []

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        prompts = batch["context"]
        self.contexts = batch["context"]
        return [
            format_entity_extraction_conversation_template(prompt) for prompt in prompts
        ]

    def format_inference_conversation(self, batch: Dict) -> List[Conversation]:
        prompts = batch["context"]
        self.contexts = batch["context"]
        return [
            [
                {"role": "system", "content": ENTITY_EXTRACTION_TUNING_INSTRUCTION},
            ]
        ]

    def format_output_rows(self, completions_batch: List[str]) -> List[Dict]:
        parsed_rows: List[ExtractiveQARow] = []
        for i, completion in enumerate(completions_batch):
            blocks = extract_json_code_blocks(completion)
            if len(blocks) != 2:
                logger.warning(f"Could not extract JSON from completion: {completion}")
                continue
            json_data, json_query = blocks
            if not isinstance(json_data, dict) or not isinstance(json_query, dict):
                logger.warning(f"Invalid JSON schema for completion: {completion}")
                continue
            field_names = set(json_data.keys()).union(json_query.keys())

            if len(field_names) == 0:
                logger.warning(f"Empty JSON schema for completion: {completion}")
                continue

            try:
                qa_row = ExtractiveQARow(
                    self.contexts[i],
                    json_query,
                    json_data,
                )
            except ValidationError as e:
                logger.warning(f"Error in formatting completion: {e}")
                continue
            parsed_rows.append(qa_row)

        out_rows = []
        for row in parsed_rows:
            row = row.__dict__
            for key in ["json_query", "json_data"]:
                row[key] = json.dumps(row[key])
            out_rows.append(row)
        return out_rows


def _filter_dolly_row(row: Dict) -> bool:
    ctx = row["context"]
    return ctx is not None and ctx != ""


class DollyEntityExtraction(SquadExtractiveQA):
    seed_data_location = "databricks/databricks-dolly-15k"
    output_dataset_name = "dolly-entity-extraction"

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        logger.info(f"Original dataset length: {len(dataset)}")
        dataset = dataset.filter(_filter_dolly_row)
        logger.info(f"Filtered dataset length: {len(dataset)}")
        return dataset


class Goody2(BaseTask):
    seed_data_format = SeedDataFormat.HF_DATASET
    seed_data_location = "yahma/alpaca-cleaned"
    dataset_columns = ["instruction", "response"]
    output_dataset_name = "open-goody2"

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        self.instructions = batch["instruction"]
        return [
            format_goody_prompt_template(instruction)
            for instruction in self.instructions
        ]

    def format_output_rows(self, completions: List[str]) -> List[Dict]:
        res = []
        for completion, instruction in zip(completions, self.instructions):
            res.append(
                {
                    "instruction": instruction,
                    "response": completion,
                }
            )
        return res


class HumanEval(DPOTask):

    def __init__(self, console) -> None:
        super().__init__(console)
        self.n_completions_per_sample = 4

    seed_data_format = SeedDataFormat.HF_DATASET
    seed_data_location = "openai/openai_humaneval"
    seed_data_split = "test"
    output_dataset_name = "humaneval-dpo-pairs"

    dataset_columns = ["chosen", "rejected", "id", "prompt"]

    def format_inference_conversation(self, sample: Dict) -> Conversation:
        fn_name, tests = sample["entry_point"], sample["test"]
        return format_code_generation_prompt(fn_name, tests)

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        fn_name, tests = batch["entry_point"], batch["test"]
        self.input_batch = dictl(batch)
        self.input_conversations = []

        for f, i in zip(fn_name, tests):
            self.input_conversations.extend(
                [format_code_generation_prompt(f, i)] * self.n_completions_per_sample
            )
        return self.input_conversations

    def format_output_rows(self, completions: List[str]) -> List[Dict]:
        res = []
        for i, completions_for_sample in enumerate(chunk_list(completions, self.n_completions_per_sample)):
            sample = self.input_batch[i]
            best_completion, best_score = None, 0
            worst_completion, worst_score = None, sys.maxsize
            for j, completion in enumerate(completions_for_sample):
                completion = completion.replace(">>>", "\n").replace("```python", "").replace("```", "")
                print_code_snippet(completion, self.console)
                err, results = evaluate_sample(
                    completion, sample["entry_point"], sample["test"], sample["entry_point"]
                )
                tests_passed = sum(results)
                if tests_passed > best_score:
                    best_score = tests_passed
                    best_completion = completion
                if tests_passed < worst_score:
                    worst_score = tests_passed
                    worst_completion = completion
            res.append(
                {
                    "chosen": best_completion,
                    "rejected": worst_completion,
                    "task_id": sample["task_id"],
                    "error": err,
                    "prompt": self.input_conversations[i + j]
                }
            )
        return res
