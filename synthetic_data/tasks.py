import json
import random
import sys
import traceback
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional

from datasets import Dataset
from loguru import logger
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown

from evaluation.code_execution import (
    CodeContestsProblem,
    CodeExecutionMode,
    EvalTask,
    HumanEvalProblem,
    MBPPProblem,
    _convert_mbpp_to_humaneval,
    evaluate_sample_against_codecontests_tests,
    evaluate_sample_ast,
    get_fn_name_from_assert,
    print_code_snippet,
)
from synthetic_data.conversion import chatml_to_conversation
from synthetic_data.generation import SHAREGPT_TO_OPENAI_ROLE
from synthetic_data.prompts import (
    format_codecontests_generation_prompt,
    format_dalle_prompt_template,
    format_entity_extraction_conversation_template,
    format_goody_prompt_template,
    format_humaneval_generation_prompt,
    get_tool_usage_prompt,
    get_toolformer_dpo_negative_completion_prompt,
)
from synthetic_data.tools import (
    DROPOUT_TYPES_JSON,
    DROPOUT_TYPES_TOOLFORMER,
    get_tool_description_json,
    get_tool_descriptions,
)
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    ExtractiveQARow,
    SyntheticToolCallDPORow,
    SyntheticToolCallRow,
    ToolFormerDPORow,
    ToolFormerRow,
    chunk_list,
    clean_message,
    dictl,
    extract_code_block,
    flatten_list,
    get_matches,
    is_valid_python,
)


class BaseTask(ABC):
    seed_data_format: DatasetFormat = DatasetFormat.SYNTHETIC
    seed_data_split = "train"

    seed_data_location: str
    output_dataset_name: str
    output_dataset_org: str = "roborovski"
    output_dataset_format: DatasetFormat = DatasetFormat.HF_DATASET

    dataset_columns: List[str] = []

    eval_tasks: List[EvalTask] = []

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

    def format_inference_conversation(
        self, sample: Dict, eval_task: Optional[EvalTask] = None
    ) -> Conversation:
        """
        Prompt template to use for generating initial seed data.
        """
        raise NotImplementedError

    def evaluate_completion(self, prompt: List[Conversation]):
        raise NotImplementedError


class DPOTask(BaseTask):

    n_dpo_completions: int = 2


class PromptUpsample(BaseTask):
    seed_data_format = DatasetFormat.TSV
    seed_data_location = "gs://openai-datasets/prompt-upsample/seed-data.tsv"
    output_dataset_name = "prompt-upsample"
    output_dataset_org = "openai"

    dataset_columns = ["Prompt", "Category", "Upsampled"]

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        prompts = batch["Prompt"]
        return [format_dalle_prompt_template(prompt) for prompt in prompts]


class Toolformer(DPOTask):
    seed_data_format = DatasetFormat.SYNTHETIC
    seed_data_location = "seed_data_files/domain_specific_tasks.csv"

    output_dataset_name = "synthetic-toolformer-dpo"
    dataset_org = "roborovski"

    dataset_columns = ["system", "question", "chosen", "rejected"]

    original_rows_batch: List[ToolFormerRow] = []

    def format_output_rows(self, completions: List[str]) -> List[Dict]:
        row = None
        for completion in completions:
            question, tool_call, call_result, agent_output = get_matches(completion)
            row = ToolFormerRow(question, tool_call, call_result, agent_output)

            assert is_valid_python(row.call_result)
            assert is_valid_python(row.tool_call)

        assert row is not None

        return [
            {
                "call_result": row.call_result,
                "tool_call": row.tool_call,
                "agent_output": row.agent_output,
            }
        ]

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
    seed_data_format = DatasetFormat.TSV
    seed_data_location = "data/domain_specific_tasks.csv"

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
    def format_output_rows(self, completions: List[str]) -> List[Dict]:
        # Get completions for each prompt, rank, and choos the 2 highest
        new_rows_batch = []
        for completion, original_row in zip(completions, self.original_rows_batch):
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

    seed_data_format = DatasetFormat.HF_DATASET
    seed_data_location = "rajpurkar/squad_v2"
    output_dataset_name = "squad-extractive-qa"
    output_dataset_org = "roborovski"

    dataset_columns = ["id", "context", "json_schema", "fields"]

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        prompts = batch["context"]
        self.contexts = batch["context"]
        return [
            format_entity_extraction_conversation_template(prompt) for prompt in prompts
        ]

    def format_output_rows(self, completions: List[str]) -> List[Dict]:
        parsed_rows: List[ExtractiveQARow] = []
        for i, completion in enumerate(completions):
            blocks = extract_code_block(completion)
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
    seed_data_format = DatasetFormat.HF_DATASET
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

    seed_data_format = DatasetFormat.HF_DATASET
    seed_data_location = "openai/openai_humaneval"
    seed_data_split = "test"
    output_dataset_name = "humaneval-dpo"

    dataset_columns = ["chosen", "rejected", "id", "prompt"]

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        fn_name, tests = batch["entry_point"], batch["test"]
        self.input_batch = dictl(batch)
        self.input_conversations = []

        for f, i in zip(fn_name, tests):
            self.input_conversations.extend(
                [format_humaneval_generation_prompt(f, i)]
                * self.n_completions_per_sample
            )
        return self.input_conversations

    def format_output_rows(self, completions: List[str]) -> List[Dict]:
        res, err, j = [], None, 0
        for i, completions_for_sample in enumerate(
            chunk_list(completions, self.n_completions_per_sample)
        ):
            sample = self.input_batch[i]
            best_completion, best_score = None, 0
            worst_completion, worst_score = None, sys.maxsize
            for j, completion in enumerate(completions_for_sample):
                completion = (
                    completion.replace(">>>", "\n")
                    .replace("```python", "")
                    .replace("```", "")
                )
                print_code_snippet(completion, self.console)
                # TODO fix
                full_code = sample["entry_point"] + "\n" + sample["test"]
                err, results = evaluate_sample_ast(full_code, 1)
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
                    "prompt": self.input_conversations[i + j],
                }
            )
        return res


class PositiveMode(Enum):
    # Generate N completions
    BEST_OF_N = "best_of_n"
    # Use the reference completion from codecontests
    REFERENCE_COMPLETION = "reference_completion"
    # Don't use a reference completion, instead return all completions as a list
    NO_COMPARISON = "no_comparison"


class CodeContests(HumanEval):

    seed_data_format = DatasetFormat.PARQUET
    seed_data_location = "dataset_samples/codeforces_problems_subset.parquet"
    seed_data_split = "train"
    output_dataset_name = "codecontests_dpo_v2"
    output_dataset_format = DatasetFormat.PARQUET

    dataset_columns = ["chosen", "rejected", "name", "prompt"]

    eval_tasks = [
        EvalTask(
            "humaneval",
            "openai/openai_humaneval",
            "humaneval",
            "exec",
            "test",
        ),
        EvalTask(
            "humaneval",
            "google-research-datasets/mbpp",
            "mbpp",
            "ast",
            "train",
        ),
    ]

    def __init__(self, console: Console) -> None:
        super().__init__(console)
        self.n_completions_per_sample = 1
        self.print_definitions = False
        self.positive_completion_mode = PositiveMode.REFERENCE_COMPLETION
        self.execution_mode: CodeExecutionMode = "exec"

    def format_inference_conversation(
        self, sample: Dict, eval_task: Optional[EvalTask] = None
    ) -> Conversation:
        if eval_task:
            problem, fn_name = None, None
            if eval_task.code_task_format == "humaneval":
                problem = HumanEvalProblem(**sample)
            elif eval_task.code_task_format == "mbpp":
                mbpp_problem = MBPPProblem(**sample)
                problem = _convert_mbpp_to_humaneval(mbpp_problem)
                fn_name = get_fn_name_from_assert(mbpp_problem.test_list[0])
                if not fn_name:
                    logger.error(
                        f"Could not find function name for problem {mbpp_problem.task_id}"
                    )
                    return [{"role": "system", "content": problem.prompt}]
                problem.entry_point = fn_name
            else:
                raise ValueError(
                    f"Invalid code task format: {eval_task.code_task_format}"
                )
            return format_codecontests_generation_prompt(problem.prompt, fn_name)
        return format_codecontests_generation_prompt(sample["description"], None)

    def format_input_conversation(self, batch: Dict) -> List[Conversation]:
        input_batch = dictl(batch)
        self.problems = [CodeContestsProblem(**row) for row in input_batch]
        self.input_conversations = []

        for i, problem in enumerate(self.problems):
            if self.print_definitions:
                self.console.print(
                    Markdown(
                        f"\n\n# Problem {i}, {problem.name}\n\n{problem.description}"
                    )
                )
            self.input_conversations.extend(
                [format_codecontests_generation_prompt(problem.description)]
                * self.n_completions_per_sample
            )
        return self.input_conversations

    def format_output_rows(self, completions: List[str]) -> List[Dict]:
        res = []
        for i, completions_for_sample in enumerate(
            chunk_list(completions, self.n_completions_per_sample)
        ):
            # Iterate through all completions, evaluate them and choose a positive and negative completion
            problem = self.problems[i]
            best_completion, best_score = None, 0
            worst_completion, worst_score = None, sys.maxsize
            for j, completion in enumerate(completions_for_sample):
                if not completion:
                    logger.error(f"Empty completion for problem {i}")
                    continue
                code_snippets = extract_code_block(completion, "python")
                if len(code_snippets) == 0:
                    logger.error(f"No code snippet found for completion {i}")
                    continue
                if len(code_snippets) != 1:
                    logger.warning(
                        f"Has more than one code snippet: {code_snippets} for completion {i}"
                    )
                completion = code_snippets[0]
                print_code_snippet(completion, self.console)
                test_results_for_completion, test_results_have_errors = (
                    evaluate_sample_against_codecontests_tests(
                        completion,
                        problem.public_tests["input"],
                        problem.public_tests["output"],
                        self.execution_mode,
                    )
                )
                flattened_tests = flatten_list(test_results_for_completion)
                n_tests_passed = sum(flattened_tests)
                logger.info(
                    f"Tests passed for completion {j}: {n_tests_passed} / {len(flattened_tests)}"
                )
                if self.positive_completion_mode == PositiveMode.BEST_OF_N:
                    if n_tests_passed > best_score:
                        best_score = n_tests_passed
                        best_completion = completion
                    if n_tests_passed < worst_score:
                        worst_score = n_tests_passed
                        worst_completion = completion
                    if best_completion is None or worst_completion is None:
                        logger.warning(
                            f"Could not find best or worst completion for problem {i}, scores: {best_score}, {worst_score}"
                        )
                        continue
                    if best_score == worst_score:
                        logger.warning(
                            f"Best and worst completions have the same score for problem {i}: {best_score}"
                        )
                        continue
                elif self.positive_completion_mode == PositiveMode.REFERENCE_COMPLETION:
                    if any(test_results_have_errors):
                        logger.warning(
                            f"Errors in tests for completion {j}, skipping..."
                        )
                        continue
                    best_completion = problem.solution
                    worst_completion = completion
                    best_score = 1
                    worst_score = 0
            if self.positive_completion_mode == PositiveMode.NO_COMPARISON:
                res.append(
                    {
                        "completions": completions_for_sample,
                    }
                )
            else:
                if self.positive_completion_mode == PositiveMode.BEST_OF_N:
                    logger.info(f"Adding row, best: {best_score}, worst: {worst_score}")
                res.append(
                    {
                        "chosen": best_completion,
                        "chosen_score": best_score,
                        "rejected": worst_completion,
                        "rejected_score": worst_score,
                        "name": problem.name,
                        "description": problem.description,
                    }
                )
        return res


class CodeContestsCoTSFT(CodeContests):
    output_dataset_name = "codecontests_cot_sft"
    dataset_columns = ["completions", "test_results", "name"]

    def __init__(self, console: Console) -> None:
        super().__init__(console)
        self.n_completions_per_sample = 1
        self.positive_completion_mode = PositiveMode.BEST_OF_N



ALL_TASKS: Dict[str, type[BaseTask]] = {
    "toolformer": Toolformer,
    "prompt_upsample": PromptUpsample,
    "synthetic_tool_calls": SyntheticToolCalls,
    "squad_extractive_qa": SquadExtractiveQA,
    "dolly_entity_extraction": DollyEntityExtraction,
    "goody": Goody2,
    "humaneval": HumanEval,
    "codecontests": CodeContests,
    "codecontests_cot_sft": CodeContestsCoTSFT,
}
