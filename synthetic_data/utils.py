import asyncio
from enum import Enum
import re
from typing import Dict, List, Optional, Sequence, Union, Any
import json
from loguru import logger
from pathlib import Path
import shutil
import ast
import traceback

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from tabulate import tabulate
from pydantic.dataclasses import dataclass


Conversation = Sequence[ChatCompletionMessageParam]
ShareGPTConversation = List[Dict[str, str]]

JSONSchemaKey = Union[str, int, float, bool, List[Any], Dict[str, Any], None]
JSONSchema = Dict[str, JSONSchemaKey]


@dataclass
class ExtractiveQARow:
    context: str
    json_query: JSONSchema
    json_data: JSONSchema


@dataclass
class ToolFormerRow:
    question: str
    call_result: str
    tool_call: str
    agent_output: str


@dataclass
class ToolFormerDPORow:
    question: str
    call_result_accepted: str
    tool_call_accepted: str
    agent_output_accepted: str
    call_result_rejected: str
    tool_call_rejected: str
    agent_output_rejected: str


@dataclass
class SyntheticToolCallRow:
    tool: str
    question: str
    tool_call: str
    call_result: str
    agent_output: str


@dataclass
class SyntheticToolCallDPORow:
    tool: str
    question: str
    tool_call_accepted: str
    call_result_accepted: str
    agent_output_accepted: str
    tool_call_rejected: str
    call_result_rejected: str
    agent_output_rejected: str


class DatasetFormat(Enum):
    TSV = "tsv"
    HF_DATASET = "hf_dataset"
    # Synthetic means the data is generated from a synthetic source, so no initial data is loaded
    SYNTHETIC = "synthetic"
    PARQUET = "parquet"
    CUSTOM = "custom"


def clean_message(message: JSONSchemaKey):
    """
    Clean up spaces, tabs, and newlines in a message with a JSON dict, so the dict is formatted nicely.
    """

    if isinstance(message, list):
        message = ", ".join(message)
    elif isinstance(message, bool):
        message = str(message)
    elif isinstance(message, (int, float)):
        message = str(message)
    elif isinstance(message, dict):
        message = json.dumps(message, indent=2)

    # Handle odd edge case where textwrap evaluates the value as a bool
    if message == "True" or message == "False":
        message = message.lower()
    if message is None:
        message = ""
    message = message.strip()
    message = message.replace("<|endoftext|>", "")
    message = re.sub(r"\n+|\t+", "", message)
    return message


def print_result_dicts(
    results: List[JSONSchema],
):
    if len(results) == 0:
        logger.warning("No results found, skipping print.")
        return
    columns = list(results[0].keys())
    new_dataset_row_elements = [
        [clean_message(row[column]) for column in columns] for row in results
    ]

    col_widths = [40] * len(columns)
    for i, column in enumerate(columns):
        col = results[0][column]
        if isinstance(col, str) and col.isdigit():
            col_widths[i] = 10

    logger.info(
        tabulate(
            new_dataset_row_elements,
            headers=columns,
            tablefmt="simple_grid",
            maxcolwidths=col_widths,
        )
    )


def extract_code_blocks(text):
    pattern = r"```(?:.*?)```|<code>(?:.*?)</code>"

    code_blocks = re.findall(pattern, text, re.DOTALL)

    clean_code_blocks = [
        block.strip("`").strip("<code>").strip("</code>").strip()
        for block in code_blocks
    ]

    code_blocks_str = "\n".join(clean_code_blocks)
    return code_blocks_str


LINE_REFIX_PATTERN = re.compile(
    r"(User:|Task:|API:|Call:|Result:|Agent:)\s*(.*)", re.IGNORECASE
)


def get_matches(text: str):
    matches = LINE_REFIX_PATTERN.findall(text)

    extracted_lines = []
    for match in matches:
        if match[1]:
            extracted_lines.append(match[1])
    return extracted_lines


# TODO refactor this into a base class with methods for prompting, parsing, etc.
# Would also have properties used for the dataclass
# don't want to do this yet until we have the full flow working


def extract_toolformer_row(text: str) -> ToolFormerRow:
    question, tool_call, call_result, agent_output = get_matches(text)
    return ToolFormerRow(question, call_result, tool_call, agent_output)


def is_valid_python(json_str: str):
    json_str = json_str.strip().replace("`", "")
    try:
        compile(json_str, "<string>", "single")
    except SyntaxError:
        traceback.print_exc()
        return False
    return True


def clean_example(text):
    cleaned_paragraph = re.sub(
        r"1\. Scenario:.*?Example API Call:|```.*?```", "", text, flags=re.DOTALL
    )
    return cleaned_paragraph.strip()


def recursive_json_parse(data: str) -> Optional[Union[Dict, str]]:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    if isinstance(data, dict):
        return {key: recursive_json_parse(value) for key, value in data.items()}
    return data


JSON_MATCH_PATTERN = r"```(?:json)?\n(.*?)\n```"


def extract_json_code_blocks(msg: str) -> List[JSONSchema]:
    """
    Parse out JSON code blocks from Markdown or plain text.
    Works even if the JSON is embedded deep in a string or with recursive serialization.
    """
    blocks = re.findall(JSON_MATCH_PATTERN, msg, re.DOTALL)

    res = []
    for match in blocks:
        match = match.strip()
        if match:
            json_obj = recursive_json_parse(match)
            res.append(json_obj)
    return res


def ensure_directory(directory: str, clear: bool = True):
    """
    Create a directory and parents if it doesn't exist, and clear it if it does.
    """
    Path(directory).mkdir(exist_ok=True, parents=True)
    if clear:
        shutil.rmtree(directory)
    Path(directory).mkdir(exist_ok=True, parents=True)


def extract_code_block(msg: str, language: str = "python") -> List[str]:
    """
    Extract a code block from a message. If none are found, treat the entire message as a code block.
    """
    match_pattern = rf"```(?:{language})?\n(.*?)\n```"
    blocks = re.findall(match_pattern, msg, re.DOTALL)

    if not blocks:
        msg = msg.lstrip("assistant\n\n")
        try:
            ast.parse(msg)
        except SyntaxError:
            return []
        blocks = [msg]

    return blocks


def extract_text_between_tags(input_text: str, tag_name: str):
    pattern = rf"<{tag_name}>(.*?)(?:</{tag_name}>|$)"
    return re.findall(pattern, input_text, re.DOTALL)


async def gather_with_concurrency_limit(n: int, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def get_class_name(obj):
    return obj.__class__.__name__


def ldictl(dict_of_lists: List[dict]):
    """
    List of dicts to dict of lists.
    """
    return (
        {key: [d[key] for d in dict_of_lists] for key in dict_of_lists[0]}
        if dict_of_lists
        else {}
    )


def dictl(dict_of_lists: dict) -> Sequence[dict]:
    """
    Dict of lists to list of dicts.
    """
    return [dict(zip(dict_of_lists.keys(), t)) for t in zip(*dict_of_lists.values())]


def chunk_list(xs: List, n: int):
    n = max(1, n)
    return (xs[i : i + n] for i in range(0, len(xs), n))


def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursively flatten sublists
        else:
            flat_list.append(item)
    return flat_list
