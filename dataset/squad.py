from typing import Tuple
from datasets.formatting.formatting import LazyBatch
import json
from unidecode import unidecode

from transformers.tokenization_utils import PreTrainedTokenizer
from synthetic_data.utils import ExtractiveQARow, ShareGPTConversation
from synthetic_data.prompts import ENTITY_EXTRACTION_TUNING_INSTRUCTION

from model.utils import SmDataset


def format_squad_extractive(sample: dict) -> Tuple[str, str]:
    sample["json_schema"] = unidecode(sample["json_schema"])
    sample["context"] = unidecode(sample["context"])

    schema: dict = json.loads(sample['json_schema'])
    json_schema = {key: type(value).__name__ for key, value in schema.items()}
    input_out = f"Extract the following information using the provided schema: \t{json.dumps(json_schema)}\tand the following context: \t{sample["context"]}\n"
    labels_out = json.dumps(schema)

    return input_out, labels_out

class SquadExtractiveQADataModule(SmDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = "dataset_caches/parti"
        self.dataset_name = "roborovski/squad-extractive-qa"

    def process_samples_batch(self, samples: LazyBatch):
        inputs, labels = [], []
        for i in range(len(samples['id'])): # type: ignore
            sample = {k: v[i] for k, v in samples.items()}
            sample_input, sample_labels = format_squad_extractive(sample)
            inputs.append(sample_input)
            labels.append(sample_labels)
        
        return self._tokenize(inputs, labels)

class SquadDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = f"dataset_caches/squad_len_{max_token_length}"
        self.dataset_name = "rajpurkar/squad_v2"

    def process_samples_batch(self, samples: LazyBatch):
        inputs, labels = [], []
        for i in range(len(samples['id'])): # type: ignore
            sample = {k: v[i] for k, v in samples.items()}
            answers = sample["answers"]["text"] 
            sample_input = f"Question: {sample['question']}\nContext: {sample['context']}\n"
            sample_labels = answers[0] if answers else "Cannot answer this question"
            inputs.append(sample_input)
            labels.append(sample_labels)
        
        return self._tokenize(inputs, labels)

class DollyEntityExtractionDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = f"dataset_caches/dolly_entity_extraction"
        self.dataset_name = "roborovski/dolly-entity-extraction"
        self.cpu_count = 1

    def process_samples_batch(self, samples: LazyBatch):
        input_ids_out, labels_out = [], []
        for i in range(len(samples['context'])): # type: ignore
            sample = {k: v[i] for k, v in samples.items()}

            conversation: ShareGPTConversation = [{
                "role": "system",
                "content": ENTITY_EXTRACTION_TUNING_INSTRUCTION,
            },
            {
                "role": "user",
                "content": sample["context"],
            },
            {
                "role": "user",
                "content": sample["json_query"],
            }]

            conversation_completion = conversation + [{
                "role": "user",
                "content": sample["json_data"],
            }]

            conversation_completion_tokenized = self.tokenizer.apply_chat_template(conversation, tokenize=False)
            conversation_no_completion_tokenized = self.tokenizer.apply_chat_template(conversation_completion, tokenize=False)
            input_ids_out.append(conversation_no_completion_tokenized)
            labels_out.append(conversation_completion_tokenized)

        return input_ids_out, labels_out

