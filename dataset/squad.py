from typing import Tuple
from datasets.formatting.formatting import LazyBatch
import json
from unidecode import unidecode

from transformers.tokenization_utils import PreTrainedTokenizer

from model.utils import SmDataset


def format_prompt(sample: dict) -> Tuple[str, str]:
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
            sample_input, sample_labels = format_prompt(sample)
            inputs.append(sample_input)
            labels.append(sample_labels)
        
        return self._tokenize(inputs, labels)


def format_squad(sample: dict) -> Tuple[str, str]:
    for k in ["context", "question", "answers"]:
        sample[k] = unidecode(sample[k])
    
    input_out = f"Question: {sample['question']}\nContext: {sample['context']}\n"
    answers = sample["answers"]["text"] 
    label_out = answers[0] if answers else "Cannot answer this question"

    return input_out, label_out


class SquadDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = "dataset_caches/squad"
        self.dataset_name = "rajpurkar/squad_v2"

    def process_samples_batch(self, samples: LazyBatch):
        inputs, labels = [], []
        for i in range(len(samples['id'])): # type: ignore
            sample = {k: v[i] for k, v in samples.items()}
            sample_input, sample_labels = format_squad(sample)
            inputs.append(sample_input)
            labels.append(sample_labels)
        
        return self._tokenize(inputs, labels)
