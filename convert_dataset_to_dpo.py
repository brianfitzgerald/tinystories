from datasets import load_dataset, Dataset
from tqdm import tqdm

alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
goody = load_dataset("roborovski/open-goody2", split="train")


original_outputs = {item["instruction"]: item["output"] for item in alpaca}

merged_data = []
for sample in tqdm(goody):
    instruction = sample["instruction"]
    if instruction in original_outputs:
        merged_item = {
            "chosen": sample["response"],
            "input": instruction,
            "rejected": original_outputs[instruction],
        }
        print(merged_item)
        merged_data.append(merged_item)


merged_dataset = Dataset.from_dict(
    {
        "chosen": [item["chosen"] for item in merged_data],
        "input": [item["input"] for item in merged_data],
        "rejected": [item["rejected"] for item in merged_data],
    }
)

merged_dataset.push_to_hub("roborovski/open-goody2-dpo")
