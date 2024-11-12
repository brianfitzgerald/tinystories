import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm

def _get_logprobs(model, tokenizer, batch):
    tokenized = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    outputs = model(**tokenized, labels=tokenized["input_ids"])
    return outputs.logits.cpu()

def main(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_name="roborovski/codecontests-dpo",
    output_file="logprobs.safetensors",
    batch_size=2,
    max_samples=100
):

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    model.eval()
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split="train")
    all_logprobs = {}
    with torch.no_grad():
        batch = []
        for i, example in enumerate(tqdm(dataset)):
            batch.append((example["chosen"], example["rejected"]))
            if len(batch) == batch_size:
                chosen_logprobs = [_get_logprobs(model, tokenizer, x[0]) for x in batch]
                rejected_logprobs = [_get_logprobs(model, tokenizer, x[1]) for x in batch]
                for i in range(len(batch)):
                    all_logprobs[f"{i}_chosen"] = chosen_logprobs[i].cpu()
                    all_logprobs[f"{i}_rejected"] = rejected_logprobs[i].cpu()
                batch = []
            if i == max_samples:
                break

    # Save logprobs
    print("Saving file...")
    save_file(all_logprobs, output_file)


if __name__ == "__main__":
    fire.Fire(main)
