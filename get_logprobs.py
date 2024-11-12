import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm


def compute_logprobs(model, tokenizer, inputs):
    """
    Compute log probabilities of the completions.
    """
    with torch.no_grad():
        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_probs = -torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            inputs["input_ids"].view(-1),
            reduction="none",
        )
        log_probs = log_probs.view(
            inputs["input_ids"].size()
        )  # Reshape to match inputs
        return log_probs


def main(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_name="roborovski/codecontests-dpo",
    dataset_split="train",
    output_file="logprobs.safetensors",
):

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    dataset = load_dataset(dataset_name, dataset_split)
    all_logprobs = {}
    for i, example in enumerate(tqdm(dataset)):
        print(f"Processing example {i+1}/{len(dataset)}...")  # type: ignore
        logprobs = compute_logprobs(model, tokenizer, example["text"])
        all_logprobs[f"example_{i}"] = logprobs.cpu()

    # Save logprobs
    print("Saving logprobs...")
    save_file(all_logprobs, output_file)


if __name__ == "__main__":
    fire.Fire(main)
