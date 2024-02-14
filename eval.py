print("Loading dependencies...")
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from huggingface_hub import login, HfApi
import os
from dotenv import load_dotenv
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch
from PIL import Image
from pathlib import Path


from model.utils import TASK_PREFIX
import pandas as pd
from fire import Fire


def format_filename(string, limit=50):
    result = string.lower().replace(" ", "_")
    result = "".join(char if char.isalnum() or char == "_" else "" for char in result)
    result = result[:limit]
    return result


def main(
    checkpoint_dir: str = "checkpoints/best_model-v1.ckpt.dir",
    batch_size: int = 8,
    upload_to_hf: bool = False,
    generate_samples: bool = False,
):

    print("Loading model...")
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        checkpoint_dir
    )

    Path("samples").mkdir(exist_ok=True)

    drawbench_df: pd.DataFrame = pd.read_csv("data/drawbench.csv")
    if generate_samples:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

    if upload_to_hf:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(file_dir, ".env"))
        token = os.getenv("HF_TOKEN")
        print(f"Logging in with token: {token}")
        login(token=token, add_to_git_credential=True)
        hf_api = HfApi(token=token)
        print("Uploading model card...")
        hf_api.upload_file(
            path_or_fileobj="MODEL_CARD.md",
            path_in_repo="README.md",
            repo_id="roborovski/superprompt-v1",
            repo_type="model",
        )

        print("Uploading model...")
        model.push_to_hub("superprompt-v1")
        return

    for i in range(0, len(drawbench_df), batch_size):

        chunk = drawbench_df[i : i + batch_size]

        prompts_with_prefix = [TASK_PREFIX + sentence for sentence in chunk["Prompt"]]

        inputs = tokenizer(prompts_with_prefix, return_tensors="pt", padding=True)

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_return_sequences=1,
        )

        out = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        for j, (prompt, upsampled) in enumerate(zip(chunk["Prompt"], out)):
            print(f"Prompt: {prompt}\nUpsampled: {upsampled}\n\n")

            if generate_samples:
                for k, txt in enumerate([prompt, upsampled]):
                    print(f"Generating sample for: {txt}")
                    image: Image.Image = pipe(txt).images[0]  # type: ignore
                    prompt_fmt = format_filename(txt)
                    label = "prompt" if k == 0 else "upsampled"
                    image.save(f"samples/{i}_{j}_{prompt_fmt}_{label}.png")


if __name__ == "__main__":
    Fire(main)
