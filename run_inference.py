import torch
from transformers.generation.configuration_utils import GenerationConfig
from threading import Thread

import fire
from typing import Optional, List

from transformers.generation.streamers import TextIteratorStreamer
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
from train_trl import LLAMA_CONFIG, TrainerWrapper
from loguru import logger


class CompletionRequest(BaseModel):
    prompts: List[str]
    max_length: Optional[int] = None


def do_inference_api(
    prompts: List[str],
    max_tokens: Optional[int],
    tokenizer,
    model,
    device: torch.device,
) -> List[str]:
    batch = tokenizer(prompts, return_tensors="pt", add_special_tokens=True)

    with torch.no_grad():
        max_tokens_val = max_tokens or 512
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens_val,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        logger.info(f"Generating with max_tokens={max_tokens_val}")
        generated = model.generate(
            inputs=batch["input_ids"].to(device),
            generation_config=generation_config,
        )
    logger.info(f"Decoding response")
    decoded_responses = tokenizer.batch_decode(generated["sequences"].cpu().tolist())
    logger.info(f"Decoded responses: {decoded_responses}")
    return decoded_responses


def do_inference_streaming(
    tokenizer,
    model,
    chat: bool,
    device: torch.device,
    max_new_tokens: int = 1024,
):

    def generate(message: str, history: List):
        if not message:
            return
        prompt = message.strip()

        history_chat_format = []
        for user_msg, assistant_msg in history:
            history_chat_format.append({"role": "user", "content": user_msg})
            history_chat_format.append({"role": "assistant", "content": assistant_msg})
        history_chat_format.append({"role": "user", "content": prompt})
        batch = tokenizer.apply_chat_template(
            history_chat_format, return_tensors="pt", add_generation_prompt=True
        )

        with torch.no_grad():
            streamer = TextIteratorStreamer(
                tokenizer, skip_special_tokens=True, skip_prompt=True, timeout=10
            )

            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            streamer = TextIteratorStreamer(tokenizer)
            generation_kwargs = {
                "inputs": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "generation_config": generation_config,
                "streamer": streamer,
            }

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            all_text = ""

            for new_text in streamer:
                all_text += new_text
                yield all_text

    if chat:
        demo = gr.ChatInterface(generate)
    else:
        demo = gr.Interface(
            fn=generate,
            inputs="textbox",
            outputs="text",
            title="Evals Chat UI",
        )

    demo.queue().launch(
        show_api=True,
        share=True,
    )


def main(
    gradio: bool = False,
):
    app = FastAPI()

    wrapper = TrainerWrapper(LLAMA_CONFIG)
    logger.info("Initializing model")
    wrapper.init_model()

    model = wrapper.model.eval()
    tokenizer = wrapper.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @app.post("/generate")
    async def generate_completion(request: CompletionRequest):
        try:
            completions = do_inference_api(
                request.prompts, request.max_length, tokenizer, model, device
            )
            return {"completions": completions}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/status")
    async def get_status():
        return {"status": "ok"}

    if gradio:
        logger.info(f"Starting Gradio interface, using device {device}")
        do_inference_streaming(tokenizer, model, chat=True, device=device)
    else:
        logger.info("Starting API")
        uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    fire.Fire(main)
