from threading import Thread
from typing import List, Optional

import fire
import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import TextIteratorStreamer

from train_trl import LLAMA_CONFIG, TrainerWrapper
from trl_wrapper.trainer_wrapper import WrapperConfig
from synthetic_data.utils import Conversation


class CompletionRequest(BaseModel):
    conversations: List[Conversation]
    max_length: Optional[int] = None


def do_inference_api(
    prompts: List[Conversation],
    max_tokens: Optional[int],
    tokenizer,
    model,
    device: torch.device,
) -> List[str]:
    # TODO allow for batch_size > 1
    batch = tokenizer.apply_chat_template(
        prompts, return_tensors="pt", tokenize=True, add_special_tokens=True
    )

    with torch.no_grad():
        max_tokens = max_tokens or 1024
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
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
        logger.info(f"Generating with max_tokens={max_tokens}")
        generated = model.generate(
            inputs=batch.to(device),
            generation_config=generation_config,
        )
    logger.info(f"Decoding response")
    decoded_responses = tokenizer.decode(generated["sequences"][0][len(batch[0]):], skip_special_tokens=True)
    logger.info(f"Decoded responses: {decoded_responses}")
    return decoded_responses


def do_inference_gradio(
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


def main(gradio: bool = False, model_dir="outputs/checkpoint-3080"):
    app = FastAPI(debug=True)

    config = WrapperConfig(
        model_id_or_path=LLAMA_CONFIG.model_id_or_path,
        adapter_path="outputs/checkpoint-3080",
    )
    wrapper = TrainerWrapper(config)
    logger.info("Initializing model")
    wrapper.init_model()

    # Have to load adapter afterwards as the embeddings need to be loaded first
    # TODO figure out why that is needed
    if config.adapter_path:
        logger.info(f"Loading adapter from {config.adapter_path}")
        wrapper.model.load_adapter(config.adapter_path)

    wrapper.model = wrapper.model.eval()
    tokenizer = wrapper.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    @app.post("/generate")
    async def generate(request: CompletionRequest):
        try:
            completions = do_inference_api(
                request.conversations,
                request.max_length,
                tokenizer,
                wrapper.model,
                device,
            )
            return {"completions": completions}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/status")
    async def get_status():
        return {"status": "ok"}

    if gradio:
        logger.info(f"Starting Gradio interface, using device {device}")
        do_inference_gradio(tokenizer, wrapper.model, chat=True, device=device)
    else:
        logger.info("Starting API")
        uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    fire.Fire(main)
