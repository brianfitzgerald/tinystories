from typing import Iterable, Optional
import gradio as gr
from openai import OpenAI
import fire
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from loguru import logger
from huggingface_hub import hf_hub_download
import json

def _get_eos_tokens():
    config_path = hf_hub_download(
        repo_id="meta-llama/Llama-3.2-3B-Instruct", filename="generation_config.json"
    )
    config_json: dict = json.load(open(config_path))
    stop_tokens = config_json["eos_token_id"]


def main(
    host: str = "localhost",
    port: int = 8001,
    vllm_host: str = "localhost",
    vllm_port: int = 8000,
    model: Optional[str] = None,
):
    oai_base_url = f"http://{vllm_host}:{vllm_port}/v1"
    logger.info(f"Creating OpenAI client with base url: {oai_base_url}")
    client = OpenAI(base_url=oai_base_url)

    def _get_model_id():
        logger.info("Fetching models from OpenAI")
        all_models = client.models.list()

        logger.info(f"Models: {[x.id for x in all_models.data]}")
        first_model_id = all_models.data[0].id
        model_id = model or first_model_id
        return model_id

    def predict(message, history):
        model_id = _get_model_id()
        # Convert chat history to OpenAI format
        history_openai_format: Iterable[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a great ai assistant."}
        ]
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})  # type: ignore
            history_openai_format.append({"role": "assistant", "content": assistant})  # type: ignore
        history_openai_format.append({"role": "user", "content": message})  # type: ignore

        stop_tokens = _get_eos_tokens()

        # Create a chat completion request and send it to the API server
        logger.info(f"Sending message: {message} to model: {model_id}")
        stream = client.chat.completions.create(
            messages=history_openai_format,
            model=model_id,
            stream=True,
            extra_body={
                "stop_token_ids": stop_tokens,
                "skip_special_tokens": False
            }
        )

        # Read and return generated text from response stream
        partial_message = ""
        for chunk in stream:
            partial_message += chunk.choices[0].delta.content or ""
            yield partial_message

    # Create and launch a chat interface with Gradio
    gr.ChatInterface(predict).queue().launch(server_name=host, server_port=port)


if __name__ == "__main__":
    fire.Fire(main)
