from typing import Iterable, Optional
import gradio as gr
from openai import OpenAI
import fire
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from loguru import logger


def main(
    host: str = "localhost",
    port: int = 8001,
    vllm_host: str = "localhost",
    vllm_port: int = 8000,
    model: Optional[str] = None,
):
    client = OpenAI(
        base_url=f"http://{vllm_host}:{vllm_port}/v1",
    )

    all_models = client.models.list()
    
    logger.info(f"Models: {[x.id for x in all_models.data]}")
    first_model_id = all_models.data[0].id

    model_id = model or first_model_id

    def predict(message, history):
        # Convert chat history to OpenAI format
        history_openai_format: Iterable[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a great ai assistant."}
        ]
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})  # type: ignore
            history_openai_format.append({"role": "assistant", "content": assistant})  # type: ignore
        history_openai_format.append({"role": "user", "content": message})  # type: ignore

        # Create a chat completion request and send it to the API server
        logger.info(f"Sending message: {message} to model: {model_id}")
        stream = client.chat.completions.create(
            messages=history_openai_format, model=model_id, stream=True
        )

        # Read and return generated text from response stream
        partial_message = ""
        for chunk in stream:
            partial_message += chunk.choices[0].delta.content or ""
            yield partial_message

    # Create and launch a chat interface with Gradio
    gr.ChatInterface(predict).queue().launch(
        server_name=host, server_port=port
    )


if __name__ == "__main__":
    fire.Fire(main)
