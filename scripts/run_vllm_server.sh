export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export HF_HOME="/weka/home-brianf/huggingface"

set -ex

LORA_DIR=${1:-"outputs/run-12-26-5-12-514725/checkpoint-6500"}
CONFIG_PATH="$LORA_DIR/adapter_config.json"
BASE_MODEL=$(jq -r '.base_model_name_or_path' "$CONFIG_PATH")

echo "Starting VLLM server with model: $BASE_MODEL and LoRA directory: $LORA_DIR"

vllm serve $BASE_MODEL --enable-lora --lora-modules dpo_lora=$LORA_DIR/ --tokenizer_mode "mistral"
# vllm serve $BASE_MODEL --tokenizer_mode "mistral"