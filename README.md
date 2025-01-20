# Smolmodels

Collection of experiments related to tuning small language models for specific tasks.

## Installation

```bash
pip install uv
uv venv
uv sync --no-install-package flash-attn
uv sync --no-build-isolation
uv pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-0.44.2.dev0-py3-none-manylinux_2_24_x86_64.whl' --no-deps
```
vLLM can be installed for evals using `uv pip install vllm`.

### Pyright dependencies

```
pyright --createstub transformers
```