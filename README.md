# Smolmodels

Collection of experiments related to tuning small language models for specific tasks.

## Installation

```bash
python3.10 -m venv venv
uv sync
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 --no-build-isolation --force-reinstall
```

## BitesandBytes
```bash
pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-0.44.2.dev0-py3-none-manylinux_2_24_x86_64.whl' --no-deps
```

### Pyright dependencies

```
pyright --createstub transformers
```