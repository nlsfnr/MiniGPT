# MiniGPT

A decoder-only language model implemented using [Jax](https://github.com/google/jax).

## Quick tour

If you want to see some code, have a look at

- [`./minigpt/nn.py`](/minigpt/nn.py) - The model and its components.
- [`./minigpt/training.py`](/minigpt/training.py) - The training loop and loss
  function.
- [`./minigpt/inference.py`](/minigpt/inference.py) - Methods to use pretrained
  models for inference.
- [`./minigpt/sidecar.py`](/minigpt/sidecar.py) - Training utilities such as
  W&B logging, auto-saving etc.
- [`./scripts/*.py`](/scripts/) - Scripts exposing CLIs to interact with MiniGPT.

## Using MiniGPT

Either run `pip install -r requirements.txt` (note that a prior [installation
of Jax](https://github.com/google/jax#installation) is assumed) or use Docker
with `docker build -t minigpt .` and then `docker run --rm -it --gpus all -v
$PWD:/workdir/ minigpt bash`. The latter requires the [Nvidia container
runtime](https://developer.nvidia.com/nvidia-container-runtime).

The scripts inside `./scripts/` should be all you need to train and run a
model. To do so, follow the steps below.

1. Create a configuration file. I recommend adapting a copy of
   [`./configs/dev.yaml`](/configs/dev.yaml) to your needs.
2. To train a model, run (we use `./configs/dev.yaml` as an example):
```bash
./scripts/train.py train \
    --config-path ./configs/dev.yaml \
    --seed 42 \
    --save-directory ./zoo/example-run/
```

The options should be self-explanatory. Run `./scripts/train.py train --help`
for an exhaustive list with help texts.

3. To generate text with a trained model, run:
```bash
./scripts/generate.py generate \
    --load ./zoo/example-run/ \
    --seed 42 \
    --temperature 0.8 \
    --top-p 0.95 \
    --prompt "A long time ago"
```
