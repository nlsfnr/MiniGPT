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
  W&B logging, autosaving etc.

## Using MiniGPT

Either run `pip install -r requirements.txt` (note that a prior [installation
of Jax](https://github.com/google/jax#installation) is assumed) or use Docker
with `docker build -t minigpt .` and `docker run --rm -it --gpus all
-v$PWD:/workdir/ minigpt bash`.

The scripts inside `./scripts/` should be all you need to train and run a
model. To do so, follow the steps below.

1. Create a configuration file. I recommend adapting a copy of
   [`./configs/dev.yaml`](/configs/dev.yaml) to your needs.
2. To train a model, run:
```bash
./scripts/train.py train \
    --config-path ./path/to/my/config.yaml \
    --seed 42 \
    --save-frequency 500 \
    --save-directory ./path/to/save/directory/
```

The options should be self-explanatory. Run `./scripts/train.py train --help`
for an exhaustive list of options with descriptions.

3. To generate text, run:
```bash
./scripts/generate.py generate \
    --load ./path/to/save/directory/ \
    --seed 42 \
    --temperature 0.8 \
    --top-p 0.95
    --prompt "A long time ago"
```
