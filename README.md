# MiniGPT

*This repository is currently getting re-written. See the `v2` branch.**

This is MiniGPT, a GPT-like language model built using *JAX*, *Haiku* and
*Optax*. The neural network components are all implemented from scratch and are
well documented. This project was loosely inspired by A. Karpathy's
[miniGPT](https://github.com/karpathy/minGPT).

## Quick tour

If you want to see the code, I encourage you to check out the following files:

1. [`./minigpt/nn.py`](/minigpt/nn.py) - The model and its components.
1. [`./minigpt/training.py`](/minigpt/training.py) - The training loop and loss
   function.
1. [`./minigpt/inference.py`](/minigpt/inference.py) - Methods to use
   pretrained models for inference.

## Getting Started

Given that JAX is already installed, run `pip install -r requirements.txt`.

Datasets are streamed during training (with the exception of smaller ones,
those are cached on the machine for faster iteration times during development).

To train a tokenizer create a config. Examples can be found in ./configs/. You
can use the pre-existing ones as a template. In the example below we use one of
them directly.

```bash
./minigpt/data.py new-tokenizer
    --config-path ./configs/default.yaml
```

Finally, to train a new model with the newly trained tokenizer run:

```bash
./minigpt/training.py train
    --config-path ./configs/default.yaml
    --save-path ./model-zoo/my-model/
    --save-frequency 500
    --csv-path ./model-zoo/my-model/telemetry-data.csv
```

To resume training from a checkpoint, run:

```bash
./minigpt/training.py train
    --load-from ./model-zoo/my-model/
    --save-path ./model-zoo/my-model/
    --save-frequency 500
    --log-frequency 10
    --csv-path ./model-zoo/my-model/telemetry-data.csv
```

To interrupt training, press Ctr+C. Once done, you can sample from the model with:

```bash
./minigpt/inference.py generate
    --load-from ./model-zoo/
    --prompt "GPT is a"
```
