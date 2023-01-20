# MiniGPT

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

Given that JAX is already installed, run `pip install -r requirements.txt`. To
download a dataset from the Huggingface hub and store it in a LMDB database,
run e.g.:

```bash
./minigpt/data.py new-dataset
    --path ./data/wikitext/
    --name wikitext
    --min-length 32
    --min-chars-per-token 3
```

This will download the Wikitext dataset and store it inside `./data/wikitext/`.
Only samples with more than 32 characters and more than 3 characters per token
after tokenization with the `bert-base-uncased` tokenizer. The latter is a
relatively reliable and easy way of ensuring that all samples consist of
natural language, taken from the [*Cramming*
paper](https://github.com/karpathy/minGPT) by Geiping et. al.

Then, to train a tokenizer on the dataset, run:

```bash
./minigpt/data.py new-tokenizer
    --path ./tokenizers/wikitext-30k
    --kind sentencepiece
    --db-path ./data/wikitext/
    --vocab-size 30000
    --min-frequency 10
```

Finally, to train a new model on the dataset with the newly trained tokenizer,
create a config file in ./configs/. You can use the pre-existing one as a
template. In the example below we use it directly. Now run:

```bash
./minigpt/training.py train
    --config-path ./configs/default.yaml
    --save-path ./model-zoo/my-model/
    --save-frequency 500
    --log-frequency 10
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
    --load-from ./model-zoo
    --prompt "GPT is a"
```
