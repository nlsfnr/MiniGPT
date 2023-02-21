#!/usr/bin/env python3
'''Inference functions for the model.'''
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Tuple

import chex
import click
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array

if __name__ == '__main__':
    # If the module is executed we need to add the parent module to the discoverable imports
    import sys
    sys.path.append('.')

from minigpt import common, data, nn

logger = logging.getLogger(common.NAME)


class InferenceConfig(nn.ModelConfig, Protocol):
    tokenizer_path: Path


def generate(rng: chex.PRNGKey,
             prompt: str,
             config: InferenceConfig,
             params: chex.ArrayTree,
             energy: float = 0.0,
             use_jit: bool = False,
             allow_eos: bool = False,
             ) -> Iterator[Tuple[str, str]]:
    '''Sample from the model.'''
    # Preparations
    tokenizer = data.get_tokenizer(config.tokenizer_path)

    def model_fn(indices: Array) -> Array:
        model = nn.Model.from_config(config)
        return model(indices, is_training=False)

    prompt_indices = tokenizer.encode(prompt, add_special_tokens=False).ids
    indices: List[int] = tokenizer.encode(data.BOS_TOKEN).ids + prompt_indices
    eos_index = tokenizer.token_to_id(data.EOS_TOKEN)
    # Execution
    with jax.disable_jit(not use_jit):
        forward = partial(hk.without_apply_rng(hk.transform(model_fn)).apply, params)
        while True:
            outputs = forward(jnp.asarray([indices]))
            logits: Array = outputs[0, -1, :]
            if not allow_eos:
                logits = logits.at[eos_index].set(-jnp.inf)
            rng, rng_sample = jax.random.split(rng)
            noise = jax.random.normal(rng_sample, logits.shape) * energy
            next_index = jnp.argmax(logits + noise, -1)
            indices.append(int(next_index))
            yield str(tokenizer.id_to_token(next_index)), str(tokenizer.decode(indices))
            if next_index == eos_index:
                break
            if len(indices) > config.max_sequence_length:
                indices.pop(0)


def get_cli() -> click.Group:
    '''Get the command line interface for this module.'''

    class Config(common.YamlConfig):

        # Data config
        tokenizer_path: Path

        # Model config
        vocab_size: int
        embedding_size: int
        max_sequence_length: int
        num_layers: int
        num_heads: int
        value_size: int
        key_size: int
        w_init_var: float
        embed_init_var: float
        mlp_size: Optional[int] = None
        model_size: Optional[int] = None
        dropout: float = 0.1

    cli = common.get_cli_group('inference')

    @cli.command('generate')
    @click.option('--prompt', '-p', default='', help='Prompt for the model')
    @click.option('--load-from', '-l', type=Path,
                  help='Path to the checkpoint to use for generation')
    @click.option('--energy', '-e', type=float, default=0.0, help='Energy of the model')
    @click.option('--seed', '-s', type=int, default=None, help='Random seed')
    def cli_generate(prompt: str,
                     load_from: Path,
                     energy: float,
                     seed: Optional[int],
                     ) -> None:
        '''Generate text from a model.'''
        rngs = common.get_rngs(seed)
        checkpoint = common.load_checkpoint(load_from, config_class=Config, for_inference=True)
        config = checkpoint['config']
        params = checkpoint['params']
        tokens = generate(rng=next(rngs),
                          params=params,
                          config=config,
                          prompt=prompt,
                          energy=energy)
        print(prompt, end='')
        for token, _ in tokens:
            print(token, end='', flush=True)

    return cli


if __name__ == '__main__':
    get_cli()()
