from functools import partial
from typing import Iterator, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from chex import ArrayTree, PRNGKey
from jax import Array

from . import data, nn
from .common import Config


def generate(
    *,
    config: Config,
    params: ArrayTree,
    rng_key: PRNGKey,
    prompt: str = "",
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> Iterator[str]:
    def fn(
        *,
        indices: Array,
        subkey: PRNGKey,
    ) -> Array:
        model = nn.Model.from_config(config)
        seq_len = indices.shape[1]
        mask = jnp.tril(jnp.full((seq_len, seq_len), True, dtype=bool))
        logits = model(indices, is_training=False, mask=mask)
        logits = logits[:, -1, :] / temperature
        probs = jax.nn.softmax(logits)
        return _sample_from_top_p(probs=probs, p=top_p, rng_key=subkey)

    # Prepare the tokenizer and the initial indices.
    tokenizer = data.tokenizer_from_config(config)
    *indices, eos_token_id = tokenizer.encode(prompt).ids
    # Prepare the model function.
    model_fn = partial(hk.without_apply_rng(hk.transform(fn)).apply, params)
    # Run the model.
    while True:
        rng_key, subkey = jax.random.split(rng_key)
        inputs = jnp.array(indices, dtype=jnp.int32)[None, :]
        index = model_fn(indices=inputs, subkey=subkey)[0]
        indices.append(int(index))
        if index == eos_token_id:
            break
        output = tokenizer.decode(indices)
        yield output


def _sample_from_top_p(
    *,
    probs: Array,
    p: float,
    rng_key: PRNGKey,
) -> Array:
    indices = jnp.argsort(probs, axis=-1)
    probs = jnp.take_along_axis(probs, indices, axis=-1)
    cumsum = jnp.cumsum(probs, axis=-1)
    probs = probs * (cumsum >= p).astype(jnp.float32)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    subkeys = jax.random.split(rng_key, probs.shape[0])
    indices = jax.vmap(jax.random.choice)(subkeys, indices, p=probs)
    return indices


def perplexity(
    *,
    config: Config,
    params: ArrayTree,
    text: str,
) -> Tuple[Array, Array, List[str]]:
    def fn(
        *,
        indices: Array,  # S
    ) -> Array:
        model = nn.Model.from_config(config)
        seq_len = indices.shape[0]
        mask = jnp.tril(jnp.full((seq_len, seq_len), True, dtype=bool))
        logits = model(indices[None, :], is_training=False, mask=mask)[0]
        return logits  # S V

    tokenizer = data.tokenizer_from_config(config)
    model_fn = partial(hk.without_apply_rng(hk.transform(fn)).apply, params)
    indices = jnp.asarray(tokenizer.encode(text).ids, dtype=jnp.int32)
    logits = model_fn(indices=indices)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, indices[..., None], axis=-1)
    tokens = [tokenizer.id_to_token(int(i)) for i in indices]
    return -selected_log_probs, indices, tokens
