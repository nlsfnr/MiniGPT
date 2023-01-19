import haiku as hk
import jax
import jax.numpy as jnp

from . import nn


def test_nolo_multihead_attention_call() -> None:
    rngs = hk.PRNGSequence(42)
    model = lambda x: nn.MultiHeadAttention(
        num_heads=2,
        key_size=3,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        value_size=4,
        model_size=5,
        name='nolo_multihead_attention')(x, True)
    model_hk = hk.transform(model)
    x = jnp.ones((2, 3, 5))
    params = model_hk.init(next(rngs), x)
    y = model_hk.apply(params, next(rngs), x)
    assert y.shape == (2, 3, 5)


def test_nolo_encoder_block_call() -> None:
    rngs = hk.PRNGSequence(42)
    model = lambda x: nn.EncoderBlock(
        num_heads=2,
        key_size=3,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        value_size=4,
        model_size=5,
        name='nolo_encoder_block')(x, True)
    model_hk = hk.transform(model)
    x = jnp.ones((2, 3, 5))
    params = model_hk.init(next(rngs), x)
    y = model_hk.apply(params, next(rngs), x)
    assert y.shape == (2, 3, 5)


def test_nolo_encoder_call() -> None:
    rngs = hk.PRNGSequence(42)
    model = lambda x: nn.Encoder(
        num_layers=2,
        num_heads=2,
        key_size=3,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        value_size=4,
        model_size=5,
        name='nolo_encoder')(x, True)
    model_hk = hk.transform(model)
    x = jnp.ones((2, 3, 5))
    params = model_hk.init(next(rngs), x)
    y = model_hk.apply(params, next(rngs), x)
    assert y.shape == (2, 3, 5)


def test_nolo_model_call() -> None:
    rngs = hk.PRNGSequence(42)
    model = lambda x: nn.Model(
        vocab_size=10,
        max_sequence_length=3,
        num_layers=2,
        num_heads=2,
        key_size=3,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        embed_init=hk.initializers.TruncatedNormal(stddev=0.02),
        mlp_size=20,
        value_size=4,
        model_size=5,
        name='nolo_model')(x, True)
    model_hk = hk.transform(model)
    indices = jax.random.randint(next(rngs), (2, 3), 0, 10)
    params = model_hk.init(next(rngs), indices)
    y = model_hk.apply(params, next(rngs), indices)
    assert y.shape == (2, 3, 10)
