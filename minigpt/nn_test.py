import haiku as hk
import jax
import jax.numpy as jnp

from . import nn


def test_multihead_attention_call() -> None:
    rngs = hk.PRNGSequence(42)
    model = lambda x: nn.MultiHeadAttention(
        num_heads=2,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        key_size=4,
        value_size=4,
        model_size=5,
        name='multihead_attention')(x, True)
    model_hk = hk.transform(model)
    x = jnp.ones((2, 3, 5))
    params = model_hk.init(next(rngs), x)
    y = model_hk.apply(params, next(rngs), x)
    assert y.shape == (2, 3, 5)


def test_decoder_block_call() -> None:
    rngs = hk.PRNGSequence(42)
    model = lambda x: nn.DecoderBlock(
        num_heads=2,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        key_size=4,
        value_size=4,
        model_size=5,
        name='decoder_block')(x, True)
    model_hk = hk.transform(model)
    x = jnp.ones((2, 3, 5))
    params = model_hk.init(next(rngs), x)
    y = model_hk.apply(params, next(rngs), x)
    assert y.shape == (2, 3, 5)


def test_decoder_call() -> None:
    rngs = hk.PRNGSequence(42)
    model = lambda x: nn.Decoder(
        num_layers=2,
        num_heads=2,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        key_size=4,
        value_size=4,
        model_size=5,
        name='decoder')(x, True)
    model_hk = hk.transform(model)
    x = jnp.ones((2, 3, 5))
    params = model_hk.init(next(rngs), x)
    y = model_hk.apply(params, next(rngs), x)
    assert y.shape == (2, 3, 5)


def test_model_call() -> None:
    rngs = hk.PRNGSequence(42)
    model = lambda x: nn.Model(
        vocab_size=10,
        embedding_size=5,
        max_sequence_length=3,
        num_layers=2,
        num_heads=2,
        key_size=4,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        embed_init=hk.initializers.TruncatedNormal(stddev=0.02),
        mlp_size=20,
        value_size=4,
        model_size=5,
        name='model')(x, True)
    model_hk = hk.transform(model)
    indices = jax.random.randint(next(rngs), (2, 3), 0, 10)
    params = model_hk.init(next(rngs), indices)
    y = model_hk.apply(params, next(rngs), indices)
    assert y.shape == (2, 3, 10)


def test_rotary_embeddings() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        re = nn.RotaryEmbedding(10)
        x = jax.random.normal(jax.random.PRNGKey(42), (1, 8, 4))
        y = re(x)

    fn()  # type: ignore
