import haiku as hk
import jax.numpy as jnp

from . import nn


def test_multihead_attention() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        model = nn.MultiHeadAttention(
            num_heads=2,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            key_size=4,
            value_size=4,
            model_size=5,
            name='multihead_attention')
        x = jnp.ones((2, 3, 5))
        y = model(x, True)
        assert y.shape == (2, 3, 5)

    fn()  # type: ignore


def test_decoder_block() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        model = nn.DecoderBlock(
            num_heads=2,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            key_size=4,
            value_size=4,
            model_size=5,
            name='decoder_block')
        x = jnp.ones((2, 3, 5))
        y = model(x, True)
        assert y.shape == (2, 3, 5)

    fn()  # type: ignore


def test_decoder() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        model = nn.Decoder(
            num_layers=2,
            num_heads=2,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            key_size=4,
            value_size=4,
            model_size=5,
            name='decoder')
        x = jnp.ones((2, 3, 5))
        y = model(x, True)
        assert y.shape == (2, 3, 5)

    fn()  # type: ignore


def test_model() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        model = nn.Model(
            vocab_size=10,
            embedding_size=4,
            max_sequence_length=8,
            num_layers=2,
            num_heads=2,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            embed_init=hk.initializers.TruncatedNormal(stddev=0.02),
            key_size=4,
            value_size=4,
            model_size=5,
            name='model')
        x = jnp.ones((2, 3), dtype=jnp.int32)
        y = model(x, True)
        assert y.shape == (2, 3, 10)

    fn()  # type: ignore


def test_rotary_embeddings() -> None:
    k = jnp.ones((1, 8, 16))
    q = jnp.ones((1, 8, 16))
    dk = nn.rotary_pos_emb(k)
    dq = nn.rotary_pos_emb(q)
    a = jnp.einsum('... i k, ... j k -> ... i j', dk, dq)
    assert a.shape == (1, 8, 8)
    # Check if a is symmetric.
    assert jnp.allclose(a, a.transpose((0, 2, 1)))
