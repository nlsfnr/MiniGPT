import haiku as hk
import jax
import jax.numpy as jnp

import minigpt.nn as nn


def test_rotary_pos_emb_symmetry() -> None:
    k = jnp.ones((1, 8, 16))
    k = nn.rotary_pos_emb(k)
    q = jnp.ones((1, 8, 16))
    q = nn.rotary_pos_emb(q)
    l = jnp.einsum("... s d, ... S d -> ... s S", q, k)
    assert l.shape == (1, 8, 8)
    assert jnp.allclose(l, l.transpose(0, 2, 1))


def test_model() -> None:
    @hk.testing.transform_and_run
    def fn() -> None:
        model = nn.Model(
            num_layers=3,
            vocabulary_size=100,
            embedding_dim=32,
            model_dim=64,
            num_heads=4,
            hidden_dim=256,
            dropout=0.1,
        )
        indices = jax.random.randint(jax.random.PRNGKey(0), (2, 10), 0, 100)
        logits = model(indices, is_training=True)
        assert logits.shape == (2, 10, 100)

    fn()  # type: ignore
