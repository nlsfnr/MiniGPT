import haiku as hk
import jax
import jax.numpy as jnp
import pytest
from chex import Array

from . import nn
from .common import Config


@pytest.mark.parametrize("collect_telemetry", [True, False])
def test_model(collect_telemetry: bool) -> None:
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
        logits, telemetry = model(
            indices, is_training=True, collect_telemetry=collect_telemetry
        )
        assert logits.shape == (2, 10, 100)
        if collect_telemetry:
            assert isinstance(telemetry, dict)
            assert "embeddings" in telemetry
            assert isinstance(telemetry["embeddings"], Array)
            assert telemetry["embeddings"].shape == (2, 10, 32)
            assert "logits" in telemetry
            assert isinstance(telemetry["logits"], Array)
            assert telemetry["logits"].shape == (2, 10, 100)
            assert "blocks" in telemetry
            assert isinstance(telemetry["blocks"], list)
            assert len(telemetry["blocks"]) == 3
            stds = jax.tree_util.tree_map(jnp.std, telemetry)
            assert all(leave > 0.0 for leave in jax.tree_util.tree_leaves(stds))
        else:
            assert telemetry is None

    fn()  # type: ignore


def test_get_params() -> None:
    config = Config(
        num_layers=3,
        vocabulary_size=100,
        embedding_dim=32,
        model_dim=64,
        num_heads=4,
        hidden_dim=256,
        dropout=0.1,
    )
    params = nn.Model.get_params(config, 0)
    assert isinstance(params, dict)
