from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from chex import ArrayTree

from . import training
from .common import Config


@pytest.fixture
def config() -> Config:
    path = Path(__file__).parent.parent / "configs/testing.yaml"
    return Config.from_yaml(path)


def test_new_from_config(config: Config, tmpdir: Path) -> None:
    trainer = training.Trainer.new_from_config(config, Path(tmpdir), 0)
    assert hasattr(trainer, "params")
    assert hasattr(trainer, "opt_state")


def tree_equal(a: ArrayTree, b: ArrayTree) -> bool:
    return jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.allclose, a, b))


def test_iterate(config: Config, tmpdir: Path) -> None:
    trainer = training.Trainer.new_from_config(config, Path(tmpdir), 0)
    next(iter(trainer))


def test_save_load(config: Config, tmpdir: Path) -> None:
    trainer = training.Trainer.new_from_config(config, Path(tmpdir), 0)
    next(iter(trainer))
    trainer.save(Path(tmpdir))
    trainer2 = training.Trainer.load(Path(tmpdir))
    assert tree_equal(trainer.params, trainer2.params)
    assert tree_equal(trainer.opt_state, trainer2.opt_state)
    assert tree_equal(trainer.rng_key, trainer2.rng_key)
    assert trainer.step == trainer2.step
    assert next(iter(trainer.dataset)) == next(iter(trainer2.dataset))
