from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple, Optional, Protocol, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import yaml
from chex import Array, ArrayTree, PRNGKey

from . import data, nn
from .common import Config, get_logger, require_implementation

logger = get_logger()


class TrainingConfig(Protocol):
    trainer: TrainerConfig
    model: nn.ModelConfig
    dataset: data.SamplesConfig


class TrainerConfig(Protocol):
    batch_size: int
    batch_buffer_size: int
    optimizer: OptimizerConfig
    yield_frequency: int
    save_frequency: int


class OptimizerConfig(Protocol):
    gradient_accumulation_steps: int
    lr_max: float
    lr_min: float
    lr_decay_steps: Optional[int]
    lr_warmup_steps: int
    gradient_clip_norm: float
    adam_b1: float
    adam_b2: float


@dataclass
class Telemetry:
    loss: float
    trainer: Trainer
    save_to: Optional[Path]
    model_telemetry: Optional[ArrayTree]
    gradients: Optional[ArrayTree]


class Trainer(Iterable[Telemetry]):
    def __init__(
        self,
        *,
        params: ArrayTree,
        opt_state: optax.MultiStepsState,
        dataset: Iterable[data.Sample],
        rng_key: PRNGKey,
        step: int,
        config: TrainingConfig,
        path: Optional[Path],
    ) -> None:
        require_implementation(config, TrainingConfig)
        self.params = params
        self.opt_state = opt_state
        self.dataset = dataset
        self.rng_key = rng_key
        self.step = step
        self.config = config
        self.path = path
        if config.trainer.save_frequency % config.trainer.yield_frequency != 0:
            raise ValueError(
                "Expected save_frequency to be a multiple of yield_frequency, got"
                f" {config.trainer.save_frequency} and"
                f" {config.trainer.yield_frequency}"
            )

    def __iter__(self) -> Iterator[Telemetry]:
        logger.info(f"Starting training at step {self.step}")
        # Prepare the data
        dict_batches = data.BatchedDataset(self.dataset, self.config.trainer.batch_size)
        indices_iter: Iterator[Optional[Array]]
        indices_iter = (
            jnp.asarray(d["input_ids"], dtype=jnp.int32) for d in dict_batches
        )
        indices_iter = chain(indices_iter, [None])
        indices = next(indices_iter)
        # Compile the training step function
        train_step = jax.jit(
            partial(_train_step, config=self.config),
            static_argnames="collect_telemetry",
        )
        # Run the training loop
        losses = []
        while True:
            if indices is None:
                break
            self.rng_key, rng_key = jax.random.split(self.rng_key)
            should_yield = self.step % self.config.trainer.yield_frequency == 0
            out = train_step(
                indices=indices,
                params=self.params,
                opt_state=self.opt_state,
                rng_key=rng_key,
                collect_telemetry=should_yield,
            )
            indices = next(indices_iter)
            self.params, self.opt_state, loss, model_telemetry, gradients = out
            losses.append(loss)
            if should_yield:
                should_save = (
                    self.path is not None
                    and self.step % self.config.trainer.save_frequency == 0
                )
                if should_save:
                    assert self.path is not None
                    self.save(self.path)
                yield Telemetry(
                    loss=float(jnp.mean(jnp.asarray(losses))),
                    trainer=self,
                    save_to=self.path if should_save else None,
                    model_telemetry=model_telemetry,
                    gradients=gradients,
                )
                losses = []
            del model_telemetry
            self.step += 1

    def save(
        self,
        path: Path,
    ) -> Trainer:
        path.mkdir(parents=True, exist_ok=True)

        def save_pickle(obj, name):
            with open(path / f"{name}.pkl", "wb") as f:
                pickle.dump(obj, f)

        save_pickle(self.params, "params")
        save_pickle(self.opt_state, "opt_state")
        save_pickle(self.rng_key, "rng_key")
        with open(path / "other.yaml", "w") as f:
            yaml.dump(dict(step=self.step), f)
        if isinstance(self.config, Config):
            self.config.to_yaml(path / "config.yaml")
        else:
            save_pickle(self.config, "config")
        logger.info(f"Saved trainer to {path} at step {self.step}")
        return self

    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        path: Optional[Path] = None,
    ) -> Trainer:
        if path is None:
            path = checkpoint_path

        def load_pickle(name) -> Any:
            with open(checkpoint_path / f"{name}.pkl", "rb") as f:
                return pickle.load(f)

        params = load_pickle("params")
        assert isinstance(params, dict)
        opt_state = load_pickle("opt_state")
        assert isinstance(opt_state, optax.MultiStepsState)
        rng_key = load_pickle("rng_key")
        assert isinstance(rng_key, Array)
        with open(checkpoint_path / "other.yaml", "r") as f:
            other = yaml.safe_load(f)
            assert isinstance(other, dict)
        step = int(other["step"])
        config = Config.from_yaml(checkpoint_path / "config.yaml")
        require_implementation(config, TrainingConfig)
        require_implementation(config.trainer, TrainerConfig)
        dataset = data.samples_from_config(config.dataset, truncate_and_pad=True)
        dataset = islice(dataset, step * config.trainer.batch_size, None)
        logger.info(f"Loaded checkpoint from {checkpoint_path} at step {step}")
        return cls(
            params=params,
            opt_state=opt_state,
            dataset=dataset,
            rng_key=rng_key,
            step=step,
            config=config,
            path=path,
        )

    @classmethod
    def new_from_config(
        cls,
        config: TrainingConfig,
        path: Path,
        seed: int,
    ) -> Trainer:
        require_implementation(config, TrainingConfig)
        rng_key, params_rng_key = jax.random.split(jax.random.PRNGKey(seed))
        params = nn.Model.get_params(config.model, params_rng_key)
        opt_state = get_optimizer(config.trainer.optimizer).init(params)
        dataset = data.samples_from_config(config.dataset, truncate_and_pad=True)
        return cls(
            params=params,
            opt_state=opt_state,
            dataset=dataset,
            rng_key=rng_key,
            step=0,
            config=config,
            path=path,
        )


class _TrainStepRV(NamedTuple):
    params: ArrayTree
    opt_state: optax.MultiStepsState
    loss: Array
    model_telemetry: nn.Telemetry
    gradients: Optional[ArrayTree]


def _train_step(
    indices: Array,
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    rng_key: PRNGKey,
    collect_telemetry: bool,
    config: TrainingConfig,
) -> _TrainStepRV:
    loss_fn = hk.transform(
        partial(_loss_fn, collect_telemetry=collect_telemetry, config=config)
    ).apply
    grad_fn = jax.grad(loss_fn, has_aux=True)
    optimizer = get_optimizer(config.trainer.optimizer)
    gradients, (loss, model_telemetry) = grad_fn(params, rng_key, indices)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return _TrainStepRV(
        params,
        opt_state,
        loss,
        model_telemetry,
        gradients if collect_telemetry else None,
    )


class _LossFnAux(NamedTuple):
    loss: Array
    model_telemetry: nn.Telemetry


def _loss_fn(
    indices: Array,
    collect_telemetry: bool,
    config: TrainingConfig,
) -> Tuple[Array, _LossFnAux]:
    model = nn.Model.from_config(config.model)
    inputs = indices[:, :-1]
    seq_len = inputs.shape[1]
    mask = jnp.triu(jnp.full((1, 1, seq_len, seq_len), -1e8))
    logits, model_telemetry = model(
        inputs, is_training=True, collect_telemetry=collect_telemetry, mask=mask
    )
    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, indices[:, 1:])
    )
    return loss, _LossFnAux(loss, model_telemetry)


def get_optimizer(config: OptimizerConfig) -> optax.MultiSteps:
    require_implementation(config, OptimizerConfig)
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(config.lr_min, config.lr_max, config.lr_warmup_steps),
            (
                optax.cosine_decay_schedule(
                    config.lr_max,
                    config.lr_decay_steps,
                    alpha=config.lr_min / config.lr_max,
                )
                if config.lr_decay_steps is not None
                else optax.constant_schedule(config.lr_max)
            ),
        ],
        [config.lr_warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.scale_by_adam(config.adam_b1, config.adam_b2),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )
    return optax.MultiSteps(optimizer, config.gradient_accumulation_steps)
