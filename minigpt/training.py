from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Tuple, TypeVar, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import Array, ArrayTree, PRNGKey

from . import data, nn
from .common import Config, get_logger

logger = get_logger()

ArrayTreeT = TypeVar("ArrayTreeT", bound=ArrayTree)


class Event:
    pass


@dataclass
class TrainStep(Event):
    step: int
    loss: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Save(Event):
    path: Path
    step: int
    config: Config
    params: ArrayTree
    opt_state: optax.MultiStepsState
    rng_key: PRNGKey
    seed: int


def to_cpu(t: ArrayTreeT) -> ArrayTreeT:
    if t is None:
        return t
    cpu = jax.devices("cpu")[0]
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device=cpu), t)


def train(
    *,
    config: Config,
    seed: int,
    rng_key: Optional[PRNGKey] = None,
    params: Optional[ArrayTree] = None,
    opt_state: Optional[optax.MultiStepsState] = None,
    step: Optional[int] = None,
    batches: Optional[Union[Iterable[Array], Iterable[np.ndarray]]] = None,
    save_frequency: Optional[int] = None,
    save_directory: Optional[Path] = None,
    log_param_size_on_init: bool = True,
) -> Iterable[Event]:
    if rng_key is None:
        rng_key = jax.random.PRNGKey(seed)
    if params is None:
        params = nn.Model.get_params(config, seed + 1, log_size=log_param_size_on_init)
    if opt_state is None:
        opt_state = _get_optimizer(config).init(params)
    if step is None:
        step = 0
    if batches is None:
        batches = islice(data.batches_from_config(config, seed + 2), step, None)
    batches = map(jnp.asarray, batches)
    assert params is not None
    assert opt_state is not None
    assert rng_key is not None
    assert step is not None
    assert batches is not None
    train_step = jax.jit(
        partial(_train_step, config=config),
        static_argnames=["with_model_telemetry", "with_gradients"],
    )

    for batch in batches:
        # Get the next batch from the input queue
        # Train on the batch
        rng_key, subkey = jax.random.split(rng_key)
        rv: _TrainStepRV = train_step(
            indices=batch,
            params=params,
            opt_state=opt_state,
            rng_key=subkey,
            with_model_telemetry=False,
            with_gradients=False,
        )
        params, opt_state, loss, model_telemetry, gradients = rv
        del model_telemetry, gradients  # TODO
        # Send the training step event to the output queue
        yield TrainStep(
            step=step,
            loss=float(loss),
        )
        # Sent the save event to the output queue if appropriate
        if save_directory is not None:
            assert save_frequency is not None
            if step % save_frequency == 0:
                yield Save(
                    path=save_directory / f"step_{step:06d}/",
                    step=step,
                    config=config,
                    params=to_cpu(params),
                    opt_state=to_cpu(opt_state),
                    rng_key=to_cpu(rng_key),
                    seed=seed,
                )
        step += 1


class _TrainStepRV(NamedTuple):
    params: ArrayTree
    opt_state: optax.MultiStepsState
    loss: Array
    model_telemetry: nn.Telemetry
    gradients: Optional[ArrayTree]


def _train_step(
    *,
    indices: Array,
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    rng_key: PRNGKey,
    config: Config,
    with_model_telemetry: bool,
    with_gradients: bool,
) -> _TrainStepRV:
    loss_fn = hk.transform(
        partial(_loss_fn, with_model_telemetry=with_model_telemetry, config=config)
    ).apply
    grad_fn = jax.grad(loss_fn, has_aux=True)
    optimizer = _get_optimizer(config)
    gradients, loss_aux = grad_fn(params, rng_key, indices=indices)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return _TrainStepRV(
        params=params,
        opt_state=opt_state,
        loss=loss_aux.loss,
        model_telemetry=loss_aux.model_telemetry,
        gradients=gradients if with_gradients else None,
    )


class _LossFnAux(NamedTuple):
    loss: Array
    model_telemetry: nn.Telemetry


def _loss_fn(
    *,
    indices: Array,
    config: Config,
    with_model_telemetry: bool,
) -> Tuple[Array, _LossFnAux]:
    model = nn.Model.from_config(config)
    inputs = indices[:, :-1]
    seq_len = inputs.shape[1]
    mask = jnp.triu(jnp.full((1, 1, seq_len, seq_len), -1e8))
    logits, model_telemetry = model(
        inputs, is_training=True, collect_telemetry=with_model_telemetry, mask=mask
    )
    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, indices[:, 1:])
    )
    return loss, _LossFnAux(loss=loss, model_telemetry=model_telemetry)


def _get_optimizer(
    config: Config,
) -> optax.MultiSteps:
    cfg = config.optimizer
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(cfg.lr_min, cfg.lr_max, cfg.lr_warmup_steps),
            (
                optax.cosine_decay_schedule(
                    cfg.lr_max,
                    cfg.lr_decay_steps,
                    alpha=cfg.lr_min / cfg.lr_max,
                )
                if cfg.lr_decay_steps is not None
                else optax.constant_schedule(cfg.lr_max)
            ),
        ],
        [cfg.lr_warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.gradient_clip_norm),
        optax.adamw(
            learning_rate=1.0,
            b1=cfg.adam_b1,
            b2=cfg.adam_b2,
            weight_decay=cfg.weight_decay,
        ),
        optax.scale_by_schedule(lr_schedule),
    )
    return optax.MultiSteps(optimizer, cfg.gradient_accumulation_steps)
