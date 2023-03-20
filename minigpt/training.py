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
import jmp
import numpy as np
import optax
from chex import Array, ArrayTree, PRNGKey
from einops import rearrange

from . import data, nn
from .common import Config, get_logger

logger = get_logger()

ArrayTreeT = TypeVar("ArrayTreeT", bound=ArrayTree)
T = TypeVar("T")


@dataclass
class TrainStep:
    step: int
    loss: float
    gradients_finite: bool
    gradients: Optional[ArrayTree] = None
    params: Optional[ArrayTree] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Save:
    path: Path
    step: int
    config: Config
    params: ArrayTree
    loss_scale: jmp.LossScale
    opt_state: optax.MultiStepsState
    rng_key: PRNGKey
    seed: int


Event = Union[TrainStep, Save]


def to_cpu(t: ArrayTreeT) -> ArrayTreeT:
    """Move a Jax array tree to the CPU.

    Args:
        t: Array tree to move to the CPU.

    Returns:
        Array tree on the CPU.
    """
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
    loss_scale: Optional[jmp.LossScale] = None,
    step: Optional[int] = None,
    batches: Optional[Union[Iterable[Array], Iterable[np.ndarray]]] = None,
    save_frequency: Optional[int] = None,
    save_directory: Optional[Path] = None,
    log_gradients_frequency: Optional[int] = None,
    log_params_frequency: Optional[int] = None,
    log_param_size_on_init: bool = True,
) -> Iterable[Event]:
    """The training loop.

    Args:
        config: Configuration.
        seed: Seed to use to initiate the training loop.
        rng_key: Current random number generator key to use, corresponding to the current step.
        params: Parameters to use.
        opt_state: Optimizer state to use.
        loss_scale: Loss scale to use.
        step: The current step of the training loop.
        batches: Iterable of batches to use.
        save_frequency: Frequency at which to send a `Save` event.
        save_directory: Directory to specify for the `Save` event.
        log_gradients_frequency: Frequency at which to send a `TrainStep` event with gradients.
        log_params_frequency: Frequency at which to send a `TrainStep` event with parameters.
        log_param_size_on_init: Whether to log the parameter size on initialization.

    Yields:
        Training events. Can be either a `TrainStep` or a `Save`.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(seed)
    if params is None:
        params = nn.Model.get_params(config, seed + 1, log_size=log_param_size_on_init)
    if opt_state is None:
        opt_state = _get_optimizer(config).init(params)
    if step is None:
        step = 0
    if loss_scale is None:
        loss_scale = _get_loss_scale(config, step)
    if batches is None:
        batches = islice(
            data.batches_from_config(config, seed + 2, extra_length=1), step, None
        )
    batches = map(partial(jnp.asarray, dtype=jnp.int32), batches)
    policy = _set_amp_policy(config)
    assert params is not None
    assert opt_state is not None
    assert rng_key is not None
    assert step is not None
    assert batches is not None
    device_count = jax.device_count()
    params = _broadcast_to_devices(params)
    opt_state = _broadcast_to_devices(opt_state)
    loss_scale = _broadcast_to_devices(loss_scale)

    train_step_with_gradients = jax.pmap(
        partial(_train_step, config=config, axis_name="device", with_gradients=True),
        axis_name="device",
    )
    train_step_without_gradients = jax.pmap(
        partial(_train_step, config=config, axis_name="device", with_gradients=False),
        axis_name="device",
    )

    for batch in batches:
        rng_key, subkey = jax.random.split(rng_key)
        subkeys = jax.random.split(subkey, num=device_count)
        batch = policy.cast_to_compute(batch)
        batch = rearrange(batch, "(d b) ... -> d b ...", d=device_count)
        with_gradients = (
            log_gradients_frequency is not None and step % log_gradients_frequency == 0
        )
        train_step = (
            train_step_with_gradients
            if with_gradients
            else train_step_without_gradients
        )
        rv: _TrainStepRV = train_step(
            indices=batch,
            params=params,
            opt_state=opt_state,
            loss_scale=loss_scale,
            rng_key=subkeys,
        )
        (
            params,
            opt_state,
            loss_scale,
            loss,
            gradients,
            gradients_finite,
        ) = rv
        log_params = (
            log_params_frequency is not None and step % log_params_frequency == 0
        )
        gffd = _get_from_first_device
        yield TrainStep(
            step=step,
            loss=float(jnp.mean(loss)),
            gradients=to_cpu(gffd(gradients)) if gradients is not None else None,
            params=to_cpu(gffd(params)) if log_params else None,
            gradients_finite=bool(gffd(gradients_finite)),
        )
        if save_directory is not None:
            assert save_frequency is not None
            if step % save_frequency == 0:
                yield Save(
                    path=save_directory,
                    step=step,
                    config=config,
                    params=to_cpu(gffd(params)),
                    opt_state=to_cpu(gffd(opt_state)),
                    loss_scale=to_cpu(gffd(loss_scale)),
                    rng_key=to_cpu(rng_key),
                    seed=seed,
                )
        step += 1


class _TrainStepRV(NamedTuple):
    params: ArrayTree
    opt_state: optax.MultiStepsState
    loss_scale: jmp.LossScale
    loss: Array
    gradients: Optional[ArrayTree]
    gradients_finite: Array


def _train_step(
    *,
    indices: Array,
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    loss_scale: jmp.LossScale,
    rng_key: PRNGKey,
    config: Config,
    axis_name: str,
    with_gradients: bool,
) -> _TrainStepRV:
    """Performs a single training step.

    Args:
        indices: Indices of the batch to use.
        params: Parameters to use.
        opt_state: Optimizer state to use.
        loss_scale: Loss scale to use.
        rng_key: Random key to use.
        config: Configuration to use.
        axis_name: Axis name across which gradients are averaged.
        with_gradients: Whether to return gradients.

    Returns:
        A tuple of the new parameters, optimizer state, loss scale, loss, gradients, and whether
        gradients are finite.
    """
    loss_fn = hk.transform(partial(_loss_fn, config=config)).apply
    grad_fn = jax.grad(loss_fn, has_aux=True)
    optimizer = _get_optimizer(config)
    gradients, loss_aux = grad_fn(
        params, rng_key, indices=indices, loss_scale=loss_scale
    )
    gradients = jax.lax.pmean(gradients, axis_name=axis_name)
    gradients = loss_scale.unscale(gradients)
    gradients_finite = jmp.all_finite(gradients)
    updates, new_opt_state = optimizer.update(gradients, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    opt_state, params = jmp.select_tree(
        gradients_finite, (new_opt_state, new_params), (opt_state, params)
    )
    return _TrainStepRV(
        params=params,
        opt_state=opt_state,
        loss_scale=loss_scale,
        loss=loss_aux.loss,
        gradients=gradients if with_gradients else None,
        gradients_finite=gradients_finite,
    )


class _LossFnAux(NamedTuple):
    loss: Array


def _loss_fn(
    *,
    indices: Array,
    loss_scale: jmp.LossScale,
    config: Config,
) -> Tuple[Array, _LossFnAux]:
    """Computes the loss for a batch.

    Args:
        indices: Indices of the batch to use.
        loss_scale: Loss scale to use.
        config: Configuration to use.

    Returns:
        A tuple of the loss and a named tuple of auxiliary values.
    """
    # Prepare the model and data
    model = nn.Model.from_config(config)
    inputs = indices[:, :-1]
    targets = indices[:, 1:]
    seq_len = inputs.shape[1]
    mask = jnp.tril(jnp.full((seq_len, seq_len), True, dtype=bool))
    is_valid = (targets != config.data.pad_token_id).astype(jnp.float32)
    # Compute the loss
    logits = model(inputs, is_training=True, mask=mask)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = jnp.mean(losses * is_valid) / jnp.mean(is_valid)
    return loss_scale.scale(loss), _LossFnAux(loss=loss)


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

    # gas = gradient accumulation steps
    step_gas_pairs = tuple(config.optimizer.gradient_accumulation_steps)
    assert all(isinstance(s, int) and isinstance(g, int) for s, g in step_gas_pairs)
    assert all(s >= 0 and g > 0 for s, g in step_gas_pairs)
    pairs = sorted(step_gas_pairs, key=lambda x: x[0])
    steps, gass = map(jnp.array, zip(*pairs))

    def _gradient_accumulation_steps_schedule(step: Array) -> Array:
        return jnp.max(jnp.where(steps <= step, gass, 1))

    return optax.MultiSteps(optimizer, _gradient_accumulation_steps_schedule)


def _get_loss_scale(
    config: Config,
    step: int,
) -> jmp.LossScale:
    if not config.mixed_precision.enable:
        return jmp.NoOpLossScale()
    return jmp.DynamicLossScale(
        loss_scale=jnp.asarray(
            2**config.mixed_precision.initial_scale_log2,
            dtype=jnp.float32,
        ),
        counter=jnp.asarray(step, dtype=jnp.int32),
        period=config.mixed_precision.scale_period,
    )


def _set_amp_policy(config: Config) -> jmp.Policy:
    full = jnp.dtype(jnp.float32)
    half = jnp.dtype(jnp.float16 if config.mixed_precision.enable else jnp.float32)
    half_policy = jmp.Policy(param_dtype=full, compute_dtype=half, output_dtype=half)
    full_policy = jmp.Policy(param_dtype=full, compute_dtype=full, output_dtype=full)
    hk.mixed_precision.set_policy(nn.Model, full_policy)
    hk.mixed_precision.set_policy(nn.Block, half_policy)
    hk.mixed_precision.set_policy(hk.Linear, half_policy)
    hk.mixed_precision.set_policy(hk.LayerNorm, full_policy)
    return half_policy


def _broadcast_to_devices(obj: T) -> T:
    device_count = jax.device_count()
    fn = lambda x: (
        jnp.broadcast_to(x, (device_count, *x.shape)) if isinstance(x, Array) else x
    )
    return jax.tree_util.tree_map(fn, obj)


def _get_from_first_device(obj: T) -> T:
    fn = lambda x: x[0] if isinstance(x, Array) else x
    return jax.tree_util.tree_map(fn, obj)
