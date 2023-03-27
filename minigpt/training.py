from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, TypeVar, Union

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax
from chex import ArrayTree, PRNGKey
from einops import rearrange
from jax import Array

from . import nn
from .common import Config, get_logger

logger = get_logger()

ArrayTreeT = TypeVar("ArrayTreeT", bound=ArrayTree)
T = TypeVar("T")


@dataclass
class TrainStep:
    step: int
    has_updated: bool
    loss: float
    gradients_finite: bool
    loss_scale_log2: float
    gradients: Optional[ArrayTree]
    params: Optional[ArrayTree]
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


class EndOfTraining:
    pass


Event = Union[TrainStep, Save, EndOfTraining]


class StopTraining(Exception):
    pass


class Trainer(threading.Thread):
    def __init__(
        self,
        *,
        batch_queue: queue.Queue,
        event_queue: queue.Queue,
        config: Config,
        seed: int,
        termination_event: Optional[threading.Event] = None,
        timeout: float = 0.1,
        rng_key: Optional[PRNGKey] = None,
        params: Optional[ArrayTree] = None,
        opt_state: Optional[optax.MultiStepsState] = None,
        loss_scale: Optional[jmp.LossScale] = None,
        step: Optional[int] = None,
        save_frequency: Optional[int] = None,
        save_directory: Optional[Path] = None,
        log_gradients_frequency: Optional[int] = None,
        log_params_frequency: Optional[int] = None,
        log_param_size_on_init: bool = True,
    ) -> None:
        super().__init__()
        self.batch_queue = batch_queue
        self.event_queue = event_queue
        self.config = config
        self.seed = seed
        self.termination_event = termination_event or threading.Event()
        self.timeout = timeout
        self.rng_key = rng_key or jax.random.PRNGKey(seed)
        self.params = params or nn.Model.get_params(config, seed + 1)
        self.opt_state = opt_state or _get_optimizer(config).init(self.params)
        self.step = step or 0
        self.loss_scale = loss_scale or _get_loss_scale(config, self.step)
        self.save_frequency = save_frequency
        self.save_directory = save_directory
        self.log_gradients_frequency = log_gradients_frequency
        self.log_params_frequency = log_params_frequency
        self.log_param_size_on_init = log_param_size_on_init
        self._exception: Optional[Exception] = None
        self._policy = _set_amp_policy(self.config)
        # Two training functions, one which returns gradients and one which does not.
        self._train_step_with_gradients = jax.pmap(
            partial(
                _train_step, config=self.config, axis_name="device", with_gradients=True
            ),
            axis_name="device",
        )
        self._train_step_without_gradients = jax.pmap(
            partial(
                _train_step,
                config=self.config,
                axis_name="device",
                with_gradients=False,
            ),
            axis_name="device",
        )

    def run(self) -> None:
        logger.info(f"Starting training loop at step {self.step}.")
        try:
            self.loop()
        except StopTraining:
            logger.info(f"Training loop terminated at step {self.step}.")
        except Exception as e:
            logger.exception(f"Training loop failed at step {self.step}.")
            self._exception = e
            raise

    def join(self, timeout: Optional[float] = None) -> None:
        super().join(timeout)
        if self._exception is not None:
            raise self._exception

    def terminate(self, timeout: Optional[float] = None) -> Trainer:
        logger.info("Terminating training thread.")
        self.termination_event.set()
        self.join(timeout)
        return self

    def save_and_terminate(self, timeout: Optional[float] = None) -> Trainer:
        self._save()
        return self.terminate(timeout)

    def emit_end_of_training_event(self) -> Trainer:
        if self.is_alive():
            raise RuntimeError(
                "Cannot emit EndOfTraining event while running."
                " Call terminate() or save_and_terminate() first."
            )
        self.event_queue.put(EndOfTraining())
        return self

    def __enter__(self) -> Trainer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self.is_alive():
            self.terminate()
        self.join()

    def loop(self) -> Trainer:
        self.params = _broadcast_to_devices(self.params)
        self.opt_state = _broadcast_to_devices(self.opt_state)
        self.loss_scale = _broadcast_to_devices(self.loss_scale)
        while True:
            with_gradients = (
                self.log_gradients_frequency is not None
                and self.step % self.log_gradients_frequency == 0
            )
            with_params = (
                self.log_params_frequency is not None
                and self.step % self.log_params_frequency == 0
            )
            self.train_step(self._fetch_batch(), with_gradients, with_params)

    def train_step(
        self,
        batch: Array,
        with_gradients: bool,
        with_params: bool,
    ) -> Trainer:
        """Performs one training step. Notably, this does not mean that the parameters will be
        updated. This is because we use a multi-step optimizer, which means that we accumulate
        gradients over multiple steps before updating the parameters."""
        # Split the batch across devices.
        device_count = jax.device_count()
        batch = self._policy.cast_to_compute(batch)
        batch = rearrange(batch, "(d b) ... -> d b ...", d=device_count)
        # Select the training function, depending on whether we want gradients or not.
        train_step = (
            self._train_step_with_gradients
            if with_gradients
            else self._train_step_without_gradients
        )
        # Get a new RNG key for each device
        self.rng_key, subkey = jax.random.split(self.rng_key)
        subkeys = jax.random.split(subkey, device_count)
        # Run the training step.
        retval = train_step(
            indices=batch,
            params=self.params,
            opt_state=self.opt_state,
            loss_scale=self.loss_scale,
            rng_key=subkeys,
        )
        (
            self.params,
            self.opt_state,
            self.loss_scale,
            loss,
            gradients,
            gradients_finite,
            has_updated,
        ) = retval
        # Emit the event.
        gffd = _get_from_first_device
        self._emit_event(
            TrainStep(
                step=self.step,
                has_updated=bool(gffd(has_updated)),
                loss=float(jnp.mean(loss)),
                gradients=gffd(gradients),
                params=gffd(self.params) if with_params else None,
                gradients_finite=bool(gffd(gradients_finite)),
                loss_scale_log2=round(
                    float(gffd(jnp.log2(self.loss_scale.loss_scale)))
                ),
            )
        )
        # If this is a gradient-accumulation step, don't update the step count.
        if not has_updated.all():
            return self
        # Update the step count.
        self.step += 1
        # Save the model.
        if (
            self.save_frequency is not None
            and self.save_directory is not None
            and self.step % self.save_frequency == 0
        ):
            self._save()
        return self

    def _save(self) -> Trainer:
        """Emits a save event."""
        if self.save_directory is None:
            return self
        gffd = _get_from_first_device
        self._emit_event(
            Save(
                path=self.save_directory,
                step=self.step,
                config=self.config,
                params=gffd(self.params),
                loss_scale=gffd(self.loss_scale),
                opt_state=gffd(self.opt_state),
                rng_key=self.rng_key,
                seed=self.seed,
            )
        )
        return self

    def _emit_event(
        self,
        event: Event,
    ) -> Trainer:
        """Emits an event."""
        while True:
            if self.termination_event.is_set():
                raise StopTraining("Termination event set when emitting event.")
            try:
                self.event_queue.put(event, timeout=self.timeout)
                break
            except queue.Full:
                pass
        return self

    def _fetch_batch(
        self,
    ) -> Array:
        """Fetches a batch from the batch queue."""
        cpu = jax.devices("cpu")[0]
        while True:
            if self.termination_event.is_set():
                raise StopTraining("Termination event set when fetching batch.")
            try:
                batch = self.batch_queue.get(timeout=self.timeout)
                batch = jnp.asarray(batch, dtype=jnp.int32)
                batch = jax.device_put(batch, device=cpu)
                return batch
            except queue.Empty:
                pass


class _TrainStepRV(NamedTuple):
    params: ArrayTree
    opt_state: optax.MultiStepsState
    loss_scale: jmp.LossScale
    loss: Array
    gradients: Optional[ArrayTree]
    gradients_finite: Array
    has_updated: Array


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
        A tuple of the new parameters, optimizer state, loss scale, loss, gradients, whether
        gradients are finite and whether the parameters have been updated.
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
    loss_scale = loss_scale.adjust(gradients_finite)
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
        has_updated=optimizer.has_updated(opt_state),
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
    # Assemble the lr schedule
    parts = [optax.linear_schedule(cfg.lr_min, cfg.lr_max, cfg.lr_warmup_steps)]
    if cfg.lr_decay_steps is None:
        parts += [optax.constant_schedule(cfg.lr_max)]
    else:
        parts += [
            (
                optax.cosine_decay_schedule(
                    cfg.lr_max, cfg.lr_decay_steps, alpha=cfg.lr_min / cfg.lr_max
                )
            )
        ]
    lr_schedule = optax.join_schedules(parts, [cfg.lr_warmup_steps])
    # Assemble the optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.gradient_clip_norm),
        # Adam + weight decay = AdamW
        optax.scale_by_adam(b1=cfg.adam_b1, b2=cfg.adam_b2),
        optax.add_decayed_weights(weight_decay=cfg.weight_decay),
        # We want gradient descent not ascent, so we negate the learning rate
        optax.scale_by_schedule(lambda step: -lr_schedule(step)),
    )
    # Assemble the multi-steps optimizer for GAS
    step_gas_pairs = tuple(config.optimizer.gradient_accumulation_steps)
    if not all(isinstance(s, int) and isinstance(g, int) for s, g in step_gas_pairs):
        raise TypeError(
            f"Expected gradient_accumulation_steps to be a sequence of (int, int) pairs, got "
            f"{step_gas_pairs}"
        )
    if not all(s >= 0 and g > 0 for s, g in step_gas_pairs):
        raise ValueError(
            f"Expected gradient_accumulation_steps to be a sequence of (int, int) pairs with "
            f"non-negative steps and positive gas, got {step_gas_pairs}"
        )
    pairs = sorted(step_gas_pairs, key=lambda x: x[0])
    steps, gass = map(jnp.array, zip(*pairs))
    return optax.MultiSteps(
        optimizer, lambda step: jnp.max(jnp.where(steps <= step, gass, 1))
    )


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
    policy = jmp.Policy(param_dtype=full, compute_dtype=half, output_dtype=half)
    hk.mixed_precision.set_policy(nn.Model, policy)
    hk.mixed_precision.set_policy(nn.Block, policy)
    hk.mixed_precision.set_policy(nn.MultiHeadAttention, policy)
    hk.mixed_precision.set_policy(nn.FeedForward, policy)
    hk.mixed_precision.set_policy(hk.Embed, policy)
    hk.mixed_precision.set_policy(hk.Linear, policy)
    hk.mixed_precision.set_policy(hk.LayerNorm, policy)
    return policy


def _broadcast_to_devices(obj: T) -> T:
    """Broadcasts a tree of arrays to all devices."""
    device_count = jax.device_count()

    def fn(x: Array) -> Array:
        x = jax.device_put(x, jax.devices("cpu")[0])
        x = jnp.broadcast_to(x, (device_count, *x.shape)) if isinstance(x, Array) else x
        return jax.pmap(lambda x: x, axis_name="batch")(x)

    return jax.tree_util.tree_map(fn, obj)


def _get_from_first_device(obj: T) -> T:
    """Gets a tree of arrays from the first device, putting it on the CPU."""
    cpu = jax.devices("cpu")[0]
    fn = lambda x: jax.device_put(x[0], cpu) if isinstance(x, Array) else x
    return jax.tree_util.tree_map(fn, obj)
