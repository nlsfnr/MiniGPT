#!/usr/bin/env python3
'''The training loop and loss function. Also implements some auxiliary
functions such as automatic logging, etc.'''
from __future__ import annotations

import csv
import logging
import sys
import time
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import count
from pathlib import Path
from typing import (Any, Dict, Iterator, Optional, Protocol, Tuple, Type,
                    TypeVar)

import click
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax
from chex import Array, ArrayTree, PRNGKey
from einops import rearrange

if __name__ == '__main__':
    # If the module is executed we need to add the parent module to the discoverable imports
    sys.path.append('.')

from minigpt import common, data, nn

logger = logging.getLogger(common.NAME)


T = TypeVar('T')


class TrainingConfig(nn.ModelConfig, Protocol):
    '''Configuration for training.'''
    batch_size: int
    use_half_precision: bool
    loss_scale_period: Optional[int]
    initial_loss_scale_log2: Optional[int]
    peak_learning_rate: float
    end_learning_rate: float
    warmup_steps: int
    total_steps: Optional[int]
    weight_decay: float

    @classmethod
    @abstractmethod
    def from_yaml(cls: Type[T], path: Path) -> T:
        raise NotImplementedError

    @abstractmethod
    def to_yaml(self: T, path: Path) -> T:
        raise NotImplementedError


@dataclass
class TelemetryData:
    '''Data to be logged during training.'''
    step: int
    epoch: int
    params: ArrayTree
    opt_state: optax.OptState
    loss_scale: jmp.LossScale
    config: TrainingConfig
    rngs: hk.PRNGSequence
    gradients: ArrayTree
    gradients_finite: bool
    loss: Array
    losses: Array
    logits: Array
    time_passed: float


def train(config: TrainingConfig,
          params: ArrayTree,
          opt_state: optax.OptState,
          dataloader: data.DataLoader,
          rngs: hk.PRNGSequence,
          loss_scale: Optional[jmp.LossScale] = None,
          step: int = 0,
          ) -> Iterator[TelemetryData]:
    '''Train the model, yielding telemetry data at each step.'''
    # Preparations
    policy = get_policy(config)
    loss_scale = get_loss_scale(config, step) if loss_scale is None else loss_scale
    train_step_jit = jax.pmap(partial(train_step, config=config, axis_name='device'),
                              axis_name='device',
                              donate_argnums=5)
    device_count = jax.device_count()
    logger.info(f'Devices found: {device_count}.')
    # Broadcast components across devices
    params = broadcast_to_devices(params)
    opt_state = broadcast_to_devices(opt_state)
    loss_scale = broadcast_to_devices(loss_scale)
    # Training loop
    t = time.perf_counter()
    for epoch in count():
        for indices in dataloader:
            indices = policy.cast_to_compute(indices)
            # Split indices and RNG betweenn devices
            indices = rearrange(indices, '(d b) ... -> d b ...', d=device_count)
            rng = jax.random.split(next(rngs), num=device_count)
            params, opt_state, loss_scale, telemetry_dict = train_step_jit(
                indices, params, opt_state, loss_scale, rng)
            yield TelemetryData(
                step=step,
                epoch=epoch,
                params=get_from_first_device(params),
                opt_state=get_from_first_device(opt_state),
                loss_scale=get_from_first_device(loss_scale),
                config=config,
                rngs=rngs,
                gradients=get_from_first_device(telemetry_dict['gradients']),
                loss=jnp.mean(telemetry_dict['loss']),
                losses=concat_from_devices(telemetry_dict['losses']),
                logits=concat_from_devices(telemetry_dict['logits']),
                gradients_finite=telemetry_dict['gradients_finite'].all(),
                time_passed=time.perf_counter() - t)
            step += 1
            t = time.perf_counter()
        logger.info(f'Epoch {epoch + 1:,} finished')


def train_step(indices: Array,
               params: ArrayTree,
               opt_state: optax.OptState,
               loss_scale: jmp.LossScale,
               rng: PRNGKey,
               *,
               config: TrainingConfig,
               axis_name: str,
               ) -> Tuple[ArrayTree,
                          optax.OptState,
                          jmp.LossScale,
                          Dict[str, Any]]:
    # Preparations
    loss_hk = hk.transform(partial(loss_fn, config=config))
    grad_fn = jax.grad(loss_hk.apply, has_aux=True)
    optimizer = get_optimizer(config)
    # Execution
    gradients, telemetry_dict = grad_fn(params, rng, indices, loss_scale)
    gradients = jax.lax.pmean(gradients, axis_name=axis_name)
    gradients = loss_scale.unscale(gradients)
    gradients_finite = jmp.all_finite(gradients)
    loss_scale = loss_scale.adjust(gradients_finite)
    updates, new_opt_state = optimizer.update(gradients, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    # Only actually update the params and opt_state if all gradients were finite
    opt_state, params = jmp.select_tree(
        gradients_finite,
        (new_opt_state, new_params),
        (opt_state, params))
    return (params,
            opt_state,
            loss_scale,
            dict(telemetry_dict,
                 gradients=gradients,
                 gradients_finite=gradients_finite))


def loss_fn(indices: Array,
            loss_scale: jmp.LossScale,
            *,
            config: TrainingConfig,
            ) -> Tuple[Array, Dict[str, Any]]:
    model = nn.Model.from_config(config)
    logits = model(indices[:, :-1], is_training=True)
    one_hot_targets = hk.one_hot(indices[:, 1:], logits.shape[-1])
    losses = optax.softmax_cross_entropy(logits, one_hot_targets)
    loss = jnp.mean(losses)
    scaled_loss = loss_scale.scale(loss)
    return scaled_loss, dict(logits=logits,
                             losses=losses,
                             loss=loss)


def get_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    '''Get the optimizer with linear warmup and cosine decay.'''
    return optax.adamw(get_learning_rate_schedule(config),
                       weight_decay=config.weight_decay)


def get_policy(config: TrainingConfig) -> jmp.Policy:
    '''Get and set the policy for mixed precision training.'''
    full = jnp.float32
    half = jnp.float16 if config.use_half_precision else jnp.float32
    half_policy = jmp.Policy(param_dtype=full,
                             compute_dtype=half,
                             output_dtype=full)
    full_policy = jmp.Policy(param_dtype=full,
                             compute_dtype=full,
                             output_dtype=full)
    hk.mixed_precision.set_policy(nn.Model, half_policy)
    hk.mixed_precision.set_policy(hk.LayerNorm, full_policy)
    return half_policy


def get_loss_scale(config: TrainingConfig,
                   step: int,
                   ) -> jmp.LossScale:
    '''Get the loss scale for mixed precision training.'''
    if not config.use_half_precision:
        return jmp.NoOpLossScale()
    msg = 'initial_loss_scale_log2 must be set for mixed precision training.'
    assert config.initial_loss_scale_log2 is not None, msg
    if config.loss_scale_period is None:
        return jmp.StaticLossScale(
            2. ** jnp.asarray(config.initial_loss_scale_log2))
    return jmp.DynamicLossScale(
        2. ** jnp.asarray(config.initial_loss_scale_log2),
        counter=jnp.asarray(step % config.loss_scale_period),
        period=config.loss_scale_period)


def get_learning_rate_schedule(config: TrainingConfig) -> optax.Schedule:
    '''Get the learning rate schedule with linear warmup and optional cosine decay.'''
    if config.total_steps is not None:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.,
            peak_value=config.peak_learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.total_steps - config.warmup_steps,
            end_value=config.end_learning_rate,
        )
    else:
        schedules = [
            optax.linear_schedule(
                init_value=0.,
                end_value=config.peak_learning_rate,
                transition_steps=config.warmup_steps),
            optax.constant_schedule(config.peak_learning_rate),
        ]
        lr_schedule = optax.join_schedules(schedules, [config.warmup_steps])
    return lr_schedule


def get_optimizer_state(config: TrainingConfig,
                        params: ArrayTree,
                        ) -> optax.OptState:
    '''Get the optimizer state.'''
    optimizer = get_optimizer(config)
    opt_state = optimizer.init(params)
    opt_state_n = hk.data_structures.tree_size(opt_state)
    opt_state_mb = round(hk.data_structures.tree_bytes(opt_state) / 1e6, 2)
    logger.info(f'Optimizer state: {opt_state_n:,} ({opt_state_mb:.2f} MB)')
    return opt_state


def broadcast_to_devices(obj: T) -> T:
    device_count = jax.device_count()
    fn = lambda x: (jnp.broadcast_to(x, (device_count, *x.shape))
                    if isinstance(x, Array) else
                    x)
    return jax.tree_util.tree_map(fn, obj)


def get_from_first_device(obj: T) -> T:
    fn = lambda x: x[0] if isinstance(x, Array) else x
    return jax.tree_util.tree_map(fn, obj)


def concat_from_devices(obj: T) -> T:
    fn = lambda x: rearrange(x, 'd b ... -> (d b) ...') if isinstance(x, Array) else x
    return jax.tree_util.tree_map(fn, obj)


def autosave(telemetry_iter: Iterator[TelemetryData],
             frequency: int,
             path: Path,
             ) -> Iterator[TelemetryData]:
    '''Save the model parameters and optimizer state etc. at regular intervals.'''
    for telemetry in telemetry_iter:
        if not isinstance(telemetry.config, common.YamlConfig):
            raise ValueError('The config must be a YamlConfig to be saved.')
        if telemetry.step % frequency == 0:
            common.save_checkpoint(path,
                                   config=telemetry.config,
                                   params=telemetry.params,
                                   opt_state=telemetry.opt_state,
                                   rngs=telemetry.rngs,
                                   loss_scale=telemetry.loss_scale,
                                   step=telemetry.step)
        yield telemetry


def autolog(telemetry_iter: Iterator[TelemetryData],
            frequency: int,
            ) -> Iterator[TelemetryData]:
    '''Log the telemetry data at the specified frequency.'''
    loss_history = []
    time_history = []
    for telemetry in telemetry_iter:
        loss_history.append(telemetry.loss)
        time_history.append(telemetry.time_passed)
        if not telemetry.gradients_finite:
            logger.warning('Non-finite gradients')
        if telemetry.step % frequency == 0 and loss_history:
            mean_loss = jnp.mean(jnp.asarray(loss_history))
            mean_time = jnp.mean(jnp.asarray(time_history))
            logger.info(f'Step: {telemetry.step:,}'
                        f' | loss: {mean_loss:.4f}'
                        f' | S/step: {mean_time:.4f}')
            loss_history.clear()
            time_history.clear()
        yield telemetry


def log_to_csv(telemetry_iter: Iterator[TelemetryData],
               path: Path,
               ) -> Iterator[TelemetryData]:
    '''Log the telemetry data to a CSV file.'''
    lr_sched = None
    path.parent.mkdir(parents=True, exist_ok=True)
    did_exist = path.exists()
    if not did_exist:
        path.touch()
    with path.open('a') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'time', 'step', 'epoch', 'loss', 'learning_rate', 'time_passed'])
        if not did_exist:
            writer.writeheader()
        for telemetry in telemetry_iter:
            if lr_sched is None:
                lr_sched = get_learning_rate_schedule(telemetry.config)
            writer.writerow(dict(time=datetime.now().isoformat(),
                                 step=telemetry.step,
                                 epoch=telemetry.epoch,
                                 loss=telemetry.loss,
                                 learning_rate=lr_sched(telemetry.step),
                                 time_passed=telemetry.time_passed))
            yield telemetry


class Config(common.YamlConfig):

    # Training config
    batch_size: int
    use_half_precision: bool
    loss_scale_period: Optional[int]
    initial_loss_scale_log2: Optional[int]
    peak_learning_rate: float
    end_learning_rate: float
    warmup_steps: int
    total_steps: Optional[int]
    weight_decay: float

    # Model config
    vocab_size: int
    embedding_size: int
    max_sequence_length: int
    num_layers: int
    num_heads: int
    value_size: int
    key_size: int
    w_init_var: float
    embed_init_var: float
    mlp_size: Optional[int] = None
    model_size: Optional[int] = None
    dropout: float = 0.1

    # Data config
    dataset_path: Path
    tokenizer_path: Path

    # DataLoader config
    num_workers: int


def get_cli() -> click.Group:
    '''Get the command line interface for this module.'''

    cli = common.get_cli_group('training')

    @cli.command('train')
    @click.option('--config-path', '-c', type=Path, default=None,
                  help='Path to the configuration file')
    @click.option('--load-from', '-l', type=Path, default=None,
                  help='Path to a checkpoint to resume training')
    @click.option('--save-path', '-o', type=Path, default=None,
                  help='Path to save checkpoints automatically')
    @click.option('--save-frequency', '-f', type=int, default=1000,
                  help='Frequency at which to save checkpoints automatically')
    @click.option('--log-frequency', type=int, default=10,
                  help='Frequency at which to log metrics automatically')
    @click.option('--csv-path', type=Path, default=None,
                  help='Path to save metrics in a CSV file')
    @click.option('--stop-at', type=int, default=None,
                  help='Stop training after this many steps')
    @click.option('--seed', type=int, default=None, help='Random seed')
    def cli_train(config_path: Optional[Path],
                  load_from: Optional[Path],
                  save_path: Optional[Path],
                  save_frequency: int,
                  log_frequency: int,
                  csv_path: Optional[Path],
                  stop_at: Optional[int],
                  seed: Optional[int],
                  ) -> None:
        '''Train a model.'''
        if config_path is None and load_from is None:
            raise ValueError('Either a configuration file or a checkpoint must be provided')
        if config_path is not None and load_from is not None:
            raise ValueError('Only one of configuration file or checkpoint must be provided')
        if config_path is not None:
            config = Config.from_yaml(config_path)
            rngs = common.get_rngs(seed)
            params = nn.Model.get_params(config, next(rngs))
            opt_state = get_optimizer_state(config, params)
            step = 0
            loss_scale = None
        else:
            assert load_from is not None
            checkpoint = common.load_checkpoint(load_from, config_class=Config)
            config = checkpoint['config']
            rngs = checkpoint['rngs']
            params = checkpoint['params']
            opt_state = checkpoint['opt_state']
            step = checkpoint['step']
            loss_scale = checkpoint['loss_scale']
        dataloader = data.LMDBDataset.from_config(config).get_dataloader_from_config(config)
        telemetry_iter = train(config=config,
                               params=params,
                               opt_state=opt_state,
                               dataloader=dataloader,
                               rngs=rngs,
                               loss_scale=loss_scale,
                               step=step)
        if save_path is not None:
            telemetry_iter = autosave(telemetry_iter, save_frequency, save_path)
        if csv_path is not None:
            telemetry_iter = log_to_csv(telemetry_iter, csv_path)
        telemetry_iter = autolog(telemetry_iter, log_frequency)
        i = stop_at if stop_at is not None else -1
        while i != 0:
            next(telemetry_iter)
            i -= 1

    return cli


if __name__ == '__main__':
    get_cli()()
