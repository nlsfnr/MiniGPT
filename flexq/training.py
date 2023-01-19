#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import pickle
import random
import time
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import count
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, Optional, Protocol, Tuple,
                    Type, TypeVar, Union)

import click
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pydantic
import yaml
from chex import Array, ArrayTree, PRNGKey
from einops import rearrange

if __name__ == '__main__':
    # If the module is executed we need to add flexq to the discoverable imports
    import sys
    sys.path.append('.')

from flexq import data, nn

logger = logging.getLogger('NoLo')


T = TypeVar('T')


class TrainingConfig(data.DataConfig,
                     data.DataLoaderConfig,
                     nn.ModelConfig,
                     Protocol):
    '''Configuration for training.'''
    batch_size: int
    gradient_accumulation_steps: int  # Must divide batch_size
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
    config: TrainingConfig
    rngs: hk.PRNGSequence
    time_taken_s: float
    gradients: ArrayTree
    loss: Array
    losses: Array
    logits: Array


def train(config: TrainingConfig,
          params: ArrayTree,
          opt_state: optax.OptState,
          rngs: hk.PRNGSequence,
          step: int = 0,
          ) -> Iterator[TelemetryData]:
    '''Train the model, yielding telemetry data at each step.'''
    # Prepare the configuration, dataset and dataloader
    dataset = data.LMDBDataset.from_config(config)
    dataloader = dataset.get_dataloader_from_config(
        config, additional_sequence_length=1)
    # Prepare the loss function
    train_step_jit = jax.jit(partial(train_step, config=config))
    # Training loop
    for epoch in count():
        for indices in dataloader:
            # Note that due to the JAX's asynchronous dispatch, the timing
            # information is not accurate within one step and should only be
            # considered across multiple steps.
            # See: https://jax.readthedocs.io/en/latest/async_dispatch.html
            start_time = time.perf_counter()
            params, opt_state, telemetry_dict = train_step_jit(
                indices, params, opt_state, next(rngs))
            end_time = time.perf_counter()
            yield TelemetryData(step=step,
                                epoch=epoch,
                                params=params,
                                opt_state=opt_state,
                                config=config,
                                rngs=rngs,
                                time_taken_s=end_time - start_time,
                                **telemetry_dict)
            step += 1
        logger.info(f'Epoch {epoch + 1:,} finished')


def train_step(indices: Iterable[Array],
               params: ArrayTree,
               opt_state: optax.OptState,
               rng: PRNGKey,
               *,
               config: TrainingConfig,
               ) -> Tuple[ArrayTree, optax.OptState, Dict[str, Any]]:
    # Preparations
    loss_hk = hk.transform(partial(loss_fn, config=config))
    grad_fn = jax.grad(loss_hk.apply, has_aux=True)
    optimizer = get_optimizer(config)
    # Execution
    gradients, telemetry_dict = grad_fn(params, rng, indices)
    updates, new_opt_state = optimizer.update(gradients, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, dict(telemetry_dict, gradients=gradients)


def get_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    '''Get the optimizer with linear warmup and cosine decay.'''
    return optax.adamw(get_learning_rate_schedule(config),
                       weight_decay=config.weight_decay)


def get_learning_rate_schedule(config: TrainingConfig) -> optax.Schedule:
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


def loss_fn(indices: Array,
            *,
            config: TrainingConfig,
            ) -> Tuple[Array, Dict[str, Any]]:
    model = nn.Model.from_config(config)
    # Accumulate the loss
    indices_splits = rearrange(indices, '(o b) ... -> o b ...',
                               o=config.gradient_accumulation_steps)
    loss = jnp.zeros((), dtype=jnp.float32)
    all_logits = []
    all_losses = []
    for split in indices_splits:
        logits = model(split[:, :-1], is_training=True)
        all_logits.append(logits)
        one_hot_targets = hk.one_hot(split[:, 1:], logits.shape[-1])
        losses = optax.softmax_cross_entropy(logits, one_hot_targets)
        all_losses.append(losses)
        loss += jnp.mean(losses)
    loss = loss / len(indices_splits)
    return loss, dict(logits=jnp.concatenate(all_logits, axis=0),
                      losses=jnp.concatenate(all_losses, axis=0),
                      loss=loss)


def autosave(telemetry_iter: Iterator[TelemetryData],
             frequency: int,
             path: Path,
             ) -> Iterator[TelemetryData]:
    '''Save the model parameters and optimizer state etc. at regular intervals.'''
    for telemetry in telemetry_iter:
        if telemetry.step % frequency == 0:
            import matplotlib.pyplot as plt
            breakpoint()
            save_checkpoint(path,
                            config=telemetry.config,
                            params=telemetry.params,
                            opt_state=telemetry.opt_state,
                            rngs=telemetry.rngs,
                            step=telemetry.step)
        yield telemetry


def autolog(telemetry_iter: Iterator[TelemetryData],
            frequency: int,
            ) -> Iterator[TelemetryData]:
    '''Log the telemetry data at the specified frequency.'''
    loss_history = []
    time_taken_history = []
    for telemetry in telemetry_iter:
        loss_history.append(telemetry.loss)
        time_taken_history.append(telemetry.time_taken_s)
        if telemetry.step % frequency == 0 and loss_history:
            mean_loss = jnp.mean(jnp.asarray(loss_history))
            mean_time_taken = jnp.mean(jnp.asarray(time_taken_history))
            logger.info(f'Step: {telemetry.step:,}'
                        f' | loss: {mean_loss:.4f}'
                        f' | S/step: {mean_time_taken:.4f}')
            loss_history.clear()
            time_taken_history.clear()
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
            'time', 'step', 'epoch', 'loss', 'time_taken_s', 'learning_rate'])
        if not did_exist:
            writer.writeheader()
        for telemetry in telemetry_iter:
            if lr_sched is None:
                lr_sched = get_learning_rate_schedule(telemetry.config)
            writer.writerow(dict(time=datetime.now().isoformat(),
                                 step=telemetry.step,
                                 epoch=telemetry.epoch,
                                 loss=telemetry.loss,
                                 time_taken_s=telemetry.time_taken_s,
                                 learning_rate=lr_sched(telemetry.step)))
            yield telemetry


def save_checkpoint(path: Path,
                    config: TrainingConfig,
                    params: ArrayTree,
                    opt_state: optax.OptState,
                    rngs: hk.PRNGSequence,
                    step: int,
                    ) -> None:
    '''Save the checkpoint to a directory.'''
    path.mkdir(parents=True, exist_ok=True)
    # Save the configuration
    config.to_yaml(path / 'config.yaml')
    # Save the parameters
    with open(path / 'params.pkl', 'wb') as f:
        pickle.dump(params, f)
    # Save the optimizer state
    with open(path / 'opt_state.pkl', 'wb') as f:
        pickle.dump(opt_state, f)
    # Save the step as a yaml file
    with open(path / 'other.yaml', 'w') as f:
        yaml.dump(dict(step=step), f)
    # Save the RNGs
    with open(path / 'rngs.pkl', 'wb') as f:
        pickle.dump(rngs, f)
    logger.info(f'Saved checkpoint to {path} at step {step:,}.')


def load_checkpoint(path: Path,
                    config_class: Type[TrainingConfig],
                    ) -> Dict[str, Any]:
    '''Load the checkpoint from a directory.'''
    config = config_class.from_yaml(path / 'config.yaml')
    # Load the parameters
    with open(path / 'params.pkl', 'rb') as f:
        params = pickle.load(f)
    # Load the optimizer state
    with open(path / 'opt_state.pkl', 'rb') as f:
        opt_state = pickle.load(f)
    # Load the step from the yaml file
    with open(path / 'other.yaml', 'r') as f:
        other = yaml.load(f, Loader=yaml.FullLoader)
    step = other['step']
    # Load the RNGs
    with open(path / 'rngs.pkl', 'rb') as f:
        rngs_internal_state = pickle.load(f).internal_state
    rngs = hk.PRNGSequence(0)
    rngs.replace_internal_state(rngs_internal_state)
    return dict(config=config,
                params=params,
                opt_state=opt_state,
                rngs=rngs,
                step=step)


def get_cli() -> click.Group:
    '''Get the command line interface for this module.'''

    class Config(pydantic.BaseModel):

        # Training config
        batch_size: int
        gradient_accumulation_steps: int  # Must divide batch_size
        peak_learning_rate: float
        end_learning_rate: float
        warmup_steps: int
        total_steps: Optional[int]
        weight_decay: float

        # Model config
        vocab_size: int
        max_sequence_length: int
        num_layers: int
        num_heads: int
        value_size: int
        w_init_var: float
        embed_init_var: float
        mlp_size: Optional[int] = None
        model_size: Optional[int] = None
        dropout: float = 0.1

        # Data config
        dataset_path: Path
        tokenizer_path: Path
        min_length: int
        min_chars_per_token: float

        # DataLoader config
        num_workers: int

        @classmethod
        def from_yaml(cls, path: Path):
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls(**data)

        def to_yaml(self, path: Path) -> Config:
            with open(path, "w") as f:
                # Use self.json() instead of self.dict() to avoid having to catet
                # to edge cases such as serializing Paths.
                yaml.dump(json.loads(self.json()), f)
            return self

    def set_debug(debug: bool) -> None:
        jax.config.update('jax_debug_nans', debug)
        jax.config.update('jax_debug_infs', debug)
        jax.config.update('jax_disable_jit', debug)

    def get_rngs(seed: Optional[Union[hk.PRNGSequence, int]] = None) -> hk.PRNGSequence:
        '''Get a PRNG sequence from an int or an existing PRNG sequence.'''
        if isinstance(seed, hk.PRNGSequence):
            return seed
        seed = (random.randint(0, 2**32 - 1)
                if seed is None else
                seed)
        logger.info(f'Using seed {seed}')
        return hk.PRNGSequence(seed)

    @click.group('NoLo-Data')
    @click.option('--log-level', default='INFO', help='Log level')
    def cli(log_level: str) -> None:
        logging.basicConfig(level=log_level,
                            format='[%(asctime)s|%(name)s|%(levelname)s] %(message)s')
        logger.info('Starting NoLo-Data')

    @cli.command('train')
    @click.option('--config-path', '-c', type=Path, default=None,
                  help='Path to the configuration file')
    @click.option('--checkpoint-path', '-f', type=Path, default=None,
                  help='Path to a checkpoint to resume training')
    @click.option('--autosave-path', '-o', type=Path, default=None,
                  help='Path to save checkpoints automatically')
    @click.option('--autosave-frequency', '-f', type=int, default=1000,
                  help='Frequency at which to save checkpoints automatically')
    @click.option('--autolog-frequency', '-l', type=int, default=10,
                  help='Frequency at which to log metrics automatically')
    @click.option('--csv-path', type=Path, default=None,
                  help='Path to save metrics in a CSV file')
    @click.option('--seed', type=int, default=None, help='Random seed')
    @click.option('--debug', '-d', is_flag=True, help='Debug mode')
    def cli_train(config_path: Optional[Path],
                  checkpoint_path: Optional[Path],
                  autosave_path: Optional[Path],
                  autosave_frequency: int,
                  autolog_frequency: int,
                  csv_path: Optional[Path],
                  seed: Optional[int],
                  debug: bool,
                  ) -> None:
        '''Train a model.'''
        set_debug(debug)
        if config_path is None and checkpoint_path is None:
            raise ValueError('Either a configuration file or a checkpoint must be provided')
        if config_path is not None and checkpoint_path is not None:
            raise ValueError('Only one of configuration file or checkpoint must be provided')
        if config_path is not None:
            config = Config.from_yaml(config_path)
            rngs = get_rngs(seed)
            params = nn.Model.get_params(config, next(rngs))
            opt_state = get_optimizer_state(config, params)
            step = 0
        else:
            assert checkpoint_path is not None
            checkpoint = load_checkpoint(checkpoint_path, config_class=Config)
            config = checkpoint['config']
            rngs = checkpoint['rngs']
            params = checkpoint['params']
            opt_state = checkpoint['opt_state']
            step = checkpoint['step']
        telemetry_iter = train(config, params, opt_state, rngs, step)
        if autosave_path is not None:
            telemetry_iter = autosave(telemetry_iter, autosave_frequency, autosave_path)
        if csv_path is not None:
            telemetry_iter = log_to_csv(telemetry_iter, csv_path)
        telemetry_iter = autolog(telemetry_iter, autolog_frequency)
        for _ in telemetry_iter:
            pass

    return cli


if __name__ == '__main__':
    # Set the parent packaget to make this file executable as a standalone file
    import sys
    sys.path.append('..')

    get_cli()()
