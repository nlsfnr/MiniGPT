#!/usr/bin/env python3
import sys
from functools import partial
from itertools import islice
from pathlib import Path
from pprint import pformat
from typing import Optional, Tuple

import click
import optax
from chex import ArrayTree, PRNGKey

sys.path.append(".")

import minigpt  # noqa: E402

logger = minigpt.get_logger()


@click.group()
@click.option("--debug", is_flag=True)
def cli(
    debug: bool,
) -> None:
    minigpt.setup_logging()
    minigpt.set_debug(debug)


# fmt: off
@cli.command("train")
@click.option("--config-path", "-c", type=Path, default=None,
              help="Path to the config file.")
@click.option("--seed", "-s", type=int, default=None,
              help="Random seed.")
@click.option("--load-from", "-l", type=Path, default=None,
              help="Path to the checkpoint to load from. Default None")
@click.option("--save-frequency", "-sf", type=int, default=250,
              help="Save frequency. Default 250.")
@click.option("--save-directory", "-sd", type=Path, default=None,
              help="Path to the directory to save checkpoints to. Default None.")
@click.option("--log-frequency", "-lf", type=int, default=10,
              help="Log frequency. Default 10.")
@click.option("--log-time-per-step-frequency", "-tf", type=int, default=100,
              help="Log time per step frequency. Default 100.")
@click.option("--log-time-per-step-percentiles", "-tp", type=int, multiple=True,
              default=(1, 50, 99), help="Percentiles to log for time per step. Default "
              "(1, 50, 99).")
@click.option("--wandb-disable", "-wd", is_flag=True,
              help="Disable wandb.")
@click.option("--wandb-project", "-wp", type=str, default="MiniGPT",
              help="Wandb project name. Default MiniGPT.")
@click.option("--wandb-group", "-wg", type=str, default=None,
              help="Wandb group name. Default None.")
@click.option("--wandb-name", "-wn", type=str, default=None,
              help="Wandb run name. Default None.")
@click.option("--wandb-tags", "-wt", type=str, multiple=True, default=[],
              help="Wandb tags. Default []")
@click.option("--data-buffer", "-db", type=int, default=10,
              help="Data buffer size. Default 10.")
@click.option("--event-buffer", "-eb", type=int, default=10,
              help="Event buffer size. Default 10.")
# fmt: on
def train_new(
    config_path: Optional[Path],
    seed: Optional[int],
    load_from: Optional[Path],
    save_frequency: int,
    save_directory: Optional[Path],
    log_frequency: int,
    log_time_per_step_frequency: int,
    log_time_per_step_percentiles: Tuple[int, ...],
    wandb_disable: bool,
    wandb_project: Optional[str],
    wandb_group: Optional[str],
    wandb_name: Optional[str],
    wandb_tags: Tuple[str, ...],
    data_buffer: int,
    event_buffer: int,
) -> None:
    rng_key: Optional[PRNGKey]
    params: Optional[ArrayTree]
    opt_state: Optional[optax.MultiStepsState]
    if config_path is None and load_from is not None:
        # Either load a checkpoint...
        cp = minigpt.load_from_directory(path=load_from)
        config = cp.config
        step = cp.step
        seed = cp.seed
        rng_key = cp.rng_key
        params = cp.params
        opt_state = cp.opt_state
        if wandb_disable:
            run = None
        else:
            run = minigpt.load_wandb_run(path=load_from)
    elif config_path is not None and load_from is None:
        # ...or create a new one
        if seed is None:
            raise ValueError("Seed must be specified if loading from config")
        config = minigpt.Config.from_yaml(config_path)
        rng_key = params = opt_state = None
        step = 0
        if wandb_disable:
            run = None
        else:
            run = minigpt.new_wandb_run(
                project=wandb_project,
                tags=wandb_tags,
                group=wandb_group,
                name=wandb_name,
                notes=pformat(config.to_dict()),
            )
    else:
        raise ValueError("Must specify either config-path or load-from")

    batches_fn = lambda: islice(
        minigpt.batches_from_config(config, seed + 1), step, None
    )
    with minigpt.BufferedIterator(batches_fn, data_buffer) as batches:
        train_fn = partial(
            minigpt.train,
            batches=batches,
            config=config,
            seed=seed,
            rng_key=rng_key,
            params=params,
            opt_state=opt_state,
            step=step,
            save_frequency=save_frequency,
            save_directory=save_directory,
        )
        with minigpt.BufferedIterator(train_fn, event_buffer) as events:
            events = minigpt.log_losses(events=events, frequency=log_frequency)
            events = minigpt.log_time_per_step(
                events=events,
                frequency=log_time_per_step_frequency,
                percentiles=log_time_per_step_percentiles,
            )
            if run is not None:
                events = minigpt.log_to_wandb(events=events, run=run)
            events = minigpt.save_to_directory(events=events)
            for _ in events:
                pass


if __name__ == "__main__":
    cli()
