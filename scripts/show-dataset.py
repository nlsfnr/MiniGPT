#!/usr/bin/env python3
import sys
from pathlib import Path

import click
from tqdm import tqdm

sys.path.append(".")

import minigpt.data as data  # noqa: E402
from minigpt.common import Config, get_logger, setup_logging  # noqa: E402

logger = get_logger()


@click.group()
def cli() -> None:
    pass


@cli.command("show")
@click.option("--config-path", "-c", type=Path, required=True)
@click.option("--seed", "-s", type=int, default=0)
@click.option("--number", "-n", type=int, default=5)
def cli_show(
    config_path: Path,
    seed: int,
    number: int,
) -> None:
    setup_logging()
    config = Config.from_yaml(config_path)
    batches = iter(data.batches_from_config(config, seed))
    for _ in range(number):
        batch = next(batches)
        print(batch)
        print()


@cli.command("perf")
@click.option("--config-path", "-c", type=Path, required=True)
@click.option("--seed", "-s", type=int, default=0)
@click.option("--number", "-n", type=int, default=-1)
def cli_perf(
    config_path: Path,
    seed: int,
    number: int,
) -> None:
    setup_logging()
    config = Config.from_yaml(config_path)
    batches = iter(data.batches_from_config(config, seed))
    batches = iter(tqdm(batches))
    while number != 0:
        next(batches)
        number -= 1


if __name__ == "__main__":
    cli()
