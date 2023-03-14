#!/usr/bin/env python3
import sys
from pathlib import Path

import click

sys.path.append(".")

import minigpt.data as data  # noqa: E402
from minigpt.common import Config, get_logger, setup_logging  # noqa: E402

logger = get_logger()


@click.command()
@click.option("--config-path", "-c", type=Path, required=True)
@click.option("--seed", "-s", type=int, default=0)
@click.option("--number", "-n", type=int, default=5)
def cli(
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


if __name__ == "__main__":
    cli()
