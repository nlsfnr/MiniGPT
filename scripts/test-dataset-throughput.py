#!/usr/bin/env python3
import sys
from pathlib import Path

import click
from tqdm import tqdm

sys.path.append(".")

import minigpt  # noqa: E402
import minigpt.data as data  # noqa: E402

logger = minigpt.get_logger()


@click.command()
@click.option("--config-path", "-c", type=Path)
def cli(
    config_path: Path,
) -> None:
    minigpt.setup_logging()
    config = minigpt.Config.from_yaml(config_path)
    dataset = data.from_config(config.dataset)
    dataset = tqdm(dataset)
    for _ in dataset:
        pass


if __name__ == "__main__":
    cli()
