#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Iterable, Union

import click
from tqdm import tqdm

sys.path.append(".")

import minigpt  # noqa: E402
import minigpt.data as data  # noqa: E402

logger = minigpt.get_logger()


@click.command()
@click.option("--config-path", "-c", type=Path)
@click.option("--samples", is_flag=True)
def cli(
    config_path: Path,
    samples: bool,
) -> None:
    minigpt.setup_logging()
    config = minigpt.Config.from_yaml(config_path)
    dataset: Union[Iterable[data.Sample], Iterable[data.Batch]]
    if samples:
        dataset = data.samples_from_config(config.dataset)
    else:
        dataset = data.batches_from_config(config.dataset)
    for x in tqdm(dataset):
        del x


if __name__ == "__main__":
    cli()
