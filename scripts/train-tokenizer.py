#!/usr/bin/env python3
import sys
from pathlib import Path

import click

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
    dataset = data.samples_from_config(config.dataset, tokenize=False, shuffle=False)
    data.train_tokenizer_from_config(dataset, config.tokenizer)


if __name__ == "__main__":
    cli()
