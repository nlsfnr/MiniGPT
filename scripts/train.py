#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional

import click

sys.path.append(".")

import minigpt  # noqa: E402

logger = minigpt.get_logger()


@click.group()
@click.option("--debug", "-d", is_flag=True)
def cli(
    debug: bool,
) -> None:
    minigpt.setup_logging()
    minigpt.set_debug(debug)


def _noop(*_: object, **__: object) -> None:
    pass


@cli.command("new")
@click.option("--path", "-p", type=Path, required=True)
@click.option("--config-path", "-c", type=Path, required=True)
@click.option("--seed", "-s", type=int, required=True)
def cli_new(
    path: Path,
    config_path: Path,
    seed: int,
) -> None:
    config = minigpt.Config.from_yaml(config_path)
    trainer = minigpt.Trainer.new_from_config(config, path, seed)
    stream_logger = (
        minigpt.StreamLogger.from_config(config.stream_logger, trainer)
        if config.stream_logger is not None
        else _noop
    )
    wandb_logger = (
        minigpt.WandBLogger.from_config(config.wandb_logger, trainer)
        if config.wandb_logger is not None
        else _noop
    )
    for telemetry in trainer:
        stream_logger(telemetry)  # type: ignore
        wandb_logger(telemetry)  # type: ignore


@cli.command("resume")
@click.option("--path", "-p", type=Path, required=True)
@click.option("--new-path", type=Path, default=None)
def cli_resume(
    path: Path,
    new_path: Optional[Path],
) -> None:
    config = minigpt.Config.from_yaml(path / "config.yaml")
    trainer = minigpt.Trainer.load(path, new_path)
    stream_logger = (
        minigpt.StreamLogger.from_config(config.stream_logger, trainer)
        if config.stream_logger is not None
        else _noop
    )
    wandb_logger = (
        minigpt.WandBLogger.load(config.wandb_logger, trainer, path)
        if config.wandb_logger is not None
        else _noop
    )
    for telemetry in trainer:
        stream_logger(telemetry)  # type: ignore
        wandb_logger(telemetry)  # type: ignore


if __name__ == "__main__":
    cli()
