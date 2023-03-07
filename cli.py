#!/usr/bin/env python3
import click

import minigpt


@click.group("MiniGPT")
def cli() -> None:
    minigpt.setup_logging()
    logger = minigpt.get_logger()
    logger.info(f"Starting MiniGPT, version {minigpt.__version__}")


@cli.command("dbg")
def cli_dbg() -> None:
    pass


if __name__ == '__main__':
    cli()
