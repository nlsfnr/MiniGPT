#!/usr/bin/env python3
import sys
from pathlib import Path

import click

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
@cli.command("perplexity")
@click.option("--load-from", "-l", type=Path, required=True,
              help="Path to the checkpoint to load from.")
@click.option("--text", "-t", type=str, default="",
              help="Text over which to compute the perplexity")
# fmt: on
def cli_generate(
    load_from: Path,
    text: str,
) -> None:
    cp = minigpt.load_from_directory(path=load_from)
    config = cp.config
    params = cp.params
    perplexities, _, tokens = minigpt.perplexity(
        config=config,
        params=params,
        text=text,
    )
    for t, p in zip(tokens, perplexities):
        print(f"{t:10s} {float(p):.3f}")
    print(f"Mean perplexity: {float(perplexities.mean()):.3f}")


if __name__ == "__main__":
    cli()
