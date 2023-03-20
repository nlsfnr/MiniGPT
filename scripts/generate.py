#!/usr/bin/env python3
import sys
from pathlib import Path

import click
import jax

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
@cli.command("generate")
@click.option("--load-from", "-l", type=Path, required=True,
              help="Path to the checkpoint to load from.")
@click.option("--prompt", "-p", type=str, default="",
              help="Prompt to start with. Default '' (empty).")
@click.option("--seed", "-s", type=int, default=0,
              help="Random seed. Default 0.")
@click.option("--temperature", "-t", type=float, default=1.0,
              help="Temperature. Default 1.0.")
@click.option("--top-p", "-t", type=float, default=0.95,
              help="Threshold for sampling tokens. Default 0.95.")
# fmt: on
def cli_generate(
    load_from: Path,
    prompt: str,
    temperature: float,
    top_p: float,
    seed: int,
) -> None:
    cp = minigpt.load_from_directory(path=load_from)
    config = cp.config
    params = cp.params
    it = minigpt.generate(
        config=config,
        params=params,
        rng_key=jax.random.PRNGKey(seed),
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
    )
    for i, x in enumerate(it):
        print(i, x)


if __name__ == "__main__":
    cli()
