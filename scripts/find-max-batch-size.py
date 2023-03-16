#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional

import click
import jax.numpy as jnp

sys.path.append(".")

import minigpt.training as training  # noqa: E402
from minigpt.common import Config, get_logger, setup_logging  # noqa: E402

logger = get_logger()


@click.command()
@click.option("--config-path", "-c", type=Path, required=True)
@click.option("--length", "-l", type=int, default=None)
@click.option("--model-dim", "-d", type=int, default=None)
@click.option("--embedding-dim", "-e", type=int, default=None)
@click.option("--low", "-L", type=int, default=0)
@click.option("--high", "-H", type=int, default=1)
@click.option("--iterations", type=int, default=3)
def cli(
    config_path: Path,
    length: Optional[int],
    model_dim: Optional[int],
    embedding_dim: Optional[int],
    low: int,
    high: int,
    iterations: int,
) -> None:
    setup_logging()
    config = Config.from_yaml(config_path)
    config.data.length = length or config.data.length
    config.model.model_dim = model_dim or config.model.model_dim
    config.model.embedding_dim = embedding_dim or config.model.embedding_dim

    def guard(batch_size: int) -> bool:
        def fn() -> None:
            batches = [
                jnp.ones((batch_size, config.data.length), dtype=jnp.int32)
            ] * iterations
            events = training.train(
                config=config, seed=0, batches=batches, log_param_size_on_init=False
            )
            events = filter(lambda e: isinstance(e, training.TrainStep), events)
            for _ in range(iterations):
                next(events)

        try:
            fn()
        except RuntimeError:
            return False
        else:
            return True

    # Find the largest power of 2 that does not result in OOM.
    while guard(high):
        logger.info(f"Batch size {low} < {high} < ? is OK.")
        low = high
        high *= 2
    # Do a binary search to find the largest batch size that does not result in OOM.
    while low < high:
        mid = (low + high) // 2
        if guard(mid):
            logger.info(f"Batch size {low} < {mid} < {high} is OK.")
            low = mid + 1
        else:
            logger.info(f"Batch size {low} < {mid} < {high} is OOM.")
            high = mid


if __name__ == "__main__":
    cli()
