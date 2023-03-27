#!/usr/bin/env python3
import queue
import sys
from pathlib import Path
from typing import Optional
import threading

import click
import jax.numpy as jnp
from chex import Array

sys.path.append(".")

import minigpt  # noqa: E402

logger = minigpt.get_logger()


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
    minigpt.setup_logging()
    config = minigpt.Config.from_yaml(config_path)
    config.data.length = length or config.data.length
    config.model.model_dim = model_dim or config.model.model_dim
    config.model.embedding_dim = embedding_dim or config.model.embedding_dim

    def guard(batch_size: int) -> bool:
        def fn() -> None:
            batches = [
                jnp.ones((batch_size, config.data.length), dtype=jnp.int32)
            ] * iterations
            batch_queue: queue.Queue[Array] = queue.Queue()
            for batch in batches:
                batch_queue.put(batch)
            event_queue: queue.Queue[minigpt.Event] = queue.Queue()
            trainer_thread = minigpt.Trainer(
                event_queue=event_queue,
                batch_queue=batch_queue,
                config=config,
                seed=0,
            )
            events = minigpt.queue_as_iterator(event_queue)
            events = filter(lambda e: isinstance(e, minigpt.TrainStep), events)
            with trainer_thread:
                for _ in range(iterations):
                    next(events)
                trainer_thread.terminate()

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
