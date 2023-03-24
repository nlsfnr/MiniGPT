import queue

import jax.numpy as jnp
from chex import Array

import minigpt


def test_training(config: minigpt.Config) -> None:
    """Test that training runs without error."""
    # Create a new trainer thread.
    batch_queue: queue.Queue[Array] = queue.Queue()
    event_queue: queue.Queue[minigpt.Event] = queue.Queue()
    trainer_thread = minigpt.Trainer(
        event_queue=event_queue,
        batch_queue=batch_queue,
        config=config,
        seed=0,
    )
    # Create a fake dataset.
    batches = [
        jnp.ones((config.data.batch_size, config.data.length), dtype=jnp.int32)
    ] * 5
    for batch in batches:
        batch_queue.put(batch)
    # Run the trainer thread.
    events = minigpt.queue_as_iterator(event_queue)
    events = filter(lambda e: isinstance(e, minigpt.TrainStep), events)
    with trainer_thread:
        for _ in range(5):
            next(events)
