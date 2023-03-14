import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import optax
from chex import ArrayTree, PRNGKey
from wandb.sdk.wandb_run import Run as WandbRun

import wandb

from .common import Config, get_logger
from .training import Event, Save, TrainStep

logger = get_logger()


def log_losses(
    *,
    events: Iterable[Event],
    frequency: int,
    log_fn: Callable[[str], None] = logger.info,
) -> Iterable[Event]:
    losses = []
    for event in events:
        if not isinstance(event, TrainStep):
            yield event
            continue
        losses.append(event.loss)
        if event.step % frequency == 0 and len(losses) >= 2:
            mean, std = statistics.mean(losses), statistics.stdev(losses)
            items = (
                f"Step: {event.step:>6}",
                f"Loss: {mean:0.6f} ± {std:0.6f}",
            )
            log_fn(" | ".join(items))
            losses = []
        yield event


def log_time_per_step(
    *,
    events: Iterable[Event],
    frequency: int,
    percentiles: Iterable[int],
    log_fn: Callable[[str], None] = logger.info,
) -> Iterable[Event]:
    if frequency < 100:
        raise ValueError(f"Expected frequency to be at least 100, got {frequency}")
    percentiles = tuple(percentiles)
    timestamps = []
    for event in events:
        if not isinstance(event, TrainStep):
            yield event
            continue
        timestamps.append(event.timestamp)
        if event.step % frequency == 0 and len(timestamps) >= max(percentiles):
            deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:])]
            deltas_seconds = [delta.total_seconds() for delta in deltas]
            points = statistics.quantiles(deltas_seconds, n=101, method="inclusive")
            points = [points[p] for p in percentiles]
            points_str = ", ".join(
                f"{p}%: {t:0.4f}s" for p, t in zip(percentiles, points)
            )
            items = (
                f"Step: {event.step:>6}",
                "s/step: " + points_str,
            )
            log_fn(" | ".join(items))
        yield event


def save_to_directory(
    *,
    events: Iterable[Event],
) -> Iterable[Event]:
    for event in events:
        if not isinstance(event, Save):
            yield event
            continue
        path = Path(event.path)
        path.mkdir(parents=True, exist_ok=True)
        event.config.to_yaml(path / "config.yaml")
        with open(path / "params.pkl", "wb") as f:
            pickle.dump(event.params, f)
        with open(path / "opt_state.pkl", "wb") as f:
            pickle.dump(event.opt_state, f)
        with open(path / "rng_key.pkl", "wb") as f:
            pickle.dump(event.rng_key, f)
        with open(path / "step.txt", "w") as f:
            f.write(str(event.step))
        with open(path / "seed.txt", "w") as f:
            f.write(str(event.seed))
        logger.info(f"Step: {event.step:>6} | Saved model to {path}")
        yield event


@dataclass
class LoadResult:
    config: Config
    step: int
    seed: int
    rng_key: PRNGKey
    params: ArrayTree
    opt_state: optax.MultiStepsState


def load_from_directory(
    *,
    path: Path,
) -> LoadResult:
    config = Config.from_yaml(path / "config.yaml")
    with open(path / "params.pkl", "rb") as f:
        params = pickle.load(f)
        assert isinstance(params, dict)
    with open(path / "opt_state.pkl", "rb") as f:
        opt_state = pickle.load(f)
        assert isinstance(opt_state, optax.MultiStepsState)
    with open(path / "rng_key.pkl", "rb") as f:
        rng_key = pickle.load(f)
    with open(path / "step.txt") as f:
        step = int(f.read().strip())
    with open(path / "seed.txt") as f:
        seed = int(f.read().strip())
    return LoadResult(
        config=config,
        step=step,
        seed=seed,
        rng_key=rng_key,
        params=params,
        opt_state=opt_state,
    )


def log_to_wandb(
    *,
    events: Iterable[Event],
    run: WandbRun,
) -> Iterable[Event]:
    for event in events:
        if isinstance(event, Save):
            event.path.mkdir(parents=True, exist_ok=True)
            with open(event.path / "wandb-run-id.txt", "w") as f:
                f.write(run.id)
        elif isinstance(event, TrainStep):
            data = dict(loss=event.loss)
            run.log(data, step=event.step)
        yield event


def load_wandb_run(
    *,
    path: Path,
) -> WandbRun:
    with open(path / "wandb-run-id.txt") as f:
        run_id = f.read().strip()
    run = wandb.init(id=run_id, resume="must")
    assert isinstance(run, WandbRun)
    return run


def new_wandb_run(
    *,
    project: Optional[str] = None,
    tags: Iterable[str] = [],
    group: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
) -> WandbRun:
    run = wandb.init(
        project=project,
        group=group,
        tags=tuple(tags),
        name=name,
        notes=notes,
    )
    assert isinstance(run, WandbRun)
    return run
