from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Protocol, Sequence, Tuple

import numpy as np
import yaml
from chex import Array, ArrayTree
from wandb.sdk.wandb_run import Run as WandbRun  # type: ignore

import wandb

from .common import get_logger
from .training import Telemetry, Trainer

logger = get_logger()


class WandBConfig(Protocol):
    histogram_frequency: int
    project: str
    group: str
    tags: Sequence[str]


class WandBLogger:
    def __init__(
        self,
        *,
        run: WandbRun,
        config: WandBConfig,
        trainer: Trainer,
    ):
        self.run = run
        self.config = config
        self.trainer = trainer

    @classmethod
    def from_config(
        cls,
        config: WandBConfig,
        trainer: Trainer,
    ) -> WandBLogger:
        run = wandb.init(
            project=config.project,
            group=config.group,
            tags=config.tags,
        )
        assert isinstance(run, WandbRun)
        return cls(
            run=run,
            config=config,
            trainer=trainer,
        )

    @classmethod
    def load(
        cls,
        config: WandBConfig,
        trainer: Trainer,
        path: Path,
    ) -> WandBLogger:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "wandb.yaml", "r") as f:
            other = yaml.safe_load(f)
            assert isinstance(other, dict)
        run_id = other["run_id"]
        run = wandb.init(
            project=config.project,
            group=config.group,
            tags=config.tags,
            id=run_id,
            resume=True,
        )
        assert isinstance(run, WandbRun)
        return cls(
            run=run,
            config=config,
            trainer=trainer,
        )

    def save(self, path: Path) -> WandBLogger:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "wandb.yaml", "w") as f:
            yaml.safe_dump(dict(run_id=self.run.id), f)
        return self

    def __call__(
        self,
        telemetry: Telemetry,
    ) -> None:
        if telemetry.save_to is not None:
            self.save(telemetry.save_to)
        data: Dict[str, Any]
        data = dict(
            loss=telemetry.loss,
        )
        if telemetry.trainer.step % self.config.histogram_frequency == 0:
            assert telemetry.model_telemetry is not None
            assert telemetry.gradients is not None
            data.update(
                dict(
                    params=self._to_histograms(telemetry.trainer.params),
                    model_telemetry=self._to_histograms(
                        telemetry.model_telemetry
                    ),
                    gradients=self._to_histograms(telemetry.gradients),
                )
            )
        self.run.log(
            data=data,
            step=telemetry.trainer.step,
        )

    @staticmethod
    def _to_histograms(
        pytree: ArrayTree,
        prefix: str = "/",
        bins: int = 64,
        min_items_per_bin: int = 10,
    ) -> Dict[str, wandb.Histogram]:
        def _fn(
            x: ArrayTree,
            p: str,
        ) -> Iterator[Tuple[str, wandb.Histogram]]:
            if isinstance(x, Array):
                x_ = np.asarray(x).flatten()
                hist = np.histogram(x_, bins=min(bins, len(x_) // min_items_per_bin))
                yield p, wandb.Histogram(np_histogram=hist)
            elif isinstance(x, dict):
                for k, v in x.items():
                    assert isinstance(k, str)
                    yield from _fn(v, f"{p}.{k.replace('/', '.')}")
            elif isinstance(x, Sequence):
                for i, v in enumerate(x):
                    yield from _fn(v, f"{p}.{i}")
            else:
                raise ValueError(f"Unsupported type {type(x)}")

        return dict(_fn(pytree, prefix))


class StreamLoggerConfig(Protocol):
    frequency: int


class StreamLogger:
    def __init__(
        self,
        *,
        config: StreamLoggerConfig,
        trainer: Trainer,
        logger: Optional[logging.Logger] = logger,
    ):
        self.config = config
        self.trainer = trainer
        self.logger = logger or get_logger()

    @classmethod
    def from_config(
        cls,
        config: StreamLoggerConfig,
        trainer: Trainer,
    ) -> StreamLogger:
        return cls(
            config=config,
            trainer=trainer,
        )

    def __call__(
        self,
        telemetry: Telemetry,
    ) -> None:
        if telemetry.trainer.step % self.config.frequency != 0:
            return
        self.logger.info(f"{telemetry.trainer.step:06d} {telemetry.loss:.4f}")
