import string
import time
from functools import partial
from itertools import tee
from typing import Iterator

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture as Benchmark  # type: ignore

from tokenizers import Tokenizer  # type: ignore

from . import data


def gibberish(
    min_length: int = 0,
    max_length: int = 1_000,
    temporal_jitter: float = 0.0,
    seed: int = 0,
) -> Iterator[data.Sample]:
    generator = np.random.default_rng(seed)
    while True:
        time.sleep(np.exp(generator.normal(0, 1.0)) * temporal_jitter)
        length = generator.integers(min_length, max_length)
        text = "".join(generator.choice(list(string.printable), length))
        yield dict(text=text)


def test_compound_dataset() -> None:
    datasets = [gibberish(seed) for seed in range(3)]
    weights = [1.0, 2.0, 3.0]
    seed = 0
    compound_dataset = data.CompoundDataset(datasets, weights, seed)
    samples = iter(compound_dataset)
    for _ in range(10):
        sample = next(samples)
        assert isinstance(sample, dict)
        assert "text" in sample
        assert isinstance(sample["text"], str)


def test_bert_tokenizer() -> None:
    dataset = gibberish()
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    tokenized_dataset = data.TokenizedDataset(dataset, tokenizer, batch_size=3)
    samples = iter(tokenized_dataset)
    sample = next(samples)
    assert isinstance(sample, dict)
    assert "text" in sample
    assert isinstance(sample["text"], str)
    assert "input_ids" in sample
    assert isinstance(sample["input_ids"], list)
    assert all(isinstance(x, int) for x in sample["input_ids"])


def test_batched_unbatched_symmetry() -> None:
    dataset, dataset_copy = tee(gibberish())
    batched_dataset = data.BatchedDataset(dataset, batch_size=3)
    unbatched_dataset = data.UnbatchedDataset(batched_dataset)
    samples = iter(unbatched_dataset)
    for _ in range(10):
        expected = next(dataset_copy)
        actual = next(samples)
        assert expected == actual


@pytest.mark.parametrize("jitter", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("buffer_size", [1, 10, 100])
def test_buffered_dataset_smoothing_effect_bench(
    benchmark: Benchmark,
    jitter: float,
    buffer_size: int,
) -> None:
    dataset_fn = partial(gibberish, temporal_jitter=jitter)
    buffered_dataset = data.BufferedDataset(dataset_fn, buffer_size=buffer_size)
    samples = iter(buffered_dataset)
    benchmark(next, samples)
