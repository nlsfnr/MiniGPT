import string
import time
from itertools import tee
from typing import Iterable, Iterator, Sequence

import numpy as np
import pytest

from . import data


@pytest.fixture
def tokenizer() -> data.TokenizerFn:
    def tokenize_batch(texts: Sequence[str]) -> Sequence[Sequence[int]]:
        return [[ord(c) for c in s] for s in texts]

    return tokenize_batch


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


def test_tokenized_dataset(
    tokenizer: data.TokenizerFn,
) -> None:
    dataset = gibberish()
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


def test_chunked_dataset(
    tokenizer: data.TokenizerFn,
) -> None:
    dataset: Iterable[data.Sample]
    dataset = gibberish()
    dataset = data.TokenizedDataset(dataset, tokenizer)
    chunked_dataset = data.ChunkedDataset(dataset, chunk_size=10)
    samples = iter(chunked_dataset)
    for _ in range(1000):
        sample = next(samples)
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert isinstance(sample["input_ids"], Sequence)
        assert all(isinstance(x, int) for x in sample["input_ids"])
        assert len(sample["input_ids"]) <= 10


def test_padded_dataset(
    tokenizer: data.TokenizerFn,
) -> None:
    dataset: Iterable[data.Sample]
    dataset = gibberish(max_length=7)
    dataset = data.TokenizedDataset(dataset, tokenizer)
    padded_dataset = data.PaddedDataset(dataset, padding_value=0, length=10)
    samples = iter(padded_dataset)
    for _ in range(1000):
        sample = next(samples)
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert isinstance(sample["input_ids"], Sequence)
        assert all(isinstance(x, int) for x in sample["input_ids"])
        assert len(sample["input_ids"]) == 10


def test_chunked_and_padded_dataset(
    tokenizer: data.TokenizerFn,
) -> None:
    dataset: Iterable[data.Sample]
    dataset = gibberish(min_length=0, max_length=1000)
    dataset = data.TokenizedDataset(dataset, tokenizer)
    dataset = data.ChunkedDataset(dataset, chunk_size=10)
    dataset = data.PaddedDataset(dataset, padding_value=0, length=10)
    samples = iter(dataset)
    for _ in range(1000):
        sample = next(samples)
        assert len(sample["input_ids"]) == 10
