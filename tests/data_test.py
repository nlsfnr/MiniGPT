import random
import string
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from tokenizers import Tokenizer

import minigpt.data as data

_TOKENIZER_PATH = "./tests/tokenizer-8192.json"


def mock_dataset_fn(
    seed: int = 0,
    min_length: int = 0,
    max_length: int = 100,
) -> Callable[..., Iterable[data.Sample]]:
    rng = random.Random(seed)

    def load_dataset_fn(*args: str, **kwargs: str) -> Iterable[data.Sample]:
        del args, kwargs
        while True:
            length = rng.randint(min_length, max_length)
            text = "".join(rng.choices(string.printable, k=length))
            yield {"text": text}

    return load_dataset_fn


def test_load_huggingface_dataset():
    dataset_fn = mock_dataset_fn()
    dataset = data.load_huggingface_dataset([], {}, dataset_fn)
    assert isinstance(dataset, Iterable)
    sample = next(iter(dataset))
    assert isinstance(sample, Mapping)
    assert "text" in sample
    assert isinstance(sample["text"], str)


def test_load_huggingface_tokenizer_from_file():
    tokenizer = data.load_huggingface_tokenizer([_TOKENIZER_PATH], {})
    out = tokenizer.encode_batch(["Hello world!"])
    assert isinstance(out, Sequence)
    assert isinstance(out[0].ids, Sequence)
    assert isinstance(out[0].ids[0], int)


def test_tokenize_samples(tokenizer: Tokenizer):
    dataset_fn = mock_dataset_fn()
    dataset = data.load_huggingface_dataset([], {}, dataset_fn)
    dataset = data.tokenize_samples(dataset, tokenizer)
    assert isinstance(dataset, Iterable)
    sample = next(iter(dataset))
    assert isinstance(sample, Mapping)
    assert "input_ids" in sample
    assert isinstance(sample["input_ids"], Sequence)
    assert isinstance(sample["input_ids"][0], int)


def test_truncate_and_pad(tokenizer: Tokenizer):
    dataset_fn = mock_dataset_fn()
    dataset = data.load_huggingface_dataset([], {}, dataset_fn)
    dataset = data.tokenize_samples(dataset, tokenizer)
    dataset = data.truncate_and_pad(dataset, 10, 0, 0)
    assert isinstance(dataset, Iterable)
    sample = next(iter(dataset))
    assert isinstance(sample, Mapping)
    assert "input_ids" in sample
    assert isinstance(sample["input_ids"], Sequence)
    assert isinstance(sample["input_ids"][0], int)
    assert len(sample["input_ids"]) == 10


def test_collate(tokenizer: Tokenizer):
    dataset_fn = mock_dataset_fn()
    dataset = data.load_huggingface_dataset([], {}, dataset_fn)
    dataset = data.tokenize_samples(dataset, tokenizer)
    dataset = data.truncate_and_pad(dataset, 10, 0, 0)
    batches = data.collate(dataset, 100)
    assert isinstance(batches, Iterable)
    batch = next(iter(batches))
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (100, 10)
    assert batch.dtype == np.int32


def test_shuffle():
    dataset_fn = mock_dataset_fn()
    dataset = data.load_huggingface_dataset([], {}, dataset_fn)
    dataset = data.shuffle(dataset, 10, 0)
    assert isinstance(dataset, Iterable)
    sample = next(iter(dataset))
    assert isinstance(sample, Mapping)
    assert "text" in sample
    assert isinstance(sample["text"], str)
