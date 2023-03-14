import itertools
import random
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    TypeVar,
)

import datasets
import numpy as np
import tokenizers
from tokenizers import Tokenizer

from .common import Config, get_logger

T = TypeVar("T")
logger = get_logger()


Sample = MutableMapping[str, Any]
Batch = MutableMapping[str, Sequence[Any]]
TokenizerFn = Callable[[Sequence[str]], Sequence[Sequence[int]]]


def _stream_huggingface_dataset(
    *args: str,
    **kwargs: str,
) -> Iterable[Sample]:
    dataset = datasets.load_dataset(*args, **kwargs, streaming=True)
    return (dict(sample) for sample in dataset)


def _load_huggingdace_tokenizer(
    *args: str,
    **kwargs: str,
) -> Tokenizer:
    if len(args) == 1 and args[0].strip().lower().endswith(".json"):
        logger.info(f"Loading tokenizer from JSON file: {args[0]}")
        return Tokenizer.from_file(*args, **kwargs)
    logger.info(f"Loading tokenizer from HuggingFace: {args}, {kwargs}")
    return tokenizers.Tokenizer.from_pretrained(*args, **kwargs)


def load_huggingface_dataset(
    args: Iterable[str],
    kwargs: Mapping[str, str],
    load_dataset_fn: Callable[..., Iterable[Sample]] = _stream_huggingface_dataset,
    repeat_forever: bool = False,
) -> Iterable[Sample]:
    """Load a HuggingFace dataset and stream its samples.

    Args:
        args: Positional arguments to pass to `datasets.load_dataset`.
        kwargs: Keyword arguments to pass to `datasets.load_dataset`.
        load_dataset_fn: Function to use to load the dataset. Defaults to
            `datasets.load_dataset`.
        repeat_forever: Whether to repeat the dataset forever. Defaults to
            `False`.

    Returns:
        Iterable of samples.
    """
    if isinstance(args, str):
        raise TypeError(f"Expected args to be a sequence of str, got {args}")
    args = list(args)
    kwargs = dict(kwargs)
    while True:
        logger.info(f"Streaming dataset from HuggingFace: {args}, {kwargs}")
        dataset = load_dataset_fn(*args, **kwargs)
        yield from (dict(sample) for sample in dataset)
        if not repeat_forever:
            break


def load_huggingface_tokenizer(
    args: Iterable[str],
    kwargs: Mapping[str, str],
    load_tokenizer_fn: Callable[..., Tokenizer] = _load_huggingdace_tokenizer,
) -> Tokenizer:
    """Load a tokenizer from HuggingFace tokenizers library. If the first
    argument is a JSON file (i.e. ends on '.json'), the tokenizer is loaded
    from that file. Otherwise, the tokenizer is loaded from HuggingFace.

    Args:
        args: Positional arguments to pass to `tokenizers.Tokenizer.from_pretrained`.
        kwargs: Keyword arguments to pass to `tokenizers.Tokenizer.from_pretrained`.
        load_tokenizer_fn: Function to use to load the tokenizer. Defaults to

    Returns:
        A tokenizer.
    """
    return load_tokenizer_fn(*args, **kwargs)


def tokenize_samples(
    samples: Iterable[Sample],
    tokenizer: Tokenizer,
    batch_size: int = 1000,
    input_key: str = "text",
    output_key: str = "input_ids",
) -> Iterable[Sample]:
    """Tokenize samples using a tokenizer. The tokenizer is applied in batches for improved
    performance.

    Args:
        samples: Samples to tokenize.
        tokenizer: Tokenizer to use.
        batch_size: Batch size to use for tokenization. Defaults to `1000`.
        input_key: Key in the sample to use as input. Defaults to `text`.
        output_key: Key in the sample to use as output. Defaults to `input_ids`.

    Returns:
        Tokenized samples.
    """

    def tokenizer_fn(texts: Sequence[str]) -> Sequence[Sequence[int]]:
        return [enc.ids for enc in tokenizer.encode_batch(texts)]

    samples, samples_ = itertools.tee(samples)
    texts = chunks((sample[input_key] for sample in samples_), batch_size)
    batched_indices = (tokenizer_fn(text) for text in texts)
    indices = (index for batch in batched_indices for index in batch)
    for sample, index in zip(samples, indices):
        sample[output_key] = index
        yield sample


def truncate_and_pad(
    samples: Iterable[Sample],
    length: int,
    min_length: int,
    pad_token_id: int,
    input_key: str = "input_ids",
    output_key: str = "input_ids",
) -> Iterable[Sample]:
    """Truncate and pad samples to the desired length.

    Args:
        samples: Samples to truncate and pad.
        length: Length to truncate and pad to.
        pad_token_id: Token ID to use for padding.
        input_key: Key in the sample to use as input. Defaults to `input_ids`.
        output_key: Key in the sample to use as output. Defaults to `input_ids`.

    Returns:
        Truncated and padded samples.
    """
    for sample in samples:
        ids = sample[input_key]
        for chunk in chunks(ids, length):
            if len(chunk) < min_length:
                continue
            if len(chunk) < length:
                chunk = list(chunk) + [pad_token_id] * (length - len(chunk))
            sample[output_key] = chunk
            yield sample


def collate(
    samples: Iterable[Sample],
    batch_size: int,
    input_key: str = "input_ids",
) -> Iterable[np.ndarray]:
    """Collate samples into batches, converting them to numpy arrays.

    Args:
        samples: Samples to collate.
        batch_size: Batch size to use.
        input_key: Key in the sample to use as input. Defaults to `input_ids`.

    Returns:
        Collated samples as numpy arrays.
    """
    values = (sample[input_key] for sample in samples)
    batched_values = chunks(values, batch_size)
    for batch in batched_values:
        if not all(len(value) == len(batch[0]) for value in batch[1:]):
            raise ValueError(f"Expected equal sample lengths, got {batch}")
        yield np.array(batch, dtype=np.int32)


def chunks(
    iterable: Iterable[T],
    size: int,
) -> Iterable[Sequence[T]]:
    """Split an iterable into chunks of a given size.

    Args:
        iterable: Iterable to split.
        size: Size of the chunks.
    """
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            break
        yield chunk


def shuffle(
    samples: Iterable[Sample],
    buffer_size: int,
    seed: int = 0,
) -> Iterable[Sample]:
    """Shuffle samples.

    Args:
        samples: Samples to shuffle.
        buffer_size: Size of the buffer to use for shuffling.
        seed: Seed to use for shuffling.

    Returns:
        Shuffled samples.
    """
    if buffer_size <= 0:
        raise ValueError(f"Expected buffer_size > 0, got {buffer_size}")
    rng = random.Random(seed)
    buffer: List[Sample] = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
        else:
            index = rng.randrange(len(buffer))
            buffer[index], sample = sample, buffer[index]
            yield sample
    rng.shuffle(buffer)
    yield from buffer


def batches_from_config(
    config: Config,
    seed: int,
) -> Iterable[np.ndarray]:
    """Pipeline to load batches from a configuration.

    Args:
        config: Configuration to use.

    Returns:
        Iterable of batches.
    """
    dataset = load_huggingface_dataset(
        args=config.dataset.args,
        kwargs=config.dataset.kwargs,
        repeat_forever=True,
    )
    tokenizer = load_huggingface_tokenizer(
        args=config.tokenizer.args,
        kwargs=config.tokenizer.kwargs,
    )
    dataset = tokenize_samples(dataset, tokenizer)
    dataset = truncate_and_pad(
        samples=dataset,
        length=config.data.length,
        min_length=config.data.min_length,
        pad_token_id=config.data.pad_token_id,
    )
    dataset = shuffle(
        samples=dataset,
        buffer_size=config.data.shuffle_buffer_size,
        seed=seed,
    )
    return collate(dataset, config.data.batch_size)
