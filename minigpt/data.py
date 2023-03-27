import itertools
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    TypeVar,
    Union,
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


def _to_rng(seed_or_rng: Union[int, np.random.Generator]) -> np.random.Generator:
    return (
        np.random.default_rng(seed_or_rng)
        if isinstance(seed_or_rng, int)
        else seed_or_rng
    )


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
    key = kwargs.pop("key", "text")
    for epoch in itertools.count():
        logger.info(
            f"Streaming dataset from HuggingFace: {args}, {kwargs} (epoch: {epoch})"
        )
        dataset = load_dataset_fn(*args, **kwargs)
        samples = (dict(text=sample[key]) for sample in dataset if sample[key].strip())
        yield from samples
        if not repeat_forever:
            break


def merge_datasets(
    *,
    datasets: Iterable[Iterable[Sample]],
    weights: Iterable[float],
    seed_or_rng: Union[int, np.random.Generator],
) -> Iterable[Sample]:
    p = np.array(list(weights), dtype=float)
    p = p / p.sum()
    rng = _to_rng(seed_or_rng)
    iterators = [iter(dataset) for dataset in datasets]
    while True:
        index = rng.choice(len(iterators), p=p)
        yield next(iterators[index])


def tokenizer_from_config(
    config: Config,
) -> Tokenizer:
    """Load a tokenizer from a config.

    Args:
        config: Config to use.

    Returns:
        A tokenizer.
    """
    return load_huggingface_tokenizer(
        config.tokenizer.args,
        config.tokenizer.kwargs,
    )


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

    samples = map(dict, samples)
    samples, samples_ = itertools.tee(samples)
    texts = chunks((sample[input_key] for sample in samples_), batch_size)
    batched_indices = (tokenizer_fn(text) for text in texts)
    indices = (index for batch in batched_indices for index in batch)
    for sample, index in zip(samples, indices):
        sample[output_key] = index
        yield sample


def chain_and_split(
    *,
    arrays: Iterable[Iterable[int]],
    length: int,
) -> Iterable[Iterable[int]]:
    """Chain token ids and split them into chunks of the desired length. Equal
    to the process in https://arxiv.org/pdf/2204.02311.pdf, section 5.

    Args:
        arrays: Arrays to chain and split.
        length: Length to split to.

    Returns:
        Chained and split samples.
    """
    ids = (id for array in arrays for id in array)
    yield from chunks(ids, length)


def chunks(
    iterable: Iterable[T],
    size: int,
    drop_last: bool = False,
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
        if drop_last and len(chunk) < size:
            break
        yield chunk


def shuffle(
    xs: Iterable[T],
    buffer_size: int,
    seed_or_rng: Union[int, np.random.Generator],
) -> Iterable[T]:
    """Shuffle an iterable using a buffer of a given size.

    Args:
        xs: Iterable to shuffle.
        buffer_size: Size of the buffer to use for shuffling.
        seed: Seed to use for shuffling.

    Returns:
        Shuffled iterable.
    """
    if buffer_size <= 0:
        raise ValueError(f"Expected buffer_size > 0, got {buffer_size}")
    rng = _to_rng(seed_or_rng)
    buffer: List[T] = list(itertools.islice(xs, buffer_size))
    for x in xs:
        index = rng.integers(len(buffer))
        buffer[index], x = x, buffer[index]
        yield x
    yield from rng.permutation(np.array(buffer), 0)


def batches_from_config(
    config: Config,
    seed_or_rng: Union[int, np.random.Generator],
    *,
    extra_length: int = 0,
) -> Iterable[np.ndarray]:
    """Pipeline to load batches from a configuration.

    Args:
        config: Configuration to use.
        seed: Seed to use for shuffling.
        extra_length: Extra length to add to the samples. This is useful for
            training with a causal language model, where the decoder needs to
            predict the next token. In this case, the extra length should be
            set to 1 since the model's input will be the first `length - 1`
            tokens and the target will be the last `length - 1` tokens.

    Returns:
        Iterable of batches.
    """
    rng = _to_rng(seed_or_rng)
    datasets = (
        load_huggingface_dataset(
            args=ds.args,
            kwargs=ds.kwargs,
            repeat_forever=True,
        )
        for ds in config.dataset
    )
    # Shuffle the samples of each dataset. This prevents the samples from
    # low-frequency datasets to be almost sequential due to a too-small buffer
    # size for the final shuffle.
    datasets = (
        shuffle(
            xs=ds,
            buffer_size=config.data.per_dataset_shuffle_buffer_size,
            seed_or_rng=rng,
        )
        for ds in datasets
    )
    weights = (ds.weight for ds in config.dataset)
    dataset = merge_datasets(datasets=datasets, weights=weights, seed_or_rng=rng)
    tokenizer = load_huggingface_tokenizer(
        args=config.tokenizer.args,
        kwargs=config.tokenizer.kwargs,
    )
    dataset = tokenize_samples(dataset, tokenizer)
    idss: Iterable[Iterable[int]]
    idss = (sample["input_ids"] for sample in dataset)
    idss = chain_and_split(arrays=idss, length=config.data.length + extra_length)
    # Shuffle the individual chunks
    idss = shuffle(
        xs=idss,
        buffer_size=config.data.shuffle_buffer_size,
        seed_or_rng=rng,
    )
    batches = chunks(idss, size=config.data.batch_size)
    arrays = (np.array(batch, dtype=np.int32) for batch in batches)
    return arrays
