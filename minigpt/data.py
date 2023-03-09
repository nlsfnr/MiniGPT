from __future__ import annotations

import queue
import threading
from functools import partial
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    runtime_checkable,
)

import datasets
import numpy as np
import tokenizers.processors

from minigpt.common import get_logger, require_implementation
from tokenizers import SentencePieceBPETokenizer, Tokenizer  # type: ignore

logger = get_logger()

SPECIAL_TOKENS = (
    PAD_TOKEN := "<pad>",
    BOS_TOKEN := "<bos>",
    EOS_TOKEN := "<eos>",
)

Sample = Mapping[str, Union[str, Sequence[int], np.ndarray]]
Batch = Mapping[
    str, Union[Sequence[str], Sequence[Sequence[int]], Sequence[np.ndarray]]
]
TokenizerFn = Callable[[Sequence[str]], Sequence[Sequence[int]]]

T = TypeVar("T")


class CompoundDataset(Iterable[Sample]):
    def __init__(
        self,
        datasets: Iterable[Iterable[Sample]],
        weights: Iterable[float],
        seed: int,
    ) -> None:
        self.datasets = datasets
        self.weights = np.asarray(weights)
        self.weights /= self.weights.sum()
        self.seed = seed
        self.generator = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[Sample]:
        """Yields a random sample from one of the datasets."""
        iterators = [iter(dataset) for dataset in self.datasets]
        while True:
            index = self.generator.choice(len(iterators), p=self.weights)
            yield next(iterators[index])


@runtime_checkable
class HuggingFaceDatasetConfig(Protocol):
    args: Sequence[str]
    kwargs: Mapping[str, Any]
    streaming: bool


class HuggingFaceDataset(Iterable[Sample]):
    def __init__(
        self,
        args: Iterable[str],
        kwargs: Mapping[str, Union[str, int, float]],
        streaming: bool = True,
        repeat_forever: bool = False,
    ) -> None:
        self.args = args
        self.kwargs = kwargs
        self.repeat_forever = repeat_forever
        self.streaming = streaming
        self.dataset_fn = partial(
            datasets.load_dataset, *args, streaming=streaming, **kwargs
        )

    @classmethod
    def from_config(
        cls,
        config: HuggingFaceDatasetConfig,
    ) -> HuggingFaceDataset:
        require_implementation(config, HuggingFaceDatasetConfig)
        return cls(
            args=config.args,
            kwargs=config.kwargs,
            streaming=config.streaming,
            repeat_forever=True,
        )

    def __iter__(self) -> Iterator[Sample]:
        """Yields a sample from the HuggingFace dataset."""
        while True:
            dataset = self.dataset_fn()
            for sample in dataset:
                sample = dict(sample)
                assert all(
                    isinstance(value, (np.ndarray, int, float, str))
                    for value in sample.values()
                )
                assert all(isinstance(key, str) for key in sample.keys())
                yield sample
            if not self.repeat_forever:
                break


@runtime_checkable
class ShuffledDatasetConfig(Protocol):
    buffer_size: int
    seed: int


class ShuffledDataset(Iterable[T]):
    def __init__(
        self,
        dataset: Iterable[T],
        buffer_size: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.seed = seed

    @classmethod
    def from_config(
        cls,
        dataset: Iterable[T],
        config: ShuffledDatasetConfig,
    ) -> ShuffledDataset:
        require_implementation(config, ShuffledDatasetConfig)
        return cls(
            dataset=dataset,
            buffer_size=config.buffer_size,
            seed=config.seed,
        )

    def __iter__(self) -> Iterator[T]:
        generator = np.random.default_rng(self.seed)
        buffer: List[T] = []
        for sample in self.dataset:
            if len(buffer) < self.buffer_size:
                buffer.append(sample)
            else:
                index = generator.integers(len(buffer))
                yield buffer[index]
                buffer[index] = sample
        while buffer:
            index = generator.integers(len(buffer))
            yield buffer.pop(index)


@runtime_checkable
class BatchedDatasetConfig(Protocol):
    size: int
    drop_last: bool


class BatchedDataset(Iterable[Batch]):
    def __init__(
        self,
        dataset: Iterable[Sample],
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    @classmethod
    def from_config(
        cls,
        dataset: Iterable[Sample],
        config: BatchedDatasetConfig,
    ) -> BatchedDataset:
        require_implementation(config, BatchedDatasetConfig)
        return cls(dataset, config.size, config.drop_last)

    def __iter__(self) -> Iterator[Batch]:
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield self._batchify(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._batchify(batch)

    def _batchify(self, samples: Sequence[Sample]) -> Batch:
        """Converts a batch of samples into a batch of tensors."""
        assert all(sample.keys() == samples[0].keys() for sample in samples)
        assert all(
            isinstance(value, type(samples[0][key]))
            for sample in samples
            for key, value in sample.items()
        )
        batch: Batch = dict()
        for key in samples[0].keys():
            batch[key] = [sample[key] for sample in samples]  # type: ignore
        return batch


class UnbatchedDataset(Iterable[Sample]):
    def __init__(
        self,
        batched_dataset: Iterable[Batch],
    ) -> None:
        self.batched_dataset = batched_dataset

    def __iter__(self) -> Iterator[Sample]:
        for batch in self.batched_dataset:
            yield from self._unbatchify(batch)

    def _unbatchify(self, batch: Batch) -> Iterator[Sample]:
        """Converts a batch of tensors into a batch of samples."""
        keys = list(batch.keys())
        assert all(len(batch[key]) == len(batch[keys[0]]) for key in keys)
        for index in range(len(batch[keys[0]])):
            sample = {key: batch[key][index] for key in keys}
            yield sample


class TokenizedDataset(Iterable[Sample]):
    def __init__(
        self,
        dataset: Iterable[Sample],
        tokenizer_fn: TokenizerFn,
        input_key: str = "text",
        output_key: str = "input_ids",
        batch_size: int = 1000,
    ) -> None:
        self.dataset = dataset
        self.tokenizer_fn = tokenizer_fn
        self.input_key = input_key
        self.output_key = output_key
        self.batch_size = batch_size

    @classmethod
    def from_tokenizer(
        cls,
        dataset: Iterable[Sample],
        tokenizer: Tokenizer,
        input_key: str = "text",
        output_key: str = "input_ids",
        batch_size: int = 1000,
    ) -> TokenizedDataset:
        return cls(
            dataset=dataset,
            tokenizer_fn=lambda strs: [e.ids for e in tokenizer.encode_batch(strs)],
            input_key=input_key,
            output_key=output_key,
            batch_size=batch_size,
        )

    def __iter__(self) -> Iterator[Sample]:
        batch_iterator = iter(BatchedDataset(self.dataset, self.batch_size))

        def inner() -> Iterator[Batch]:
            for batch in batch_iterator:
                texts = batch[self.input_key]
                assert all(isinstance(text, str) for text in texts)
                ids = self.tokenizer_fn(texts)  # type: ignore
                batch = dict(**batch, **{self.output_key: ids})
                yield batch

        yield from UnbatchedDataset(inner())


class ChunkedDataset(Iterable[Sample]):
    """Split each sample into chunks of a given size. Notably, this deletes all
    other fields in the sample."""

    def __init__(
        self,
        dataset: Iterable[Sample],
        chunk_size: int,
        input_key: str = "input_ids",
        output_key: str = "input_ids",
    ) -> None:
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.input_key = input_key
        self.output_key = output_key

    def __iter__(self) -> Iterator[Sample]:
        for sample in self.dataset:
            xs = sample[self.input_key]
            assert isinstance(xs, Sequence)
            while xs:
                yield {self.output_key: xs[: self.chunk_size]}
                xs = xs[self.chunk_size :]


class PaddedDataset(Iterable[Sample]):
    def __init__(
        self,
        dataset: Iterable[Sample],
        padding_value: int,
        length: int,
        input_key: str = "input_ids",
        output_key: str = "input_ids",
    ) -> None:
        self.dataset = dataset
        self.padding_value = padding_value
        self.length = length
        self.input_key = input_key
        self.output_key = output_key

    def __iter__(self) -> Iterator[Sample]:
        for sample in self.dataset:
            xs = sample[self.input_key]
            assert isinstance(xs, Sequence)
            yield {self.output_key: self._pad(xs)}  # type: ignore

    def _pad(self, xs: Sequence[int]) -> Sequence[int]:
        if isinstance(xs, np.ndarray):
            return np.pad(
                xs, (0, self.length - len(xs)), constant_values=self.padding_value
            )
        else:
            return list(xs) + [self.padding_value] * (self.length - len(xs))


@runtime_checkable
class BufferedDatasetConfig(Protocol):
    buffer_size: int
    timeout: float


class BufferedDataset(Iterable[T]):
    """Moves the function generating the samples in a dataset into a background
    (Python-) thread. This mitigates the effect of inconsistent I/O performance
    on the main thread."""

    def __init__(
        self,
        dataset_fn: Callable[[], Iterable[T]],
        buffer_size: int = 1000,
        timeout: float = 0.1,
    ) -> None:
        self.dataset_fn = dataset_fn
        self.buffer_size = buffer_size
        self.timeout = timeout

    @classmethod
    def from_config(
        cls, dataset_fn: Callable[[], Iterable[T]], config: BufferedDatasetConfig
    ) -> BufferedDataset:
        require_implementation(config, BufferedDatasetConfig)
        return cls(
            dataset_fn=dataset_fn,
            buffer_size=config.buffer_size,
            timeout=config.timeout,
        )

    def __iter__(self) -> Iterator[T]:
        terminate = threading.Event()
        sentinel = object()
        buffer: queue.Queue[Union[T, object]] = queue.Queue(maxsize=self.buffer_size)

        def _fill_buffer() -> None:
            try:
                if terminate.is_set():
                    return
                for sample in self.dataset_fn():
                    while True:
                        if terminate.is_set():
                            return
                        try:
                            buffer.put(sample, timeout=self.timeout)
                            break
                        except queue.Full:
                            continue
            finally:
                buffer.put(sentinel)

        thread = threading.Thread(target=_fill_buffer)
        thread.start()
        try:
            while True:
                sample = buffer.get()
                if sample is sentinel:
                    break
                yield sample  # type: ignore
        except Exception:
            terminate.set()
            raise
        else:
            terminate.set()
            thread.join()


@runtime_checkable
class TruncateAndPadDatasetConfig(Protocol):
    sequence_length: int
    min_sequence_length: int


class TruncateAndPadDataset(Iterable[Sample]):
    def __init__(
        self,
        dataset: Iterable[Sample],
        sequence_length: int,
        min_sequence_length: int,
        padding_value: int,
        input_key: str = "input_ids",
    ) -> None:
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.padding_value = padding_value
        self.input_key = input_key

    @classmethod
    def from_config(
        cls,
        dataset: Iterable[Sample],
        config: TruncateAndPadDatasetConfig,
        padding_value: int,
        input_key: str = "input_ids",
    ) -> TruncateAndPadDataset:
        return cls(
            dataset=dataset,
            sequence_length=config.sequence_length,
            min_sequence_length=config.min_sequence_length,
            padding_value=padding_value,
            input_key=input_key,
        )

    def __iter__(self) -> Iterator[Sample]:
        chunked_dataset = ChunkedDataset(
            dataset=self.dataset,
            chunk_size=self.sequence_length,
            input_key=self.input_key,
            output_key="chunks",
        )
        padded_dataset = PaddedDataset(
            dataset=chunked_dataset,
            length=self.sequence_length,
            padding_value=self.padding_value,
            input_key="chunks",
        )
        yield from padded_dataset


def load_tokenizer(path: Path) -> Tokenizer:
    """Loads a tokenizer from a given path."""
    tokenizer = Tokenizer.from_file(str(path))
    logger.info(f"Loaded tokenizer from {path}")
    return tokenizer


@runtime_checkable
class TokenizerLoadingConfig(Protocol):
    path: str


def load_tokenizer_from_config(config: TokenizerLoadingConfig) -> Tokenizer:
    """Loads a tokenizer from a given config."""
    require_implementation(config, TokenizerLoadingConfig)
    return load_tokenizer(Path(config.path))


def train_tokenizer(
    dataset: Iterable[Sample],
    vocabulary_size: int,
    path: Path,
    min_token_frequency: int,
    max_samples: Optional[int] = None,
    key: str = "text",
) -> Tokenizer:
    logger.info(
        f"Training tokenizer on {'all' if max_samples is None else max_samples} samples"
    )
    if max_samples is not None:
        dataset = islice(dataset, None, max_samples)
    tokenizer = SentencePieceBPETokenizer()  # type: ignore
    tokenizer.train_from_iterator(
        (sample[key] for sample in dataset),
        special_tokens=list(SPECIAL_TOKENS),
        vocab_size=vocabulary_size,
        min_frequency=min_token_frequency,
    )
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(  # type: ignore
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            (PAD_TOKEN, tokenizer.token_to_id(PAD_TOKEN)),
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    logger.info(f"Saved tokenizer to {path}")
    return tokenizer


@runtime_checkable
class TokenizerTrainingConfig(Protocol):
    vocabulary_size: int
    min_token_frequency: int
    max_samples: int
    key: str
    path: str


def train_tokenizer_from_config(
    dataset: Iterable[Sample],
    config: TokenizerTrainingConfig,
) -> Tokenizer:
    require_implementation(config, TokenizerTrainingConfig)
    return train_tokenizer(
        dataset=dataset,
        vocabulary_size=config.vocabulary_size,
        path=Path(config.path),
        min_token_frequency=config.min_token_frequency,
        max_samples=config.max_samples,
        key=config.key,
    )


@runtime_checkable
class SamplesConfig(Protocol):
    huggingface: HuggingFaceDatasetConfig
    buffering: Optional[BufferedDatasetConfig]
    shuffle: Optional[ShuffledDatasetConfig]
    tokenizer: Optional[TokenizerLoadingConfig]
    truncate_and_pad: Optional[TruncateAndPadDatasetConfig]


def samples_from_config(
    config: SamplesConfig,
    tokenize: bool = True,
    shuffle: bool = True,
    buffer: bool = True,
    truncate_and_pad: bool = False,
    tokenizer: Optional[Tokenizer] = None,
) -> Iterable[Sample]:
    require_implementation(config, SamplesConfig)
    if tokenize:
        if config.tokenizer is None:
            raise ValueError("Tokenizer config is missing")
        tokenizer = tokenizer or load_tokenizer_from_config(config.tokenizer)
    else:
        tokenizer = None
    if shuffle and config.shuffle is None:
        raise ValueError("Shuffle config is missing")
    if buffer and config.buffering is None:
        raise ValueError("Buffering config is missing")
    if truncate_and_pad and config.truncate_and_pad is None:
        raise ValueError("Truncate and pad config is missing")

    def dataset_fn() -> Iterable[Sample]:
        dataset: Iterable[Sample]
        dataset = HuggingFaceDataset.from_config(config.huggingface)
        if tokenizer is not None:
            dataset = TokenizedDataset.from_tokenizer(dataset, tokenizer)
        if shuffle:
            assert config.shuffle is not None
            dataset = ShuffledDataset.from_config(dataset, config.shuffle)
        if truncate_and_pad:
            assert tokenizer is not None
            pad_token_id = int(tokenizer.token_to_id(PAD_TOKEN))
            assert config.truncate_and_pad is not None
            dataset = TruncateAndPadDataset.from_config(
                dataset, config.truncate_and_pad, pad_token_id
            )
        return dataset

    dataset: Iterable[Sample]
    if buffer:
        assert config.buffering is not None
        dataset = BufferedDataset.from_config(dataset_fn, config.buffering)
    else:
        dataset = dataset_fn()
    return dataset
