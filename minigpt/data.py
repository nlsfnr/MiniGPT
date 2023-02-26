#!/usr/bin/env python3
from __future__ import annotations

import logging
import random
import sys
import threading
from functools import partial
from itertools import islice, tee
from pathlib import Path
from queue import Queue
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Protocol, TypeVar, Union)

import click
import datasets
import numpy as np
import tokenizers.processors
import torch
from torch.utils.data import DataLoader, IterableDataset

import tokenizers  # type: ignore
from tokenizers import Tokenizer  # type: ignore

if __name__ == '__main__':
    # If the module is executed we need to add the parent module to the discoverable imports
    sys.path.append('.')

from minigpt import common

LMDB_MAP_SIZE = 1 << 40
UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'
PAD_TOKEN = '[PAD]'


logger = logging.getLogger(common.NAME)


# Disable CUDA for torch since we only want Jax to use it
torch.cuda.is_available = lambda: False


class DataConfig(Protocol):
    '''Protocol for data configuration'''
    datasets: List[str]
    dataset_weights: List[float]
    tokenizer_path: Path
    vocab_size: int
    min_frequency: int
    min_length: int
    tokenizer_kind: str


class DataLoaderConfig(DataConfig, Protocol):
    '''Protocol for data loader configuration'''
    batch_size: int
    num_workers: int
    max_sequence_length: int
    shuffle_buffer_size: int


TokenizerLike = Union[Path, Tokenizer]
T = TypeVar('T')


def get_tokenizer(t: TokenizerLike) -> Tokenizer:
    '''Ensures that the tokenizer is a transformers.PreTrainedTokenizer'''
    if isinstance(t, Path):
        if not t.exists():
            logger.error(f'Tokenizer {t} does not exist')
            raise FileNotFoundError(f'Could not find tokenizer at {t}')
        return Tokenizer.from_file(str(t))
    return t


def get_batches(config: DataLoaderConfig,
                ) -> Iterator[np.ndarray]:
    tokenizer = get_tokenizer(config.tokenizer_path)

    def fn(dataset_name: str) -> Iterator[np.ndarray]:
        samples = stream_hf_dataset(dataset_name, repeat_when_done=True)
        samples = tokenize_samples(samples, tokenizer)
        ids: Iterator[List[int]]
        ids = (sample['input_ids'] for sample in samples)
        ids = (chunk
               for sample in ids
               for chunk in into_chunks(sample, config.max_sequence_length + 1))
        ids = length_filter(ids, config.min_length)
        np_ids = (np.asarray(chunk) for chunk in ids)
        return shuffle(np_ids, config.shuffle_buffer_size)

    all_ids = [fn(name) for name in config.datasets]
    ids = merge_samples(all_ids, config.dataset_weights)
    # Pad each sample
    pad_token = tokenizer.token_to_id(PAD_TOKEN)
    ids = (np.pad(chunk, (0, config.max_sequence_length + 1 - len(chunk)),
                  constant_values=pad_token)
           for chunk in ids)
    dl = DataLoader(IterDataset(ids),
                    batch_size=config.batch_size,
                    num_workers=1,
                    pin_memory=True,
                    drop_last=True,
                    shuffle=False)
    return iter(dl)


class IterDataset(IterableDataset):

    def __init__(self, samples: Iterator[np.ndarray]) -> None:
        self.samples = samples

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.samples


def stream_hf_dataset(name: str,
                      repeat_when_done: bool = False,
                      ) -> Iterator[Dict[str, Any]]:
    # Construct a function that returns an interator over the dataset.
    fn: Callable[[], Iterable[Dict[str, Any]]]
    if name == 'imdb':
        ds = datasets.load_dataset('imdb', split='unsupervised')
        fn = lambda: iter(ds)
    elif name == 'wikitext':
        ds = datasets.load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        fn = lambda: iter(ds)
    elif name == 'c4':
        fn_1 = partial(datasets.load_dataset, 'c4', 'en', split='train', streaming=True)
        fn = lambda: buffer(iter(fn_1()), 5000)
    elif name == 'bookcorpus':
        fn_1 = partial(datasets.load_dataset, 'bookcorpus', split='train', streaming=True)
        fn = lambda: buffer(iter(fn_1()), 5000)
    elif name == 'openwebtext':
        fn_1 = partial(datasets.load_dataset, 'openwebtext', split='train', streaming=True)
        fn = lambda: buffer(iter(fn_1()), 5000)
    elif name == 'dummy':
        fn = lambda: iter([dict(text='Hello world!')] * 5)
    else:
        raise ValueError(f'Unknown dataset: {name}')
    logger.info(f'Loading dataset {name}')
    while True:
        for sample in fn():
            yield sample
        if not repeat_when_done:
            break
        logger.info(f'Repeating dataset {name}')


def buffer(it: Iterator[T],
           buffer_size: int,
           ) -> Iterator[T]:
    queue: Queue[T] = Queue(maxsize=buffer_size)

    def producer() -> None:
        for x in it:
            queue.put(x)

    thread = threading.Thread(target=producer)
    thread.start()
    try:
        while True:
            yield queue.get()
    finally:
        thread.join()


def shuffle(samples: Iterator[T],
            buffer_size: int,
            ) -> Iterator[T]:
    logger.info(f'Filling shuffle buffer of size {buffer_size}')
    buffer = list(islice(samples, buffer_size))
    logger.info('Buffer filled')
    for next_sample in samples:
        index = random.randint(0, len(buffer) - 1)
        chosen_sample, buffer[index] = buffer[index], next_sample
        yield chosen_sample
    yield from random.sample(buffer, len(buffer))


def merge_samples(sample_streams: List[Iterator[T]],
                  weights: List[float],
                  seed: Optional[int] = None,
                  ) -> Iterator[T]:
    msg = (f'Number of sample streams ({len(sample_streams)}) does not match number of '
           f'weights ({len(weights)})')
    assert len(sample_streams) == len(weights), msg
    rng = random.Random(seed)
    while sum(weights) > 0:
        index = rng.choices(range(len(sample_streams)), weights=weights)[0]
        stream = sample_streams[index]
        try:
            yield next(stream)
        except StopIteration:
            logger.error(f'Sample stream exhausted: {index}')
            weights[index] = 0


def length_filter(samples: Iterator[T],
                  min_length: int,
                  verbose: bool = True,
                  key: Callable[[T], int] = len,  # type: ignore
                  ) -> Iterator[T]:
    '''Filters out samples that are too short.'''
    n_removed = 0
    for sample in samples:
        if key(sample) >= min_length:
            yield sample
        else:
            n_removed += 1
    if verbose:
        logger.info(f'Removed {n_removed} samples that were too short')


def chars_per_token_filter(samples: Iterator[Dict[str, Any]],
                           min_chars_per_token: float,
                           tokenizer: Optional[TokenizerLike] = None,
                           verbose: bool = True,
                           ) -> Iterator[Dict[str, Any]]:
    '''Filters out samples that have too few characters per token. Similar to
    the compression score from https://arxiv.org/abs/2212.14034'''
    if tokenizer is None:
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")  # type: ignore
        assert isinstance(tokenizer, Tokenizer)
    else:
        tokenizer = get_tokenizer(tokenizer)
    s1, s2 = tee(samples)
    n_removed = 0
    for original_sample, tokenized_sample in zip(s1, tokenize_samples(s2, tokenizer)):
        ids = tokenized_sample['input_ids']
        text = tokenized_sample['text']
        chars_per_token = len(text) / len(ids)
        if chars_per_token >= min_chars_per_token:
            yield original_sample
        else:
            n_removed += 1
    if verbose:
        logger.info(f'Removed {n_removed} samples that had too few characters per token')


def tokenize_samples(samples: Iterator[Dict[str, Any]],
                     tokenizer: TokenizerLike,
                     ) -> Iterator[Dict[str, Any]]:
    '''Tokenize a dataset and store the result in a new database. Uses batching
    to improve performance.'''
    tokenizer = get_tokenizer(tokenizer)
    samples_1, samples_2 = tee(samples)
    text_batches = into_chunks((sample['text']
                                for sample in samples_1), 1000)
    indices = (encoding.ids
               for text_batch in text_batches
               for encoding in tokenizer.encode_batch(text_batch))
    samples = (dict(sample, input_ids=indices)
               for sample, indices in zip(samples_2, indices))
    return samples


def into_chunks(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    '''Yield successive n-sized chunks from iterable.'''
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def new_tokenizer(samples: Iterator[Dict[str, Any]],
                  vocab_size: int = 1 << 15,
                  min_frequency: int = 10,
                  tokenizer_kind: str = 'sentencepiece',
                  ) -> Tokenizer:
    '''Create a new tokenizer from a stream of samples.'''
    # Instantiate a new tokenizer.
    if tokenizer_kind == 'sentencepiece':
        tokenizer = tokenizers.SentencePieceBPETokenizer()  # type: ignore
    elif tokenizer_kind == 'wordpiece':
        tokenizer = tokenizers.BertWordPieceTokenizer()  # type: ignore
    else:
        raise ValueError(f'Unknown tokenizer kind: {tokenizer_kind}')
    tokenizer.train_from_iterator((sample['text'] for sample in samples),
                                  special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN],
                                  vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f'{BOS_TOKEN} $A {EOS_TOKEN}',
        special_tokens=[(UNK_TOKEN, tokenizer.token_to_id(UNK_TOKEN)),
                        (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
                        (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
                        (PAD_TOKEN, tokenizer.token_to_id(PAD_TOKEN))])
    return tokenizer


def save_tokenizer(tokenizer: TokenizerLike,
                   path: Path,
                   ) -> Tokenizer:
    tokenizer = get_tokenizer(tokenizer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def get_cli() -> click.Group:
    '''Get the command line interface for this module.'''

    cli = common.get_cli_group('data')

    class DLConfig(common.YamlConfig):
        datasets: List[str]
        dataset_weights: List[float]
        tokenizer_path: Path
        vocab_size: int
        min_frequency: int
        min_length: int
        tokenizer_kind: str
        batch_size: int
        num_workers: int
        max_sequence_length: int
        shuffle_buffer_size: int

    @cli.command('show-samples')
    @click.option('--config-path', '-c', type=Path,
                  help='Path to the configuration file.')
    @click.option('--num-samples', '-n', type=int, default=10, help='Number of samples to show')
    def cli_show_samples(config_path: Path,
                         num_samples: int,
                         ) -> None:
        '''Show a few samples from a dataset.'''
        config = DLConfig.from_yaml(config_path)
        batches = get_batches(config)
        tokenizer = get_tokenizer(config.tokenizer_path)
        samples = (sample
                   for batch in batches
                   for sample in batch)
        for _ in range(num_samples):
            sample = next(samples)
            print(sample)
            print(tokenizer.decode(list(sample)))

    class Config(common.YamlConfig):
        datasets: List[str]
        dataset_weights: List[float]
        tokenizer_path: Path
        vocab_size: int
        min_frequency: int
        min_length: int
        tokenizer_kind: str

    @cli.command('new-tokenizer')
    @click.option('--config-path', '-c', type=Path, required=True,
                  help='Path to the config file')
    @click.option('--max-samples', '-m', type=int, default=None,
                  help='Maximum number of samples to use for training the tokenizer.')
    def cli_new_tokenizer(config_path: Path,
                          max_samples: Optional[int],
                          ) -> None:
        '''Create a new tokenizer from a database.'''
        config: DataConfig = Config.from_yaml(config_path)
        sample_streams = [stream_hf_dataset(dataset)
                          for dataset in config.datasets]
        sample_streams = [length_filter(stream, config.min_length, key=lambda x: len(x['text']))
                          for stream in sample_streams]
        samples = merge_samples(sample_streams, config.dataset_weights)
        if max_samples is not None:
            samples = islice(samples, max_samples)
        tokenizer = new_tokenizer(samples,
                                  vocab_size=config.vocab_size,
                                  min_frequency=config.min_frequency,
                                  tokenizer_kind=config.tokenizer_kind)
        save_tokenizer(tokenizer, config.tokenizer_path)
        logger.info(f'Saved tokenizer to {config.tokenizer_path}')

    return cli


if __name__ == '__main__':
    get_cli()()
