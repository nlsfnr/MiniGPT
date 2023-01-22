#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
from itertools import islice, tee
from multiprocessing import cpu_count
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Protocol,
                    Tuple, TypeVar, Union)

import click
import datasets
import lmdb
import numpy as np
import tokenizers.processors
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import tokenizers  # type: ignore
from tokenizers import Tokenizer  # type: ignore

if __name__ == '__main__':
    # If the module is executed we need to add the parent module to the discoverable imports
    sys.path.append('.')

from minigpt import common

logger = logging.getLogger(common.NAME)


# Disable CUDA for torch, since we only want Jax to use it
torch.cuda.is_available = lambda: False


class DataConfig(Protocol):
    '''Protocol for data configuration'''
    dataset_path: Path
    tokenizer_path: Path


class DataLoaderConfig(Protocol):
    '''Protocol for data loader configuration'''
    batch_size: int
    num_workers: int
    max_sequence_length: int


LMDB_MAP_SIZE = 1 << 40
UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'
PAD_TOKEN = '[PAD]'


TokenizerLike = Union[Path, Tokenizer]
DatasetLike = Union[Path, 'LMDBDataset']
T = TypeVar('T')


class LMDBDataset(Dataset):

    @classmethod
    def from_config(cls,
                    config: DataConfig,
                    ) -> LMDBDataset:
        return cls(config.dataset_path, config.tokenizer_path)

    def __init__(self,
                 db_path: Path,
                 tokenizer: TokenizerLike,
                 ) -> None:
        self.db_path = db_path
        if not db_path.exists():
            logger.error(f'LMDB database {db_path} does not exist')
            raise FileNotFoundError(db_path)
        self.tokenizer = get_tokenizer(tokenizer)
        self.env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)
        self.keys = json.loads(self.txn.get('keys'.encode()).decode())
        self.length = len(self.keys)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        value = self.txn.get(self.keys[index].encode())
        sample = json.loads(value.decode())
        return sample

    def get_dataloader_from_config(self,
                                   config: DataLoaderConfig,
                                   additional_sequence_length: int = 0,
                                   ) -> DataLoader:
        return self.get_dataloader(config.batch_size,
                                   config.max_sequence_length + additional_sequence_length)

    def get_dataloader(self,
                       batch_size: int,
                       sequence_length: int,
                       ) -> DataLoader:
        self.tokenizer.enable_truncation(sequence_length)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(PAD_TOKEN),
                                      length=sequence_length)

        def collate_fn(samples: List[Dict[str, Any]]) -> np.ndarray:
            encodings = self.tokenizer.encode_batch([sample['text'] for sample in samples])
            return np.asarray([encoding.ids for encoding in encodings])

        return DataLoader(self,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=True,
                          num_workers=cpu_count() // 2)


def get_dataset(d: DatasetLike, t: TokenizerLike) -> LMDBDataset:
    '''Ensures that the dataset is an LMDBDataset'''
    if isinstance(d, Path):
        return LMDBDataset(d, t)
    return d


def get_tokenizer(t: TokenizerLike) -> Tokenizer:
    '''Ensures that the tokenizer is a transformers.PreTrainedTokenizer'''
    if isinstance(t, Path):
        if not t.exists():
            logger.error(f'Tokenizer {t} does not exist')
            raise FileNotFoundError(f'Could not find tokenizer at {t}')
        return Tokenizer.from_file(str(t))
    return t


def load_hf_dataset(name: str,
                    ) -> Iterator[Dict[str, Any]]:
    '''Loads a dataset from HuggingFace and stores it in an LMDB database'''
    dataset: Iterable[Dict[str, Any]]
    if name == 'imdb':
        dataset = datasets.load_dataset('imdb', split='unsupervised')
    elif name == 'wikitext':
        dataset = datasets.load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    elif name == 'dummy':
        dataset = [dict(text='Hello world!')] * 5
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return (dict(text=sample['text']) for sample in dataset)


def length_filter(samples: Iterator[Dict[str, Any]],
                  min_length: int,
                  verbose: bool = True,
                  ) -> Iterator[Dict[str, Any]]:
    '''Filters out samples that are too short.'''
    n_removed = 0
    for sample in samples:
        if len(sample['text']) >= min_length:
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
        tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")  # type: ignore
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


def tokenize_samples(samples_or_db_path: Union[Iterator[Dict[str, Any]], Path],
                     tokenizer: TokenizerLike,
                     ) -> Iterator[Dict[str, Any]]:
    '''Tokenize a dataset and store the result in a new database. Uses batching
    to improve performance.'''
    if isinstance(samples_or_db_path, Path):
        samples = load_samples(samples_or_db_path)
    else:
        samples = samples_or_db_path
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


def into_chunks(iterable: Iterator[T], size: int) -> Iterator[List[T]]:
    '''Yield successive n-sized chunks from iterable.'''
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def new_tokenizer(samples_or_db_path: Union[Iterator[Dict[str, Any]], Path],
                  path: Path,
                  vocab_size: int = 1 << 15,
                  min_frequency: int = 10,
                  tokenizer_kind: str = 'sentencepiece',
                  ) -> None:
    '''Create a new tokenizer from a stream of samples.'''
    # Resolce the samples
    if isinstance(samples_or_db_path, Path):
        samples = load_samples(samples_or_db_path)
    else:
        samples = samples_or_db_path
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
    t2id = tokenizer.token_to_id
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f'{BOS_TOKEN} $A {EOS_TOKEN}',
        special_tokens=[(UNK_TOKEN, t2id(UNK_TOKEN)),
                        (BOS_TOKEN, t2id(BOS_TOKEN)),
                        (EOS_TOKEN, t2id(EOS_TOKEN)),
                        (PAD_TOKEN, t2id(PAD_TOKEN))])
    # Save the tokenizer
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def load_samples(db_path: Path) -> Iterator[Dict[str, Any]]:
    '''Load samples from a database.'''
    env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        keys = json.loads(txn.get('keys'.encode()).decode())
        samples = (json.loads(txn.get(key.encode()).decode()) for key in keys)
        yield from samples


def store_samples(samples: Iterator[Dict[str, Any]],
                  db_path: Path,
                  ) -> None:
    '''Store samples in an LMDB database'''
    db_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(db_path), map_size=LMDB_MAP_SIZE)
    with env.begin(write=True) as txn:
        keys = []
        for i, sample in tqdm(enumerate(samples)):
            key = str(i)
            txn.put(key.encode(), json.dumps(sample).encode())
            keys.append(key)
        txn.put('keys'.encode(), json.dumps(keys).encode())


def get_n_tokens(db_like: DatasetLike,
                 tokenizer_like: TokenizerLike,
                 max_samples: Optional[int] = None,
                 ) -> Tuple[int, float]:
    '''Get the number of tokens in a dataset. To avoid tokenizing the entire
    dataset only max_sampes are tokenzied and the total number of tokens in the
    dataset are extrapolated from there. Returns the estimated number of tokens
    and the variance of tokens pers sample across the tokenized samples.'''
    tokenizer = get_tokenizer(tokenizer_like)
    ds = get_dataset(db_like, tokenizer)
    keys = (ds.keys
            if max_samples is None else
            np.random.permutation(ds.keys)[:max_samples])
    n_tokens = []
    for key in keys:
        n_tokens.append(len(ds[key]['input_ids']))
    return int(sum(n_tokens) * len(ds) / len(keys)), float(np.var(n_tokens))


def get_cli() -> click.Group:
    '''Get the command line interface for this module.'''

    cli = common.get_cli_group('data')

    @cli.command('new-dataset')
    @click.option('--path', '-p', type=Path, required=True, help='Where to save the dataset')
    @click.option('--name', '-n', type=str, required=True, help='Dataset name')
    @click.option('--min-length', '-l', type=int, default=0, help='Minimum length of a sample')
    @click.option('--min-chars-per-token', '-c', type=int, default=0.0,
                  help='Maximum number of characters per token')
    def cli_new_dataset(path: Path,
                        name: str,
                        min_length: int,
                        min_chars_per_token: float,
                        ) -> None:
        '''Create a new dataset from Huggingface.'''
        logger.info(f'Creating a new dataset at {path} with name {name}')
        samples = load_hf_dataset(name)
        if min_length > 0:
            samples = length_filter(samples, min_length)
        if min_chars_per_token > 0.0:
            samples = chars_per_token_filter(samples, min_chars_per_token)
        store_samples(samples, path)
        logger.info('Done')

    @cli.command('show-samples')
    @click.option('--dataset-path', '-d', type=Path, required=True, help='Path to the dataset')
    @click.option('--num-samples', '-n', type=int, default=10, help='Number of samples to show')
    def cli_show_samples(dataset_path: Path,
                         num_samples: int,
                         ) -> None:
        '''Show a few samples from a dataset.'''
        samples = load_samples(dataset_path)
        for _ in range(num_samples):
            print(next(samples))

    @cli.command('new-tokenizer')
    @click.option('--path', '-p', type=Path, required=True, help='Where to save the tokenizer')
    @click.option('--kind', '-k', type=str, required=True, help='Tokenizer kind')
    @click.option('--db-path', '-d', type=Path, required=True, help='Path to the database')
    @click.option('--vocab-size', '-v', type=int, default=1 << 15, help='Vocabulary size')
    @click.option('--min-frequency', '-m', type=int, default=10,
                  help='Minimum frequency of a word to be included in the vocabulary')
    def cli_new_tokenizer(path: Path,
                          kind: str,
                          db_path: Path,
                          vocab_size: int,
                          min_frequency: int,
                          ) -> None:
        '''Create a new tokenizer from a database.'''
        logger.info(f'Creating a new tokenizer at {path} with kind {kind}, '
                    f'vocab size {vocab_size}, min frequency {min_frequency}')
        new_tokenizer(db_path, path, vocab_size, min_frequency, kind)
        logger.info('Done')

    return cli

    # TODO: Add CLI endpoint for n_token estimation


if __name__ == '__main__':
    get_cli()()
