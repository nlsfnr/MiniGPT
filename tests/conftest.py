from pathlib import Path

import pytest
from tokenizers import Tokenizer

import minigpt.data as data
from minigpt import Config

_TOKENIZER_PATH = "./tests/tokenizer-8192.json"
_CONFIG_PATH = "./tests/config.yaml"


@pytest.fixture
def tokenizer() -> Tokenizer:
    return data.load_huggingface_tokenizer([_TOKENIZER_PATH], {})


@pytest.fixture
def config() -> Config:
    return Config.from_yaml(Path(_CONFIG_PATH))
