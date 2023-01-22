'''Common functionality shared across modules.'''
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import chex
import click
import haiku as hk
import jax
import jmp
import optax
import pydantic
import yaml

NAME = 'MiniGPT'

logger = logging.getLogger(NAME)


def set_debug(debug: bool) -> None:
    jax.config.update('jax_debug_nans', debug)
    jax.config.update('jax_debug_infs', debug)
    jax.config.update('jax_disable_jit', debug)
    if debug:
        logger.warn('Running in debug mode')


def get_rngs(seed: Optional[Union[hk.PRNGSequence, int]] = None,
             ) -> hk.PRNGSequence:
    '''Get a PRNG sequence from an int or an existing PRNG sequence.'''
    if isinstance(seed, hk.PRNGSequence):
        return seed
    seed = (random.randint(0, 2**32 - 1)
            if seed is None else
            seed)
    logger.info(f'Using seed {seed}')
    return hk.PRNGSequence(seed)


T = TypeVar('T', bound='YamlConfig')


class YamlConfig(pydantic.BaseModel):

    @classmethod
    def from_yaml(cls, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self: T, path: Path) -> T:
        with open(path, "w") as f:
            # Use self.json() instead of self.dict() to avoid having to catet
            # to edge cases such as serializing Paths.
            yaml.dump(json.loads(self.json()), f)
        return self


def save_checkpoint(path: Path,
                    config: YamlConfig,
                    params: chex.ArrayTree,
                    opt_state: optax.OptState,
                    rngs: hk.PRNGSequence,
                    loss_scale: jmp.LossScale,
                    step: int,
                    ) -> None:
    '''Save the checkpoint to a directory.'''
    path.mkdir(parents=True, exist_ok=True)
    # Save the configuration
    config.to_yaml(path / 'config.yaml')
    # Save the parameters
    with open(path / 'params.pkl', 'wb') as f:
        pickle.dump(params, f)
    # Save the optimizer state
    with open(path / 'opt_state.pkl', 'wb') as f:
        pickle.dump(opt_state, f)
    # Save the step as a yaml file
    with open(path / 'other.yaml', 'w') as f:
        yaml.dump(dict(step=step), f)
    # Save the RNGs
    with open(path / 'rngs.pkl', 'wb') as f:
        pickle.dump(rngs, f)
    # Save the loss scale
    with open(path / 'loss_scale.pkl', 'wb') as f:
        pickle.dump(loss_scale, f)
    logger.info(f'Saved checkpoint to {path} at step {step:,}.')


def load_checkpoint(path: Path,
                    config_class: Type[YamlConfig],
                    ) -> Dict[str, Any]:
    '''Load the checkpoint from a directory.'''
    config = config_class.from_yaml(path / 'config.yaml')
    # Load the parameters
    with open(path / 'params.pkl', 'rb') as f:
        params = pickle.load(f)
    # Load the optimizer state
    with open(path / 'opt_state.pkl', 'rb') as f:
        opt_state = pickle.load(f)
    # Load the step from the yaml file
    with open(path / 'other.yaml', 'r') as f:
        other = yaml.load(f, Loader=yaml.FullLoader)
    step = other['step']
    # Load the RNGs
    with open(path / 'rngs.pkl', 'rb') as f:
        rngs_internal_state = pickle.load(f).internal_state
    rngs = hk.PRNGSequence(0)
    rngs.replace_internal_state(rngs_internal_state)
    # Load the loss scale
    with open(path / 'loss_scale.pkl', 'rb') as f:
        loss_scale = pickle.load(f)
    return dict(config=config,
                params=params,
                opt_state=opt_state,
                rngs=rngs,
                step=step,
                loss_scale=loss_scale)


def get_cli_group(name: str) -> click.Group:
    '''Get a click group with a common set of options.'''
    full_name = f'{NAME} - {name}'

    @click.group(full_name)
    @click.option('--log-level', default='INFO', help='Log level')
    @click.option('--log-to-stdout', is_flag=True, help='Log to stdout instead of stderr')
    @click.option('--logfile', type=Path, default=Path('./logs.log'), help='Log file')
    @click.option('--debug', '-d', is_flag=True, help='Debug mode')
    def cli(log_level: str,
            log_to_stdout: bool,
            logfile: Optional[Path],
            debug: bool,
            ) -> None:
        '''MiniGPT, a GPT-like language model'''
        handlers: List[logging.Handler] = []
        handlers.append(logging.StreamHandler(sys.stdout if log_to_stdout else sys.stderr))
        if logfile is not None:
            handlers.append(logging.FileHandler(logfile))
        logging.basicConfig(level=log_level,
                            format='[%(asctime)s|%(name)s|%(levelname)s] %(message)s',
                            handlers=handlers)
        logger.info(f'Starting {full_name}')
        set_debug(debug)

    return cli
