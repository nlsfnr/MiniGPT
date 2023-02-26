'''Implementation of the model and its components'''
from __future__ import annotations

import logging
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Optional, Protocol, Type, TypeVar

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array
from einops import rearrange, repeat

from . import common

logger = logging.getLogger(common.NAME)


T = TypeVar('T')


def rotary_pos_emb(x: Array,  # B H S D
                   ) -> Array:
    dim = x.shape[-1]
    seq = x.shape[-2]
    # Near eq. 15 in https://arxiv.org/abs/2104.09864, equivalent to those
    # in https://arxiv.org/abs/1706.03762
    ts = jnp.arange(0, dim, 2, dtype=jnp.float32)
    inv_freqs = 10_000 ** (-ts / dim)
    grid = jnp.einsum('s, d -> s d', jnp.arange(seq), inv_freqs)
    # Eq. 34 in https://arxiv.org/abs/2104.09864
    sin_embs = repeat(jnp.sin(grid), 's d -> 1 s (d 2)')
    cos_embs = repeat(jnp.cos(grid), 's d -> 1 s (d 2)')
    # Pairwise swap with alternating signs
    x1, x2 = x[..., ::2], x[..., 1::2]  # [x1, x3, x5, ...], [x2, x4, x6, ...]
    x1x2 = jnp.stack([-x2, x1], axis=-1)  # [[-x2, x1], [-x4, x3], ...]
    xs = rearrange(x1x2, '... d two -> ... (d two)', two=2)  # [-x2, x1, -x4, x3, ...]
    out = x * cos_embs + xs * sin_embs
    return out


class MultiHeadAttention(hk.Module):

    def __init__(self,
                 num_heads: int,
                 key_size: int,
                 w_init: hk.initializers.Initializer,
                 value_size: Optional[int] = None,
                 model_size: Optional[int] = None,
                 dropout: float = 0.1,
                 use_rotary_embedding: bool = False,
                 name: Optional[str] = None,
                 ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init = w_init
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding

    def __call__(self,
                 x: Array,  # B L V
                 is_training: bool,
                 ) -> Array:
        chex.assert_rank(x, 3)
        # Projections
        projection = partial(hk.Linear, w_init=self.w_init, with_bias=False)
        q_proj = projection(self.key_size * self.num_heads, name='q_proj')
        k_proj = projection(self.key_size * self.num_heads, name='k_proj')
        v_proj = projection(self.value_size * self.num_heads, name='v_proj')
        o_proj = projection(self.model_size, name='o_proj')
        # Q, K, V
        q = q_proj(x) / x.shape[-1] ** 0.5  # B L H K
        q = rearrange(q, 'b l (h k) -> b h l k', h=self.num_heads)
        k = k_proj(x)  # B L H K
        k = rearrange(k, 'b l (h k) -> b h l k', h=self.num_heads)
        v = v_proj(x)  # B L H V
        v = rearrange(v, 'b l (h v) -> b h l v', h=self.num_heads)
        if self.use_rotary_embedding:
            q = rotary_pos_emb(q)
            k = rotary_pos_emb(k)
        # Attention weights
        l: Array = jnp.einsum('b h i k, b h j k -> b h i j', q, k)  # B H L L
        mask = jnp.tril(jnp.ones_like(l))
        l = jnp.where(mask, l, -1e8)
        if is_training:
            l = hk.dropout(hk.next_rng_key(), self.dropout, l)
        a = jax.nn.softmax(l, axis=-1)  # B H L L
        # Attention output
        y = jnp.einsum('b h i j, b h j v -> b h i v', a, v)  # B H L V
        y = rearrange(y, 'b h l v -> b l (h v)')  # B L (H V)
        return o_proj(y)  # B L M


class DecoderBlock(hk.Module):

    def __init__(self,
                 num_heads: int,
                 key_size: int,
                 w_init: hk.initializers.Initializer,
                 mlp_size: Optional[int] = None,
                 value_size: Optional[int] = None,
                 model_size: Optional[int] = None,
                 dropout: float = 0.1,
                 use_rotary_embedding: bool = False,
                 name: Optional[str] = None,
                 ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init = w_init
        self.mlp_size = mlp_size or 4 * (model_size or key_size * num_heads)
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding

    def __call__(self,
                 x: Array,  # B L V
                 is_training: bool,
                 ) -> Array:
        chex.assert_rank(x, 3)
        mha_ln = hk.LayerNorm(-1, True, False, name='mha_ln')
        mha = MultiHeadAttention(self.num_heads,
                                 self.key_size,
                                 self.w_init,
                                 self.value_size,
                                 self.model_size,
                                 self.dropout,
                                 self.use_rotary_embedding,
                                 name='mha')
        mlp = hk.Sequential([
            hk.LayerNorm(-1, True, False, name='mlp_ln'),
            hk.Linear(self.mlp_size, w_init=self.w_init, name='mlp_1'),
            jax.nn.gelu,
            hk.Linear(self.model_size, w_init=self.w_init, name='mlp_2'),
        ], name='mlp_seq')
        y = mha(mha_ln(x), is_training)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        y = x + y
        z = mlp(y)
        if is_training:
            z = hk.dropout(hk.next_rng_key(), self.dropout, z)
        return y + z


class Decoder(hk.Module):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 key_size: int,
                 w_init: hk.initializers.Initializer,
                 mlp_size: Optional[int] = None,
                 value_size: Optional[int] = None,
                 model_size: Optional[int] = None,
                 dropout: float = 0.1,
                 use_rotary_embedding: bool = False,
                 name: Optional[str] = None,
                 ) -> None:
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init = w_init
        self.mlp_size = mlp_size or 4 * (model_size or key_size * num_heads)
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding

    def __call__(self,
                 x: Array,  # B L V
                 is_training: bool,
                 ) -> Array:
        chex.assert_rank(x, 3)
        for i in range(self.num_layers):
            x = DecoderBlock(num_heads=self.num_heads,
                             key_size=self.key_size,
                             w_init=self.w_init,
                             mlp_size=self.mlp_size,
                             value_size=self.value_size,
                             model_size=self.model_size,
                             dropout=self.dropout,
                             use_rotary_embedding=i == 0 and self.use_rotary_embedding,
                             name=f'block_{i}')(x, is_training)
        return x


class ModelConfig(Protocol):
    vocab_size: int
    embedding_size: int
    max_sequence_length: int
    num_layers: int
    num_heads: int
    key_size: int
    value_size: int
    w_init_var: float
    embed_init_var: float
    use_rotary_embedding: bool
    mlp_size: Optional[int] = None
    model_size: Optional[int] = None
    dropout: float = 0.1

    @classmethod
    @abstractmethod
    def from_yaml(cls: Type[T], path: Path) -> T:
        raise NotImplementedError

    @abstractmethod
    def to_yaml(self: T, path: Path) -> T:
        raise NotImplementedError


class Model(hk.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 max_sequence_length: int,
                 num_layers: int,
                 num_heads: int,
                 key_size: int,
                 w_init: hk.initializers.Initializer,
                 embed_init: hk.initializers.Initializer,
                 mlp_size: Optional[int] = None,
                 value_size: Optional[int] = None,
                 model_size: Optional[int] = None,
                 dropout: float = 0.1,
                 use_rotary_embedding: bool = False,
                 name: Optional[str] = None,
                 ) -> None:
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init = w_init
        self.embed_init = embed_init
        self.mlp_size = mlp_size or 4 * (model_size or key_size * num_heads)
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding
        if self.value_size * self.num_heads != self.model_size:
            logger.warning('value_size * num_heads != model_size: '
                           f'{self.value_size} * {self.num_heads} != {self.model_size}')

    @classmethod
    def from_config(cls,
                    config: ModelConfig,
                    ) -> Model:
        return cls(vocab_size=config.vocab_size,
                   embedding_size=config.embedding_size,
                   max_sequence_length=config.max_sequence_length,
                   num_layers=config.num_layers,
                   num_heads=config.num_heads,
                   key_size=config.key_size,
                   w_init=hk.initializers.TruncatedNormal(config.w_init_var),
                   embed_init=hk.initializers.TruncatedNormal(config.embed_init_var),
                   mlp_size=config.mlp_size,
                   model_size=config.model_size,
                   dropout=config.dropout,
                   use_rotary_embedding=config.use_rotary_embedding)

    def __call__(self,
                 indices: Array,
                 is_training: bool,
                 ) -> Array:
        chex.assert_rank(indices, 2)
        wte = hk.Embed(self.vocab_size,
                       self.embedding_size,
                       w_init=self.embed_init,
                       name='embedding')
        x = wte(indices)
        if not self.use_rotary_embedding:
            pte = hk.Embed(self.max_sequence_length,
                           self.embedding_size,
                           w_init=self.embed_init,
                           name='positional_embedding')
            x = x + pte(jnp.arange(indices.shape[1])[None, :])
        if self.model_size != self.embedding_size:
            x = hk.Linear(self.model_size,
                          with_bias=False,
                          name='emb_to_model')(x)
        x = Decoder(num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    key_size=self.key_size,
                    value_size=self.value_size,
                    w_init=self.w_init,
                    mlp_size=self.mlp_size,
                    model_size=self.model_size,
                    dropout=self.dropout,
                    use_rotary_embedding=self.use_rotary_embedding,
                    name='decoder')(x, is_training)
        x = hk.LayerNorm(-1, True, False, name='layer_norm')(x)
        x = hk.Linear(self.embedding_size,
                      with_bias=False,
                      name='model_to_emb')(x)
        logits = x @ wte.embeddings.T
        return logits

    @classmethod
    def get_params(cls,
                   config: ModelConfig,
                   rng: chex.PRNGKey,
                   ) -> chex.ArrayTree:
        indices = jnp.zeros((1, config.max_sequence_length), dtype=jnp.int32)

        def model_fn() -> None:
            model = cls.from_config(config)
            model(indices, is_training=False)

        model_hk = hk.transform(model_fn)
        params = model_hk.init(rng)
        params_n = hk.data_structures.tree_size(params)
        params_mb = round(hk.data_structures.tree_bytes(params) / 1e6, 2)
        logger.info(f'Model parameters: {params_n:,} ({params_mb:.2f} MB)')
        return params
