from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Protocol

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array
from einops import rearrange

logger = logging.getLogger('NoLo')


class MultiHeadAttention(hk.Module):

    def __init__(self,
                 num_heads: int,
                 value_size: int,
                 w_init: hk.initializers.Initializer,
                 model_size: Optional[int] = None,
                 dropout: float = 0.1,
                 name: Optional[str] = None,
                 ) -> None:
        '''Initialize the module.

        Args:
            num_heads: Number of attention heads.
            value_size: Size of the value vectors.
            w_init: Initializer for the attention weights.
            model_size: Size of the model. If None, use the value size multiplied
                by the number of heads.
            name: Name of the module.
        '''
        super().__init__(name=name)
        self.num_heads = num_heads
        self.w_init = w_init
        self.value_size = value_size
        self.model_size = model_size or value_size * num_heads
        self.dropout = dropout

    def __call__(self,
                 x: Array,  # B L V
                 is_training: bool,
                 ) -> Array:
        '''Compute multi-head attention.

        Args:
            x: Input tensor of shape [batch_size, length, model_size].
            is_training: Whether the model is in training mode.

        Returns:
            The attention output. Shape: [batch_size, query_length, model_size].
        '''
        chex.assert_rank(x, 3)
        # Projections
        projection = partial(hk.Linear, w_init=self.w_init, with_bias=False)
        q_proj = projection(self.value_size * self.num_heads, name='q_proj')
        kv_proj = projection(self.value_size * self.num_heads, name='kv_proj')
        o_proj = projection(self.model_size, name='o_proj')
        # Query and key/value
        q = q_proj(x) / x.shape[-1] ** 0.5  # B L H V
        q = rearrange(q, 'b l (h k) -> b h l k', h=self.num_heads)
        kv = kv_proj(x)  # B L H K
        kv = rearrange(kv, 'b l (h k) -> b h l k', h=self.num_heads)
        # Attention logits
        k_weights = hk.get_parameter('k_weights',
                                     shape=(1, self.num_heads, 1, self.value_size),
                                     init=hk.initializers.Constant(0.))  # 1 H 1 V
        l: Array = jnp.einsum('b h i k, b h j k -> b h i j', q, kv * (1. + k_weights))  # B H L L
        # Mask the attention scores
        mask = jnp.tril(jnp.ones_like(l))
        l = jnp.where(mask, l, -1e6)  # type: ignore
        if is_training:
            l = hk.dropout(hk.next_rng_key(), self.dropout, l)
        a = jax.nn.softmax(l, axis=-1)  # B H L L
        # Attention output
        y = jnp.einsum('b h i j, b h j v -> b h i v', a, kv)  # B H L V
        y = rearrange(y, 'b h l v -> b l (h v)')  # B L (H V)
        return o_proj(y)  # B L M


class EncoderBlock(hk.Module):

    def __init__(self,
                 num_heads: int,
                 value_size: int,
                 w_init: hk.initializers.Initializer,
                 mlp_size: Optional[int] = None,
                 model_size: Optional[int] = None,
                 dropout: float = 0.1,
                 name: Optional[str] = None,
                 ) -> None:
        '''Initialize the module.

        Args:
            num_heads: Number of attention heads.
            value_size: Size of the value vectors.
            w_init: Initializer for the attention weights.
            mlp_size: Size of the MLP hidden layer. If None, use four times the model size.
            model_size: Size of the model. If None, use the key size multiplied
                by the number of heads.
            name: Name of the module.
        '''
        super().__init__(name=name)
        self.num_heads = num_heads
        self.w_init = w_init
        self.mlp_size = mlp_size or 4 * (model_size or value_size * num_heads)
        self.value_size = value_size
        self.model_size = model_size or value_size * num_heads
        self.dropout = dropout

    def __call__(self,
                 x: Array,  # B L V
                 is_training: bool,
                 ) -> Array:
        '''Compute the output of a transformer encoder block.

        Args:
            x: Input vectors. Shape: [batch_size, sequence_length, model_size].
            is_training: Whether the model is in training mode.

        Returns:
            The output vectors. Shape: [batch_size, sequence_length, model_size].
        '''
        chex.assert_rank(x, 3)
        mha_ln = hk.LayerNorm(-1, False, False, name='mha_ln')
        mha = MultiHeadAttention(self.num_heads,
                                 self.value_size,
                                 self.w_init,
                                 self.model_size,
                                 self.dropout,
                                 name='mha')
        mlp = hk.Sequential([
            hk.LayerNorm(-1, False, False, name='mlp_ln'),
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


class Encoder(hk.Module):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 value_size: int,
                 w_init: hk.initializers.Initializer,
                 mlp_size: Optional[int] = None,
                 model_size: Optional[int] = None,
                 dropout: float = 0.1,
                 name: Optional[str] = None,
                 ) -> None:
        '''Initialize the module.

        Args:
            num_layers: Number of layers.
            num_heads: Number of attention heads.
            value_size: Size of the value vectors.
            w_init: Initializer for the attention weights.
            model_size: Size of the model. If None, use the key size multiplied
                by the number of heads.
            name: Name of the module.
        '''
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.w_init = w_init
        self.mlp_size = mlp_size or 4 * (model_size or value_size * num_heads)
        self.value_size = value_size
        self.model_size = model_size or value_size * num_heads
        self.dropout = dropout

    def __call__(self,
                 x: Array,  # B L V
                 is_training: bool,
                 ) -> Array:
        '''Compute the output of a transformer encoder stack.

        Args:
            x: Input vectors. Shape: [batch_size, sequence_length, model_size].
            is_training: Whether the model is in training mode.

        Returns:
            The output vectors. Shape: [batch_size, sequence_length, model_size].
        '''
        chex.assert_rank(x, 3)
        for i in range(self.num_layers):
            x = EncoderBlock(num_heads=self.num_heads,
                             value_size=self.value_size,
                             w_init=self.w_init,
                             mlp_size=self.mlp_size,
                             model_size=self.model_size,
                             dropout=self.dropout,
                             name=f'block_{i}')(x, is_training)
        return x


class LearnedPositionalEncoding(hk.Module):

    def __init__(self,
                 max_sequence_length: int,
                 dim: int,
                 name: Optional[str] = None,
                 ) -> None:
        super().__init__(name=name)
        self.max_sequence_length = max_sequence_length
        self.dim = dim

    def __call__(self,
                 x: Array,
                 ) -> Array:
        pos_emb = hk.get_parameter('pos_emb', [self.max_sequence_length, self.dim], x.dtype,
                                   init=hk.initializers.RandomNormal(stddev=0.02))
        return x + pos_emb[None, :x.shape[1], :]


class ModelConfig(Protocol):
    vocab_size: int
    max_sequence_length: int
    num_layers: int
    num_heads: int
    value_size: int
    w_init_var: float
    embed_init_var: float
    mlp_size: Optional[int] = None
    model_size: Optional[int] = None
    dropout: float = 0.1


class Model(hk.Module):

    def __init__(self,
                 vocab_size: int,
                 max_sequence_length: int,
                 num_layers: int,
                 num_heads: int,
                 value_size: int,
                 w_init: hk.initializers.Initializer,
                 embed_init: hk.initializers.Initializer,
                 mlp_size: Optional[int] = None,
                 model_size: Optional[int] = None,
                 dropout: float = 0.1,
                 name: Optional[str] = None,
                 ) -> None:
        '''Initialize the module.

        Args:
            vocab_size: Size of the vocabulary.
            max_sequence_length: Maximum sequence length.
            num_layers: Number of stacked encoder blocks.
            num_heads: Number of attention heads.
            value_size: Size of the value vectors.
            w_init: Initializer for the attention weights.
            embed_init: Initializer for the embedding weights.
            mlp_size: Size of the MLP hidden layer. If None, use four times the model size.
            model_size: Size of the model. If None, use the key size multiplied
                by the number of heads.
            name: Name of the module.
        '''
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.value_size = value_size
        self.w_init = w_init
        self.embed_init = embed_init
        self.mlp_size = mlp_size or 4 * (model_size or value_size * num_heads)
        self.model_size = model_size or value_size * num_heads
        self.dropout = dropout

    @classmethod
    def from_config(cls,
                    config: ModelConfig,
                    ) -> Model:
        '''Create a model from a configuration object.'''
        return cls(vocab_size=config.vocab_size,
                   max_sequence_length=config.max_sequence_length,
                   num_layers=config.num_layers,
                   num_heads=config.num_heads,
                   value_size=config.value_size,
                   w_init=hk.initializers.TruncatedNormal(config.w_init_var),
                   embed_init=hk.initializers.TruncatedNormal(config.embed_init_var),
                   mlp_size=config.mlp_size,
                   model_size=config.model_size,
                   dropout=config.dropout)

    def __call__(self,
                 indices: Array,
                 is_training: bool,
                 ) -> Array:
        '''Compute the output of a transformer encoder stack.

        Args:
            indices: Input indices. Shape: [batch_size, sequence_length].
            is_training: Whether the model is in training mode.

        Returns:
            The output vectors. Shape: [batch_size, sequence_length, model_size].
        '''
        chex.assert_rank(indices, 2)
        x = hk.Embed(self.vocab_size,
                     self.model_size,
                     w_init=self.embed_init,
                     name='embedding')(indices)
        x = LearnedPositionalEncoding(self.max_sequence_length,
                                      self.model_size)(x)
        x = Encoder(num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    value_size=self.value_size,
                    w_init=self.w_init,
                    mlp_size=self.mlp_size,
                    model_size=self.model_size,
                    dropout=self.dropout,
                    name='encoder')(x, is_training)
        x = hk.LayerNorm(-1, False, False, name='layer_norm')(x)
        logits = hk.Linear(self.vocab_size, w_init=self.embed_init, name='logits')(x)
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
