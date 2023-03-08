from __future__ import annotations

from functools import partial
from typing import (Dict, List, Optional, Protocol, Tuple, Union,
                    runtime_checkable)

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, PRNGKey
from einops import rearrange, repeat

_DEFAULT_W_INIT = hk.initializers.VarianceScaling(0.01)


Telemetry = Union[None, Array, List["Telemetry"], Dict[str, "Telemetry"]]


def rotary_pos_emb(
    x: Array,  # B H S D
) -> Array:
    dim = x.shape[-1]
    seq = x.shape[-2]
    # Near eq. 15 in https://arxiv.org/abs/2104.09864, equivalent to those
    # in https://arxiv.org/abs/1706.03762
    ts = jnp.arange(0, dim, 2, dtype=jnp.float32)  # D/2
    inv_freqs = 10_000 ** (-ts / dim)  # D/2
    grid = jnp.einsum("s, d -> s d", jnp.arange(seq), inv_freqs)  # S D/2
    # Eq. 34 in https://arxiv.org/abs/2104.09864
    sin_embs = repeat(jnp.sin(grid), "s d -> 1 s (d 2)")  # B S D
    cos_embs = repeat(jnp.cos(grid), "s d -> 1 s (d 2)")  # B S D
    # Pairwise swap with alternating signs
    x1, x2 = x[..., ::2], x[..., 1::2]  # [x1, x3, x5, ...], [x2, x4, x6, ...]
    x1x2 = jnp.stack([-x2, x1], axis=-1)  # [[-x2, x1], [-x4, x3], ...]
    xs = rearrange(x1x2, "... d two -> ... (d two)", two=2)  # [-x2, x1, -x4, x3, ...]
    out = x * cos_embs + xs * sin_embs
    return out


class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        num_heads: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads

    def __call__(
        self,
        x: Array,
        collect_telemetry: bool,
        mask: Optional[Array] = None,
    ) -> Tuple[Array, Telemetry]:
        model_dim = x.shape[-1]
        key_size = model_dim // self.num_heads
        # Projections
        projection = partial(hk.Linear, w_init=_DEFAULT_W_INIT, with_bias=False)
        q_proj = projection(key_size * self.num_heads, name="q_proj")
        k_proj = projection(key_size * self.num_heads, name="k_proj")
        v_proj = projection(key_size * self.num_heads, name="v_proj")
        o_proj = projection(model_dim, name="o_proj")
        # Q, K, V
        q = q_proj(x) / x.shape[-1] ** 0.5  # B L H K
        q = rearrange(q, "b l (h k) -> b h l k", h=self.num_heads)
        q = rotary_pos_emb(q)
        k = k_proj(x)  # B L H K
        k = rearrange(k, "b l (h k) -> b h l k", h=self.num_heads)
        k = rotary_pos_emb(k)
        v = v_proj(x)  # B L H V
        v = rearrange(v, "b l (h v) -> b h l v", h=self.num_heads)
        # Attention weights
        l: Array = jnp.einsum("b h i k, b h j k -> b h i j", q, k)  # B H L L
        if mask is not None:
            l = l + mask
        a = jax.nn.softmax(l, axis=-1)  # B H L L
        # Attention output
        y = jnp.einsum("b h i j, b h j v -> b h i v", a, v)  # B H L V
        y = rearrange(y, "b h l v -> b l (h v)")  # B L (H V)
        o = o_proj(y)  # B L M
        telemetry: Telemetry = (
            dict(q=q, k=k, v=v, l=l, y=y, o=o) if collect_telemetry else None
        )
        return o, telemetry


class FeedForward(hk.Module):
    def __init__(
        self,
        hidden_dim: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.hidden_dim = hidden_dim

    def __call__(
        self,
        x: Array,
        collect_telemetry: bool,
    ) -> Tuple[Array, Telemetry]:
        model_dim = x.shape[-1]
        # Projections
        projection = partial(hk.Linear, w_init=_DEFAULT_W_INIT, with_bias=False)
        w1 = projection(self.hidden_dim, name="w1")
        w2 = projection(self.hidden_dim, name="w2")
        w3 = projection(model_dim, name="w3")
        # PaLM-like SwiGLU
        h = jax.nn.silu(w1(x)) * w2(x)  # B L H
        y = w3(h)  # B L M
        telemetry: Telemetry = dict(h=h, y=y) if collect_telemetry else None
        return y, telemetry


class Block(hk.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def __call__(
        self,
        x: Array,
        is_training: bool,
        collect_telemetry: bool,
        mask: Optional[Array] = None,
    ) -> Tuple[Array, Telemetry]:
        mha = MultiHeadAttention(self.num_heads, name="mha")
        mha_ln = hk.LayerNorm(-1, True, False, name="mha_ln")
        ff = FeedForward(self.hidden_dim, name="ff")
        ff_ln = hk.LayerNorm(-1, True, False, name="ff_ln")
        # Multi-head attention
        y, mha_telemetry = mha(mha_ln(x), collect_telemetry, mask)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        x = x + y
        # Feed-forward
        z, ff_telemetry = ff(ff_ln(x), collect_telemetry)
        if is_training:
            z = hk.dropout(hk.next_rng_key(), self.dropout, z)
        out = x + z
        telemetry: Telemetry = (
            dict(mha=mha_telemetry, ff=ff_telemetry, mha_out=y, ff_out=z, out=out)
            if collect_telemetry
            else None
        )
        return out, telemetry


@runtime_checkable
class ModelConfig(Protocol):
    num_layers: int
    vocabulary_size: int
    embedding_dim: int
    model_dim: int
    num_heads: int
    hidden_dim: int
    dropout: float


class Model(hk.Module):
    def __init__(
        self,
        num_layers: int,
        vocabulary_size: int,
        embedding_dim: int,
        model_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_layers = num_layers
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    @classmethod
    def from_config(cls, config: ModelConfig) -> Model:
        return cls(
            num_layers=config.num_layers,
            vocabulary_size=config.vocabulary_size,
            embedding_dim=config.embedding_dim,
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    def __call__(
        self,
        indices: Array,
        is_training: bool,
        collect_telemetry: bool,
        mask: Optional[Array] = None,
    ) -> Tuple[Array, Telemetry]:
        embedding = hk.Embed(
            self.vocabulary_size,
            self.embedding_dim,
            w_init=_DEFAULT_W_INIT,
            name="embedding",
        )
        embedding_proj = (
            hk.Linear(
                self.model_dim,
                w_init=_DEFAULT_W_INIT,
                with_bias=False,
                name="embedding_proj",
            )
            if self.embedding_dim != self.model_dim
            else None
        )
        blocks = [
            Block(self.num_heads, self.hidden_dim, self.dropout, name=f"block_{i}")
            for i in range(self.num_layers)
        ]
        out_ln = hk.LayerNorm(-1, True, False, name="out_ln")
        out_proj = hk.Linear(
            self.vocabulary_size,
            with_bias=False,
            w_init=_DEFAULT_W_INIT,
            name="out_proj",
        )
        # Execution
        embeddings = embedding(indices)
        h = embedding_proj(embeddings) if embedding_proj is not None else embeddings
        block_telemetries: List[Telemetry] = []
        for block in blocks:
            h, block_telemetry = block(h, is_training, collect_telemetry, mask)
            block_telemetries.append(block_telemetry)
        # Output
        logits = out_proj(out_ln(h))
        telemetry: Telemetry = (
            dict(embeddings=embeddings, logits=logits, blocks=block_telemetries)
            if collect_telemetry
            else None
        )
        return logits, telemetry

    @classmethod
    def get_params(
        cls,
        config: ModelConfig,
        rng_or_seed: Union[int, PRNGKey],
    ) -> ArrayTree:
        assert isinstance(config, ModelConfig)

        def fn() -> None:
            model = cls.from_config(config)
            model(jnp.zeros((1, 1), dtype=jnp.int32), False, False)

        rng = (
            jax.random.PRNGKey(rng_or_seed)
            if isinstance(rng_or_seed, int)
            else rng_or_seed
        )
        return hk.transform(fn).init(rng)
