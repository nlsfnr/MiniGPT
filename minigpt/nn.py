from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, PRNGKey
from einops import rearrange, repeat

from .common import Config, get_logger

_DEFAULT_W_INIT = hk.initializers.VarianceScaling(0.01)

logger = get_logger()


def full_precision(fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    def inner(x: Array) -> Array:
        return fn(x.astype(jnp.float32)).astype(x.dtype)

    return inner


def rotary_pos_emb(
    x: Array,  # B H S D
) -> Array:
    dim, seq = x.shape[-1], x.shape[-2]
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
        *,
        num_heads: int,
        pos_emb_portion: float,
        name: str,
    ) -> None:
        """Multi-head attention.

        Args:
            num_heads: Number of attention heads.
            pos_emb_portion: Portion of the dimension to use for rotary positional
                embeddings.
            name: Name of the module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.pos_emb_portion = pos_emb_portion

    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
    ) -> Array:
        """Applies multi-head attention.

        Args:
            x: Input array of shape [batch, sequence, features].
            mask: Mask array of shape [sequence, sequence]. If
                provided, the attention will be masked out for the masked tokens. The
                mask should be broadcastable to the shape of the attention logits.

        Returns:
            Output array of shape [batch, sequence, features].
        """
        # Constants
        D, H = x.shape[-1], self.num_heads
        if D % H != 0:
            raise ValueError(f"Dimension {D} must be divisible by number of heads {H}")
        K = D // H
        # Projections
        projection = partial(hk.Linear, with_bias=False)
        q_proj = projection(K * H, name="q_proj")
        k_proj = projection(K * H, name="k_proj")
        v_proj = projection(K * H, name="v_proj")
        o_proj = projection(D, name="o_proj")
        # Q, K, V
        p = int(K * self.pos_emb_portion)
        q = q_proj(x) / K**0.5  # B L H K
        q = rearrange(q, "b l (h k) -> b h l k", h=H)
        q = jnp.concatenate([rotary_pos_emb(q[..., :p]), q[..., p:]], axis=-1)
        k = k_proj(x)  # B L H K
        k = rearrange(k, "b l (h k) -> b h l k", h=H)
        k = jnp.concatenate([rotary_pos_emb(k[..., :p]), k[..., p:]], axis=-1)
        v = v_proj(x)  # B L H V
        v = rearrange(v, "b l (h v) -> b h l v", h=H)
        # Attention weights
        l: Array = jnp.einsum("b h i k, b h j k -> b h i j", q, k)  # B H L L

        def _logits_to_weights(l_: Array) -> Array:
            nonlocal mask
            if mask is not None:
                mask = jnp.broadcast_to(mask, l_.shape)  # B 1 L L
                l_ = hk.remat(jnp.where)(mask, l_, -1e30)
            return jax.nn.softmax(l_, axis=-1)  # B H L L

        a = full_precision(_logits_to_weights)(l)  # B H L L
        # Attention output
        y = jnp.einsum("b h i j, b h j v -> b h i v", a, v)  # B H L V
        y = rearrange(y, "b h l v -> b l (h v)")  # B L (H V)
        o = o_proj(y)  # B L M
        return o


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
    ) -> Array:
        model_dim = x.shape[-1]
        # Projections
        projection = partial(hk.Linear, with_bias=False)
        w1 = projection(self.hidden_dim, name="w1")
        w2 = projection(self.hidden_dim, name="w2")
        w3 = projection(model_dim, name="w3")
        # PaLM-like SwiGLU
        a, b = w1(x), w2(x)
        h = hk.remat(lambda a_, b_: jax.nn.silu(a_) * b_)(a, b)
        y = w3(h)  # B L M
        return y


class Block(hk.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        hidden_dim: int,
        pos_emb_portion: float,
        dropout: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.pos_emb_portion = pos_emb_portion
        self.dropout = dropout

    def __call__(
        self,
        x: Array,
        is_training: bool,
        mask: Optional[Array] = None,
    ) -> Array:
        mha = MultiHeadAttention(
            num_heads=self.num_heads, pos_emb_portion=self.pos_emb_portion, name="mha"
        )
        mha_ln = hk.remat(hk.LayerNorm(-1, True, False, name="mha_ln"))
        ff = FeedForward(self.hidden_dim, name="ff")
        ff_ln = hk.remat(hk.LayerNorm(-1, True, False, name="ff_ln"))
        # Multi-head attention
        y = mha(mha_ln(x), mask)
        y = hk.dropout(hk.next_rng_key(), self.dropout, y) if is_training else y
        x = x + y
        # Feed-forward
        z = ff(ff_ln(x))
        z = hk.dropout(hk.next_rng_key(), self.dropout, z) if is_training else z
        out = x + z
        return out


class Model(hk.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        vocabulary_size: int,
        embedding_dim: int,
        model_dim: int,
        num_heads: int,
        pos_emb_portion: float,
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
        self.pos_emb_portion = pos_emb_portion
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    @classmethod
    def from_config(cls, config: Config) -> Model:
        cfg = config.model
        return cls(
            num_layers=int(cfg.num_layers),
            vocabulary_size=int(cfg.vocabulary_size),
            embedding_dim=int(cfg.embedding_dim),
            model_dim=int(cfg.model_dim),
            num_heads=int(cfg.num_heads),
            pos_emb_portion=float(cfg.pos_emb_portion),
            hidden_dim=int(cfg.hidden_dim),
            dropout=float(cfg.dropout),
        )

    def __call__(
        self,
        indices: Array,
        is_training: bool,
        mask: Optional[Array] = None,
    ) -> Array:
        embedding = hk.Embed(
            self.vocabulary_size,
            self.embedding_dim,
            w_init=_DEFAULT_W_INIT,
            name="embedding",
        )
        embedding_proj = (
            hk.Linear(
                self.model_dim,
                with_bias=False,
                name="embedding_proj",
            )
            if self.embedding_dim != self.model_dim
            else lambda x: x
        )
        blocks = [
            Block(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                pos_emb_portion=self.pos_emb_portion,
                dropout=self.dropout,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]
        out_ln = hk.LayerNorm(-1, True, False, name="out_ln")
        out_proj = hk.Linear(
            self.embedding_dim,
            with_bias=False,
            w_init=_DEFAULT_W_INIT,
            name="out_proj",
        )
        # Execution
        embeddings = embedding(indices)
        h = embedding_proj(embeddings)  # type: ignore
        for block in blocks:
            h = block(h, is_training, mask)
        final_hidden = out_proj(out_ln(h))
        logits = jnp.einsum("b s m, v m -> b s v", final_hidden, embedding.embeddings)
        return logits

    @classmethod
    def get_params(
        cls,
        config: Config,
        rng_or_seed: Union[int, PRNGKey],
        log_size: bool = True,
    ) -> ArrayTree:
        def fn() -> None:
            model = cls.from_config(config)
            model(jnp.zeros((1, 1), dtype=jnp.int32), False)

        rng = (
            jax.random.PRNGKey(rng_or_seed)
            if isinstance(rng_or_seed, int)
            else rng_or_seed
        )
        params = hk.transform(fn).init(rng)
        if log_size:
            params_n = hk.data_structures.tree_size(params)
            params_mb = round(hk.data_structures.tree_bytes(params) / 1e6, 2)
            logger.info(f"Model parameters: {params_n:,} ({params_mb:.2f} MB)")
        return params
