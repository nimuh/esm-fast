"""Transformer encoder layer and stack.

The layer supports both pre-norm (``norm_first=True``, the ESM-2 arrangement)
and post-norm (``norm_first=False``, the default of
:class:`torch.nn.TransformerEncoderLayer`). The post-norm path is written to
match PyTorch's block ordering exactly so weights can be copied across for a
parity test — see ``tests/test_encoder.py``.
"""

from __future__ import annotations

import copy

from torch import Tensor, nn

from esm_fast.config import ModelConfig
from esm_fast.modules.attention import MultiHeadAttention
from esm_fast.modules.feed_forward import FeedForward
from esm_fast.modules.layer_norm import LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout=attention_dropout)
        self.feed_forward = FeedForward(dim, ffn_dim, activation=activation, dropout=dropout)
        self.norm1 = LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = LayerNorm(dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _sa_block(
        self, x: Tensor, attn_mask: Tensor | None, key_padding_mask: Tensor | None
    ) -> Tensor:
        x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.feed_forward(x))

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask, key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        layer = TransformerEncoderLayer(
            dim=config.dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            norm_first=config.norm_first,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_layers)]
        )
        # Final norm is standard for pre-norm transformers (ESM-2, GPT, ...).
        self.final_norm = LayerNorm(config.dim, eps=config.layer_norm_eps) if config.norm_first else None

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x
