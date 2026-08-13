"""Multi-head self-attention built from scratch.

The projection layout uses separate ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj``
Linear layers. This keeps the code readable while remaining trivially loadable
from :class:`torch.nn.MultiheadAttention`, whose combined ``in_proj_weight`` is
just the vertical concatenation of the three projections. See
``tests/test_attention.py`` for the weight-copying parity test.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from esm_fast.functional import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (batch, seq, dim) -> (batch, num_heads, seq, head_dim)
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (batch, num_heads, seq, head_dim) -> (batch, seq, dim)
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, self.dim)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Self-attention over ``x`` of shape ``(batch, seq, dim)``.

        ``key_padding_mask`` is ``(batch, seq)`` with True marking padding
        positions to be ignored (same convention as nn.MultiheadAttention).
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        mask = _merge_masks(attn_mask, key_padding_mask, self.num_heads)

        dropout_p = self.dropout if self.training else 0.0
        attn = scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal
        )
        return self.out_proj(self._merge_heads(attn))


def _merge_masks(
    attn_mask: Tensor | None,
    key_padding_mask: Tensor | None,
    num_heads: int,
) -> Tensor | None:
    """Combine an attention mask and a key-padding mask into one broadcastable
    boolean mask of shape ``(batch, num_heads, q, k)`` where True == keep.
    """
    mask: Tensor | None = None
    if attn_mask is not None:
        m = attn_mask if attn_mask.dtype == torch.bool else attn_mask == 0
        mask = m
    if key_padding_mask is not None:
        # (batch, seq) True==pad -> (batch, 1, 1, seq) True==keep
        keep = ~key_padding_mask.bool()
        keep = keep[:, None, None, :]
        mask = keep if mask is None else (mask & keep)
    return mask
