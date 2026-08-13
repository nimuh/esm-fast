"""Pure-functional building blocks.

These are the reference numerics for the from-scratch model. Each function is
written to match a specific ``torch.nn.functional`` counterpart exactly (up to
floating point), which is what the tests assert.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """Gaussian Error Linear Unit.

    ``approximate="none"`` is the exact erf formulation used by ESM-2 and is the
    default of :func:`torch.nn.functional.gelu`. ``approximate="tanh"`` selects
    the tanh approximation.
    """
    if approximate == "none":
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    if approximate == "tanh":
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
        )
    raise ValueError(f"unknown gelu approximation: {approximate!r}")


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    """Reference scaled dot-product attention.

    Matches the semantics of :func:`torch.nn.functional.scaled_dot_product_attention`:

    * ``query/key/value`` are ``(..., seq, head_dim)``.
    * ``attn_mask`` may be boolean (True == keep) or additive float.
    * ``is_causal`` applies a lower-triangular mask and must not be combined with
      an explicit ``attn_mask``.
    """
    if is_causal and attn_mask is not None:
        raise ValueError("attn_mask and is_causal cannot both be set")

    scale = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = (query @ key.transpose(-2, -1)) * scale

    L, S = query.size(-2), key.size(-2)
    if is_causal:
        causal = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ value


ACTIVATIONS = {
    "gelu": gelu,
    "relu": F.relu,
}


def get_activation(name: str):
    try:
        return ACTIVATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown activation: {name!r}") from exc
