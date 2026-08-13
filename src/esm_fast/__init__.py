"""esm-fast: a transformer built from scratch in PyTorch with Triton kernels.

The public surface mirrors the pieces we validate against PyTorch's official
implementations:

    from esm_fast import (
        ModelConfig,
        LayerNorm,
        FeedForward,
        MultiHeadAttention,
        TransformerEncoderLayer,
        TransformerEncoder,
    )

Triton kernels live under ``esm_fast.kernels`` and are only importable/usable on
a CUDA device; importing this top-level package never requires Triton.
"""

from esm_fast.config import ModelConfig
from esm_fast.functional import gelu, scaled_dot_product_attention
from esm_fast.modules.attention import MultiHeadAttention
from esm_fast.modules.encoder import TransformerEncoder, TransformerEncoderLayer
from esm_fast.modules.feed_forward import FeedForward
from esm_fast.modules.layer_norm import LayerNorm

__all__ = [
    "ModelConfig",
    "LayerNorm",
    "FeedForward",
    "MultiHeadAttention",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "gelu",
    "scaled_dot_product_attention",
]

__version__ = "0.0.1"
