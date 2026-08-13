"""Parity of the encoder layer/stack against torch.nn.TransformerEncoderLayer.

We use the post-norm arrangement (``norm_first=False``) because that is the
default of PyTorch's layer and lets us copy weights across one-to-one.
"""

from __future__ import annotations

import torch

from esm_fast.config import ModelConfig
from esm_fast.modules.encoder import TransformerEncoder, TransformerEncoderLayer
from tests.conftest import ATOL, RTOL


def _copy_layer_weights(ours: TransformerEncoderLayer, ref: torch.nn.TransformerEncoderLayer) -> None:
    with torch.no_grad():
        w_q, w_k, w_v = ref.self_attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = ref.self_attn.in_proj_bias.chunk(3, dim=0)
        ours.self_attn.q_proj.weight.copy_(w_q)
        ours.self_attn.q_proj.bias.copy_(b_q)
        ours.self_attn.k_proj.weight.copy_(w_k)
        ours.self_attn.k_proj.bias.copy_(b_k)
        ours.self_attn.v_proj.weight.copy_(w_v)
        ours.self_attn.v_proj.bias.copy_(b_v)
        ours.self_attn.out_proj.weight.copy_(ref.self_attn.out_proj.weight)
        ours.self_attn.out_proj.bias.copy_(ref.self_attn.out_proj.bias)

        ours.feed_forward.fc1.weight.copy_(ref.linear1.weight)
        ours.feed_forward.fc1.bias.copy_(ref.linear1.bias)
        ours.feed_forward.fc2.weight.copy_(ref.linear2.weight)
        ours.feed_forward.fc2.bias.copy_(ref.linear2.bias)

        ours.norm1.weight.copy_(ref.norm1.weight)
        ours.norm1.bias.copy_(ref.norm1.bias)
        ours.norm2.weight.copy_(ref.norm2.weight)
        ours.norm2.bias.copy_(ref.norm2.bias)


def _make_pair(dim=64, heads=8, ffn=256):
    ours = TransformerEncoderLayer(dim, heads, ffn, activation="gelu", norm_first=False).eval()
    ref = torch.nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=ffn,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    ).eval()
    _copy_layer_weights(ours, ref)
    return ours, ref


def test_encoder_layer_matches_torch():
    ours, ref = _make_pair()
    x = torch.randn(3, 11, 64)
    torch.testing.assert_close(ours(x), ref(x), atol=ATOL, rtol=RTOL)


def test_encoder_layer_matches_torch_with_padding():
    ours, ref = _make_pair()
    x = torch.randn(2, 10, 64)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, 6:] = True
    torch.testing.assert_close(
        ours(x, key_padding_mask=mask),
        ref(x, src_key_padding_mask=mask),
        atol=ATOL,
        rtol=RTOL,
    )


def test_encoder_stack_matches_torch():
    config = ModelConfig(dim=64, num_heads=8, ffn_dim=256, num_layers=3, norm_first=False)
    ours = TransformerEncoder(config).eval()

    ref_layer = torch.nn.TransformerEncoderLayer(
        d_model=config.dim,
        nhead=config.num_heads,
        dim_feedforward=config.ffn_dim,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    ref = torch.nn.TransformerEncoder(ref_layer, num_layers=config.num_layers).eval()

    for ours_layer, ref_layer in zip(ours.layers, ref.layers, strict=True):
        _copy_layer_weights(ours_layer, ref_layer)

    x = torch.randn(2, 12, config.dim)
    torch.testing.assert_close(ours(x), ref(x), atol=ATOL, rtol=RTOL)
