"""Parity of the from-scratch MultiHeadAttention against torch.nn.MultiheadAttention.

torch.nn.MultiheadAttention stores a single combined ``in_proj_weight`` of shape
``(3*dim, dim)`` which is ``concat([Wq, Wk, Wv])``. We split it back into our
separate projections and check the outputs agree.
"""

from __future__ import annotations

import torch

from esm_fast.modules.attention import MultiHeadAttention
from tests.conftest import ATOL, RTOL


def _copy_weights(ours: MultiHeadAttention, ref: torch.nn.MultiheadAttention) -> None:
    with torch.no_grad():
        w_q, w_k, w_v = ref.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = ref.in_proj_bias.chunk(3, dim=0)
        ours.q_proj.weight.copy_(w_q)
        ours.q_proj.bias.copy_(b_q)
        ours.k_proj.weight.copy_(w_k)
        ours.k_proj.bias.copy_(b_k)
        ours.v_proj.weight.copy_(w_v)
        ours.v_proj.bias.copy_(b_v)
        ours.out_proj.weight.copy_(ref.out_proj.weight)
        ours.out_proj.bias.copy_(ref.out_proj.bias)


def test_self_attention_matches_torch():
    dim, heads = 64, 8
    ours = MultiHeadAttention(dim, heads).eval()
    ref = torch.nn.MultiheadAttention(dim, heads, batch_first=True).eval()
    _copy_weights(ours, ref)

    x = torch.randn(3, 12, dim)
    out_ref, _ = ref(x, x, x, need_weights=False)
    torch.testing.assert_close(ours(x), out_ref, atol=ATOL, rtol=RTOL)


def test_self_attention_matches_torch_with_key_padding_mask():
    dim, heads = 32, 4
    ours = MultiHeadAttention(dim, heads).eval()
    ref = torch.nn.MultiheadAttention(dim, heads, batch_first=True).eval()
    _copy_weights(ours, ref)

    x = torch.randn(2, 9, dim)
    key_padding_mask = torch.zeros(2, 9, dtype=torch.bool)
    key_padding_mask[0, 7:] = True  # pad the tail of the first sequence
    key_padding_mask[1, 5:] = True

    out_ref, _ = ref(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
    torch.testing.assert_close(
        ours(x, key_padding_mask=key_padding_mask), out_ref, atol=ATOL, rtol=RTOL
    )


def test_self_attention_matches_torch_causal():
    dim, heads = 32, 4
    ours = MultiHeadAttention(dim, heads).eval()
    ref = torch.nn.MultiheadAttention(dim, heads, batch_first=True).eval()
    _copy_weights(ours, ref)

    x = torch.randn(2, 7, dim)
    causal = torch.nn.Transformer.generate_square_subsequent_mask(7)
    out_ref, _ = ref(x, x, x, attn_mask=causal, need_weights=False)
    torch.testing.assert_close(ours(x, is_causal=True), out_ref, atol=ATOL, rtol=RTOL)
