"""Parity of the Triton rotary-embedding kernel against an eager reference.

The kernel in ``esm_fast.kernels.rope`` is still a stub; this file pins the
contract it has to satisfy:

    rope(x, cos, sin) -> Tensor

with ``x`` of shape ``(batch, heads, seq, head_dim)`` (float32, CUDA, even
``head_dim``) and ``cos``/``sin`` of shape ``(seq, head_dim)`` -- i.e. each of
the ``head_dim // 2`` inverse frequencies already repeated twice, so they line
up elementwise with ``x``.

The rotation convention is the split-halves ("rotate_half") one used by RoFormer
and by ESM-2's ``RotaryEmbedding``, *not* the interleaved GPT-J pairing:

    out = x * cos + rotate_half(x) * sin,  rotate_half([x1, x2]) = [-x2, x1]

``_rotate_half``/``_rope_reference`` below are that reference. They belong in
``esm_fast.functional`` once rotary embeddings land in the model (per the repo
convention that the math has one home); until then this file carries them so the
kernel has something to be pinned against.

Skipped automatically on CPU-only machines (no CUDA / no Triton).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from tests.conftest import ATOL, RTOL, requires_triton

pytestmark = [pytest.mark.gpu, requires_triton]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_reference(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Eager rotary embedding; ``cos``/``sin`` broadcast over batch and heads."""
    return x * cos + _rotate_half(x) * sin


def _cos_sin(seq_len: int, head_dim: int, base: float = 10000.0, device: str = "cuda"):
    """The standard RoPE tables, each frequency repeated twice along head_dim."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (seq, head_dim // 2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq, head_dim)
    return emb.cos(), emb.sin()


@pytest.mark.parametrize(
    "batch,heads,seq,head_dim",
    [
        (1, 1, 16, 8),
        (2, 8, 128, 64),
        (4, 4, 37, 32),  # sequence length not a multiple of any block size
    ],
)
def test_triton_rope_matches_reference(batch, heads, seq, head_dim):
    from esm_fast.kernels.rope import rope

    x = torch.randn(batch, heads, seq, head_dim, device="cuda")
    cos, sin = _cos_sin(seq, head_dim)

    torch.testing.assert_close(
        rope(x, cos, sin), _rope_reference(x, cos, sin), atol=ATOL, rtol=RTOL
    )


def test_triton_rope_applies_to_query_and_key_alike():
    """q and k share one table; the kernel must not special-case either."""
    from esm_fast.kernels.rope import rope

    q = torch.randn(2, 4, 64, 32, device="cuda")
    k = torch.randn(2, 4, 64, 32, device="cuda")
    cos, sin = _cos_sin(64, 32)

    torch.testing.assert_close(
        rope(q, cos, sin), _rope_reference(q, cos, sin), atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        rope(k, cos, sin), _rope_reference(k, cos, sin), atol=ATOL, rtol=RTOL
    )


def test_triton_rope_is_norm_preserving():
    """RoPE is a rotation: per-position vector norms are unchanged."""
    from esm_fast.kernels.rope import rope

    x = torch.randn(2, 4, 128, 64, device="cuda")
    cos, sin = _cos_sin(128, 64)

    out = rope(x, cos, sin)

    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-4)


def test_triton_rope_encodes_relative_position():
    """``<rope(q)[m], rope(k)[n]>`` may depend on ``m - n`` only.

    Shifting both positions by the same offset must leave the attention score
    unchanged -- the property rotary embeddings exist for, and the one an
    off-by-one in the position index would break.
    """
    from esm_fast.kernels.rope import rope

    seq, head_dim, shift = 64, 32, 5
    cos, sin = _cos_sin(seq, head_dim)

    # Same vector repeated at every position, so only the rotation varies.
    q_vec = torch.randn(head_dim, device="cuda")
    k_vec = torch.randn(head_dim, device="cuda")
    q = q_vec.expand(1, 1, seq, head_dim).contiguous()
    k = k_vec.expand(1, 1, seq, head_dim).contiguous()

    q_rot = rope(q, cos, sin)[0, 0]
    k_rot = rope(k, cos, sin)[0, 0]

    m, n = 20, 12
    torch.testing.assert_close(
        q_rot[m] @ k_rot[n],
        q_rot[m + shift] @ k_rot[n + shift],
        atol=1e-4,
        rtol=1e-4,
    )


def test_triton_rope_is_identity_at_position_zero():
    """Position 0 has cos=1, sin=0, so the first row must come back untouched."""
    from esm_fast.kernels.rope import rope

    x = torch.randn(1, 2, 8, 16, device="cuda")
    cos, sin = _cos_sin(8, 16)

    torch.testing.assert_close(rope(x, cos, sin)[:, :, 0], x[:, :, 0], atol=ATOL, rtol=RTOL)
