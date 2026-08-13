"""Parity of the fused residual + dropout + LayerNorm kernel.

The kernel in ``esm_fast.kernels.fused_res_dropout_layernorm`` is still a stub;
this file pins the contract it has to satisfy:

    fused_res_dropout_layernorm(x, residual, weight, bias, p=0.0, eps=1e-5, seed=0)

computing, for 2D float32 CUDA tensors,

    layer_norm(residual + dropout(x, p), (n_cols,), weight, bias, eps)

i.e. dropout applies to the *sublayer output* ``x`` and not to the residual
stream -- the ``norm1(x + dropout1(sa_block(x)))`` ordering used by
:class:`esm_fast.modules.encoder.TransformerEncoderLayer` in its post-norm path.

Dropout uses the kernel's own counter-based RNG, so it cannot be compared
elementwise against ``torch.nn.functional.dropout``. It is pinned three ways
instead: exact parity at ``p=0``, reproducibility/variation across seeds, and
the drop rate itself measured from a mask the test can reconstruct.

Skipped automatically on CPU-only machines (no CUDA / no Triton).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import ATOL, RTOL, requires_triton

pytestmark = [pytest.mark.gpu, requires_triton]


def _eager_reference(x, residual, weight, bias, eps):
    """The unfused chain, with dropout disabled."""
    return F.layer_norm(residual + x, (x.shape[-1],), weight, bias, eps=eps)


@pytest.mark.parametrize("shape", [(128, 64), (64, 512), (7, 33)])
def test_fused_res_dropout_layernorm_matches_eager_without_dropout(shape):
    from esm_fast.kernels.fused_res_dropout_layernorm import fused_res_dropout_layernorm

    n_cols = shape[1]
    x = torch.randn(*shape, device="cuda")
    residual = torch.randn(*shape, device="cuda")
    weight = torch.randn(n_cols, device="cuda")
    bias = torch.randn(n_cols, device="cuda")

    out = fused_res_dropout_layernorm(x, residual, weight, bias, p=0.0, eps=1e-5)

    torch.testing.assert_close(
        out, _eager_reference(x, residual, weight, bias, eps=1e-5), atol=ATOL, rtol=RTOL
    )


def test_fused_res_dropout_layernorm_matches_encoder_sublayer():
    """Tie-in with the post-norm block ordering the encoder uses."""
    from esm_fast.kernels.fused_res_dropout_layernorm import fused_res_dropout_layernorm

    dim = 128
    x = torch.randn(64, dim, device="cuda")
    residual = torch.randn(64, dim, device="cuda")
    norm = torch.nn.LayerNorm(dim, eps=1e-5).to("cuda")
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()

    out = fused_res_dropout_layernorm(x, residual, norm.weight, norm.bias, p=0.0, eps=1e-5)

    torch.testing.assert_close(out, norm(residual + x), atol=ATOL, rtol=RTOL)


def test_fused_res_dropout_layernorm_seed_controls_the_mask():
    from esm_fast.kernels.fused_res_dropout_layernorm import fused_res_dropout_layernorm

    x = torch.randn(256, 128, device="cuda")
    residual = torch.randn(256, 128, device="cuda")
    weight = torch.ones(128, device="cuda")
    bias = torch.zeros(128, device="cuda")

    def run(seed):
        return fused_res_dropout_layernorm(x, residual, weight, bias, p=0.5, eps=1e-5, seed=seed)

    # Same seed -> bit-identical; different seed -> a different mask.
    torch.testing.assert_close(run(0), run(0), atol=0.0, rtol=0.0)
    assert not torch.allclose(run(0), run(1234), atol=ATOL, rtol=RTOL)

    # And dropout must actually be doing something at p=0.5.
    no_drop = fused_res_dropout_layernorm(x, residual, weight, bias, p=0.0, eps=1e-5)
    assert not torch.allclose(run(0), no_drop, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_fused_res_dropout_layernorm_drop_rate_matches_p(p):
    """Measure the realised drop rate through a problem with a readable mask.

    With ``x`` all-ones and no residual, each pre-norm element is either ``0``
    (dropped) or ``1 / (1 - p)`` (kept, inverted-dropout scaling). LayerNorm is
    monotone within a row, so the dropped entries stay the row minimum and can
    simply be counted.
    """
    from esm_fast.kernels.fused_res_dropout_layernorm import fused_res_dropout_layernorm

    rows, cols = 1024, 1024
    x = torch.ones(rows, cols, device="cuda")
    residual = torch.zeros(rows, cols, device="cuda")
    weight = torch.ones(cols, device="cuda")
    bias = torch.zeros(cols, device="cuda")

    out = fused_res_dropout_layernorm(x, residual, weight, bias, p=p, eps=1e-5, seed=0)

    dropped = (out < out.mean(dim=-1, keepdim=True)).float().mean().item()
    # ~1M Bernoulli draws: the sampling error here is O(1e-3).
    assert abs(dropped - p) < 0.02, f"realised drop rate {dropped:.4f} != p={p}"
