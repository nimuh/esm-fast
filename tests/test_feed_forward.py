"""Parity of the feed-forward block against an equivalent nn.Sequential."""

from __future__ import annotations

import torch

from esm_fast.modules.feed_forward import FeedForward
from tests.conftest import ATOL, RTOL


def test_feed_forward_matches_reference():
    dim, hidden = 32, 128
    ff = FeedForward(dim, hidden, activation="gelu")

    ref = torch.nn.Sequential(
        torch.nn.Linear(dim, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, dim),
    )
    with torch.no_grad():
        ref[0].weight.copy_(ff.fc1.weight)
        ref[0].bias.copy_(ff.fc1.bias)
        ref[2].weight.copy_(ff.fc2.weight)
        ref[2].bias.copy_(ff.fc2.bias)

    x = torch.randn(4, 10, dim)
    torch.testing.assert_close(ff(x), ref(x), atol=ATOL, rtol=RTOL)
