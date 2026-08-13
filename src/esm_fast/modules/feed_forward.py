"""Position-wise feed-forward network (the transformer MLP block)."""

from __future__ import annotations

from torch import Tensor, nn

from esm_fast.functional import get_activation


class FeedForward(nn.Module):
    """``fc2(activation(fc1(x)))`` with optional dropout, matching the sublayer
    used inside :class:`torch.nn.TransformerEncoderLayer`.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))
