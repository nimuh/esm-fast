"""LayerNorm implemented from scratch.

Numerically equivalent to :class:`torch.nn.LayerNorm` over the last dimension
(biased variance, same eps placement inside the sqrt).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        # Biased (population) variance, matching torch.nn.LayerNorm.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            normed = normed * self.weight
        if self.bias is not None:
            normed = normed + self.bias
        return normed

    def extra_repr(self) -> str:
        return f"{self.normalized_shape[0]}, eps={self.eps}, affine={self.elementwise_affine}"
