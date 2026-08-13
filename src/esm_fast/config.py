"""Model configuration.

Defaults are deliberately small so tests run quickly on CPU. The field names and
semantics follow ESM-2 / RoBERTa so the config can later drive a real protein
masked-language-model without changing the module code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Vocabulary: ESM-2 uses a 33-token alphabet (20 amino acids + special/rare).
    vocab_size: int = 33

    # Core transformer dimensions.
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 512
    max_seq_len: int = 1024

    # Regularization.
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Numerics / structure.
    layer_norm_eps: float = 1e-5
    # ESM-2 places LayerNorm before the sublayers (pre-norm). Toggle to False to
    # reproduce the post-norm arrangement of torch.nn.TransformerEncoderLayer.
    norm_first: bool = True
    activation: str = "gelu"

    # Special token ids (ESM-2 alphabet ordering).
    pad_token_id: int = 1
    mask_token_id: int = 32
    bos_token_id: int = 0
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads
