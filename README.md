# esm-fast

A transformer implemented **from scratch in PyTorch**, with hand-written **Triton
kernels** benchmarked and validated against the PyTorch implementation. The
end goal is to train it on protein sequences in the style of
[ESM-2](https://www.science.org/doi/10.1126/science.ade2574) (masked language
modeling over an amino-acid alphabet).

## Project status

Milestone 1 (current): Triton kernels, each checked
for numerical parity against PyTorch's official implementation.

- ✅ `gelu`, scaled dot-product attention (`esm_fast.functional`)
- ✅ `LayerNorm`, `FeedForward`, `MultiHeadAttention` (`esm_fast.modules`)
- ✅ `TransformerEncoderLayer` / `TransformerEncoder`
- ✅ Triton fused `softmax` and `layer_norm` kernels (`esm_fast.kernels`)
- ⏳ Rotary embeddings, ESM-2 masked-LM head, tokenizer, training loop

### Triton kernels planned

Stub modules exist in `src/esm_fast/kernels/`; each still needs the kernel plus a
parity test against the PyTorch op it replaces.

- [ ] `matmul.py` — tiled blocked GEMM, parity vs. `torch.matmul`
- [ ] `layer_norm_bwd.py` — backward pass for the fused `layer_norm` kernel
      (dx / dweight / dbias), parity vs. autograd through `F.layer_norm`
- [ ] `fused_bias_gelu.py` — bias add + GELU in one pass, parity vs.
      `F.gelu(x + bias)` (used by `FeedForward`)
- [ ] `fused_res_dropout_layernorm.py` — residual add + dropout + LayerNorm
      fused, parity vs. the eager chain (dropout compared with `p=0` / fixed seed)
- [ ] `rope.py` — rotary position embeddings applied to q/k, parity vs. the
      reference implementation in `esm_fast.functional`
- [ ] `fused_attention.py` — FlashAttention-style fused SDPA, parity vs.
      `F.scaled_dot_product_attention`

## Layout

```
src/esm_fast/
  config.py          ModelConfig (ESM-2 / RoBERTa-style defaults)
  functional.py      reference ops that mirror torch.nn.functional exactly
  modules/           from-scratch nn.Modules (the model)
  kernels/           Triton kernels (CUDA-only, optional)
tests/               parity tests vs. torch's official implementations
```

## Setup

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

```bash
make install       # editable install + dev tools (CPU)
make install-gpu   # also installs Triton (Linux + CUDA only)
```

## Testing

Every from-scratch component is tested by copying weights out of the equivalent
PyTorch module (or calling the equivalent `torch.nn.functional`) and asserting
the outputs match within float32 tolerance.

```bash
make test        # full suite
make test-cpu    # skips GPU/Triton tests (default on macOS)
make test-gpu    # only the Triton kernel tests (needs CUDA)

# single test:
pytest tests/test_attention.py::test_self_attention_matches_torch
```

The Triton tests are marked `gpu` and skip automatically when no CUDA device /
Triton install is present.
