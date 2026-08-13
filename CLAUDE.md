# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A transformer built **from scratch** in PyTorch, paired with hand-written
**Triton kernels**. The organizing principle: every from-scratch component must be
provably numerically equivalent to its PyTorch official counterpart. The
long-term goal is ESM-2-style protein masked language modeling, but that work
comes *after* the transformer core and kernels are correct and tested.

## Commands

```bash
make install       # editable install + dev extras (CPU)
make install-gpu   # also installs Triton (Linux + CUDA only)
make test-cpu      # run everything except GPU/Triton tests (use this on macOS)
make test          # full suite
make test-gpu      # only Triton kernel tests (needs CUDA)
make check         # lint + typecheck + test-cpu
make lint          # ruff check
make format        # ruff format

# single test / single case
pytest tests/test_attention.py
pytest tests/test_attention.py::test_self_attention_matches_torch
```

There is no CUDA on the typical dev machine (macOS), so `make test-cpu` is the
working loop; the 6 Triton tests skip automatically.

## The core invariant: parity with PyTorch

This is the whole point of the repo. When adding or changing a from-scratch
module, there must be a test that pins it to PyTorch's official implementation.
The established patterns (see `tests/`):

- **Functional ops** are compared directly against `torch.nn.functional`
  (`gelu` vs `F.gelu`, our SDPA vs `F.scaled_dot_product_attention`).
- **Modules** are compared by *copying weights* out of the equivalent
  `torch.nn` module and asserting equal outputs (and sometimes grads).
- Tolerances live in `tests/conftest.py` (`ATOL=1e-5`, `RTOL=1e-4`) via
  `torch.testing.assert_close`.

Non-obvious weight-layout gotcha, relied on by the tests: `torch.nn.MultiheadAttention`
stores one combined `in_proj_weight` of shape `(3*dim, dim)` = `concat([Wq, Wk, Wv])`.
Our `MultiHeadAttention` uses **separate** `q_proj/k_proj/v_proj/out_proj` Linear
layers, so parity tests `.chunk(3, dim=0)` the reference weight to load ours.

## Architecture notes

- `src/esm_fast/functional.py` holds the reference numerics. Modules are thin
  `nn.Module` wrappers over these; keep the math here so it has one home and one
  parity test.
- `norm_first` (in `ModelConfig` and `TransformerEncoderLayer`) switches between
  **pre-norm** (`True`, the ESM-2 arrangement, adds a `final_norm` in the stack)
  and **post-norm** (`False`, matches `torch.nn.TransformerEncoderLayer`'s block
  ordering exactly). Parity tests against PyTorch use `norm_first=False` because
  that is what's directly comparable; the default for the model is `True`.
- Mask convention follows PyTorch: `key_padding_mask` is `(batch, seq)` with
  `True == ignore`. Internally `_merge_masks` converts everything to a single
  boolean `(batch, heads, q, k)` mask where `True == keep`.

## Triton kernels (`src/esm_fast/kernels/`)

- `triton` is an **optional** dependency (`[gpu]` extra); it is not installed on
  CPU-only machines and importing `esm_fast` must never require it.
- Every kernel module imports `triton` **lazily inside functions** (`_build()`),
  never at module top level, so the package stays importable on CPU. Preserve
  this pattern when adding kernels.
- Guard call sites and tests with `esm_fast.kernels.utils.triton_available()`
  (True only when Triton imports *and* a CUDA device exists). Tests use the
  `@requires_triton` skip marker plus `pytest.mark.gpu` from `tests/conftest.py`.
- Each kernel gets a parity test vs. the PyTorch op it fuses (`softmax` vs
  `torch.softmax`, `layer_norm` vs `F.layer_norm`).

## Conventions

- `ModelConfig` defaults are deliberately tiny for fast CPU tests but use ESM-2
  field semantics (33-token alphabet, pad/mask/bos/eos ids). Don't bloat the
  defaults; pass a config to scale up.
- Python ≥ 3.10, `from __future__ import annotations` at the top of modules.
- ruff is the linter/formatter (config in `pyproject.toml`); run `make check`
  before considering a change done.
