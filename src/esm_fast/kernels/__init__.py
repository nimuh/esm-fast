"""Hand-written Triton kernels.

Everything here requires a CUDA device and a working Triton install (the ``gpu``
optional dependency). Use :func:`triton_available` to guard call sites and tests
so the package stays importable on CPU-only machines.
"""

from esm_fast.kernels.utils import triton_available

__all__ = ["triton_available"]
