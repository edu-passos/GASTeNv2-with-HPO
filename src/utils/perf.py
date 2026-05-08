"""Performance knobs that we want to opt into per-experiment.

- ``cudnn.benchmark = True`` lets cuDNN pick the fastest convolution algorithm
  per input shape. Speeds up training when input shapes are stable; introduces
  non-determinism, so it's mutually exclusive with reproducibility mode.

- ``cudnn.deterministic = True`` forces deterministic algorithms; *slower* than
  benchmark mode.

- ``tf32_matmul`` enables TF32 on Ampere+ matmul/conv; default on in PyTorch
  for cuDNN but opt-in for matmul. Cheap quality cost, real speed win.

- ``torch.compile`` graphs and fuses the model. Significant speedup once the
  graph is traced and cached; first iteration pays a compile cost.

The compile wrapper changes attribute access slightly: parameters live under
``model._orig_mod`` after compile. Use :func:`unwrap_compiled` to reach the raw
module — important for EMA bookkeeping where parameter names matter.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def apply_perf_settings(cfg_perf: dict | None) -> dict:
    """Apply cuDNN / TF32 settings from a small config dict. Idempotent.

    Recognised keys (all optional)::

        cudnn_benchmark:     bool   # default: leave unchanged
        cudnn_deterministic: bool   # default: leave unchanged
        tf32_matmul:         bool   # default: False

    Returns the resolved settings (for logging).
    """
    cfg_perf = cfg_perf or {}
    out: dict[str, Any] = {}

    if "cudnn_benchmark" in cfg_perf:
        v = bool(cfg_perf["cudnn_benchmark"])
        torch.backends.cudnn.benchmark = v
        out["cudnn_benchmark"] = v

    if "cudnn_deterministic" in cfg_perf:
        v = bool(cfg_perf["cudnn_deterministic"])
        torch.backends.cudnn.deterministic = v
        out["cudnn_deterministic"] = v

    if cfg_perf.get("tf32_matmul", False):
        torch.set_float32_matmul_precision("high")
        out["tf32_matmul"] = True

    if out:
        print(f"[perf] applied {out}")
    return out


def maybe_compile(model: nn.Module, mode) -> nn.Module:
    """Wrap with :func:`torch.compile` if ``mode`` is truthy.

    ``mode`` accepts True (== 'default'), False/None (no-op), or one of
    'default'|'reduce-overhead'|'max-autotune'. Falls back to eager if compile
    raises at construction OR at first forward (e.g. host has no CUDA toolkit
    and Inductor cannot codegen Triton kernels).
    """
    if not mode:
        return model
    if isinstance(mode, bool):
        mode = "default"
    if not hasattr(torch, "compile"):
        return model

    # Suppress runtime InductorErrors so production gracefully degrades to
    # eager instead of killing the whole training run if codegen fails.
    try:
        import torch._dynamo as _dynamo  # noqa: WPS433
        _dynamo.config.suppress_errors = True
    except Exception:  # pragma: no cover - defensive
        pass

    try:
        compiled = torch.compile(model, mode=mode)
        print(f"[perf] torch.compile(mode={mode!r}) applied")
        return compiled
    except Exception as e:  # pragma: no cover - defensive
        print(f"[perf] torch.compile(mode={mode!r}) failed: {e!r}; using eager")
        return model


def unwrap_compiled(m: nn.Module) -> nn.Module:
    """Return the underlying module behind a ``torch.compile`` wrapper.

    EMA bookkeeping iterates ``model.named_parameters()`` and uses those names
    as identifiers across save/load. After ``torch.compile`` the wrapper adds
    an ``_orig_mod.`` prefix to every parameter name, so we want to point EMA
    at the raw module instead.
    """
    return getattr(m, "_orig_mod", m)
