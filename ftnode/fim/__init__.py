"""FIM-ODE backbone for the FIM x ftnode-structure factorial (cells 3 and 4).

Wraps the installed ``fim`` package's pretrained ``FIMODE`` model as a differentiable
physical-state field: a context is bound once from the measurement window, then the
drift is queried at arbitrary states during the RK4 rollout.  This subpackage requires
the optional ``fim`` dependency group (``uv sync --group fim``).
"""
from __future__ import annotations

from .backbone import FIMBackbone, load_base_model
from .fields import FreeFIMField, StructuredFIMField, freeze_uncertainty_head

__all__ = [
    "FIMBackbone",
    "load_base_model",
    "FreeFIMField",
    "StructuredFIMField",
    "freeze_uncertainty_head",
]
