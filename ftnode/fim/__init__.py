"""FIM-ODE backbone for the FIM x ftnode-structure factorial (cells 3, 4, and 5).

Wraps the installed ``fim`` package's pretrained ``FIMODE`` model as a differentiable
physical-state field: a context is bound once from the measurement window, then the
drift is queried at arbitrary states during the RK4 rollout.  This subpackage requires
the optional ``fim`` dependency group (``uv sync --group fim``).
"""
from __future__ import annotations

from .backbone import FIMBackbone, load_base_model
from .fields import FreeFIMField, MLPHeadFIMField, StructuredFIMField, freeze_uncertainty_head
from .quick import (
    DistillConfig,
    RetuneConfig,
    distill_structured,
    drift_mse_on_grid,
    drift_mse_vs_true,
    retune_short_rollout,
    short_horizon_qmse,
    zero_shot_snapshot,
)

__all__ = [
    "FIMBackbone",
    "load_base_model",
    "FreeFIMField",
    "MLPHeadFIMField",
    "StructuredFIMField",
    "freeze_uncertainty_head",
    "RetuneConfig",
    "DistillConfig",
    "retune_short_rollout",
    "distill_structured",
    "drift_mse_vs_true",
    "drift_mse_on_grid",
    "short_horizon_qmse",
    "zero_shot_snapshot",
]
