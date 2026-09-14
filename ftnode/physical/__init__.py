"""Physical-state Duffing identification -- the observed-state counterpart to ``ftnode.latent``.

The model works directly in the physical state ``x = (q, q_dot)``.  There is no
encoder and no decoder: the measured output is ``q = x[0]``, and the rollout's
initial condition is a finite-difference estimate of ``(q, q_dot)`` from the
``q``-window, because ``q_dot`` is never measured.

The field parameterizations reuse the generic, dimension-``m`` building blocks from
``ftnode.latent`` (``ClampOperator``, ``BoundedTanhG``, ``MLP``).  The latent-named
containers ``LatentNODE``/``LatentFTNODE`` are deliberately not imported -- this
module defines its own thin containers instead.
"""
from __future__ import annotations

from .model import (
    PhysicalField,
    StructuredField,
    UnstructuredField,
    estimate_x0,
)

__all__ = [
    "estimate_x0",
    "UnstructuredField",
    "StructuredField",
    "PhysicalField",
]
