"""Model sizing for the physical-state cells, and the builders that make them.

Mirrors ``ftnode.latent.config``: one config object names the shared dimensions and
per-field sizing, and the builders own construction.  The cold cells build now; the
FIM cells require the vendored OpenFIM backbone (``ftnode.fim``) and raise a clear
error until it is present.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..latent.operator import KappaBudget, spectral_clamp
from .model import PhysicalField, StructuredField, UnstructuredField

__all__ = [
    "PhysicalConfig",
    "build_cold_unstructured",
    "build_cold_structured",
    "build_fim_unstructured",
    "build_fim_structured",
    "build_fim_mlp_head",
]


@dataclass(frozen=True)
class PhysicalConfig:
    """Shared dimensions, the integration step, and per-field sizing.

    Defaults match the Duffing settings the factorial uses: physical dimension 2,
    scalar input, ``h = 0.05``, ``tau = 8``.  The free field is sized to the
    structured field's capacity (``hidden_free=95, depth_free=4``), matching the
    ``ftnode.latent.LatentNODE`` sizing, so the cold row isolates structure.
    """

    m: int = 2
    q: int = 1
    h: float = 0.05
    tau: int = 8
    activation: str = "silu"
    # Free (unstructured) field.
    hidden_free: int = 95
    depth_free: int = 4
    # Structured field and its operator/equilibrium budget.
    hidden: int = 64
    depth: int = 3
    sigma_min: float = 0.1
    kappa_max: float = 25.0
    skew_frac: float = 0.6
    R_g: float = 2.0

    def __post_init__(self):
        if self.m != 2:
            raise ValueError(f"physical state is 2D (q, q_dot); got m={self.m}")
        if self.h <= 0.0:
            raise ValueError(f"h must be positive; got {self.h}")
        if self.tau < 2:
            raise ValueError(f"tau must be >= 2 for the finite-difference estimate; got {self.tau}")

    def budget(self) -> KappaBudget:
        """The conditioning budget for the structured field's operator."""
        return KappaBudget(self.sigma_min, self.kappa_max, self.skew_frac, self.m)


def build_cold_unstructured(cfg: PhysicalConfig) -> PhysicalField:
    """Cell 1: a cold, unstructured free field ``MLP([x,u])``."""
    field = UnstructuredField(cfg.m, cfg.q, cfg.hidden_free, cfg.depth_free, cfg.activation)
    return PhysicalField(field, cfg.h)


def build_cold_structured(cfg: PhysicalConfig, clamp_fn=spectral_clamp) -> PhysicalField:
    """Cell 2: a cold, structured field ``A(x)(x - g(x,u))`` (g built before A)."""
    field = StructuredField(
        cfg.m, cfg.q, cfg.hidden, cfg.depth, cfg.sigma_min, cfg.R_g,
        cfg.activation, cfg.budget(), clamp_fn=clamp_fn,
    )
    return PhysicalField(field, cfg.h)


def _mlp_numel(in_dim: int, out_dim: int, hidden: int, depth: int) -> int:
    """Parameter count of ``ftnode.latent.nets.MLP(in, out, hidden, depth)``."""
    n = in_dim * hidden + hidden
    n += (depth - 1) * (hidden * hidden + hidden)
    n += hidden * out_dim + out_dim
    return n


def _structured_head_numel(cfg: PhysicalConfig) -> int:
    """Total parameters in the cell-4 head: two ``L/S`` nets plus the ``g`` net."""
    d, q, H, D = cfg.m, cfg.q, cfg.hidden, cfg.depth
    feat = 2 * d
    return 2 * _mlp_numel(feat, d * d, H, D) + _mlp_numel(feat + q, d, H, D)


def _matched_free_hidden(cfg: PhysicalConfig, in_dim: int | None = None) -> int:
    """Free-head width whose parameter count is closest to the structured head's.

    The free head is ``MLP(in_dim -> m, hidden, cfg.depth)``, with ``in_dim = 2m`` for
    the cell-3 residual (``[drift, x]``) and ``2m + q`` for the cell-5 head
    (``[drift, x, u]``).  We search hidden widths and pick the one minimizing the
    parameter-count gap, so the free FIM heads match cell 4's head capacity and the
    contrasts isolate structure.
    """
    target = _structured_head_numel(cfg)
    feat = 2 * cfg.m if in_dim is None else in_dim
    d, D = cfg.m, cfg.depth
    best_h, best_gap = cfg.hidden, None
    for h in range(1, 4096):
        gap = abs(_mlp_numel(feat, d, h, D) - target)
        if best_gap is None or gap < best_gap:
            best_gap, best_h = gap, h
        elif gap > best_gap:
            break  # numel is increasing in h, so the gap only grows past the minimum
    return best_h


def build_fim_unstructured(cfg: PhysicalConfig, backbone=None) -> PhysicalField:
    """Cell 3: the pretrained FIM backbone with a residual (identity-init) head.

    ``ftnode.fim`` is imported lazily so the cold cells stay importable without the
    optional ``fim`` dependency.  A fresh backbone is loaded when none is passed, so a
    reseed before the build only affects the head init, not the pretrained weights.
    The free head width is matched to the cell-4 structured head's parameter count.
    """
    from ..fim import FIMBackbone, FreeFIMField

    if backbone is None:
        backbone = FIMBackbone.from_pretrained(state_dim=cfg.m)
    field = FreeFIMField(backbone, d=cfg.m, hidden=_matched_free_hidden(cfg),
                         depth=cfg.depth, activation=cfg.activation)
    return PhysicalField(field, cfg.h)


def build_fim_structured(cfg: PhysicalConfig, backbone=None) -> PhysicalField:
    """Cell 4: the pretrained FIM backbone with a structured ``A(x)(x - g)`` head."""
    from ..fim import FIMBackbone, StructuredFIMField

    if backbone is None:
        backbone = FIMBackbone.from_pretrained(state_dim=cfg.m)
    field = StructuredFIMField(
        backbone, d=cfg.m, q=cfg.q, hidden=cfg.hidden, depth=cfg.depth,
        sigma_min=cfg.sigma_min, R_g=cfg.R_g, kappa_max=cfg.kappa_max,
        skew_frac=cfg.skew_frac, activation=cfg.activation,
    )
    return PhysicalField(field, cfg.h)


def build_fim_mlp_head(cfg: PhysicalConfig, backbone=None) -> PhysicalField:
    """Cell 5: the pretrained FIM backbone with a free MLP head on ``[drift, x, u]``.

    The unstructured control for cell 4: the same head inputs, the head width matched
    to the structured head's parameter count, and no drift pass-through.
    """
    from ..fim import FIMBackbone, MLPHeadFIMField

    if backbone is None:
        backbone = FIMBackbone.from_pretrained(state_dim=cfg.m)
    field = MLPHeadFIMField(
        backbone, d=cfg.m, q=cfg.q,
        hidden=_matched_free_hidden(cfg, in_dim=2 * cfg.m + cfg.q),
        depth=cfg.depth, activation=cfg.activation,
    )
    return PhysicalField(field, cfg.h)
