"""The FIM-backed fields for the warm row of the factorial (cells 3, 4, and 5).

All wrap the same :class:`FIMBackbone` and differ only in the head:

- :class:`FreeFIMField` (cell 3) -- a residual MLP head on the FIM drift, zero-init so
  the field starts exactly at the pretrained drift.
- :class:`StructuredFIMField` (cell 4) -- a structured head that reads the FIM drift as
  a feature and forms ``A(x)(x - g(x,u))`` with ``A`` negative definite by construction
  (the same clamp math as ``ClampOperator``, composed from the generic blocks).
- :class:`MLPHeadFIMField` (cell 5) -- a free MLP head that reads the same FIM feature
  and ``u`` as the structured head, with no pass-through of the drift.  It is the
  unstructured control for cell 4: same inputs, matched capacity, no structure.

All expose the physical-field interface the rollout needs -- ``prepare(window, h)`` and
``F(x, u)`` -- plus ``param_groups`` so the FIM backbone gets a low learning rate. The
uncertainty head is off the drift path and is frozen.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..latent.nets import MLP
from ..latent.operator import KappaBudget, spectral_clamp
from .backbone import FIMBackbone

__all__ = ["FreeFIMField", "MLPHeadFIMField", "StructuredFIMField", "freeze_uncertainty_head"]


def freeze_uncertainty_head(backbone: FIMBackbone) -> None:
    """Freeze the FIM uncertainty head; it is not on the drift path under the q-loss."""
    if hasattr(backbone.fim, "u_model"):
        for p in backbone.fim.u_model.parameters():
            p.requires_grad_(False)


def _param_groups(module, backbone: FIMBackbone, lr, lr_backbone):
    """Head params at ``lr``, trainable backbone params at ``lr_backbone`` (or ``lr``)."""
    backbone_ids = {id(p) for p in backbone.parameters()}
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in module.parameters() if p.requires_grad and id(p) not in backbone_ids]
    groups = [{"params": head_params, "lr": lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone if lr_backbone else lr})
    return groups


class FreeFIMField(nn.Module):
    """Cell 3: ``F(x,u) = drift(x) + residual([drift(x), x])``, residual zero-init.

    ``u`` enters through the context (bound in :meth:`prepare`), not the field, so it is
    unused here.  The field starts at the pretrained FIM drift and learns a residual.
    The residual reads the same ``[drift, x]`` feature the structured head reads, and its
    width is chosen by the builder so its parameter count matches the structured head's,
    which keeps the warm-row interaction from confounding structure with capacity.
    """

    def __init__(self, backbone: FIMBackbone, d=2, hidden=64, depth=2, activation="silu"):
        super().__init__()
        self.backbone = backbone
        self.d = d
        self.res = MLP(2 * d, d, hidden, depth, last_zero=True, activation=activation)
        freeze_uncertainty_head(backbone)

    def prepare(self, window, h):
        self.backbone.prepare(window, h)

    def F(self, x, u):
        drift = self.backbone.drift(x)
        feat = torch.cat([drift, x], dim=-1)
        return drift + self.res(feat)

    def param_groups(self, lr, lr_backbone=None):
        return _param_groups(self, self.backbone, lr, lr_backbone)


class MLPHeadFIMField(nn.Module):
    """Cell 5: ``F(x,u) = MLP([drift(x), x, u])``, default init, no drift pass-through.

    The head reads exactly what the cell-4 structured head reads -- the differentiable
    FIM drift, the state, and ``u`` -- and its width is chosen by the builder so its
    parameter count matches the structured head's.  So cell 4 against cell 5 differs
    only in the structure.  Unlike cell 3, the field does not start at the FIM drift.
    """

    def __init__(self, backbone: FIMBackbone, d=2, q=1, hidden=64, depth=2, activation="silu"):
        super().__init__()
        self.backbone = backbone
        self.d, self.q = d, q
        self.net = MLP(2 * d + q, d, hidden, depth, activation=activation)
        freeze_uncertainty_head(backbone)

    def prepare(self, window, h):
        self.backbone.prepare(window, h)

    def F(self, x, u):
        if u.dim() == x.dim() - 1:
            u = u.unsqueeze(-1)
        return self.net(torch.cat([self.backbone.drift(x), x, u], dim=-1))

    def param_groups(self, lr, lr_backbone=None):
        return _param_groups(self, self.backbone, lr, lr_backbone)


class StructuredFIMField(nn.Module):
    """Cell 4: ``F(x,u) = A(x)(x - g(x,u))`` with heads that read the FIM drift.

    ``A = -(sigma_min I + P) + K`` with ``P = L L^T`` PSD and ``K`` skew, both spectrally
    clamped to the kappa budget, so ``sym(A) <= -sigma_min I`` by construction at every
    ``x`` -- independent of the FIM feature.  The heads consume the differentiable FIM
    drift concatenated with ``x`` (never the detached ``FIMODEOutput.D``).
    """

    def __init__(self, backbone: FIMBackbone, d=2, q=1, hidden=64, depth=2,
                 sigma_min=0.1, R_g=2.0, kappa_max=25.0, skew_frac=0.6, activation="silu"):
        super().__init__()
        self.backbone = backbone
        self.d, self.q = d, q
        self.sigma_min, self.R_g = sigma_min, R_g
        budget = KappaBudget(sigma_min, kappa_max, skew_frac, d)
        self.c_P, self.c_K = float(budget.c_P), float(budget.c_K)
        feat = d + d  # [drift, x]
        self.L_net = MLP(feat, d * d, hidden, depth, activation=activation)
        self.S_net = MLP(feat, d * d, hidden, depth, last_zero=True, activation=activation)
        self.g_net = MLP(feat + q, d, hidden, depth, last_zero=True, activation=activation)
        self.register_buffer("_eye", torch.eye(d))
        freeze_uncertainty_head(backbone)

    def prepare(self, window, h):
        self.backbone.prepare(window, h)

    def _A_from_feat(self, feat):
        Lc = spectral_clamp(self.L_net(feat).view(-1, self.d, self.d), self.c_P**0.5)
        P = Lc @ Lc.transpose(1, 2)
        Mr = self.S_net(feat).view(-1, self.d, self.d)
        K = spectral_clamp(Mr - Mr.transpose(1, 2), self.c_K)
        return -(self.sigma_min * self._eye + P) + K

    def _g_from_feat(self, feat, u):
        if u.dim() == feat.dim() - 1:
            u = u.unsqueeze(-1)
        return self.R_g * torch.tanh(self.g_net(torch.cat([feat, u], dim=-1)))

    def F(self, x, u):
        drift = self.backbone.drift(x)
        feat = torch.cat([drift, x], dim=-1)
        A = self._A_from_feat(feat)
        g = self._g_from_feat(feat, u)
        return torch.einsum("bij,bj->bi", A, x - g)

    def A(self, x):
        """The operator ``A(x)`` (needs :meth:`prepare`), for diagnostics and tests."""
        feat = torch.cat([self.backbone.drift(x), x], dim=-1)
        return self._A_from_feat(feat)

    def param_groups(self, lr, lr_backbone=None):
        return _param_groups(self, self.backbone, lr, lr_backbone)
