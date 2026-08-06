"""The frozen identified plant the control stage designs against.

In ``examples/duffing/_proto_ctrl.py`` the identified model is a module-level
``idm`` that is **loaded from disk at import time**, and every control function
closes over it.  That makes ``import`` do I/O, pins the working directory, and
makes it impossible to hold two models at once.  :class:`FrozenLatentPlant`
replaces that global with an explicit object.
"""
from __future__ import annotations

import pathlib

import torch

from ..latent import (
    KappaBudget,
    LatentModelConfig,
    LatentSysID,
    build_clamp,
    migrate_flat_state_dict,
    spectral_clamp,
)

__all__ = ["FrozenLatentPlant"]


class FrozenLatentPlant:
    """A trained :class:`~ftnode.latent.LatentSysID`, frozen, with the derivatives control needs.

    "Frozen" is the whole point of the control stage: the identified field
    ``F_theta`` represents the plant, and training the splitting must not be able
    to move it.  Construction puts the model in ``eval()`` mode and clears
    ``requires_grad`` on every parameter, so gradients from the control loss can
    only ever reach the splitting operator.
    """

    def __init__(self, model: LatentSysID):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.m = model.dynamics.m

    @property
    def device(self):
        return next(self.model.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        path,
        cfg: LatentModelConfig | None = None,
        budget: KappaBudget | None = None,
        clamp_fn=spectral_clamp,
        map_location="cpu",
    ) -> "FrozenLatentPlant":
        """Rebuild the SVD-clamp architecture and load a state dict into it.

        The committed checkpoints are bare state dicts -- they carry no
        ``c_P``/``c_K``/``sigma_min``/``m``, so the architecture has to be
        re-derived from ``cfg`` and ``budget`` (their defaults are the values the
        notebooks trained with).  Loads with ``strict=True``: a key mismatch here
        means the architecture drifted from what produced the checkpoint, and
        should fail loudly rather than silently leave layers at their init.

        The committed checkpoints predate the operator/equilibrium split, so the
        state dict is re-keyed by
        :func:`~ftnode.latent.migrate_flat_state_dict` first.  That is a no-op for
        anything already in the current layout.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {path}\n"
                "Train one first, or run from the directory holding it "
                "(the committed checkpoints live in examples/duffing/)."
            )
        cfg = cfg or LatentModelConfig()
        budget = budget or KappaBudget()
        model = build_clamp(cfg, budget, clamp_fn=clamp_fn)
        sd = migrate_flat_state_dict(torch.load(path, map_location=map_location), model)
        model.load_state_dict(sd, strict=True)
        return cls(model.to(map_location))

    # ------------------------------------------------------------------ field

    def F(self, z, u):
        """The frozen identified latent field ``F_theta(z, u)``.  ``u`` may be ``(b,)`` or ``(b, 1)``."""
        return self.model.dynamics.F(z, u)

    def g(self, z, u):
        """The identified equilibrium map ``g_theta``."""
        return self.model.dynamics.g(z, u)

    def A(self, z):
        """The identified operator ``A_theta(z)``."""
        return self.model.dynamics.A(z)

    def DuF(self, z, u):
        """``D_u F_theta(z, u)`` as a ``(b, m)`` column, by forward-mode jvp in ``u``.

        Exact and single-pass.  ``A_theta`` does not depend on ``u``, so this
        equals :meth:`DuF_closed`; they are cross-checked in the tests.
        """
        u1 = u.unsqueeze(-1) if u.dim() == z.dim() - 1 else u
        _, jv = torch.func.jvp(lambda uu: self.F(z, uu), (u1,), (torch.ones_like(u1),))
        return jv

    def DuF_closed(self, z, u):
        """The same quantity via the split form ``-A(z) @ dg/du``, as a cross-check."""
        u1 = u.unsqueeze(-1) if u.dim() == z.dim() - 1 else u
        _, dg = torch.func.jvp(lambda uu: self.g(z, uu), (u1,), (torch.ones_like(u1),))
        return -torch.einsum("bij,bj->bi", self.A(z), dg)

    # ---------------------------------------------------------------- codec

    def encode(self, w):
        return self.model.encode(w)

    def decode(self, z):
        return self.model.decode(z)
