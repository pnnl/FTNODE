"""Operators ``A(z)`` -- one half of the split ``F = A(z)(z - g(z,u))``.

Writing ``A = -(sigma_min I + P) + K`` with ``P`` symmetric PSD and ``K`` skew
gives ``sym(A) <= -sigma_min I`` unconditionally.  Bounding ``||P||_2 <= c_P``
and ``||K||_2 <= c_K`` then caps the conditioning,

    kappa(A) = sigma_max(A) / lambda_min(-sym A) <= (sigma_min + c_P + c_K) / sigma_min,

which is what :class:`KappaBudget` derives.  Every bound here holds by
construction, not by penalty.

Each operator is a **standalone module** selected by name from :data:`A_KINDS`,
exactly as the equilibrium maps are selected from
:data:`~ftnode.latent.equilibrium.G_KINDS`.  The two axes are peers: any operator
composes with any equilibrium map, and each is free to carry as much internal
structure as its math needs.  They are consumed by
:class:`~ftnode.latent.model.LatentFTNODE`, which never looks inside either.

Named to mirror :mod:`ftnode.control.operator`, which carries the same concept for
the control stage -- ``SplitOperator`` there is structurally :class:`ClampOperator`
with :func:`spectral_clamp_safe` wired in unconditionally.

.. warning::
   ``L_net``, ``S_net``, ``W_net``, ``b_net`` and the index buffers are
   checkpoint keys, now reached under ``dynamics.operator.``.  See
   :mod:`ftnode.latent`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .nets import MLP

__all__ = [
    "KappaBudget",
    "spectral_clamp",
    "spectral_clamp_safe",
    "OperatorBase",
    "UnboundedOperator",
    "ClampOperator",
    "YoulaOperator",
    "A_KINDS",
    "resolve_operator",
]


# --------------------------------------------------------------------- kappa budget


@dataclass(frozen=True)
class KappaBudget:
    """Splits the conditioning budget between the symmetric and skew parts of ``A``.

    Given the target cap ``kappa_max`` on ``kappa(A)`` and the contraction floor
    ``sigma_min``, the total spectral budget is ``sigma_min * (kappa_max - 1)``,
    apportioned by ``skew_frac``.  The notebook defaults
    ``KappaBudget(0.1, 25.0, 0.6, 4)`` give ``budget=2.4``, ``c_P=0.96``,
    ``c_K=1.44``, hence ``sigma_max = 0.1 + 0.96 + 1.44 = 2.5`` and ``kappa <= 25``.

    Lives beside the operators because it constrains ``A`` specifically.  An
    equilibrium map that needs its own budget -- a bounded-gradient ``g``, say --
    should define one in :mod:`~ftnode.latent.equilibrium` rather than extend
    this.
    """

    sigma_min: float = 0.1
    kappa_max: float = 25.0
    skew_frac: float = 0.6
    m: int = 4

    @property
    def budget(self) -> float:
        """Total spectral budget ``c_P + c_K``."""
        return self.sigma_min * (self.kappa_max - 1.0)

    @property
    def c_K(self) -> float:
        """Skew spectral budget: ``||K||_2 <= c_K``."""
        return self.skew_frac * self.budget

    @property
    def c_P(self) -> float:
        """Symmetric spread budget: ``||P||_2 <= c_P``."""
        return (1.0 - self.skew_frac) * self.budget

    @property
    def sigma_max(self) -> float:
        """Resulting cap on ``||A||_2``."""
        return self.sigma_min + self.c_P + self.c_K

    @property
    def l_bound(self) -> float:
        """Entry bound for the Youla variant's tanh-bounded Cholesky factor.

        Chosen so the closed-form worst case ``l_bound^2 * m(m+1)/2`` equals
        ``c_P``, matching the clamped variant's symmetric budget without an SVD.
        """
        return (2.0 * self.c_P / (self.m * (self.m + 1))) ** 0.5

    def beta_min(self, frac: float = 0.0) -> float:
        """Floor on the Youla block magnitudes, as a fraction of ``c_K``."""
        return frac * self.c_K


# ------------------------------------------------------------------ spectral clamps


def spectral_clamp(Bmat, cap):
    """Scale each ``(m, m)`` matrix so its spectral norm is at most ``cap`` (batched SVD).

    NOTE: ``matrix_norm(ord=2)`` is degenerate on skew inputs (singular values come
    in equal pairs), so its backward is ill-conditioned -- exactly what the Youla
    variant avoids structurally and what :func:`spectral_clamp_safe` avoids
    numerically.  Kept as the reference implementation the notebooks were run
    with; do not "fix" it here, or the frozen results stop reproducing.
    """
    s = torch.linalg.matrix_norm(Bmat, ord=2)
    return Bmat * torch.clamp(cap / (s + 1e-12), max=1.0).view(-1, 1, 1)


def spectral_clamp_safe(Bmat, cap, eps=1e-12):
    """Same projection as :func:`spectral_clamp`, via ``eigvalsh`` instead of the SVD.

    The spectral norm is obtained as ``sqrt(lambda_max(B^T B))``.  ``B^T B`` is
    symmetric PSD by construction, and ``eigvalsh`` is stable on it.

    WHY this exists: under psi-training the *forward* SVD in
    :func:`spectral_clamp` fails outright --
    ``torch._C._LinAlgError: linalg.svd: failed to converge (too many repeated
    singular values)`` -- because the skew argument has paired singular values.
    The clamp is mathematically identical; only the route to ``||B||_2`` changes.

    Two changes on promotion out of ``examples/duffing/_proto_ctrl.py``:

    * The reshape is rank-agnostic, matching the ``transpose(-1, -2)`` above it,
      instead of hardcoding a 3-D batch via ``.view(-1, 1, 1)``.
    * The degenerate case is handled by ``clamp_min(eps)`` on the eigenvalue
      rather than ``sqrt(lam + eps)``.  Both keep the gradient finite at
      ``lam = 0`` -- ``clamp_min`` by zeroing it there, which is the right answer
      since a vanishing matrix is never the active constraint -- but the clamp
      stays *exact* for any ``cap`` above ``sqrt(eps)``, where the additive form
      turned conservative below ``cap ~ 1e-4``.
    """
    G = Bmat.transpose(-1, -2) @ Bmat
    lam = torch.linalg.eigvalsh(G)[..., -1].clamp_min(eps)
    s = torch.sqrt(lam)
    scale = torch.clamp(cap / s, max=1.0)
    return Bmat * scale.reshape(*Bmat.shape[:-2], 1, 1)


# ------------------------------------------------------------------------ operators


class OperatorBase(nn.Module):
    """Common state for an ``A(z)`` operator: dimensions, floor, sub-network sizing.

    The uniform constructor contract every entry in :data:`A_KINDS` must satisfy is

        ``Op(m, sigma_min, hidden, depth, activation, budget, **kwargs)``

    with ``forward(z) -> (b, m, m)``.  ``budget`` is a :class:`KappaBudget` or
    ``None``; operators that impose no cap ignore it, and operators that need one
    raise if it is missing.  Subclassing this is a convenience for the shared
    fields, not part of the contract -- a registry entry only has to honour the
    signature.
    """

    def __init__(self, m=4, sigma_min=0.1, hidden=64, depth=3, activation="silu", budget=None):
        super().__init__()
        self.m = m
        self.sigma_min = sigma_min
        self.hidden, self.depth = hidden, depth
        self.activation = activation
        self.register_buffer("_eye", torch.eye(m))

    @staticmethod
    def _require_budget(budget, who):
        if budget is None:
            raise ValueError(f"{who} needs a KappaBudget to derive its caps; got None")
        return budget

    def forward(self, z):
        raise NotImplementedError


class UnboundedOperator(OperatorBase):
    """Baseline: ``A = -(L L^T + sigma_min I) + (S - S^T)``, skew UNCONSTRAINED.

    ``L`` is lower-triangular with a softplus diagonal, so ``sym(A) <= -sigma_min I``
    still holds -- but nothing caps ``||K||_2``, so ``kappa(A)`` is free to blow up.
    This is the variant the kappa-bounded operators are measured against, and the
    only one that ignores ``budget``.
    """

    def __init__(self, m=4, sigma_min=0.1, hidden=64, depth=3, activation="silu", budget=None):
        super().__init__(m, sigma_min, hidden, depth, activation, budget)
        tril = torch.tril_indices(m, m)
        self.register_buffer("_tr", tril[0])
        self.register_buffer("_tc", tril[1])
        self.register_buffer("_diag_mask", (tril[0] == tril[1]).float())
        self.L_net = MLP(m, m * (m + 1) // 2, hidden, depth,
                         activation=activation) # Cholesky fact sym
        self.S_net = MLP(m, m * m, hidden, depth,
                         last_zero=True, activation=activation) # skew net

    def _L(self, z):
        B = z.shape[0]
        raw = self.L_net(z)
        flat = Fnn.softplus(raw) * self._diag_mask + raw * (1.0 - self._diag_mask)
        Lm = z.new_zeros(B, self.m, self.m)
        Lm[:, self._tr, self._tc] = flat
        return Lm

    def forward(self, z):
        L = self._L(z)
        M = self.S_net(z).view(-1, self.m, self.m)
        return -(L @ L.transpose(1, 2) + self.sigma_min * self._eye) + (M - M.transpose(1, 2))


class ClampOperator(OperatorBase):
    """``A = -(sigma_min I + P) + K`` with ``||P||_2 <= c_P``, ``||K||_2 <= c_K``.

    Both bounds are imposed by projecting through ``clamp_fn``, so
    ``kappa(A) <= (sigma_min + c_P + c_K)/sigma_min = kappa_max``.

    ``clamp_fn`` defaults to the SVD-based :func:`spectral_clamp` the frozen
    notebooks were run with.  Pass :func:`spectral_clamp_safe` to get the
    eigvalsh route, which is what the control stage needs (see
    :class:`ftnode.control.SplitOperator`).

    .. note::
       ``clamp_fn`` is a *callable*, so it is deliberately not reachable from a
       config file -- a YAML must not be able to change which clamp the frozen
       results were produced with.  Route it through the builder instead, or
       register a second :data:`A_KINDS` entry.
    """

    def __init__(self, m=4, sigma_min=0.1, hidden=64, depth=3, activation="silu", budget=None,
                 clamp_fn=spectral_clamp):
        super().__init__(m, sigma_min, hidden, depth, activation, budget)
        budget = self._require_budget(budget, type(self).__name__)
        self.c_P, self.c_K = float(budget.c_P), float(budget.c_K)
        self.clamp_fn = clamp_fn
        self.L_net = MLP(m, m * m, hidden, depth, activation=activation)
        self.S_net = MLP(m, m * m, hidden, depth, last_zero=True, activation=activation)

    def forward(self, z):
        Lc = self.clamp_fn(self.L_net(z).view(-1, self.m, self.m), self.c_P**0.5)
        P = Lc @ Lc.transpose(1, 2)
        Mr = self.S_net(z).view(-1, self.m, self.m)
        K = self.clamp_fn(Mr - Mr.transpose(1, 2), self.c_K)
        return -(self.sigma_min * self._eye + P) + K


class YoulaOperator(OperatorBase):
    """SVD-free kappa bound: ``A = -(L L^T + sigma_min I) + K``.

    The symmetric factor has ``|L_ij| <= l_bound`` via ``tanh``, giving the
    closed-form worst case ``||L L^T||_2 <= l_bound^2 * m(m+1)/2 = c_P``.

    The skew part is Youla-parameterized: ``K = Q Sigma Q^T`` with
    ``Q = matrix_exp(W - W^T)`` in ``SO(m)`` and ``Sigma`` block-diagonal with
    ``2x2`` blocks ``beta_j * [[0, 1], [-1, 0]]``, where
    ``beta_j = beta_min + (c_K - beta_min) * sigmoid(.)`` lies in ``[beta_min, c_K)``.
    The singular values of such a ``K`` are the doubled set ``{beta_j, beta_j}``,
    so ``||K||_2 = max_j beta_j <= c_K`` -- **without ever computing an SVD**.
    That is the point: it reaches the same cap as :class:`ClampOperator` while
    avoiding the ill-conditioned backward through ``matrix_norm(ord=2)`` on a skew
    argument.

    ``beta_min_frac`` is config-reachable (a plain float); requires an even ``m``.
    """

    def __init__(self, m=4, sigma_min=0.1, hidden=64, depth=3, activation="silu", budget=None,
                 beta_min_frac=0.0):
        super().__init__(m, sigma_min, hidden, depth, activation, budget)
        budget = self._require_budget(budget, type(self).__name__)
        if m % 2 != 0:
            raise ValueError(f"YoulaOperator needs an even latent dim, got m={m}")
        self.l_bound = float(budget.l_bound)
        self.c_K = float(budget.c_K)
        self.beta_min = float(budget.beta_min(beta_min_frac))
        self.n_blk = m // 2
        tril = torch.tril_indices(m, m)
        self.register_buffer("_tr", tril[0])
        self.register_buffer("_tc", tril[1])
        r = 2 * torch.arange(self.n_blk)
        c = r + 1
        self.register_buffer("_br", r)
        self.register_buffer("_bc", c)
        act = activation
        self.L_net = MLP(m, m * (m + 1) // 2, hidden, depth, activation=act)  # symmetric factor (bounded)
        self.W_net = MLP(m, m * m, hidden, depth, last_zero=True, activation=act)  # skew gen (Q starts at I)
        self.b_net = MLP(m, self.n_blk, hidden, depth, activation=act)  # block magnitudes

    def _A_sym(self, z):
        B = z.shape[0]
        ent = self.l_bound * torch.tanh(self.L_net(z))  # bounded -> closed-form sigma_max
        Lm = z.new_zeros(B, self.m, self.m)
        Lm[:, self._tr, self._tc] = ent
        return -(Lm @ Lm.transpose(1, 2) + self.sigma_min * self._eye)

    def _K(self, z):
        B = z.shape[0]
        Wr = self.W_net(z).view(B, self.m, self.m)
        Q = torch.matrix_exp(Wr - Wr.transpose(1, 2))  # in SO(m)
        beta = self.beta_min + (self.c_K - self.beta_min) * torch.sigmoid(self.b_net(z))
        Sig = z.new_zeros(B, self.m, self.m)
        Sig[:, self._br, self._bc] = beta
        Sig[:, self._bc, self._br] = -beta
        return Q @ Sig @ Q.transpose(1, 2)

    def forward(self, z):
        return self._A_sym(z) + self._K(z)


#: Selectable operators, keyed by the string a config stores.  Peer of
#: :data:`~ftnode.latent.equilibrium.G_KINDS`; see :class:`OperatorBase` for the
#: constructor contract.
A_KINDS: dict[str, type] = {
    "svd_clamp": ClampOperator,
    "youla": YoulaOperator,
    "unbounded": UnboundedOperator,
}


def resolve_operator(spec):
    """Turn an operator spec into the class to construct.

    Accepts a name from :data:`A_KINDS`, an ``nn.Module`` subclass, or any
    callable returning a module.  Mirrors
    :func:`~ftnode.latent.equilibrium.resolve_g`.
    """
    if isinstance(spec, str):
        try:
            return A_KINDS[spec]
        except KeyError:
            raise ValueError(
                f"unknown operator kind {spec!r}; choose from {sorted(A_KINDS)} "
                "or pass an nn.Module subclass"
            ) from None
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec
    if callable(spec):
        return spec
    raise TypeError(f"operator kind must be a name, nn.Module subclass, or factory; got {spec!r}")
