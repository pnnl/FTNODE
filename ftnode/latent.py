"""Latent forward-tracking NODEs with spectrally-bounded ``A(z)``.

Promoted from the two kappa-bounded duffing notebooks:

* ``examples/duffing/duffing_kappa_svdclamp_vs_ln_2variant_10seed.ipynb``
* ``examples/duffing/duffing_kappa_bounded_youla_skew_3variant_10seed.ipynb``

The model family here is *not* the x-space :class:`ftnode.node.FTNODE`
(``f(x) * (x - g(x, u))``, elementwise).  It is the latent, matrix-valued form

    F(z, u) = A(z) @ (z - g(z, u)),

where ``z`` is an information state produced by an encoder from a window of past
measurements, ``g`` is the bounded equilibrium map, and ``A(z)`` carries the
stability structure.  The subclasses differ only in how ``A(z)`` is built.

Writing ``A = -(sigma_min I + P) + K`` with ``P`` symmetric PSD and ``K`` skew
gives ``sym(A) <= -sigma_min I`` unconditionally.  Bounding ``||P||_2 <= c_P``
and ``||K||_2 <= c_K`` then caps the conditioning,

    kappa(A) = sigma_max(A) / lambda_min(-sym A) <= (sigma_min + c_P + c_K) / sigma_min,

which is what :class:`KappaBudget` derives.  Every bound in this module holds by
construction, not by penalty.

.. warning::
   The ``nn.Module`` attribute names in this file (``encoder``/``dynamics``/
   ``decoder``, ``net``, ``g_net``, ``L_net``, ``S_net``, ``c``, and the ``_eye``
   buffer) are load-bearing: the checkpoints committed under ``examples/duffing/``
   are flat state dicts keyed by them.  Renaming any of them silently breaks
   ``load_state_dict``.  ``tests/test_checkpoints.py`` guards this.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

__all__ = [
    "ACTIVATIONS",
    "resolve_activation",
    "is_lipschitz_1",
    "MLP",
    "Encoder",
    "LinearDecoder",
    "LatentSysID",
    "KappaBudget",
    "spectral_clamp",
    "spectral_clamp_safe",
    "LatentFTNODEBase",
    "LatentFTNODEUnbounded",
    "LatentFTNODEClamp",
    "LatentFTNODEYoula",
    "LatentNODE",
    "LatentModelConfig",
    "build_clamp",
    "build_youla",
    "build_unbounded",
    "build_latent_node",
]


# -------------------------------------------------------------------- activations

#: Selectable hidden activations, keyed by the string a config stores.
#:
#: The ``lipschitz_1`` flag records whether the activation is 1-Lipschitz, which
#: matters whenever a downstream argument composes per-layer bounds into a bound
#: on the whole network -- contraction and conditioning arguments do exactly
#: that, and they are only valid if every nonlinearity in the path has unit
#: gain.  Note that the default ``silu`` is **not** 1-Lipschitz (its derivative
#: peaks near 1.0998 around x = 2.4), and neither is ``gelu`` (~1.1289).  They
#: remain available because the committed results were produced with ``silu``.
#: Use :func:`ftnode.diagnostics.empirical_lipschitz` to check any of these
#: numerically rather than trusting the table.
ACTIVATIONS: dict[str, tuple[type, bool]] = {
    "silu": (nn.SiLU, False),
    "gelu": (nn.GELU, False),
    "tanh": (nn.Tanh, True),
    "relu": (nn.ReLU, True),
    "leaky_relu": (nn.LeakyReLU, True),
    "elu": (nn.ELU, True),
    "softplus": (nn.Softplus, True),
}


def resolve_activation(spec):
    """Turn an activation spec into a zero-argument factory producing fresh modules.

    Accepts a name from :data:`ACTIVATIONS`, an ``nn.Module`` subclass, or any
    callable returning a module.  A *factory* rather than a shared instance so
    every layer gets its own module -- harmless for the parameterless
    activations here, but it keeps a stateful or parameterized activation (say
    ``nn.PReLU``) from being silently tied across layers.
    """
    if isinstance(spec, str):
        try:
            return ACTIVATIONS[spec][0]
        except KeyError:
            raise ValueError(
                f"unknown activation {spec!r}; choose from {sorted(ACTIVATIONS)} "
                "or pass an nn.Module subclass"
            ) from None
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec
    if callable(spec):
        return spec
    raise TypeError(f"activation must be a name, nn.Module subclass, or factory; got {spec!r}")


def is_lipschitz_1(spec) -> bool:
    """Whether a named activation is 1-Lipschitz.  Unknown/custom specs return ``False``."""
    return ACTIVATIONS.get(spec, (None, False))[1] if isinstance(spec, str) else False


# --------------------------------------------------------------------------- nets


class MLP(nn.Module):
    """``Linear -> activation`` repeated ``depth`` times, then a linear readout.

    ``last_zero`` zeroes the final weight and bias, so the module starts at the
    zero map -- used for ``g`` (equilibrium map starts at the origin) and for the
    skew generators (``Q`` starts at the identity).

    ``activation`` takes a name from :data:`ACTIVATIONS`, an ``nn.Module``
    subclass, or a factory.  It defaults to ``"silu"`` because that is what
    produced the committed checkpoints and the frozen notebook results -- **not**
    because it is the best choice.  SiLU is not 1-Lipschitz, so any argument that
    composes per-layer Lipschitz bounds needs one of the ``lipschitz_1``
    activations instead.

    Swapping the activation does not disturb checkpoints: state-dict keys are
    ``net.0.weight``, ``net.2.weight``, ... -- indices into the ``Sequential`` --
    and every activation here is parameterless, so it occupies its slot without
    contributing or shifting a single key.

    Distinct from :class:`ftnode.node.terms.MLP`, which takes a ``dims`` list and
    serves the x-space model.  The two never meet: ``ftnode.node`` deliberately
    does not export its ``MLP``, and this one is reached as ``ftnode.latent.MLP``.
    """

    def __init__(self, in_dim, out_dim, hidden=64, depth=3, last_zero=False, activation="silu"):
        super().__init__()
        act = resolve_activation(activation)
        self.activation = activation
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
        if last_zero:
            with torch.no_grad():
                self.net[-1].weight.zero_()
                self.net[-1].bias.zero_()

    def forward(self, z):
        return self.net(z)


class Encoder(nn.Module):
    """Information state from a window of past measurements: ``z = z_scale * tanh(MLP(w))``.

    The ``tanh`` confines the latent to the box ``[-z_scale, z_scale]^m``, which
    is what makes the forward-invariance diagnostics well posed.

    .. note::
       **The encoder is not 1-Lipschitz, and choosing a 1-Lipschitz hidden
       activation does not make it one.**  The output map is
       ``z_scale * tanh(.)``, whose gain is ``z_scale`` (2.0 by default), since
       ``tanh`` has unit slope at the origin and the scaling sits outside it.
       So the encoder's Lipschitz constant is ``z_scale * L(MLP)``, bounded below
       by ``z_scale`` no matter what ``activation`` is set to.

       This is separate from the per-layer activation gain that
       :data:`ACTIVATIONS` tracks, and it is *not* currently addressed anywhere in
       the package.  Any end-to-end bound that needs a non-expansive encoder has
       to account for this factor explicitly, or the encoder has to be
       reparameterized (e.g. fold ``z_scale`` into the latent metric, or
       normalize the readout).  Flagged rather than fixed: changing it would
       alter the latent scale and invalidate every committed checkpoint.
    """

    def __init__(self, tau, m, hidden=64, depth=2, z_scale=2.0, activation="silu"):
        super().__init__()
        self.net = MLP(tau, m, hidden, depth, activation=activation)
        self.z_scale = z_scale

    def forward(self, w):
        return self.z_scale * torch.tanh(self.net(w))


class LinearDecoder(nn.Module):
    """Affine readout of the measured output ``y = q`` from the latent state."""

    def __init__(self, m):
        super().__init__()
        self.c = nn.Linear(m, 1, bias=True)

    def forward(self, z):
        return self.c(z).squeeze(-1)


class LatentSysID(nn.Module):
    """Container tying an encoder, a latent vector field, and a decoder together."""

    def __init__(self, encoder, dynamics, decoder):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder

    def encode(self, w):
        return self.encoder(w)

    def decode(self, z):
        return self.decoder(z)

    def F(self, z, u):
        return self.dynamics.F(z, u)


# --------------------------------------------------------------------- kappa budget


@dataclass(frozen=True)
class KappaBudget:
    """Splits the conditioning budget between the symmetric and skew parts of ``A``.

    Given the target cap ``kappa_max`` on ``kappa(A)`` and the contraction floor
    ``sigma_min``, the total spectral budget is ``sigma_min * (kappa_max - 1)``,
    apportioned by ``skew_frac``.  The notebook defaults
    ``KappaBudget(0.1, 25.0, 0.6, 4)`` give ``budget=2.4``, ``c_P=0.96``,
    ``c_K=1.44``, hence ``sigma_max = 0.1 + 0.96 + 1.44 = 2.5`` and ``kappa <= 25``.
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


# ---------------------------------------------------------------------- FT models


class LatentFTNODEBase(nn.Module):
    """Shared ``g(z,u) = R_g * tanh(MLP)`` and ``F(z,u) = A(z)(z - g)``.

    Subclasses implement :meth:`A`.  The ``tanh`` confines the image of ``g`` --
    i.e. the model's equilibria -- to the box ``[-R_g, R_g]^m``.
    """

    def __init__(
        self,
        m=4,
        q=1,
        hidden=64,
        depth=3,
        sigma_min=0.1,
        R_g=2.0,
        activation="silu",
        op_hidden=64,
        op_depth=3,
    ):
        super().__init__()
        self.m, self.q, self.sigma_min, self.R_g = m, q, sigma_min, R_g
        self.activation = activation
        # `hidden`/`depth` size the equilibrium map g only.  `op_hidden`/`op_depth`
        # size the operator sub-networks that subclasses build to assemble A(z).
        # The two were a single hardcoded (64, 3) in the source notebooks.
        self.op_hidden, self.op_depth = op_hidden, op_depth
        self.g_net = MLP(m + q, m, hidden, depth, last_zero=True, activation=activation)
        self.register_buffer("_eye", torch.eye(m))

    def g(self, z, u):
        """Bounded equilibrium map.  ``u`` may be ``(b,)`` or ``(b, q)``."""
        if u.dim() == z.dim() - 1:
            u = u.unsqueeze(-1)
        return self.R_g * torch.tanh(self.g_net(torch.cat([z, u], -1)))

    def A(self, z):
        raise NotImplementedError

    def F(self, z, u):
        return torch.einsum("bij,bj->bi", self.A(z), z - self.g(z, u))


class LatentFTNODEUnbounded(LatentFTNODEBase):
    """Baseline: ``A = -(L L^T + sigma_min I) + (S - S^T)``, skew UNCONSTRAINED.

    ``L`` is lower-triangular with a softplus diagonal, so ``sym(A) <= -sigma_min I``
    still holds -- but nothing caps ``||K||_2``, so ``kappa(A)`` is free to blow up.
    This is the variant the kappa-bounded models are measured against.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        m = self.m
        tril = torch.tril_indices(m, m)
        self.register_buffer("_tr", tril[0])
        self.register_buffer("_tc", tril[1])
        self.register_buffer("_diag_mask", (tril[0] == tril[1]).float())
        self.L_net = MLP(m, m * (m + 1) // 2, self.op_hidden, self.op_depth,
                         activation=self.activation) # Cholesky fact sym
        self.S_net = MLP(m, m * m, self.op_hidden, self.op_depth,
                         last_zero=True, activation=self.activation) # skew net

    def _L(self, z):
        B = z.shape[0]
        raw = self.L_net(z)
        flat = Fnn.softplus(raw) * self._diag_mask + raw * (1.0 - self._diag_mask)
        Lm = z.new_zeros(B, self.m, self.m)
        Lm[:, self._tr, self._tc] = flat
        return Lm

    def A(self, z):
        L = self._L(z)
        M = self.S_net(z).view(-1, self.m, self.m)
        return -(L @ L.transpose(1, 2) + self.sigma_min * self._eye) + (M - M.transpose(1, 2))


class LatentFTNODEClamp(LatentFTNODEBase):
    """``A = -(sigma_min I + P) + K`` with ``||P||_2 <= c_P``, ``||K||_2 <= c_K``.

    Both bounds are imposed by projecting through ``clamp_fn``, so
    ``kappa(A) <= (sigma_min + c_P + c_K)/sigma_min = kappa_max``.

    ``clamp_fn`` defaults to the SVD-based :func:`spectral_clamp` the frozen
    notebooks were run with.  Pass :func:`spectral_clamp_safe` to get the
    eigvalsh route, which is what the control stage needs (see
    :class:`ftnode.control.SplitOperator`).
    """

    def __init__(self, c_P, c_K, clamp_fn=spectral_clamp, **kw):
        super().__init__(**kw)
        self.c_P, self.c_K = float(c_P), float(c_K)
        self.clamp_fn = clamp_fn
        m = self.m
        self.L_net = MLP(m, m * m, self.op_hidden, self.op_depth, activation=self.activation)
        self.S_net = MLP(m, m * m, self.op_hidden, self.op_depth,
                         last_zero=True, activation=self.activation)

    @classmethod
    def from_budget(cls, budget: KappaBudget, clamp_fn=spectral_clamp, **kw):
        kw.setdefault("m", budget.m)
        kw.setdefault("sigma_min", budget.sigma_min)
        return cls(c_P=budget.c_P, c_K=budget.c_K, clamp_fn=clamp_fn, **kw)

    def A(self, z):
        Lc = self.clamp_fn(self.L_net(z).view(-1, self.m, self.m), self.c_P**0.5)
        P = Lc @ Lc.transpose(1, 2)
        Mr = self.S_net(z).view(-1, self.m, self.m)
        K = self.clamp_fn(Mr - Mr.transpose(1, 2), self.c_K)
        return -(self.sigma_min * self._eye + P) + K


class LatentFTNODEYoula(LatentFTNODEBase):
    """SVD-free kappa bound: ``A = -(L L^T + sigma_min I) + K``.

    The symmetric factor has ``|L_ij| <= l_bound`` via ``tanh``, giving the
    closed-form worst case ``||L L^T||_2 <= l_bound^2 * m(m+1)/2 = c_P``.

    The skew part is Youla-parameterized: ``K = Q Sigma Q^T`` with
    ``Q = matrix_exp(W - W^T)`` in ``SO(m)`` and ``Sigma`` block-diagonal with
    ``2x2`` blocks ``beta_j * [[0, 1], [-1, 0]]``, where
    ``beta_j = beta_min + (c_K - beta_min) * sigmoid(.)`` lies in ``[beta_min, c_K)``.
    The singular values of such a ``K`` are the doubled set ``{beta_j, beta_j}``,
    so ``||K||_2 = max_j beta_j <= c_K`` -- **without ever computing an SVD**.
    That is the point: it reaches the same cap as :class:`LatentFTNODEClamp`
    while avoiding the ill-conditioned backward through ``matrix_norm(ord=2)`` on
    a skew argument.

    Requires an even ``m``.
    """

    def __init__(self, l_bound, c_K, beta_min=0.0, **kw):
        super().__init__(**kw)
        m = self.m
        if m % 2 != 0:
            raise ValueError(f"LatentFTNODEYoula needs an even latent dim, got m={m}")
        self.l_bound, self.c_K, self.beta_min = float(l_bound), float(c_K), float(beta_min)
        self.n_blk = m // 2
        tril = torch.tril_indices(m, m)
        self.register_buffer("_tr", tril[0])
        self.register_buffer("_tc", tril[1])
        r = 2 * torch.arange(self.n_blk)
        c = r + 1
        self.register_buffer("_br", r)
        self.register_buffer("_bc", c)
        act = self.activation
        self.L_net = MLP(m, m * (m + 1) // 2, self.op_hidden, self.op_depth, activation=act)  # symmetric factor (bounded)
        self.W_net = MLP(m, m * m, self.op_hidden, self.op_depth, last_zero=True, activation=act)  # skew gen (Q starts at I)
        self.b_net = MLP(m, self.n_blk, self.op_hidden, self.op_depth, activation=act)  # block magnitudes

    @classmethod
    def from_budget(cls, budget: KappaBudget, beta_min_frac: float = 0.0, **kw):
        kw.setdefault("m", budget.m)
        kw.setdefault("sigma_min", budget.sigma_min)
        return cls(
            l_bound=budget.l_bound,
            c_K=budget.c_K,
            beta_min=budget.beta_min(beta_min_frac),
            **kw,
        )

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

    def A(self, z):
        return self._A_sym(z) + self._K(z)


class LatentNODE(nn.Module):
    """Unstructured latent NODE baseline: ``F(z, u) = MLP([z, u])``, no structural prior.

    ``hidden=95, depth=4`` makes its dynamics parameter count (~28.3k) match
    :class:`LatentFTNODEClamp`, so the comparison isolates structure rather than
    capacity.  Deliberately exposes neither ``.A`` nor ``.g``, which is how
    :func:`ftnode.train.train_one` knows the residual regularizer is inert for it
    and how the kappa diagnostics know to skip it.
    """

    def __init__(self, m=4, q=1, hidden=95, depth=4, activation="silu"):
        super().__init__()
        self.m = m
        self.q = q
        self.activation = activation
        self.f_net = MLP(m + q, m, hidden=hidden, depth=depth, activation=activation)

    def F(self, z, u):
        if u.dim() == z.dim() - 1:
            u = u.unsqueeze(-1)
        return self.f_net(torch.cat([z, u], dim=-1))


# ------------------------------------------------------------------------ builders


@dataclass(frozen=True)
class LatentModelConfig:
    """Architecture settings shared by every variant.

    Defaults are the values the duffing notebooks use.  ``tau`` must match the
    dataset's window length (:attr:`ftnode.systems.DuffingDataConfig.tau`).

    ``activation`` is a **string** rather than a class so the config stays
    round-trippable through :func:`ftnode.utils.save_config` (``yaml.safe_dump``
    cannot serialize a class object).  It defaults to ``"silu"`` for
    reproducibility of the committed results, which is not the same as it being
    the right choice -- see :data:`ACTIVATIONS` and :attr:`lipschitz_1`.

    There are three independent width/depth pairs, one per *role*.  Which
    sub-network each pair reaches:

    ==========================  ==========================================
    field pair                  sizes
    ==========================  ==========================================
    ``enc_hidden/enc_depth``    :class:`Encoder` ``net``
    ``hidden/depth``            ``g_net`` -- the equilibrium map ``g(z,u)``
    ``op_hidden/op_depth``      the operator sub-networks assembling ``A(z)``:
                                ``L_net`` and ``S_net`` for
                                :class:`LatentFTNODEUnbounded` and
                                :class:`LatentFTNODEClamp`; ``L_net``,
                                ``W_net`` and ``b_net`` for
                                :class:`LatentFTNODEYoula`
    ==========================  ==========================================

    ``op_hidden``/``op_depth`` replace a literal ``(64, 3)`` that the source
    notebooks hardcoded at every operator sub-network, so before this the
    ``hidden``/``depth`` pair reached ``g_net`` *only* -- setting
    ``hidden=128`` widened under a third of the model and silently left the
    operator networks at 64.

    Does **not** apply to :class:`LatentNODE`: it has no ``A(z)``, so it has no
    operator sub-networks.  Its width comes from ``build_latent_node``'s own
    ``hidden``/``depth`` arguments (95/4, chosen to parameter-match the clamp).
    """

    m: int = 4
    q: int = 1
    hidden: int = 64
    depth: int = 3
    sigma_min: float = 0.1
    R_g: float = 2.0
    z_scale: float = 2.0
    tau: int = 8
    enc_hidden: int = 64
    enc_depth: int = 2
    op_hidden: int = 64
    op_depth: int = 3
    activation: str = "silu"

    def __post_init__(self):
        # Fail at construction rather than deep inside a layer build.
        resolve_activation(self.activation)

    @property
    def lipschitz_1(self) -> bool:
        """Whether every hidden nonlinearity in this configuration is 1-Lipschitz.

        Composed per-layer Lipschitz bounds -- the kind contraction and
        conditioning arguments rely on -- are only valid when this is ``True``.
        It is ``False`` under the default ``"silu"``.
        """
        return is_lipschitz_1(self.activation)

    def _encoder(self) -> Encoder:
        return Encoder(
            self.tau,
            self.m,
            self.enc_hidden,
            self.enc_depth,
            self.z_scale,
            activation=self.activation,
        )

    def _base_kw(self) -> dict:
        return dict(
            m=self.m,
            q=self.q,
            hidden=self.hidden,
            depth=self.depth,
            sigma_min=self.sigma_min,
            R_g=self.R_g,
            activation=self.activation,
            op_hidden=self.op_hidden,
            op_depth=self.op_depth,
        )


def _assemble(cfg: LatentModelConfig, make_dynamics) -> LatentSysID:
    """Build encoder, then dynamics, then decoder -- in that order.

    The order is load-bearing, not stylistic: these modules draw from the global
    torch RNG as they initialize, so building them in a different sequence gives
    different weights for the same ``torch.manual_seed(s)``.  This is the order
    the notebooks' ``build_*`` closures use (Python evaluates
    ``LatentSysID(Encoder(...), Dynamics(...), Decoder(...))`` left to right), and
    matching it is what lets a seed reproduce a frozen notebook run.
    """
    encoder = cfg._encoder()
    dynamics = make_dynamics() # Initilaize different builds 
    decoder = LinearDecoder(cfg.m)
    return LatentSysID(encoder, dynamics, decoder)


def build_clamp(cfg: LatentModelConfig, budget: KappaBudget, clamp_fn=spectral_clamp) -> LatentSysID:
    """The kappa-bounded SVD-clamp model.  This is the ID architecture the control stage builds on."""
    return _assemble(
        cfg,
        lambda: LatentFTNODEClamp(
            c_P=budget.c_P, c_K=budget.c_K, clamp_fn=clamp_fn, **cfg._base_kw()
        ),
    )


def build_youla(
    cfg: LatentModelConfig, budget: KappaBudget, beta_min_frac: float = 0.0
) -> LatentSysID:
    """The kappa-bounded Youla model: same cap, no SVD anywhere."""
    return _assemble(
        cfg,
        lambda: LatentFTNODEYoula(
            l_bound=budget.l_bound,
            c_K=budget.c_K,
            beta_min=budget.beta_min(beta_min_frac),
            **cfg._base_kw(),
        ),
    )


def build_unbounded(cfg: LatentModelConfig) -> LatentSysID:
    """The unconstrained-skew FT baseline (stable, but kappa is unbounded)."""
    return _assemble(cfg, lambda: LatentFTNODEUnbounded(**cfg._base_kw()))


def build_latent_node(cfg: LatentModelConfig, hidden: int = 95, depth: int = 4) -> LatentSysID:
    """The unstructured, parameter-matched latent NODE baseline."""
    return _assemble(
        cfg,
        lambda: LatentNODE(
            m=cfg.m, q=cfg.q, hidden=hidden, depth=depth, activation=cfg.activation
        ),
    )
