"""Equilibrium maps ``g(z, u)`` -- one half of the split ``F = A(z)(z - g(z,u))``.

Selecting a map by name from :data:`G_KINDS` is what lets any ``g`` compose with
any operator (see :mod:`ftnode.latent.operator`) instead of needing a class per
pair.  The frozen ``duffing_partial_obs_latent_ftnode_superposition_g_*`` notebooks
show the cost of the alternative: they declare ``LatentFTNODESuper`` with ``A``
re-implemented inline, usable with one operator out of three.

.. warning::
   The module attached here is reached under a fixed attribute name on the
   dynamics module, and that name is a checkpoint key.  See :mod:`ftnode.latent`.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .nets import MLP, lipschitz_bound
from .operator import spectral_clamp_safe

__all__ = ["BoundedTanhG", "GradPotentialG", "G_KINDS", "resolve_g"]


class BoundedTanhG(MLP):
    """Default equilibrium map: ``g(z, u) = R_g * tanh(MLP([z, u]))``.

    The ``tanh`` confines the image of ``g`` -- i.e. the model's equilibria -- to
    the box ``[-R_g, R_g]^m``.  This is the form every duffing notebook and every
    committed checkpoint was produced with.

    It **subclasses** :class:`~ftnode.latent.nets.MLP` rather than wrapping one,
    so its weights sit at ``equilibrium.net.*`` and its parameter draws are
    bitwise identical to the bare ``MLP`` this replaced.  A new kind is under no
    obligation to do the same -- a checkpoint written by one map was never
    loadable into another.
    """

    def __init__(self, m=4, q=1, hidden=64, depth=3, R_g=2.0, activation="silu"):
        super().__init__(m + q, m, hidden, depth, last_zero=True, activation=activation)
        self.m, self.q = m, q
        self.R_g = R_g

    def forward(self, z, u):
        """``u`` may be ``(b,)`` or ``(b, q)``; broadcasts over rollout dims too."""
        if u.dim() == z.dim() - 1:
            u = u.unsqueeze(-1)
        return self.R_g * torch.tanh(self.net(torch.cat([z, u], -1)))


class GradPotentialG(MLP):
    """Symmetric-Jacobian equilibrium map: ``g(z, u) = grad_z Phi(z, u)``.

    ``Phi`` is a scalar-output MLP over ``[z, u]``, so ``J_g = grad^2 Phi`` is a Hessian
    and therefore **symmetric by construction**.  By the Poincare lemma that is not one
    modelling choice among several -- every symmetric-``J_g`` map is a potential gradient,
    so this is *the* form, and the only freedom is how ``Phi`` is parameterized.

    What the symmetry buys, at frozen ``u``, writing ``V = 0.5||z||^2 - Phi`` so that
    ``z - g = grad V`` and ``F = A(z) grad V``:

    * ``V`` decreases along every trajectory -- ``dV/dt = <grad V, A grad V> <=
      -sigma_min ||grad V||^2``, the skew part of ``A`` cancelling identically.  The field
      becomes a generalized gradient system: no limit cycles, no chaos.
    * The equilibria are the critical points of ``V``, and their stability is read off
      ``V`` alone.  Since ``M J + J^T M = 2 M (sym A) M < 0`` for ``M = grad^2 V``, the
      inertia theorem gives ``#{unstable directions of F} = Morse index of V`` for *any*
      ``A`` with ``sym A < 0``.  So ``A`` cannot create, destroy, or restabilize an
      equilibrium -- it owns transients only.  A general (non-symmetric) ``J_g`` has no
      such guarantee.
    * Multistability survives: the critical set may be as large as you like, and a saddle
      is exactly a point where ``lambda_max(J_g) > 1``.  Do **not** additionally constrain
      ``lambda_max(J_g) < 1`` -- that makes ``V`` strictly convex and deletes the
      pitchfork.

    **The bound is l2, not the l-infinity box.**  ``BoundedTanhG`` confines ``g`` to
    ``[-R_g, R_g]^m``; capping the spectral norm of every weight matrix instead confines it
    to a *ball*, ``||g||_2 <= g_cap``, which is what the forward-invariance results want
    (``r_in = g_cap`` rather than ``R_g sqrt(m)``).  ``R_g`` is accepted to satisfy the
    shared constructor contract and is deliberately **ignored**: reinterpreting it would
    silently change what a shared config means for the two kinds.

    Args:
        cap: Enforce the a priori bound.  ``False`` gives the free potential, whose
            Lipschitz constant is a measured training outcome rather than a design
            parameter -- useful as a diagnostic, but it gives up the trapping radius.
        g_cap: The certified bound on ``||g||_2`` when ``cap`` is set.  An absolute
            default rather than one derived from ``R_g``: the learned equilibria of this
            plant sit at ``||z*||_2`` up to 2.41, and ``z* = g(z*, u)`` forces
            ``g_cap >= max||z*||``, so inheriting ``R_g = 2.0`` would make the pitchfork
            unrepresentable.  The useful window is ``[2.5, 4)`` -- below the floor the
            equilibria do not fit, and at 4 the resulting Nagumo ball
            ``R = kappa_max * g_cap`` stops beating the incumbent box.
        w1_cap: Cap on the first weight matrix; the remaining caps are derived so the
            product is exactly ``g_cap``.  Deliberately **unbalanced**: reachable curvature
            scales with ``||W_1||`` at fixed product, so equal caps buy a weaker model for
            the same certificate.  But the trade is two-sided -- a large ``w1_cap`` forces
            the remaining caps down, and hidden layers capped well below 1 attenuate the
            signal badly.  Measured headroom at ``g_cap=3, depth=3``: 1.06 at ``w1_cap=1.5``,
            2.20 at 2, **3.38 at 3**, 5.09 at 10, against the ~2.2 a saddle needs; and
            training at ``w1_cap=10`` reaches a quarter the ``||g||`` that ``w1_cap=2``
            does.  The default takes the middle.
        init_scale: Standard deviation of the readout initialization.  **Not zero**, and
            that matters more than it looks.  ``last_zero`` would give ``Phi == 0``, hence
            ``g == 0`` -- the "equilibrium map starts at the origin" property
            :class:`BoundedTanhG` has -- but with a zero readout *every hidden layer gets
            exactly zero gradient*, so the map only starts learning once the readout moves
            off zero.  For ``BoundedTanhG`` that happens fast; here the readout enters
            ``grad Phi`` multiplied by the product of the capped hidden norms, which
            measured **33x smaller** gradient than the incumbent's at the same point.  The
            result is a cold start the map does not escape within a short run: at 15 epochs,
            ``init_scale=0`` reaches ``max||g||`` of 0.011 against 0.377 at 0.5.  Values
            above the readout's own cap are equivalent, since ``project_`` clamps them.

    .. note::
       ``grad_z Phi`` does not depend on the readout bias, so ``net[-1].bias`` receives no
       gradient, ever, and stays at its ``last_zero`` value of 0.  Harmless.  Do not "fix"
       it by dropping the bias -- that would fork :class:`~ftnode.latent.nets.MLP`.  Note
       the trapping-radius argument's ``Phi(0,u) = 0`` normalization is *not* what this
       provides and does not need to be: the argument compares ``V(z(t))`` to ``V(z_0)``,
       so an additive constant in ``Phi`` cancels.

    .. warning::
       A checkpoint written by :class:`BoundedTanhG` will not load into this map: the
       readout is ``(1, hidden)`` here against ``(m, hidden)`` there.  That is true of any
       two equilibrium maps and fails loudly at ``load_state_dict``.
    """

    def __init__(self, m=4, q=1, hidden=64, depth=3, R_g=2.0, activation="silu",
                 cap=True, g_cap=3.0, w1_cap=3.0, init_scale=0.5):
        if activation in ("relu", "leaky_relu"):
            raise ValueError(
                f"{type(self).__name__} cannot use {activation!r}: a piecewise-linear Phi "
                "has grad^2 Phi = 0 almost everywhere, so J_g vanishes, M = I, and no "
                "saddle -- hence no multistability -- can exist.  Training would run and "
                "every bound would hold trivially.  Use tanh, elu or softplus for a "
                "1-Lipschitz choice with a nonzero second derivative."
            )
        super().__init__(m + q, 1, hidden, depth, last_zero=True, activation=activation)
        self.m, self.q = m, q
        self.R_g = R_g  # part of the shared contract; unused, see the class docstring
        self.cap = bool(cap)
        self.g_cap, self.w1_cap = float(g_cap), float(w1_cap)

        # Derive from the built module, not from `depth`: MLP(depth=d) lays down d+1
        # weight matrices and d activations, and the certificate is stated per layer.
        n_lin = len(self._linears())
        n_act = n_lin - 1
        l_sigma = lipschitz_bound(activation) if isinstance(activation, str) else 1.0
        rest = (self.g_cap / (l_sigma**n_act * self.w1_cap)) ** (1.0 / n_act)
        self._caps = [self.w1_cap] + [rest] * n_act
        prod = 1.0
        for c in self._caps:
            prod *= c
        #: Certified bound on ``||g||_2``; ``inf`` when uncapped.
        self.g_bound = l_sigma**n_act * prod if self.cap else float("inf")
        self.l_sigma = l_sigma
        self.init_scale = float(init_scale)
        if self.init_scale > 0.0:
            # Undo `last_zero` on the readout WEIGHT only -- see the `init_scale` note.
            # The bias stays zero, which is where it would stay regardless.
            with torch.no_grad():
                self._linears()[-1].weight.normal_(0.0, self.init_scale)
        # Hold the invariant from construction, not from the first optimizer step: the
        # default Linear init overshoots the derived hidden caps.
        self.project_()

    def _linears(self):
        return [layer for layer in self.net if isinstance(layer, nn.Linear)]

    def phi(self, z, u):
        """The scalar potential itself, shape ``(...)``.  Broadcasts as ``forward`` does."""
        if u.dim() == z.dim() - 1:
            u = u.unsqueeze(-1)
        return self.net(torch.cat([z, u], -1)).squeeze(-1)

    def forward(self, z, u):
        """``grad_z Phi``, shape ``(..., m)``.  ``u`` may be ``(b,)``, ``(b, q)`` or rollout-ranked.

        Implemented as the gradient of the **summed** potential.  ``Phi`` for sample ``b``
        depends only on row ``b`` -- an :class:`~ftnode.latent.nets.MLP` has no
        cross-sample layer -- so ``d(sum_b Phi_b)/dz_b = dPhi_b/dz_b`` and one backward
        pass yields every per-sample gradient.  This needs no ``vmap`` and no reshaping,
        which is what makes it rank-agnostic: the rollout's ``(b, L+1, m)`` and the
        field's ``(b, m)`` both work unchanged.

        ``torch.func.grad`` rather than ``torch.autograd.grad``: the control stage runs
        ``torch.func.jvp`` straight through this map
        (:meth:`ftnode.control.FrozenLatentPlant.DuF_closed`), and an ``autograd.grad``
        body raises there -- functorch refuses a nested ``requires_grad_``.  It also
        sidesteps the ``create_graph`` trap, where gating on ``self.training`` would
        silently detach ``g`` during every eval rollout.

        .. warning::
           The summed-gradient identity holds only while ``MLP`` stays free of
           cross-sample layers.  Adding a ``BatchNorm`` inside it would make this map
           silently wrong; ``tests/test_grad_potential.py`` pins the identity for exactly
           that reason.
        """
        if u.dim() == z.dim() - 1:
            u = u.unsqueeze(-1)
        return torch.func.grad(lambda zz: self.net(torch.cat([zz, u], -1)).sum())(z)

    @torch.no_grad()
    def project_(self):
        """Project every weight matrix back onto its spectral cap, in place.

        Called after each optimizer step (see :func:`ftnode.train.train_one`) rather than
        inside :meth:`forward`.  The projection costs ~1 ms and does not depend on the
        batch, so paying it per field evaluation would add ~0.8 s to every ``L=200``
        rollout -- roughly 800 evaluations -- against ~1 s per *seed* when it runs
        post-step.  The weights then lie in the constraint set at every forward, so the
        certificate holds as an invariant rather than as an output of the forward pass.

        Uses :func:`~ftnode.latent.operator.spectral_clamp_safe` and not
        :func:`~ftnode.latent.operator.spectral_clamp`: the SVD route can fail outright on
        the forward pass when singular values repeat, and its ``.view(-1, 1, 1)`` returns
        ``(1, out, in)`` for a 2-D weight, so it cannot be copied back at all.
        """
        if not self.cap:
            return
        for layer, c in zip(self._linears(), self._caps):
            layer.weight.copy_(spectral_clamp_safe(layer.weight, c))

    def curvature_headroom(self, n=64, steps=300, lr=5e-2, box=2.0, seed=0):
        """Largest ``lambda_max(J_g)`` reachable under the current caps.  Diagnostic only.

        A saddle of ``V`` needs ``lambda_max(J_g) > 1``, so a configuration whose headroom
        sits below that cannot represent multistability no matter how it is trained -- it
        would train cleanly, satisfy every bound, and silently never form the pitchfork.
        At the default caps this returns roughly 17, against the 2.15--2.2 the incumbent
        map reaches, so it guards against a bad ``w1_cap``/``g_cap`` allocation rather than
        against the method.

        Runs on a **copy**, so the caller's weights are untouched.  Maximizes a directional
        curvature ``v^T (grad^2 Phi) v`` by Hessian-vector product -- no eigendecomposition
        inside the loop, which would trip over the repeated eigenvalues this very
        construction produces.  Being an optimizer result it is a *lower* bound on what the
        constraint set admits, in the same sense as
        :func:`ftnode.diagnostics.empirical_lipschitz`.
        """
        probe = copy.deepcopy(self).to(torch.float64)
        gen = torch.Generator().manual_seed(seed)
        # `last_zero` leaves the readout at zero, hence Phi == 0 and a flat start with no
        # curvature to climb from.  Re-seed it: the question is what the constraint set
        # admits, not what is reachable from this particular initialization.
        with torch.no_grad():
            last = probe._linears()[-1]
            last.weight.normal_(0.0, 1.0, generator=gen)
        probe.project_()
        z = (torch.rand(n, self.m, generator=gen, dtype=torch.float64) * 2 - 1) * box
        u = torch.zeros(n, self.q, dtype=torch.float64)
        v = torch.randn(n, self.m, generator=gen, dtype=torch.float64)
        v = (v / v.norm(dim=-1, keepdim=True)).requires_grad_(True)
        opt = torch.optim.Adam(list(probe.parameters()) + [v], lr=lr)
        best = 0.0
        for _ in range(steps):
            vn = v / v.norm(dim=-1, keepdim=True)
            _, hv = torch.func.jvp(lambda zz: probe(zz, u), (z,), (vn,))
            quad = (hv * vn).sum(-1)
            best = max(best, quad.max().item())
            opt.zero_grad()
            (-quad.max()).backward()
            opt.step()
            probe.project_()
        return best


#: Selectable equilibrium maps, keyed by the string a config stores.
#:
#: A string rather than a class for the same reason ``activation`` is one: the
#: config has to round-trip through :func:`ftnode.utils.save_config`, and
#: ``yaml.safe_dump`` cannot serialize a class object.
#:
#: Every entry must accept ``(m, q, hidden, depth, R_g, activation)`` positionally
#: plus any variant-specific keywords, and expose ``forward(z, u) -> (..., m)``.
#: Give those six defaults, as :class:`BoundedTanhG` does and as the operators do:
#: it costs nothing and makes the map constructible by hand for anyone building a
#: model directly rather than through a config.
#: That signature is the whole contract -- the model builds whichever map it is
#: told to and never looks inside, so an entry is free to carry as much internal
#: structure as its math needs.
#:
#: .. note::
#:    The equilibrium map is constructed *before* the operator, so a kind with a
#:    different parameter count shifts the RNG stream the operator draws from.  At
#:    a fixed seed, "same ``A``, different ``g``" gives a structurally identical
#:    but numerically different ``A(z)`` at initialization.  That is expected --
#:    the order is fixed for checkpoint compatibility -- but it means a ``g``
#:    comparison is only fair across seeds, not within one.
G_KINDS: dict[str, type] = {
    "tanh_mlp": BoundedTanhG,
    "grad_potential": GradPotentialG,
}


def resolve_g(spec):
    """Turn an equilibrium-map spec into the class to construct.

    Accepts a name from :data:`G_KINDS`, an ``nn.Module`` subclass, or any
    callable returning a module.  Mirrors
    :func:`~ftnode.latent.nets.resolve_activation`.
    """
    if isinstance(spec, str):
        try:
            return G_KINDS[spec]
        except KeyError:
            raise ValueError(
                f"unknown g_kind {spec!r}; choose from {sorted(G_KINDS)} "
                "or pass an nn.Module subclass"
            ) from None
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec
    if callable(spec):
        return spec
    raise TypeError(f"g_kind must be a name, nn.Module subclass, or factory; got {spec!r}")
