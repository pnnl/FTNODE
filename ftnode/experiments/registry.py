"""Named model variants -- a point on the ``A(z)`` axis crossed with one on ``g(z,u)``.

A variant is not a class here, it is a *pair of registry keys*: one from
:data:`ftnode.latent.A_KINDS` and one from :data:`ftnode.latent.G_KINDS`.  That is
what makes an experiment file able to sweep a new equilibrium map against every
operator without anything being added to this module.

:data:`REGISTRY` holds the four variants the duffing notebooks compare, under the
slugs their checkpoint filenames already use, so ``pkg-l-ft-k-svd-clamp-seed0.pth``
naming carries over.  Anything outside that set is expressible inline -- see
:func:`variant_from_spec`.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..latent import A_KINDS, G_KINDS

__all__ = ["Variant", "REGISTRY", "variant_from_spec", "resolve_variants"]


@dataclass(frozen=True)
class Variant:
    """One model to train: which operator, which equilibrium map, and how to label it.

    ``slug`` is the **identifier**: it names the variant's directory inside a run,
    so it must stay stable -- changing one orphans every checkpoint written under
    the old name.  ``name`` is the **display label** for plots and tables.

    ``operator`` and ``equilibrium`` are ``None`` together for the unstructured
    :class:`~ftnode.latent.LatentNODE` baseline, which is on neither axis -- it has
    no ``A(z)`` and no ``g(z,u)``, and the residual regularizer and the kappa
    diagnostics both duck-type on that absence.  Naming one axis but not the other
    is a spec error, not a variant.

    Carries no styling.  Colors, line styles and markers are presentation and
    belong wherever the figure is drawn -- putting a color here would mean editing
    the package to restyle a plot, and would still not cover line style or marker.
    """

    slug: str
    name: str
    operator: str | None
    equilibrium: str | None

    def __post_init__(self):
        if (self.operator is None) != (self.equilibrium is None):
            raise ValueError(
                f"variant {self.slug!r}: operator and equilibrium must be given together "
                f"(got operator={self.operator!r}, equilibrium={self.equilibrium!r}). "
                "Omit both for the unstructured LatentNODE baseline."
            )
        if self.operator is not None and self.operator not in A_KINDS:
            raise ValueError(
                f"variant {self.slug!r}: unknown operator {self.operator!r}; "
                f"choose from {sorted(A_KINDS)}"
            )
        if self.equilibrium is not None and self.equilibrium not in G_KINDS:
            raise ValueError(
                f"variant {self.slug!r}: unknown equilibrium {self.equilibrium!r}; "
                f"choose from {sorted(G_KINDS)}"
            )

    @property
    def is_baseline(self) -> bool:
        """Whether this is the unstructured baseline rather than a split model."""
        return self.operator is None

    @property
    def kind(self) -> str:
        """``'ln'`` for the unstructured baseline, ``'ft'`` for a split model.

        The duffing notebooks branch on this to decide which diagnostics apply --
        the kappa and skew figures are meaningless without an ``A(z)``.
        """
        return "ln" if self.is_baseline else "ft"


#: The four variants the frozen kappa notebooks compare.
#:
#: Named shortcuts, not a gate: any operator/equilibrium pair is usable straight
#: from an experiment file without appearing here (see :func:`variant_from_spec`).
#: Slugs match the names the notebooks' checkpoint files already used, so a run
#: directory and a notebook figure agree.  Changing one orphans checkpoints.
REGISTRY: dict[str, Variant] = {
    v.slug: v
    for v in (
        #       slug,                 name,                 operator,     equilibrium
        Variant("l-n",                "L-N",                None,         None),
        Variant("l-ft-unbounded",     "L-FT (unbounded)",   "unbounded",  "tanh_mlp"),
        Variant("l-ft-k-svd-clamp",   "L-FT-k (SVD-clamp)", "svd_clamp",  "tanh_mlp"),
        Variant("l-ft-k-youla",       "L-FT-k (Youla)",     "youla",      "tanh_mlp"),
    )
}


def variant_from_spec(spec) -> Variant:
    """Resolve one ``variants:`` entry from an experiment file.

    Two forms are accepted::

        - l-ft-k-svd-clamp                          # a REGISTRY slug
        - {operator: youla, equilibrium: sym_jac}   # an ad-hoc pair

    The second form is the point: a newly registered equilibrium map can be swept
    against every operator immediately, with **no entry added to** :data:`REGISTRY`
    -- only the module and its ``G_KINDS`` line in
    :mod:`ftnode.latent.equilibrium`.  Its slug and label are derived as
    ``operator+equilibrium``; ``slug`` and ``name`` may be given explicitly to
    override that.

    Args:
        spec (str | dict): A registry slug, or a mapping with at least
            ``operator`` and ``equilibrium``.

    Returns:
        Variant

    Raises:
        ValueError: For an unknown slug, an unknown kind on either axis, or only
            one of the two axes being named.
        TypeError: For anything that is neither a string nor a mapping.
    """
    if isinstance(spec, str):
        try:
            return REGISTRY[spec]
        except KeyError:
            raise ValueError(
                f"unknown variant {spec!r}; choose from {sorted(REGISTRY)} "
                "or give an explicit {operator: ..., equilibrium: ...} mapping"
            ) from None

    if not isinstance(spec, dict):
        raise TypeError(f"a variant must be a slug or a mapping; got {type(spec).__name__}")

    known = {"slug", "name", "operator", "equilibrium"}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(
            f"variant mapping has no fields {sorted(unknown)}"
            + ("; styling belongs where the figure is drawn, not in the spec"
               if "color" in unknown else "")
        )

    operator, equilibrium = spec.get("operator"), spec.get("equilibrium")
    if operator is None and equilibrium is None and "slug" not in spec:
        raise ValueError(
            "a variant mapping needs `operator` and `equilibrium` "
            "(or a `slug` naming the baseline)"
        )
    derived = f"{operator}+{equilibrium}" if operator is not None else "l-n"
    return Variant(
        slug=spec.get("slug", derived),
        name=spec.get("name", derived),
        operator=operator,
        equilibrium=equilibrium,
    )


def resolve_variants(specs) -> tuple[Variant, ...]:
    """Resolve a whole ``variants:`` list, rejecting duplicate slugs.

    Duplicates matter because the slug is a directory name: two variants sharing
    one would train in sequence into the same files, and the second would silently
    win.
    """
    out = [variant_from_spec(s) for s in specs]
    seen = {}
    for v in out:
        if v.slug in seen:
            raise ValueError(f"duplicate variant slug {v.slug!r}; slugs name run directories")
        seen[v.slug] = v
    if not out:
        raise ValueError("a spec must list at least one variant")
    return tuple(out)
