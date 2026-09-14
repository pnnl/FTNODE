"""The FIM-ODE backbone as a differentiable physical-state field.

``FIMBackbone`` wraps a pretrained ``fim.models.ode.FIMODE`` so it fits the Route D
rollout.  Usage per minibatch:

1. ``prepare(window, h)`` reconstructs a 2D ``(q, q_dot)`` context from the ``q``-window
   and encodes it once (differentiable in the encoder parameters).
2. ``drift(x)`` queries the physical drift at state ``x`` at each RK4 stage.

Two details make the query correct and differentiable:

- The query location is padded to ``dim_max_trajectory`` with a plain ``cat`` of zeros,
  not ``FIMODE.pad_if_necessary`` (which runs under ``no_grad`` and would detach the
  state from the rollout graph).
- ``function_decoding`` returns drift in normalized coordinates.  ``renormalize()``
  maps it back to physical units and keeps the autograd graph, unlike
  ``get_prediction_for_eval`` which detaches.
"""
from __future__ import annotations

import pathlib

import torch
import torch.nn as nn
import torch.utils.checkpoint

from fim.models.ode import FIMODE, load_fim_ode_hf, load_fim_ode_local
from fim.models.ode_trainer import ODEConcepts

__all__ = ["FIMBackbone", "load_base_model"]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _REPO_ROOT / "weights" / "openfim"


def load_base_model(device: str = "cpu", cache_dir=None) -> FIMODE:
    """Load the pretrained ``FIM4Science/fim-ode`` ``base_model`` via the fim package.

    Reuses ``fim.models.ode.load_fim_ode_hf`` (download + safetensors + strict load, no
    ``trust_remote_code``).  Caches under ``weights/openfim`` by default, which is
    git-ignored.
    """
    cache = pathlib.Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    return load_fim_ode_hf(device=device, cache_dir=cache)


class FIMBackbone(nn.Module):
    """Pretrained FIM-ODE as a context-bound, differentiable field over ``x = (q, q_dot)``.

    ``state_dim`` is the physical dimension (2 for Duffing); the model pads internally
    to ``dim_max_trajectory``.  Call :meth:`prepare` before :meth:`drift`.
    """

    def __init__(self, fim: FIMODE, state_dim: int = 2, use_checkpoint: bool = True):
        super().__init__()
        self.fim = fim
        self.d = state_dim
        self.M = int(fim.model_config.dim_max_trajectory)
        # Gradient-checkpoint the per-step drift so backprop-through-time over a long
        # rollout stays memory-bounded (activations are recomputed in backward instead
        # of stored for every RK4 stage). Without it, batch x L exhausts GPU memory.
        self.use_checkpoint = use_checkpoint
        # Bound context, set by prepare(): encoder output, its feature mask, and the
        # state/time normalization stats. Kept as plain attributes, not buffers.
        self._D = None
        self._feature_mask = None
        self._sns = None  # states_norm_stats
        self._tns = None  # times_norm_stats

    @classmethod
    def from_pretrained(cls, device: str = "cpu", cache_dir=None, state_dim: int = 2):
        return cls(load_base_model(device=device, cache_dir=cache_dir), state_dim=state_dim)

    @classmethod
    def from_local(cls, checkpoint_dir, device: str = "cpu", state_dim: int = 2):
        return cls(load_fim_ode_local(checkpoint_dir, device=device), state_dim=state_dim)

    def _reconstruct_context(self, window, h):
        """A 2D ``(q, q_dot)`` context trajectory from the scalar ``q``-window.

        ``window`` is ``(B, tau)``.  ``q_dot`` is a central finite difference of the
        window.  Returns ``trajectories (B,1,tau,2)``, ``times (B,1,tau,1)``,
        ``mask (B,1,tau,1)`` bool, matching FIMODE's context contract (one trajectory
        per item, so the constant input ``u`` is encoded implicitly).
        """
        B, tau = window.shape
        q = window
        qd = torch.gradient(q, spacing=h, dim=1)[0]
        traj = torch.stack([q, qd], dim=-1).unsqueeze(1)  # (B,1,tau,2)
        t = torch.arange(tau, device=window.device, dtype=window.dtype) * h
        times = t.view(1, 1, tau, 1).expand(B, 1, tau, 1).contiguous()
        mask = torch.ones(B, 1, tau, 1, dtype=torch.bool, device=window.device)
        return traj, times, mask

    def prepare(self, window, h):
        """Encode the context once. Call before :meth:`drift`, once per minibatch.

        The encoding is not persisted across optimizer steps -- recompute it each
        minibatch so its autograd graph is not reused after being freed.
        """
        traj, times, mask = self._reconstruct_context(window, h)
        D, feature_mask, concept = self.fim.trajectory_encoding(traj, times, mask)
        self._D = D
        self._feature_mask = feature_mask
        self._sns = concept._states_norm_stats
        self._tns = concept._times_norm_stats

    def _fresh_concept(self):
        """A concept builder referencing the bound norm stats, for one decode.

        Cheaper than deep-copying the encode-time builder, and correct: each decode
        sets its own locations/drift on a fresh builder that shares the stats tensors.
        """
        c = ODEConcepts.builder()
        c.states_norm(self.fim.spatial_norm).states_norm_stats(self._sns)
        c.times_norm(self.fim.temporal_norm).times_norm_stats(self._tns)
        return c

    def _decode(self, loc_norm):
        """Attention decode only: normalized locations -> normalized drift ``(B,1,M)``.

        This is the memory-heavy, checkpoint-safe part.  It must not contain the
        normalization derivatives, which use ``torch.func``/``vmap`` and are incompatible
        with checkpoint's saved-tensor hooks.
        """
        out = self.fim.function_decoding(loc_norm, self._feature_mask, self._D, self._fresh_concept())
        return out.predictions.drift

    def _drift_impl(self, x):
        loc = x[:, None, :]  # (B, 1, d)
        if self.M > self.d:
            pad = loc.new_zeros(loc.shape[:-1] + (self.M - self.d,))
            loc = torch.cat([loc, pad], dim=-1)  # (B, 1, M), differentiable
        # Forward and inverse normalization stay OUTSIDE the checkpoint (they use vmap).
        loc_norm = self.fim.spatial_norm.normalization_map(loc, self._sns)
        if self.use_checkpoint and torch.is_grad_enabled():
            drift_norm = torch.utils.checkpoint.checkpoint(self._decode, loc_norm, use_reentrant=False)
        else:
            drift_norm = self._decode(loc_norm)
        # Renormalize to physical units with the library's own transform (it uses vmap,
        # so it stays outside the checkpoint), rather than replicating the math.
        concept = ODEConcepts(
            locations=loc_norm, drift=drift_norm, normalized=True,
            states_norm=self.fim.spatial_norm, states_norm_stats=self._sns,
            times_norm=self.fim.temporal_norm, times_norm_stats=self._tns,
        )
        concept.renormalize()
        return concept.drift[:, 0, : self.d]  # (B, d)

    def drift(self, x):
        """Physical drift at state ``x`` ``(B, d)``, differentiable. Requires :meth:`prepare`."""
        if self._D is None:
            raise RuntimeError("call prepare(window, h) before drift(x)")
        return self._drift_impl(x)
