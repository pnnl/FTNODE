"""A context-conditioned field pretrained across systems (the structured foundation-model line).

A set encoder reads a context of measured trajectories and returns one system code ``c``;
a head turns ``c`` into a field ``F(x, u; c)``.  Pretraining is drift matching with exact
labels on a generated family of damped mechanical systems (:mod:`.generator`), with
Duffing held out.  Every random draw of the pretraining data is a function of
``(seed, stream, index)``, so any batch can be regenerated.
"""
from .encoder import SetEncoder
from .generator import (
    DUFFING_HOLDOUT,
    DUFFING_PARAMS,
    TRAIN_STREAM,
    VAL_STREAM,
    MechanicalFamilyConfig,
    SystemBatch,
    batch_generator,
    context_tokens,
    in_duffing_holdout,
    mechanical_field,
    sample_batch,
    sample_params,
    simulate_q,
)
from .heads import FreeCodeHead, StructuredCodeHead, matched_free_hidden, mlp_numel
from .model import ContextField, build_context_field, tokens_from_q

__all__ = [
    "SetEncoder",
    "DUFFING_HOLDOUT",
    "DUFFING_PARAMS",
    "TRAIN_STREAM",
    "VAL_STREAM",
    "MechanicalFamilyConfig",
    "SystemBatch",
    "batch_generator",
    "context_tokens",
    "in_duffing_holdout",
    "mechanical_field",
    "sample_batch",
    "sample_params",
    "simulate_q",
    "FreeCodeHead",
    "StructuredCodeHead",
    "matched_free_hidden",
    "mlp_numel",
    "ContextField",
    "build_context_field",
    "tokens_from_q",
]
