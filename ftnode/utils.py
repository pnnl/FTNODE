from tqdm.auto import tqdm 
import torch
import numpy as np
import random
import dataclasses
import typing
import yaml

def _load_loop_wrapper(show_progress:bool):
    if show_progress:
        return tqdm
    return lambda x: x 


def set_global_seed(seed: int, deterministic: bool = True):
    """
    Set seeds for reproducibility.
    
    Args:
        seed (int): Random seed.
        deterministic (bool): 
            If True, enforce deterministic algorithms (slower but reproducible).
            If False, allow faster algorithms (results may differ slightly).
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Strict reproducibility (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[Seed] Deterministic mode enabled (may reduce speed).")
    else:
        # Faster training, reproducibility not bitwise guaranteed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        print("[Seed] Non-deterministic (fast) mode enabled.")


def save_config(cfg, path):
    """Write a frozen-dataclass config to YAML.

    The duffing work configures runs through dataclasses
    (:class:`ftnode.systems.DuffingDataConfig`,
    :class:`ftnode.latent.KappaBudget`, :class:`ftnode.train.TrainConfig`,
    :class:`ftnode.control.ControlConfig`); this records exactly which one
    produced a given checkpoint, since the checkpoints themselves are bare state
    dicts that carry no settings.

    Args:
        cfg: Any dataclass instance.
        path (str | pathlib.Path): Destination ``.yaml`` file.
    """
    if not dataclasses.is_dataclass(cfg):
        raise TypeError(f"save_config expects a dataclass instance, got {type(cfg).__name__}")
    with open(path, "w") as fh:
        yaml.safe_dump(dataclasses.asdict(cfg), fh, sort_keys=False)


def _from_dict(cls, data):
    """Build ``cls`` from a plain dict, recursing into dataclass-typed fields.

    Unknown keys raise at every level rather than being dropped, so a config
    written by an older version fails loudly instead of silently falling back to
    defaults.

    The type lookup goes through ``typing.get_type_hints`` rather than
    ``field.type``: the config modules use ``from __future__ import annotations``,
    which makes every annotation a *string*, so ``field.type`` would be
    ``"EncoderConfig"`` and no nested section would ever be recognized.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__} has no fields {sorted(unknown)}")

    hints = typing.get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        hint = hints.get(key)
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            kwargs[key] = _from_dict(hint, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(cls, path):
    """Read a YAML file back into the dataclass ``cls``.

    Nested dataclass fields are reconstructed recursively, so a config split into
    sections -- :class:`ftnode.latent.LatentModelConfig` and its ``encoder`` /
    ``operator`` / ``equilibrium`` sub-configs -- round-trips through
    :func:`save_config` unchanged.  ``dataclasses.asdict`` already recurses on the
    way out.

    Args:
        cls (type): The dataclass to construct.
        path (str | pathlib.Path): Source ``.yaml`` file.

    Returns:
        An instance of ``cls``.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return _from_dict(cls, data)
