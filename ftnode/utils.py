from tqdm.auto import tqdm 
import torch
import numpy as np
import random

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
