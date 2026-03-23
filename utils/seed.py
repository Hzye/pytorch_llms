"""
Centralised seeding for reproducibility.

Usage:
    from utils.seed import set_seed
    set_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all RNGs used by the training stack.

    Args:
        seed:          Integer seed (e.g. cfg.seed).
        deterministic: Forces cuDNN determinism for exact reproducibility.
                       Set False for faster training when exact reproducibility
                       isn't needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic