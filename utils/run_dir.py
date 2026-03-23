"""
Timestamped experiment directory management.

Creates a unique directory for each run with a consistent internal layout.

Layout:
    runs/
    └── transformer__2024-03-21_14-30-22/
        ├── checkpoints/
        ├── logs/
        └── outputs/

Usage:
    from utils.run_dir import make_run_dir

    run_dir = make_run_dir("runs", "transformer_small")
    setup_logging(log_dir=run_dir / "logs")
    save_checkpoint(run_dir / "checkpoints", ...)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def make_run_dir(base_dir: str | Path = "runs", experiment_name: str = "exp") -> Path:
    """
    Create a timestamped run directory with checkpoints/, logs/, and outputs/ inside.

    Args:
        base_dir:        Root directory for all runs.
        experiment_name: Human-readable label, e.g. "transformer_small".

    Returns:
        Path to the newly created run directory.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"{experiment_name}__{stamp}"

    for subdir in ("checkpoints", "logs", "outputs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    log.info("Run directory: %s", run_dir)
    return run_dir