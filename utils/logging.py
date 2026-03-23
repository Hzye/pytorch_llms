"""
Logging setup for training runs.

Call setup_logging once at the start of your script. Then use the standard
logging.getLogger(__name__) pattern anywhere in the codebase.

Usage:
    from utils.logging import setup_logging
    import logging

    setup_logging(log_dir=run_dir / "logs")

    log = logging.getLogger(__name__)
    log.info("epoch=%d  loss=%.4f", epoch, loss)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured — prevent duplicate handlers in tests
    """
    Attach a console handler and (optionally) a file handler to the root logger.
    Safe to call multiple times — subsequent calls are no-ops.

    Args:
        log_dir: Directory to write train.log into. Pass None for console-only.
        level:   Minimum log level (default: INFO).
    """
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir is not None:
        log_path = Path(log_dir) / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)