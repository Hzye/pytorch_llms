"""
Checkpoint saving and loading.

Saves a complete training snapshot so a run can be resumed exactly where
it left off.

Layout:
    runs/exp_001/checkpoints/
    ├── epoch_001.pt
    ├── epoch_002.pt
    ├── best.pt       ← best validation loss
    └── latest.pt     ← most recent epoch

Usage:
    from utils.checkpoint import save_checkpoint, load_checkpoint

    save_checkpoint(run_dir / "checkpoints", model, optimiser, scheduler,
                    epoch=epoch, step=step, loss=val_loss, cfg=cfg, is_best=True)

    state = load_checkpoint("runs/exp_001/checkpoints/best.pt", model, optimiser)
    start_epoch = state["epoch"] + 1
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    loss: float,
    cfg: Optional[Any] = None,
    is_best: bool = False,
) -> Path:
    """Save model, optimiser, and scheduler state to disk."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if cfg is not None:
        state["cfg"] = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg

    epoch_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
    torch.save(state, epoch_path)
    log.info("Checkpoint saved → %s (loss=%.4f)", epoch_path, loss)

    shutil.copy2(epoch_path, checkpoint_dir / "latest.pt")

    if is_best:
        shutil.copy2(epoch_path, checkpoint_dir / "best.pt")
        log.info("New best checkpoint (loss=%.4f)", loss)

    return epoch_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimiser: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """
    Restore a training snapshot from disk.

    Returns the raw state dict so callers can read epoch, step, loss, cfg, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device or torch.device("cpu"), weights_only=False)

    model.load_state_dict(state["model_state_dict"])
    log.info("Loaded checkpoint from %s (epoch %d)", path, state.get("epoch", -1))

    if optimiser is not None and "optimiser_state_dict" in state:
        optimiser.load_state_dict(state["optimiser_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return state