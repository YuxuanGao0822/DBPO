"""
Unified logging utilities for DBPO.

All metrics are emitted under two namespaces:
  train/*   — per-step training scalars
  eval/*    — per-evaluation episode metrics

WandB is the primary backend; a plain-text fallback is always active.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_wandb_run = None  # module-level handle


def init_wandb(cfg) -> None:
    """Initialise a WandB run from a Hydra config node."""
    global _wandb_run
    if cfg is None:
        return
    try:
        import wandb
        offline = cfg.get("offline_mode", False)
        run_dir = cfg.get("dir", "./wandb_offline" if offline else "./wandb")
        os.makedirs(run_dir, exist_ok=True)
        from omegaconf import OmegaConf
        _wandb_run = wandb.init(
            entity=cfg.get("entity", None),
            project=cfg.get("project", "dbpo"),
            name=cfg.get("run", None),
            config=OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "_metadata") else dict(cfg),
            mode="offline" if offline else "online",
            dir=run_dir,
        )
        log.info("WandB initialised: run_id=%s", wandb.run.id)
    except Exception as exc:
        log.warning("WandB init failed (%s). Continuing without WandB.", exc)


def log_metrics(metrics: Dict[str, Any], step: int, commit: bool = True) -> None:
    """
    Log a flat dict of scalars.

    Keys should already carry their namespace prefix, e.g.
      {"train/loss": 0.42, "train/dbp_global_scale": 0.1}
    """
    # Always log to Python logger
    msg = "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items())
    log.info("[step %d] %s", step, msg)

    # WandB
    if _wandb_run is not None:
        try:
            import wandb
            wandb.log(data=metrics, step=step, commit=commit)
        except Exception as exc:
            log.warning("WandB log failed: %s", exc)


def finish_wandb() -> None:
    if _wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


def create_bordered_text(text: str) -> str:
    """Wrap text in a simple ASCII border for console readability."""
    lines = text.splitlines()
    width = max(len(l) for l in lines) + 4
    border = "=" * width
    padded = [f"| {l:<{width - 4}} |" for l in lines]
    return "\n".join([border] + padded + [border])
