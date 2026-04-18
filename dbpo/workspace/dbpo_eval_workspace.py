"""
DBPO evaluation workspace.

Loads a checkpoint, runs the env_runner, and logs results.
Works for both pre-trained DBP and fine-tuned DBPO policies.

Logging namespace:
  eval/mean_score
  eval/success_rate
  eval/avg_episode_reward
  eval/avg_episode_length
"""
from __future__ import annotations

import logging
import os

import torch

from dbpo.utils.logging_util import init_wandb, log_metrics, finish_wandb, create_bordered_text

log = logging.getLogger(__name__)


class DBPOEvalWorkspace:
    """
    Standalone evaluation workspace for DBP and DBPO policies.

    Config contract
    ---------------
    cfg.policy          : LowdimDBPPolicy, ImageDBPPolicy, or PointCloudDBPPolicy
    cfg.env_runner      : evaluation runner
    cfg.checkpoint_path : path to .pt checkpoint
    cfg.use_ema         : bool — load EMA weights if available
    cfg.logdir          : output directory
    cfg.device          : torch device string
    cfg.wandb           : WandB config (optional)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.logdir = cfg.logdir
        os.makedirs(self.logdir, exist_ok=True)
        init_wandb(cfg.get("wandb", None))

        import hydra
        self.policy = hydra.utils.instantiate(cfg.policy).to(self.device)
        self.env_runner = hydra.utils.instantiate(cfg.env_runner)

        # Load checkpoint
        ckpt_path = cfg.checkpoint_path
        use_ema = cfg.get("use_ema", True)
        self._load_checkpoint(ckpt_path, use_ema=use_ema)

    def run(self):
        self.policy.eval()
        log.info("Running evaluation with %s", self.policy.__class__.__name__)
        with torch.no_grad():
            log_data = self.env_runner.run(self.policy)

        eval_metrics = {f"eval/{k}": v for k, v in log_data.items()}
        log_metrics(eval_metrics, step=0)
        log.info(create_bordered_text(
            "Evaluation Results\n" +
            "\n".join(
                f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}"
                for k, v in eval_metrics.items()
                if not k.endswith("video")
            )
        ))
        finish_wandb()
        return eval_metrics

    def _load_checkpoint(self, path: str, use_ema: bool = True):
        log.info("Loading checkpoint from %s (use_ema=%s)", path, use_ema)
        data = torch.load(path, map_location=self.device, weights_only=True)

        # Try EMA weights first
        if use_ema and "ema" in data and data["ema"] is not None:
            self.policy.load_state_dict(data["ema"])
            log.info("Loaded EMA weights.")
        elif "model" in data:
            # For DBPOPPOWrapper checkpoints, extract actor policy weights
            state_dict = {}
            for k, v in data["model"].items():
                if k.startswith("actor_ft.policy."):
                    state_dict[k.replace("actor_ft.policy.", "")] = v
                elif k.startswith("actor_old."):
                    state_dict[k.replace("actor_old.", "")] = v
            if state_dict:
                self.policy.load_state_dict(state_dict, strict=False)
                log.info("Loaded actor policy weights from DBPOPPOWrapper checkpoint.")
            else:
                self.policy.load_state_dict(data["model"])
                log.info("Loaded model weights.")
        elif "policy" in data:
            self.policy.load_state_dict(data["policy"])
            log.info("Loaded policy weights.")
        else:
            raise ValueError(f"Checkpoint at {path} has no recognised weight keys.")
