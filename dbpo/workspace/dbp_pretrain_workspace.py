"""
DBP supervised pre-training workspace.

Handles all benchmark suites: Robomimic, D4RL, Adroit, MetaWorld, PushT,
and FrankaKitchen.

Training objective: compute_dbp_loss (single-step, no iterative sampling).

Logging namespace:
  train/loss          — mean DBP loss per batch
  train/dbp_*         — per-temperature force magnitudes and global scale
  train/lr            — current learning rate
  eval/mean_score     — mean max reward from env runner (if configured)
  eval/success_rate   — task success rate (if runner provides it)
"""
from __future__ import annotations

import logging
import os
import random
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from dbpo.model.common.ema_model import EMAModel
from dbpo.utils.logging_util import init_wandb, log_metrics, finish_wandb, create_bordered_text
from dbpo.utils.scheduler import CosineAnnealingWarmupRestarts

log = logging.getLogger(__name__)


class DBPPretrainWorkspace:
    """
    DBP supervised pre-training loop.

    Instantiated entirely from Hydra config via hydra.utils.instantiate.
    The policy, dataset, and optional env_runner are all config-driven.

    Config contract
    ---------------
    cfg.policy          : LowdimDBPPolicy, ImageDBPPolicy, or PointCloudDBPPolicy
    cfg.train_dataset   : any BaseLowdimDataset / BaseImageDataset
    cfg.train           : training hyperparameters (see pretrain.yaml)
    cfg.wandb           : WandB config (optional)
    cfg.env_runner      : evaluation runner (optional)
    cfg.logdir          : output directory
    cfg.seed            : random seed
    cfg.device          : torch device string
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = OmegaConf.create(kwargs)
        elif kwargs:
            merged = OmegaConf.create(kwargs)
            cfg = OmegaConf.merge(cfg, merged)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Reproducibility
        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Logging
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.logdir, "render"), exist_ok=True)
        init_wandb(cfg.get("wandb", None))

        # Build policy
        import hydra
        self.policy: nn.Module = hydra.utils.instantiate(cfg.policy).to(self.device)
        log.info("Policy: %s  params=%.2fM",
                 self.policy.__class__.__name__,
                 sum(p.numel() for p in self.policy.parameters()) / 1e6)

        # EMA
        ema_cfg = cfg.get("ema", None)
        self.ema_policy: Optional[nn.Module] = None
        if ema_cfg is not None:
            self.ema_policy = deepcopy(self.policy)
            self.ema_policy.eval()
            self.ema_policy.requires_grad_(False)
            self.ema = EMAModel(
                model=self.ema_policy,
                update_after_step=ema_cfg.get("update_after_step", 0),
                inv_gamma=ema_cfg.get("inv_gamma", 1.0),
                power=ema_cfg.get("power", 0.75),
                min_value=ema_cfg.get("min_value", 0.0),
                max_value=ema_cfg.get("max_value", 0.9999),
            )

        # Dataset
        self.dataset = hydra.utils.instantiate(cfg.train_dataset)
        log.info("Dataset: %s  len=%d", self.dataset.__class__.__name__, len(self.dataset))

        num_workers = cfg.train.get("num_workers", 4)
        batch_size = cfg.train.batch_size
        # Point-cloud validation runs may intentionally use large batch sizes
        # that exceed the small released dataset size. In that case, keep the
        # last partial batch instead of silently producing an empty epoch.
        drop_last = len(self.dataset) >= batch_size
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
            drop_last=drop_last,
        )

        # Fit normalizer from dataset
        if hasattr(self.dataset, "get_normalizer"):
            normalizer = self.dataset.get_normalizer()
            self.policy.set_normalizer(normalizer)
            if self.ema_policy is not None:
                self.ema_policy.set_normalizer(normalizer)
            # Loading a fresh normalizer creates new CPU ParameterDict entries.
            # Move policy trees back to the requested runtime device afterwards.
            self.policy = self.policy.to(self.device)
            if self.ema_policy is not None:
                self.ema_policy = self.ema_policy.to(self.device)
            log.info("Normalizer fitted from dataset.")

        # Optimizer & scheduler
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.get("weight_decay", 1e-6),
        )
        lr_cfg = cfg.train.lr_scheduler
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=lr_cfg.first_cycle_steps,
            max_lr=cfg.train.learning_rate,
            min_lr=lr_cfg.min_lr,
            warmup_steps=lr_cfg.warmup_steps,
        )

        # Optional env runner for periodic evaluation
        self.env_runner = None
        if cfg.get("env_runner", None) is not None:
            self.env_runner = hydra.utils.instantiate(cfg.env_runner)

        # Training state
        self.n_epochs = cfg.train.n_epochs
        self.save_model_freq = cfg.train.save_model_freq
        self.val_freq = cfg.train.get("val_freq", 100)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.update_ema_freq = cfg.train.get("update_ema_freq", 10)
        self.epoch_start_ema = cfg.train.get("epoch_start_ema", 20)
        self.epoch = 0
        self.global_step = 0

        # Resume
        resume_path = cfg.get("base_policy_path", None)
        if resume_path is not None:
            self._load(resume_path)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def run(self):
        log.info("Starting pre-training for %d epochs.", self.n_epochs)
        for epoch in range(self.epoch, self.epoch + self.n_epochs):
            self.epoch = epoch
            self.policy.train()
            epoch_losses = []
            metrics = {}

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False):
                batch = self._batch_to_device(batch)
                self.optimizer.zero_grad()
                loss, metrics = self.policy.compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
                self.optimizer.step()

                # EMA update
                if self.ema_policy is not None and self.global_step % self.update_ema_freq == 0:
                    if epoch >= self.epoch_start_ema:
                        self.ema.step(self.policy)
                    else:
                        self.ema_policy.load_state_dict(self.policy.state_dict())

                epoch_losses.append(loss.item())
                self.global_step += 1

            self.lr_scheduler.step()
            if not epoch_losses:
                raise RuntimeError(
                    "Empty training epoch: dataloader yielded no batches. "
                    "Check dataset length, batch size, and drop_last behavior."
                )
            mean_loss = float(np.mean(epoch_losses))

            # --- Logging ---
            if epoch % self.log_freq == 0:
                train_metrics = {
                    "train/loss": mean_loss,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                }
                # Propagate last-batch DBP diagnostics under train/dbp_*
                for k, v in metrics.items():
                    train_metrics[f"train/dbp_{k}"] = float(v) if hasattr(v, "item") else v
                log_metrics(train_metrics, step=self.global_step)

            # --- Periodic evaluation ---
            if self.env_runner is not None and epoch % self.val_freq == 0:
                eval_policy = self.ema_policy if self.ema_policy is not None else self.policy
                eval_policy.eval()
                with torch.no_grad():
                    eval_log = self.env_runner.run(eval_policy)
                eval_metrics = {f"eval/{k}": v for k, v in eval_log.items()}
                log_metrics(eval_metrics, step=self.global_step)
                log.info(create_bordered_text(
                    f"Eval epoch {epoch}\n" +
                    "\n".join(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}"
                              for k, v in eval_metrics.items())
                ))
                self.policy.train()

            # --- Checkpointing ---
            self._save_last()
            if epoch % self.save_model_freq == 0 or epoch == self.epoch + self.n_epochs - 1:
                self._save(epoch)

        finish_wandb()
        log.info("Pre-training complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _batch_to_device(self, batch):
        """Recursively move batch tensors to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self._batch_to_device(v) for k, v in batch.items()}
        if hasattr(batch, "_fields"):  # namedtuple (StitchedSequenceDataset)
            return type(batch)(*[self._batch_to_device(f) for f in batch])
        return batch

    def _save(self, epoch: int):
        data = self._build_checkpoint_payload(epoch)
        path = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        torch.save(data, path)
        log.info("Saved checkpoint to %s", path)

    def _save_last(self):
        data = self._build_checkpoint_payload(self.epoch)
        torch.save(data, os.path.join(self.checkpoint_dir, "last.pt"))

    def _load(self, path: str):
        log.info("Resuming from %s", path)
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.epoch = data.get("epoch", 0) + 1
        self.global_step = data.get("global_step", 0)
        self.policy.load_state_dict(data["model"])
        if self.ema_policy is not None and data.get("ema") is not None:
            self.ema_policy.load_state_dict(data["ema"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        if "lr_scheduler" in data:
            self.lr_scheduler.load_state_dict(data["lr_scheduler"])
        log.info("Resumed from epoch %d, step %d", self.epoch, self.global_step)

    def _build_checkpoint_payload(self, epoch: int) -> dict:
        task_cfg = self.cfg.get("task", OmegaConf.create())
        normalizer_state = None
        if hasattr(self.policy, "normalizer") and hasattr(self.policy.normalizer, "state_dict"):
            normalizer_state = self.policy.normalizer.state_dict()

        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.policy.state_dict(),
            "ema": self.ema_policy.state_dict() if self.ema_policy is not None else None,
            "normalizer": normalizer_state,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "task_name": task_cfg.get("name", "unknown"),
            "task_family": task_cfg.get("env_suite", "unknown"),
            "modality": task_cfg.get("modality", "unknown"),
            "policy_class": self.policy.__class__.__name__,
            "obs_dim": task_cfg.get("obs_dim", None),
            "action_dim": task_cfg.get("action_dim", None),
            "horizon": task_cfg.get("horizon_steps", None),
            "n_obs_steps": task_cfg.get("n_obs_steps", None),
            "n_action_steps": task_cfg.get("n_action_steps", None),
        }
