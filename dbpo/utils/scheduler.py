"""
Learning-rate schedulers.
No upstream imports.
"""
from __future__ import annotations
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with linear warmup and optional restarts.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        return [
            base_lr
            + (self.max_lr - base_lr)
            * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                            / (self.cur_cycle_steps - self.warmup_steps)))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class CustomScheduler:
    """
    Simple constant-with-warmup or cosine schedule.
    Supports reset() for visualization before training starts.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = "constant_warmup",
        max: float = 1e-3,
        min: float = 1e-5,
        warmup_steps: int = 0,
        hold_steps: int = 0,
        anneal_steps: int = 1000,
    ):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.max_lr = max
        self.min_lr = min
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.anneal_steps = anneal_steps
        self._step = 0
        self._set_lr(min)

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self):
        s = self._step
        if self.schedule_type == "constant_warmup":
            if s < self.warmup_steps:
                lr = self.min_lr + (self.max_lr - self.min_lr) * s / max(self.warmup_steps, 1)
            else:
                lr = self.max_lr
        elif self.schedule_type == "cosine":
            if s < self.warmup_steps:
                lr = self.min_lr + (self.max_lr - self.min_lr) * s / max(self.warmup_steps, 1)
            elif s < self.warmup_steps + self.hold_steps:
                lr = self.max_lr
            else:
                t = s - self.warmup_steps - self.hold_steps
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1 + math.cos(math.pi * min(t, self.anneal_steps) / self.anneal_steps)
                )
        else:
            lr = self.max_lr
        self._set_lr(lr)
        self._step += 1

    def reset(self):
        self._step = 0
        self._set_lr(self.min_lr)

    def state_dict(self):
        return {"_step": self._step}

    def load_state_dict(self, state: dict):
        self._step = state["_step"]
