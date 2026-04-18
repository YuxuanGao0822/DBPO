#!/usr/bin/env python3
"""Server-test helpers for stage-1 DBP pretrain."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_CHECKPOINT_KEYS = (
    "model",
    "ema",
    "normalizer",
    "optimizer",
    "lr_scheduler",
    "epoch",
    "global_step",
    "task_name",
    "task_family",
    "modality",
    "policy_class",
    "obs_dim",
    "action_dim",
    "horizon",
    "n_obs_steps",
    "n_action_steps",
)


def _load_cfg(task: str, overrides: Iterable[str] | None = None):
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = hydra.compose(
            config_name="pretrain",
            overrides=[f"task={task}", *(overrides or [])],
            return_hydra_config=False,
        )
    return cfg


def _write_report(path: Path, text: str) -> None:
    if str(path) == "-":
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolve_dataset_path(cfg) -> str | None:
    train_dataset = cfg.get("train_dataset", None)
    if train_dataset is None:
        return None
    for key in ("zarr_path", "hdf5_path", "dataset_path", "path", "dataset_dir"):
        value = train_dataset.get(key, None)
        if value is not None:
            value = str(value).strip()
            return value if value else ""
    task_cfg = cfg.get("task", None)
    if task_cfg is not None:
        for key in ("offline_dataset_path", "env_dataset_path", "dataset_path", "dataset_dir", "path"):
            value = task_cfg.get(key, None)
            if value is not None:
                value = str(value).strip()
                return value if value else ""
    return None


def _dataset_field_name(cfg) -> str | None:
    train_dataset = cfg.get("train_dataset", None)
    if train_dataset is not None:
        for key in ("zarr_path", "hdf5_path", "dataset_path", "path", "dataset_dir"):
            value = train_dataset.get(key, None)
            if value is not None:
                return key
    task_cfg = cfg.get("task", None)
    if task_cfg is not None:
        for key in ("offline_dataset_path", "env_dataset_path", "dataset_path", "dataset_dir", "path"):
            value = task_cfg.get(key, None)
            if value is not None:
                return key
    return None


def _legacy_dataset_candidates(raw_path: str) -> list[str]:
    candidates: list[str] = []
    if "/blockpush/" in raw_path:
        candidates.append(raw_path.replace("/blockpush/", "/block_pushing/"))
    return candidates


def dataset_status_dict(task: str, overrides: Iterable[str] | None = None) -> dict:
    cfg = _load_cfg(task, overrides=overrides)
    raw_path = _resolve_dataset_path(cfg)
    if raw_path is None:
        return {
            "task": task,
            "status": "UNSPECIFIED",
            "dataset_path": None,
            "reason": "train_dataset has no explicit path field",
        }

    expanded = os.path.expanduser(raw_path)
    if not expanded:
        return {
            "task": task,
            "status": "MISSING",
            "dataset_path": "",
            "reason": "resolved dataset path is empty",
        }

    exists = Path(expanded).exists()
    resolved_reason = None
    if not exists:
        for candidate in _legacy_dataset_candidates(expanded):
            if Path(candidate).exists():
                expanded = candidate
                exists = True
                resolved_reason = "resolved via legacy block_pushing layout"
                break
    return {
        "task": task,
        "status": "AVAILABLE" if exists else "MISSING",
        "dataset_path": expanded,
        "reason": resolved_reason if exists else "dataset path does not exist",
    }


def run_compose(task: str, report: Path) -> None:
    from omegaconf import OmegaConf

    cfg = _load_cfg(task, overrides=["wandb.offline_mode=true", "train.num_workers=0"])
    required_top_level = ("policy", "train_dataset", "env_runner", "task", "train")
    missing = [key for key in required_top_level if key not in cfg]
    required_task = (
        "name",
        "modality",
        "dataset_source",
        "supports_pretrain",
        "supports_finetune",
        "train_dataset",
        "policy",
    )
    missing_task = [key for key in required_task if key not in cfg.task]
    text = [
        f"TASK={task}",
        "STATUS=OK",
        f"MISSING_TOP_LEVEL={missing}",
        f"MISSING_TASK_KEYS={missing_task}",
        "RESOLVED_CONFIG_BEGIN",
        OmegaConf.to_yaml(cfg, resolve=True),
        "RESOLVED_CONFIG_END",
    ]
    _write_report(report, "\n".join(text))


def run_instantiate(
    task: str,
    report: Path,
    device: str,
    batch_size_override: int | None,
    skip_env_runner: bool,
) -> None:
    import hydra
    import torch

    overrides = [
        "wandb.offline_mode=true",
        f"device={device}",
        "train.num_workers=0",
        f"logdir={REPO_ROOT / 'reports' / 'instantiate_runtime' / task}",
    ]
    if batch_size_override is not None:
        overrides.append(f"train.batch_size={batch_size_override}")

    cfg = _load_cfg(task, overrides=overrides)
    if skip_env_runner:
        cfg.env_runner = None
    dataset_status = dataset_status_dict(task, overrides=overrides)
    if dataset_status["status"] != "AVAILABLE":
        _write_report(report, json.dumps({
            "task": task,
            "status": "SKIPPED",
            "reason": dataset_status["reason"],
            "dataset_path": dataset_status["dataset_path"],
        }, indent=2, sort_keys=True) + "\n")
        return
    dataset_path = dataset_status["dataset_path"]
    field_name = _dataset_field_name(cfg)
    if dataset_path and field_name is not None:
        train_dataset = cfg.get("train_dataset", None)
        if train_dataset is not None and field_name in train_dataset:
            train_dataset[field_name] = dataset_path
        task_cfg = cfg.get("task", None)
        if task_cfg is not None and field_name in task_cfg:
            task_cfg[field_name] = dataset_path
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        workspace = hydra.utils.instantiate(cfg, _recursive_=False)
        batch = next(iter(workspace.dataloader))
        batch = workspace._batch_to_device(batch)
        loss, metrics = workspace.policy.compute_loss(batch)
        result = {
            "task": task,
            "status": "OK",
            "policy_class": workspace.policy.__class__.__name__,
            "dataset_class": workspace.dataset.__class__.__name__,
            "runner_class": None if workspace.env_runner is None else workspace.env_runner.__class__.__name__,
            "loss": float(loss.item()),
            "metrics": {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()},
            "resolved_name": cfg.task.name,
            "resolved_modality": cfg.task.modality,
            "resolved_dataset_source": cfg.task.dataset_source,
            "logdir": str(cfg.logdir),
        }
        del workspace
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_report(
        report,
        "INSTANTIATE_RESULT_BEGIN\n"
        + json.dumps(result, indent=2, sort_keys=True)
        + "\nINSTANTIATE_RESULT_END\n"
        + buffer.getvalue(),
    )


def verify_checkpoint(checkpoint: Path, expected_task: str, report: Path | None) -> None:
    import torch

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in data]
    result = {
        "checkpoint": str(checkpoint),
        "expected_task": expected_task,
        "task_name": data.get("task_name"),
        "task_family": data.get("task_family"),
        "policy_class": data.get("policy_class"),
        "missing_keys": missing,
        "status": "OK" if not missing and data.get("task_name") == expected_task else "FAILED",
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if report is None:
        print(text)
    else:
        _write_report(report, text + "\n")
    if result["status"] != "OK":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 server testing helper for DBP pretrain.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    compose_parser = subparsers.add_parser("compose", help="Compose and resolve a task config.")
    compose_parser.add_argument("--task", required=True)
    compose_parser.add_argument("--report", required=True)

    instantiate_parser = subparsers.add_parser("instantiate", help="Instantiate workspace and run one loss pass.")
    instantiate_parser.add_argument("--task", required=True)
    instantiate_parser.add_argument("--report", required=True)
    instantiate_parser.add_argument("--device", default="cpu")
    instantiate_parser.add_argument("--batch-size-override", type=int, default=None)
    instantiate_parser.add_argument("--skip-env-runner", action="store_true")

    checkpoint_parser = subparsers.add_parser("verify-checkpoint", help="Verify checkpoint metadata contract.")
    checkpoint_parser.add_argument("--checkpoint", required=True)
    checkpoint_parser.add_argument("--task", required=True)
    checkpoint_parser.add_argument("--report", default=None)

    dataset_parser = subparsers.add_parser("dataset-status", help="Report whether a task dataset path exists.")
    dataset_parser.add_argument("--task", required=True)
    dataset_parser.add_argument("--report", default=None)

    args = parser.parse_args()

    if args.cmd == "compose":
        run_compose(task=args.task, report=Path(args.report))
        return
    if args.cmd == "instantiate":
        run_instantiate(
            task=args.task,
            report=Path(args.report),
            device=args.device,
            batch_size_override=args.batch_size_override,
            skip_env_runner=args.skip_env_runner,
        )
        return
    if args.cmd == "verify-checkpoint":
        verify_checkpoint(
            checkpoint=Path(args.checkpoint),
            expected_task=args.task,
            report=None if args.report is None else Path(args.report),
        )
        return
    if args.cmd == "dataset-status":
        result = dataset_status_dict(task=args.task)
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.report is None:
            print(text)
        else:
            _write_report(Path(args.report), text + "\n")
        if result["status"] == "AVAILABLE":
            return
        raise SystemExit(10)


if __name__ == "__main__":
    main()
