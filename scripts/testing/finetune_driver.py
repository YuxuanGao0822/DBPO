#!/usr/bin/env python3
"""Server-test helpers for stage-2 DBPO finetune."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_FINETUNE_CHECKPOINT_KEYS = (
    "itr",
    "cnt_train_steps",
    "model",
    "actor_optimizer",
    "critic_optimizer",
    "actor_lr_scheduler",
    "critic_lr_scheduler",
)


def _load_cfg(task: str, actor_policy_path: str, overrides: Iterable[str] | None = None):
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = hydra.compose(
            config_name="finetune",
            overrides=[f"task={task}", f"actor_policy_path={actor_policy_path}", *(overrides or [])],
            return_hydra_config=False,
        )
    return cfg


def _write_report(path: Path, text: str) -> None:
    if str(path) == "-":
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_compose(task: str, actor_policy_path: str, report: Path) -> None:
    from omegaconf import OmegaConf

    cfg = _load_cfg(
        task,
        actor_policy_path=actor_policy_path,
        overrides=["wandb.offline_mode=true"],
    )
    required_top_level = ("model", "env", "train", "task", "actor_policy_path")
    missing = [key for key in required_top_level if key not in cfg]
    required_task = (
        "name",
        "supports_finetune",
        "finetune_actor",
        "env_cfg",
        "normalization_path",
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
    actor_policy_path: str,
    report: Path,
    device: str,
    n_envs_override: int,
    n_steps_override: int,
    batch_size_override: int,
    n_train_itr_override: int,
) -> None:
    import hydra
    import torch

    overrides = [
        "wandb.offline_mode=true",
        f"device={device}",
        f"logdir={REPO_ROOT / 'reports' / 'instantiate_finetune_runtime' / task}",
        f"hydra.run.dir={REPO_ROOT / 'reports' / 'instantiate_finetune_runtime' / task}",
        f"task.env_cfg.n_envs={n_envs_override}",
        f"train.n_steps={n_steps_override}",
        f"train.batch_size={batch_size_override}",
        f"train.n_train_itr={n_train_itr_override}",
        "train.update_epochs=1",
        "train.val_freq=1",
        "train.save_model_freq=1",
        "train.n_critic_warmup_itr=0",
    ]
    cfg = _load_cfg(task, actor_policy_path=actor_policy_path, overrides=overrides)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        workspace = hydra.utils.instantiate(cfg, _recursive_=False)
        obs_venv = workspace.venv.reset_arg([{} for _ in range(workspace.n_envs)])
        cond = {
            "state": torch.tensor(
                obs_venv["state"], device=workspace.device, dtype=torch.float32
            )
        }
        actions, chains, logprobs = workspace.model.get_actions(
            cond=cond,
            eval_mode=True,
            save_chains=True,
            normalize_act_space_dimension=workspace.normalize_act_space_dim,
            clip_intermediate_actions=workspace.clip_intermediate_actions,
        )
        action_venv = actions.cpu().numpy()[:, : workspace.cfg.act_steps]
        step_result = workspace.venv.step(action_venv)
        obs_next, rewards, terminated, truncated, infos = step_result
        result = {
            "task": task,
            "status": "OK",
            "workspace_class": workspace.__class__.__name__,
            "model_class": workspace.model.__class__.__name__,
            "n_envs": workspace.n_envs,
            "obs_shape": list(obs_venv["state"].shape),
            "action_shape": list(actions.shape),
            "chains_shape": list(chains.shape),
            "logprob_shape": list(logprobs.shape),
            "step_reward_shape": list(rewards.shape),
            "step_terminated_shape": list(terminated.shape),
            "step_truncated_shape": list(truncated.shape),
            "next_obs_shape": list(obs_next["state"].shape),
            "info_len": len(infos),
            "resolved_name": cfg.task.name,
            "resolved_env_type": cfg.env.env_type,
            "actor_policy_path": actor_policy_path,
            "logdir": str(cfg.logdir),
        }
        if hasattr(workspace.venv, "close"):
            workspace.venv.close()
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
    missing = [key for key in REQUIRED_FINETUNE_CHECKPOINT_KEYS if key not in data]
    model_keys = tuple(data.get("model", {}).keys()) if isinstance(data.get("model"), dict) else ()
    result = {
        "checkpoint": str(checkpoint),
        "expected_task": expected_task,
        "missing_keys": missing,
        "has_actor_old": any(key.startswith("actor_old.") for key in model_keys),
        "has_actor_ft_policy": any(key.startswith("actor_ft.policy.") for key in model_keys),
        "has_critic": any(key.startswith("critic.") for key in model_keys),
        "status": "OK"
        if not missing
        and any(key.startswith("actor_old.") for key in model_keys)
        and any(key.startswith("actor_ft.policy.") for key in model_keys)
        and any(key.startswith("critic.") for key in model_keys)
        else "FAILED",
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if report is None:
        print(text)
    else:
        _write_report(report, text + "\n")
    if result["status"] != "OK":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 server testing helper for DBPO finetune.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    compose_parser = subparsers.add_parser("compose", help="Compose and resolve a finetune config.")
    compose_parser.add_argument("--task", required=True)
    compose_parser.add_argument("--actor-policy-path", required=True)
    compose_parser.add_argument("--report", required=True)

    instantiate_parser = subparsers.add_parser(
        "instantiate",
        help="Instantiate finetune workspace, vectorized env, and one-step actor interaction.",
    )
    instantiate_parser.add_argument("--task", required=True)
    instantiate_parser.add_argument("--actor-policy-path", required=True)
    instantiate_parser.add_argument("--report", required=True)
    instantiate_parser.add_argument("--device", default="cpu")
    instantiate_parser.add_argument("--n-envs-override", type=int, default=2)
    instantiate_parser.add_argument("--n-steps-override", type=int, default=2)
    instantiate_parser.add_argument("--batch-size-override", type=int, default=4)
    instantiate_parser.add_argument("--n-train-itr-override", type=int, default=1)

    checkpoint_parser = subparsers.add_parser(
        "verify-checkpoint",
        help="Verify stage-2 finetune checkpoint contract.",
    )
    checkpoint_parser.add_argument("--checkpoint", required=True)
    checkpoint_parser.add_argument("--task", required=True)
    checkpoint_parser.add_argument("--report", default=None)

    args = parser.parse_args()

    if args.cmd == "compose":
        run_compose(
            task=args.task,
            actor_policy_path=args.actor_policy_path,
            report=Path(args.report),
        )
        return
    if args.cmd == "instantiate":
        run_instantiate(
            task=args.task,
            actor_policy_path=args.actor_policy_path,
            report=Path(args.report),
            device=args.device,
            n_envs_override=args.n_envs_override,
            n_steps_override=args.n_steps_override,
            batch_size_override=args.batch_size_override,
            n_train_itr_override=args.n_train_itr_override,
        )
        return
    if args.cmd == "verify-checkpoint":
        verify_checkpoint(
            checkpoint=Path(args.checkpoint),
            expected_task=args.task,
            report=None if args.report is None else Path(args.report),
        )
        return


if __name__ == "__main__":
    main()
