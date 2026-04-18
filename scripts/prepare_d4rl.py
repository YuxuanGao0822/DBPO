#!/usr/bin/env python3
"""Prepare DBPO D4RL locomotion datasets using the D4RL processing contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RELEASED_D4RL_ENVS = (
    "hopper-medium-v2",
    "ant-medium-expert-v2",
    "walker2d-medium-v2",
)


def probe_env_registration(env_name: str) -> tuple[bool, str | None]:
    import gym
    import d4rl  # noqa: F401

    try:
        gym.spec(env_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a public D4RL locomotion dataset for DBPO. "
            "The output follows the D4RL data-processing standard and writes "
            "train.npz plus normalization.npz."
        )
    )
    parser.add_argument(
        "--env",
        required=True,
        help="D4RL env name from the default public gym set.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Target directory. The script writes train.npz and normalization.npz here.",
    )
    args = parser.parse_args()
    is_registered, reason = probe_env_registration(args.env)
    if not is_registered:
        print(json.dumps({
            "env": args.env,
            "status": "UNSUPPORTED_ENV",
            "reason": reason,
        }, indent=2, sort_keys=True))
        return
    from data_process.normalize import prepare_d4rl_dataset

    prepare_d4rl_dataset(env_name=args.env, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
