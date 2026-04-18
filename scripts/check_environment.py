#!/usr/bin/env python3
"""Static environment checks for the stage-1 DBPO repository."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GROUPS = {
    "native_suite": [
        "torch",
        "torchvision",
        "hydra",
        "omegaconf",
        "zarr",
        "h5py",
        "robomimic",
        "robosuite",
        "pygame",
        "pymunk",
        "shapely",
        "pybullet",
        "cv2",
        "skimage",
        "dbpo.env.pusht.pusht_image_env",
        "dbpo.env.block_pushing.block_pushing_multimodal",
        "dbpo.env.kitchen.v0",
    ],
    "adroit_metaworld_suite": [
        "open3d",
        "metaworld",
        "mj_envs",
        "adroit_metaworld_runtime",
        "adroit_metaworld_support.generators.adroit",
        "adroit_metaworld_support.generators.metaworld",
        "dbpo.model.vision.pointnet_extractor",
        "dbpo.models.encoders.pointcloud",
        "dbpo.dataset.adroit_dataset",
        "dbpo.dataset.metaworld_dataset",
    ],
    "gym_suite": [
        "gym",
        "d4rl",
        "mujoco",
        "mujoco_py",
        "dm_control",
        "dbpo.dataset.d4rl_sequence_dataset",
        "dbpo.methods.dbp.policies",
    ],
}


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _report_system_requirements() -> list[tuple[str, bool, str]]:
    mujoco210 = Path.home() / ".mujoco" / "mujoco210"
    return [
        ("path ~/.mujoco/mujoco210", mujoco210.exists(), str(mujoco210)),
        ("env LD_LIBRARY_PATH", bool(os.environ.get("LD_LIBRARY_PATH")), os.environ.get("LD_LIBRARY_PATH", "<unset>")),
        ("env MUJOCO_GL", bool(os.environ.get("MUJOCO_GL")), os.environ.get("MUJOCO_GL", "<unset>")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Static import/version/path checks for stage-1 DBPO.")
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted(GROUPS),
        required=True,
        help="Check one or more dependency groups.",
    )
    args = parser.parse_args()

    failed = False
    print(f"Python: {sys.version.split()[0]}")
    for group in args.group:
        print(f"\n[{group}]")
        for module_name in GROUPS[group]:
            ok, detail = _check_import(module_name)
            status = "OK" if ok else "MISSING"
            print(f"  - {module_name}: {status} ({detail})")
            failed = failed or not ok

    print("\n[system]")
    for label, ok, detail in _report_system_requirements():
        status = "OK" if ok else "WARN"
        print(f"  - {label}: {status} ({detail})")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
