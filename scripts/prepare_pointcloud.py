#!/usr/bin/env python3
"""Prepare Adroit / MetaWorld point-cloud demonstrations for DBPO."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


DBPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = DBPO_ROOT / ".assets"
DEFAULT_VRL3_GOOGLE_DRIVE_URL = os.environ.get(
    "DBPO_ADROIT_VRL3_CKPT_URL",
    "https://drive.google.com/file/d/1iNkSrLD_N4NrezLx58L1YoBBqYYg-33u/view",
)
DEFAULT_VRL3_ONEDRIVE_URL = os.environ.get("DBPO_ADROIT_VRL3_CKPT_ONEDRIVE_URL", "")
DEFAULT_VRL3_LOCAL_ARCHIVE = os.environ.get("DBPO_ADROIT_VRL3_CKPT_ARCHIVE", "")
DEFAULT_VRL3_LOCAL_DIR = os.environ.get("DBPO_ADROIT_VRL3_CKPT_DIR", "")
ADROIT_TASKS = ("door", "hammer", "pen")


def _asset_root() -> Path:
    return Path(os.environ.get("DBPO_ASSET_DIR", DEFAULT_ASSET_DIR)).expanduser().resolve()


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in url:
        subprocess.run(
            [sys.executable, "-m", "gdown", "--fuzzy", url, "-O", str(output_path)],
            check=True,
        )
        return

    if shutil.which("wget"):
        subprocess.run(["wget", "-O", str(output_path), url], check=True)
        return
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", url, "-o", str(output_path)], check=True)
        return
    raise RuntimeError("Neither wget nor curl is available for checkpoint download.")


def _resolve_downloaded_ckpt(task: str, ckpt_dir: Path, ckpt_path: Path) -> Path:
    candidates = list(ckpt_dir.rglob(f"vrl3_{task}.pt"))
    if candidates:
        chosen = candidates[0]
        if chosen != ckpt_path:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(chosen), str(ckpt_path))
        return ckpt_path
    return ckpt_path


def _ensure_adroit_ckpt(task: str, dry_run: bool) -> Path:
    ckpt_dir = _asset_root() / "adroit" / "vrl3_ckpts"
    ckpt_path = ckpt_dir / f"vrl3_{task}.pt"
    if ckpt_path.exists():
        return ckpt_path

    if DEFAULT_VRL3_LOCAL_DIR:
        local_ckpt = Path(DEFAULT_VRL3_LOCAL_DIR).expanduser().resolve() / f"vrl3_{task}.pt"
        if local_ckpt.exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_ckpt, ckpt_path)
            return ckpt_path

    if DEFAULT_VRL3_LOCAL_ARCHIVE:
        archive_path = Path(DEFAULT_VRL3_LOCAL_ARCHIVE).expanduser().resolve()
        if archive_path.exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(ckpt_dir)
            _resolve_downloaded_ckpt(task, ckpt_dir, ckpt_path)
            if ckpt_path.exists():
                return ckpt_path

    remote_source = DEFAULT_VRL3_GOOGLE_DRIVE_URL or DEFAULT_VRL3_ONEDRIVE_URL
    zip_path = ckpt_dir / "vrl3_ckpts.zip"

    print(f"Missing Adroit expert checkpoint: {ckpt_path}")
    if dry_run and not remote_source:
        print(
            "No default checkpoint URL is configured. "
            "Dry-run will continue without downloading."
        )
        return ckpt_path

    if not remote_source:
        raise FileNotFoundError(
            "VRL3 expert checkpoints are not present and no download URL is configured. "
            "Set DBPO_ADROIT_VRL3_CKPT_URL, DBPO_ADROIT_VRL3_CKPT_ONEDRIVE_URL, "
            "DBPO_ADROIT_VRL3_CKPT_ARCHIVE, or DBPO_ADROIT_VRL3_CKPT_DIR, "
            f"or place {task} checkpoint under {ckpt_dir}."
        )

    print(f"Checkpoint source: {remote_source}")
    if dry_run:
        return ckpt_path

    print(f"Checkpoint archive target: {zip_path}")
    _download_file(remote_source, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ckpt_dir)
    _resolve_downloaded_ckpt(task, ckpt_dir, ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Downloaded checkpoint archive but did not find expected file: {ckpt_path}"
        )
    return ckpt_path


def _build_adroit_command(
    task: str, episodes: int, root_dir: Path, dry_run: bool
) -> tuple[list[str], Path, Path]:
    ckpt = _ensure_adroit_ckpt(task, dry_run=dry_run)
    source = root_dir / f"adroit_{task}_expert.zarr"
    target = root_dir / "adroit" / task / "train.zarr"
    command = [
        sys.executable,
        "-m",
        "adroit_metaworld_support.generators.adroit",
        "--env_name",
        task,
        "--num_episodes",
        str(episodes),
        "--root_dir",
        str(root_dir),
        "--expert_ckpt_path",
        str(ckpt),
        "--img_size",
        "84",
        "--not_use_multi_view",
        "--use_point_crop",
    ]
    return command, source, target


def _build_metaworld_command(task: str, episodes: int, root_dir: Path) -> tuple[list[str], Path, Path]:
    source = root_dir / f"metaworld_{task}_expert.zarr"
    target = root_dir / "metaworld" / task / "train.zarr"
    command = [
        sys.executable,
        "-m",
        "adroit_metaworld_support.generators.metaworld",
        "--env_name",
        task,
        "--num_episodes",
        str(episodes),
        "--root_dir",
        str(root_dir),
    ]
    return command, source, target


def _move_generated_zarr(source: Path, target: Path, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Generator finished but expected source zarr is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {target}")
        shutil.rmtree(target)
    shutil.move(str(source), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Adroit / MetaWorld point-cloud datasets for DBPO.")
    parser.add_argument("--suite", choices=("adroit", "metaworld"), required=True)
    parser.add_argument("--task", required=True, help="Task name, e.g. door or assembly")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DBPO_DATA_DIR", str(DBPO_ROOT / "data")),
        help="DBPO data root. Final outputs are written under <data-dir>/<suite>/<task>/train.zarr",
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    root_dir = data_dir

    if args.suite == "adroit":
        if args.task not in ADROIT_TASKS:
            raise ValueError(f"Unsupported Adroit task: {args.task}")
        command, source, target = _build_adroit_command(
            args.task, args.num_episodes, root_dir, dry_run=args.dry_run
        )
    else:
        command, source, target = _build_metaworld_command(args.task, args.num_episodes, root_dir)

    print(f"Suite: {args.suite}")
    print(f"Task: {args.task}")
    print(f"Command: {' '.join(command)}")
    print(f"Expected source zarr: {source}")
    print(f"Final target: {target}")
    print(f"Asset root: {_asset_root()}")

    if args.dry_run:
        return

    subprocess.run(command, check=True)
    _move_generated_zarr(source=source, target=target, overwrite=args.overwrite)
    print(f"Moved generated zarr to {target}")


if __name__ == "__main__":
    main()
