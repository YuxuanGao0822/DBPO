#!/usr/bin/env python3
"""Prepare D4RL FrankaKitchen datasets for DBPO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a D4RL kitchen dataset for DBPO.")
    parser.add_argument("--env", default="kitchen-mixed-v0", help="D4RL kitchen env name")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Target directory. The script writes observations_seq.npy, actions_seq.npy, and existence_mask.npy here.",
    )
    parser.add_argument(
        "--write_summary",
        action="store_true",
        help="Also write dataset_summary.json for bookkeeping.",
    )
    args = parser.parse_args()
    from data_process.process_kitchen import prepare_kitchen_dataset

    prepare_kitchen_dataset(
        env_name=args.env,
        output_dir=args.output_dir,
        write_summary=args.write_summary,
    )


if __name__ == "__main__":
    main()
