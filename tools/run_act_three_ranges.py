#!/usr/bin/env python3
"""Run the agreed 0-24, 25-49, and 0-49 ACT experiments sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_act_experiment.py"
RANGES = ((0, 24), (25, 49), (0, 49))


def run_phase(args, epochs: int, label: str) -> None:
    for start, end in RANGES:
        command = [
            sys.executable,
            str(RUNNER),
            "--start", str(start),
            "--end", str(end),
            "--eval-episode", str(args.eval_episode),
            "--source-dir", str(args.source_dir),
            "--view-root", str(args.view_root),
            "--run-root", str(args.run_root),
            "--epochs", str(epochs),
            "--seed", str(args.seed),
            "--run-name", f"act_ep{start:03d}_{end:03d}_seed{args.seed}_{label}",
        ]
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full", "all"), default="all")
    parser.add_argument("--eval-episode", type=int, default=50)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "act" / "datasets")
    parser.add_argument("--view-root", type=Path, default=ROOT / "act" / "dataset_views")
    parser.add_argument("--run-root", type=Path, default=ROOT / "act" / "runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.phase in ("smoke", "all"):
        run_phase(args, 2, "smoke")
    if args.phase in ("full", "all"):
        run_phase(args, 3000, "full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
