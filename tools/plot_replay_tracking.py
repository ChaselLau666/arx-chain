#!/usr/bin/env python3
"""Plot a replay run against the trajectory it was replaying.

Reads the .npz written by replay.py --execute and draws, per joint, the
commanded trajectory over what the arms actually did. Gripper feedback is
shifted by whole turns first: its motor count has no absolute reference, so it
can sit 2*pi away from the frame the commands were written in without anything
being wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "act"))

from replay_support import ARM_INDICES, GRIPPER_INDICES, best_lag, tracking_report

TWO_PI = 2.0 * np.pi
NAMES = ([f"L_j{i}" for i in range(6)] + ["L_grip"]
         + [f"R_j{i}" for i in range(6)] + ["R_grip"])


def moving_columns(command, threshold):
    """Columns whose commanded travel is more than noise - the ones worth plotting."""
    travel = command.max(axis=0) - command.min(axis=0)
    return [i for i in range(command.shape[1]) if travel[i] > threshold]


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(args.log, allow_pickle=True)
    t, command, actual = data["t"], data["command"], data["actual"]
    t = t - t[0]
    fps = float(data["frame_rate"]) if "frame_rate" in data else args.frame_rate

    report = tracking_report(command, actual)
    lag = report["lag_frames"]

    # Put the gripper feedback in the commanded frame before drawing it.
    aligned = actual.copy()
    for column, turns in zip(GRIPPER_INDICES, report["gripper_turns"]):
        aligned[:, column] -= turns * TWO_PI

    cols = args.columns if args.columns else moving_columns(command, args.min_travel)
    if not cols:
        print("nothing moved by more than --min-travel; use --columns to force")

        return 1

    print(f"frames {report['frames']}  lag {lag} 帧 ({lag / fps * 1000:.1f} ms)")
    print(f"arm rmse {report['arm_rmse']:.4f} rad, max {report['arm_max']:.4f} rad")
    print(f"gripper turns {report['gripper_turns']}, "
          f"residual rmse {np.round(report['gripper_rmse'], 4)}")
    print(f"plotting columns: {[NAMES[i] for i in cols]}")

    rows = len(cols)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.1 * rows), squeeze=False)
    for row, col in enumerate(cols):
        c, a = command[:, col], aligned[:, col]
        shifted = np.full_like(a, np.nan)
        if lag:
            shifted[:len(a) - lag] = a[lag:]
        else:
            shifted[:] = a

        ax = axes[row][0]
        ax.plot(t, c, lw=1.3, label="commanded", color="#1f77b4")
        ax.plot(t, a, lw=1.0, label="actual", color="#ff7f0e", alpha=.85)
        ax.set_ylabel(NAMES[col], fontsize=9)
        ax.grid(alpha=.3)
        if row == 0:
            ax.legend(fontsize=8, loc="best")
            ax.set_title("commanded vs actual", fontsize=11)

        ax = axes[row][1]
        ax.plot(t, a - c, lw=0.9, color="#7f7f7f", label="raw")
        ax.plot(t, shifted - c, lw=1.1, color="#d62728", label=f"lag-aligned ({lag}f)")
        ax.axhline(0, color="k", lw=.4)
        ax.grid(alpha=.3)
        if row == 0:
            ax.legend(fontsize=8, loc="best")
            ax.set_title("error (actual - commanded)", fontsize=11)

    axes[-1][0].set_xlabel("time (s)")
    axes[-1][1].set_xlabel("time (s)")
    episode = str(data["episode"]) if "episode" in data else ""
    fig.suptitle(f"replay tracking  |  {Path(episode).name}  |  "
                 f"arm rmse {report['arm_rmse']:.4f} rad, lag {lag / fps * 1000:.0f} ms",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}")

    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path,
                        help=".npz written by replay.py --execute")
    parser.add_argument("--out", type=Path, default=Path("replay_tracking.png"))
    parser.add_argument("--columns", type=int, nargs="+",
                        help="column indices to plot; default is everything that moved")
    parser.add_argument("--min-travel", type=float, default=0.02,
                        help="skip columns whose commanded travel is below this (rad)")
    parser.add_argument("--frame-rate", type=int, default=60,
                        help="used only when the log has no frame_rate")
    parser.add_argument("--dpi", type=int, default=95)

    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
