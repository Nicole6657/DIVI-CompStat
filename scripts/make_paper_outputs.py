#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convenience wrapper: aggregate runs and emit paper-ready CSV/LaTeX/figures.")
    ap.add_argument("--root", type=str, default="outputs_revision")
    ap.add_argument("--outdir", type=str, default="outputs_revision/aggregated")
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--plot-formats", type=str, default="pdf,png")
    args = ap.parse_args()

    script = Path(__file__).with_name("aggregate_divi_revision_results.py")
    cmd = [
        sys.executable,
        str(script),
        "--root", args.root,
        "--outdir", args.outdir,
        "--plot-formats", args.plot_formats,
    ]
    if args.make_plots:
        cmd.append("--make-plots")
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
