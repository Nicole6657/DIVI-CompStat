#!/usr/bin/env python3
"""Re-run one inexpensive experiment and check it against the committed results.

The split-interval sensitivity grid is used because it is cheap (a few minutes on
CPU), it exercises the whole `divi/` package -- data generation, Step A, the
gated mixture, split growth and the metric pipeline -- and its committed values
include quantities with zero replicate variance (the terminal component count is
exactly 31, 16, 8 and 4), which makes agreement unambiguous.

Usage, from the repository root:

    python tools/verify_reproduction.py

    python tools/verify_reproduction.py --keep      # keep the fresh output
    python tools/verify_reproduction.py --atol 0.02 # looser ARI tolerance

Exit status is 0 if every checked quantity agrees within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
COMMITTED = REPO / "results" / "sensitivity_synth" / "Tsplit" / "summary.csv"
DRIVER = REPO / "experiments" / "fig03_sensitivity.py"
CONFIG = REPO / "divi" / "configs" / "defaults_divi.yaml"

KEY = ["dataset_variant", "sensitivity_value"]

# column -> absolute tolerance. final_K and split_count are integer-valued with
# zero replicate variance in the committed run, so they must match exactly.
CHECKS = {
    "final_K__mean": 0.0,
    "split_count__mean": 0.0,
    "ari__mean": 0.01,
    "nmi__mean": 0.01,
    "f1_feature__mean": 0.01,
    "selected_dims_count__mean": 0.5,
}


def run_experiment(outdir: Path, verbose: bool) -> None:
    cmd = [
        sys.executable, str(DRIVER),
        "--factor", "Tsplit",
        "--values", "10,20,40,80",
        "--outdir", str(outdir),
        "--n-list", "200,1000",
        "--D", "100",
        "--n-signal", "10",
        "--noise-scale", "3.0",
        "--n-runs", "10",
        "--prior-mode", "1",
        "--beta-mult", "1.0",
        "--split-interval", "80",
        "--max-epochs", "300",
        "--lr", "0.01",
        "--temp-start", "1.0",
        "--temp-end", "0.1",
        "--config", str(CONFIG),
    ]
    print("Running:\n  " + " ".join(cmd) + "\n")
    env_note = "" if verbose else "  (stdout suppressed; use --verbose to see it)\n"
    print(env_note)
    subprocess.run(
        cmd,
        cwd=REPO,
        check=True,
        stdout=None if verbose else subprocess.DEVNULL,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "divi")},
    )


def compare(fresh_csv: Path, atol_ari: float) -> bool:
    old = pd.read_csv(COMMITTED)
    new = pd.read_csv(fresh_csv)

    checks = dict(CHECKS)
    for name in ("ari__mean", "nmi__mean", "f1_feature__mean"):
        checks[name] = atol_ari

    merged = old.merge(new, on=KEY, suffixes=("_committed", "_fresh"))
    if len(merged) != len(old):
        print(f"! row mismatch: committed {len(old)}, matched {len(merged)}")
        return False

    ok = True
    header = f"{'variant':<36} {'Tsplit':>7}  {'quantity':<26} {'committed':>11} {'fresh':>11}   result"
    print(header)
    print("-" * len(header))

    for _, row in merged.sort_values(KEY).iterrows():
        variant = row["dataset_variant"].replace("synthetic_", "")
        for col, tol in checks.items():
            a, b = row[f"{col}_committed"], row[f"{col}_fresh"]
            passed = abs(a - b) <= tol
            ok &= bool(passed)
            print(
                f"{variant:<36} {row['sensitivity_value']:>7.0f}  {col:<26} "
                f"{a:>11.4f} {b:>11.4f}   {'ok' if passed else 'DIFFERS'}"
            )
        print()

    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep", action="store_true", help="keep the freshly generated output directory")
    p.add_argument("--atol", type=float, default=0.01, help="absolute tolerance for ARI, NMI and feature F1")
    p.add_argument("--verbose", action="store_true", help="show the experiment's stdout")
    args = p.parse_args()

    if not COMMITTED.exists():
        print(f"Committed reference not found: {COMMITTED}")
        return 1

    outroot = Path(tempfile.mkdtemp(prefix="divi_verify_")) if not args.keep else REPO / "verification_run"
    outroot.mkdir(parents=True, exist_ok=True)

    try:
        run_experiment(outroot, args.verbose)
        fresh = outroot / "Tsplit" / "summary.csv"
        if not fresh.exists():
            print(f"Expected output not produced: {fresh}")
            return 1

        print()
        ok = compare(fresh, args.atol)
        print("=" * 72)
        if ok:
            print("PASS — the re-run reproduces the committed split-interval results.")
            print("Record the date and commit ENVIRONMENT.md and requirements-frozen.txt.")
        else:
            print("MISMATCH — at least one quantity differs beyond tolerance.")
            print("Structural quantities (final_K, split_count) are compared exactly;")
            print("a difference there indicates a genuine behavioural change, not noise.")
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(outroot, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
