#!/usr/bin/env python3
"""Record the environment that produced a set of results.

The experiments in this repository were executed on Google Colab, which does
not pin package versions: the same notebook run at a different date resolves to
a different set of wheels. This script captures the versions present at the
moment it is run, so that a verification run can be tied to a concrete
environment.

Usage (from the repository root):

    python tools/capture_environment.py

Writes:

    ENVIRONMENT.md            human-readable summary, including R if available
    requirements-frozen.txt   output of `pip freeze`

In Colab, place this at the top of the notebook that re-runs an experiment:

    !python tools/capture_environment.py

Then re-run one inexpensive experiment, confirm that the numbers match the
committed results, and commit both files together with a note of the date.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DIRECT_DEPENDENCIES = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "yaml",
    "matplotlib",
    "psutil",
    "sentence_transformers",
]

# Import name -> distribution name, where they differ.
DIST_NAME = {"sklearn": "scikit-learn", "yaml": "PyYAML"}


def package_versions() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for name in DIRECT_DEPENDENCIES:
        label = DIST_NAME.get(name, name)
        try:
            module = importlib.import_module(name)
        except ImportError:
            rows.append((label, "not installed"))
            continue
        rows.append((label, getattr(module, "__version__", "unknown")))
    return rows


def accelerator() -> str:
    try:
        import torch
    except ImportError:
        return "unknown (torch not installed)"
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(0)} (torch.version.cuda={torch.version.cuda})"
    return "CPU only"


def pip_freeze() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def r_environment() -> str | None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    code = (
        'cat(R.version.string, "\\n"); '
        'for (p in c("sparcl", "jsonlite", "Seurat")) '
        '  if (requireNamespace(p, quietly = TRUE)) '
        '    cat(p, as.character(packageVersion(p)), "\\n")'
    )
    result = subprocess.run(
        [rscript, "-q", "-e", code], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Environment",
        "",
        f"Captured {stamp} by `tools/capture_environment.py`.",
        "",
        "## Platform",
        "",
        f"- Python: {platform.python_version()} ({platform.python_implementation()})",
        f"- OS: {platform.platform()}",
        f"- Machine: {platform.machine()}",
        f"- Accelerator: {accelerator()}",
        "",
        "## Direct Python dependencies",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for name, version in package_versions():
        lines.append(f"| {name} | {version} |")

    r_info = r_environment()
    lines += ["", "## R (Sparse K-means, PBMC preprocessing)", ""]
    if r_info:
        lines += ["```", r_info, "```"]
    else:
        lines.append("Rscript not found on PATH; R components were not exercised here.")

    lines += [
        "",
        "## Full package list",
        "",
        "See `requirements-frozen.txt` (output of `pip freeze`).",
        "",
        "## Note on Colab",
        "",
        "The originally reported results were produced on Google Colab, which does",
        "not pin package versions. The versions above describe the environment in",
        "which the accompanying verification run was performed, not necessarily the",
        "one used for the first execution of every experiment.",
        "",
    ]

    Path("ENVIRONMENT.md").write_text("\n".join(lines), encoding="utf-8")
    Path("requirements-frozen.txt").write_text(pip_freeze(), encoding="utf-8")

    print("\n".join(lines))
    print("\nWrote ENVIRONMENT.md and requirements-frozen.txt")


if __name__ == "__main__":
    main()
