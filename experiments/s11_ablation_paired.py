#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired ARI tests for the DIVI component ablation.

Primary comparisons
-------------------
Reference: Full-DIVI

Comparators:
- No-StepA
- No-Gating
- Unsafe-Split

Pairing unit:
- the same (setting, seed) synthetic dataset

Primary family:
- 2 settings x 3 comparisons = 6 ARI tests
- Holm adjustment is applied jointly across these six pre-specified tests.

Reported quantities
-------------------
- Full-DIVI and comparator means
- mean paired difference: Full-DIVI minus comparator
- percentile paired-bootstrap 95% CI
- paired Cohen's d_z
- matched-pairs rank-biserial correlation
- two-sided Wilcoxon signed-rank p-value
- Holm-adjusted p-value
- wins / ties / losses for Full-DIVI

Input
-----
component_ablation_raw.csv produced by run_component_ablation_v3.py

Outputs
-------
component_ablation_paired_ari.csv
component_ablation_paired_ari_table.tex
component_ablation_paired_data.csv

Example
-------
python run_component_ablation_paired_tests.py \
  --input_csv /content/drive/MyDrive/component_ablation/component_ablation_raw.csv \
  --output_dir /content/drive/MyDrive/component_ablation/paired_tests \
  --bootstrap_reps 20000 \
  --bootstrap_seed 20260716
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


REFERENCE = "Full-DIVI"
COMPARATORS = ["No-StepA", "No-Gating", "Unsafe-Split"]
PRIMARY_METRIC = "ARI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to component_ablation_raw.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--bootstrap_reps",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=20260716,
    )
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["matched", "correlated_noise"],
    )
    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "setting",
        "variant",
        "seed",
        "status",
        PRIMARY_METRIC,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    clean = df.loc[df["status"].astype(str) == "ok"].copy()
    clean["seed"] = pd.to_numeric(clean["seed"], errors="raise").astype(int)
    clean[PRIMARY_METRIC] = pd.to_numeric(
        clean[PRIMARY_METRIC],
        errors="coerce",
    )

    clean = clean.dropna(subset=[PRIMARY_METRIC])

    duplicate_mask = clean.duplicated(
        subset=["setting", "variant", "seed"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = clean.loc[
            duplicate_mask,
            ["setting", "variant", "seed"],
        ].sort_values(["setting", "variant", "seed"])

        raise ValueError(
            "Duplicate successful rows found for the same "
            "(setting, variant, seed). Resolve them before testing.\n"
            + duplicates.to_string(index=False)
        )

    return clean


def paired_bootstrap_ci(
    differences: np.ndarray,
    reps: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    if n == 0:
        return np.nan, np.nan

    # Chunking avoids allocating a very large bootstrap index matrix.
    chunk_size = min(5000, reps)
    means: List[np.ndarray] = []
    completed = 0

    while completed < reps:
        current = min(chunk_size, reps - completed)
        indices = rng.integers(0, n, size=(current, n))
        means.append(differences[indices].mean(axis=1))
        completed += current

    bootstrap_means = np.concatenate(means)
    alpha = 1.0 - confidence

    low, high = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(low), float(high)


def paired_cohens_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)

    if len(differences) < 2:
        return np.nan

    sd = differences.std(ddof=1)
    mean_difference = differences.mean()

    if np.isclose(sd, 0.0):
        if np.isclose(mean_difference, 0.0):
            return 0.0
        return float(np.sign(mean_difference) * np.inf)

    return float(mean_difference / sd)


def matched_rank_biserial(
    differences: np.ndarray,
    tie_tolerance: float,
) -> float:
    """Matched-pairs rank-biserial correlation.

    r_rb = (sum positive ranks - sum negative ranks)
           / (sum positive ranks + sum negative ranks)

    Exact zero differences are omitted, matching the usual Wilcoxon
    signed-rank convention.
    """
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[np.abs(differences) > tie_tolerance]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero), method="average")
    positive_sum = ranks[nonzero > 0].sum()
    negative_sum = ranks[nonzero < 0].sum()
    denominator = positive_sum + negative_sum

    if np.isclose(denominator, 0.0):
        return 0.0

    return float((positive_sum - negative_sum) / denominator)


def wilcoxon_two_sided(
    differences: np.ndarray,
    tie_tolerance: float,
) -> Tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences.copy()
    differences[np.abs(differences) <= tie_tolerance] = 0.0

    if np.all(differences == 0.0):
        return 0.0, 1.0

    try:
        result = wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
    except ValueError:
        # Conservative fallback for pathological all-tie configurations.
        return np.nan, 1.0

    return float(result.statistic), float(result.pvalue)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down family-wise error-rate adjustment."""
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)

    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return adjusted

    ordered_valid = valid[np.argsort(p[valid])]
    m = len(ordered_valid)

    running_max = 0.0
    for rank_index, original_index in enumerate(ordered_valid):
        candidate = (m - rank_index) * p[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(running_max, 1.0)

    return adjusted


def make_pairs(
    df: pd.DataFrame,
    setting: str,
    comparator: str,
) -> pd.DataFrame:
    subset = df.loc[
        (df["setting"] == setting)
        & (df["variant"].isin([REFERENCE, comparator])),
        ["setting", "variant", "seed", PRIMARY_METRIC],
    ].copy()

    wide = subset.pivot(
        index=["setting", "seed"],
        columns="variant",
        values=PRIMARY_METRIC,
    ).reset_index()

    required_columns = [REFERENCE, comparator]
    for column in required_columns:
        if column not in wide.columns:
            raise ValueError(
                f"No successful rows found for {column!r} "
                f"in setting {setting!r}."
            )

    paired = wide.dropna(subset=required_columns).copy()
    paired["comparator"] = comparator
    paired["difference"] = paired[REFERENCE] - paired[comparator]
    return paired


def analyze_comparison(
    paired: pd.DataFrame,
    setting: str,
    comparator: str,
    bootstrap_reps: int,
    rng: np.random.Generator,
    tie_tolerance: float,
) -> Dict[str, float | int | str]:
    differences = paired["difference"].to_numpy(dtype=float)
    reference_values = paired[REFERENCE].to_numpy(dtype=float)
    comparator_values = paired[comparator].to_numpy(dtype=float)

    ci_low, ci_high = paired_bootstrap_ci(
        differences=differences,
        reps=bootstrap_reps,
        rng=rng,
    )
    wilcoxon_stat, raw_p = wilcoxon_two_sided(
        differences=differences,
        tie_tolerance=tie_tolerance,
    )

    wins = int(np.sum(differences > tie_tolerance))
    ties = int(np.sum(np.abs(differences) <= tie_tolerance))
    losses = int(np.sum(differences < -tie_tolerance))

    return {
        "setting": setting,
        "reference": REFERENCE,
        "comparator": comparator,
        "metric": PRIMARY_METRIC,
        "n_pairs": int(len(differences)),
        "reference_mean": float(reference_values.mean()),
        "reference_sd": (
            float(reference_values.std(ddof=1))
            if len(reference_values) > 1
            else np.nan
        ),
        "comparator_mean": float(comparator_values.mean()),
        "comparator_sd": (
            float(comparator_values.std(ddof=1))
            if len(comparator_values) > 1
            else np.nan
        ),
        "mean_difference": float(differences.mean()),
        "difference_sd": (
            float(differences.std(ddof=1))
            if len(differences) > 1
            else np.nan
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "cohens_dz": paired_cohens_dz(differences),
        "rank_biserial": matched_rank_biserial(
            differences,
            tie_tolerance=tie_tolerance,
        ),
        "wilcoxon_statistic": wilcoxon_stat,
        "p_value_raw": raw_p,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    result = value
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def fmt_number(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    if np.isposinf(value):
        return r"$+\infty$"
    if np.isneginf(value):
        return r"$-\infty$"
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def write_latex_table(results: pd.DataFrame, output_path: Path) -> None:
    setting_labels = {
        "matched": "Matched",
        "correlated_noise": "Correlated noise",
    }
    comparator_labels = {
        "No-StepA": "No Step A",
        "No-Gating": "No gating",
        "Unsafe-Split": "Unsafe split",
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        (
            r"\caption{Primary paired ARI comparisons for the component "
            r"ablation. Differences are Full DIVI minus the ablated variant. "
            r"Confidence intervals are percentile-bootstrap 95\% intervals "
            r"based on paired resampling. Wilcoxon signed-rank $p$-values are "
            r"Holm-adjusted across the six pre-specified comparisons.}"
        ),
        r"\label{tab:component_ablation_paired}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        (
            r"Setting & Comparator & $n$ & Full & Comparator & "
            r"$\Delta$ ARI (95\% CI) & $d_z$ & $r_{\mathrm{rb}}$ & "
            r"Holm $p$ \\"
        ),
        r"\midrule",
    ]

    previous_setting = None

    for _, row in results.iterrows():
        setting = str(row["setting"])

        if previous_setting is not None and setting != previous_setting:
            lines.append(r"\addlinespace")

        setting_text = (
            setting_labels.get(setting, latex_escape(setting))
            if setting != previous_setting
            else ""
        )
        comparator = comparator_labels.get(
            str(row["comparator"]),
            latex_escape(str(row["comparator"])),
        )

        delta_ci = (
            f"{fmt_number(row['mean_difference'])} "
            f"([{fmt_number(row['bootstrap_ci_low'])}, "
            f"{fmt_number(row['bootstrap_ci_high'])}])"
        )

        lines.append(
            f"{setting_text} & {comparator} & "
            f"{int(row['n_pairs'])} & "
            f"{fmt_number(row['reference_mean'])} & "
            f"{fmt_number(row['comparator_mean'])} & "
            f"{delta_ci} & "
            f"{fmt_number(row['cohens_dz'])} & "
            f"{fmt_number(row['rank_biserial'])} & "
            f"{fmt_p(row['p_value_holm'])} \\\\"
        )

        previous_setting = setting

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.bootstrap_reps < 1000:
        raise ValueError(
            "Use at least 1,000 paired-bootstrap replicates; "
            "20,000 is recommended for the paper."
        )

    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    raw = pd.read_csv(input_path)
    clean = validate_input(raw)

    expected_variants = {REFERENCE, *COMPARATORS}
    observed_variants = set(clean["variant"].astype(str))
    missing_variants = expected_variants.difference(observed_variants)

    if missing_variants:
        raise ValueError(
            "The following required variants are absent from successful rows: "
            + ", ".join(sorted(missing_variants))
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    result_rows: List[Dict[str, float | int | str]] = []
    paired_frames: List[pd.DataFrame] = []

    for setting in args.settings:
        for comparator in COMPARATORS:
            paired = make_pairs(
                df=clean,
                setting=setting,
                comparator=comparator,
            )

            if len(paired) < 2:
                raise ValueError(
                    f"Only {len(paired)} complete pairs for "
                    f"{setting} / {comparator}."
                )

            paired_frames.append(paired)
            result_rows.append(
                analyze_comparison(
                    paired=paired,
                    setting=setting,
                    comparator=comparator,
                    bootstrap_reps=args.bootstrap_reps,
                    rng=rng,
                    tie_tolerance=args.tie_tolerance,
                )
            )

    results = pd.DataFrame(result_rows)

    # One pre-specified primary family: all 6 ARI comparisons.
    results["p_value_holm"] = holm_adjust(
        results["p_value_raw"].to_numpy(dtype=float)
    )
    results["significant_holm_0_05"] = (
        results["p_value_holm"] < 0.05
    )

    setting_order = {
        setting: index
        for index, setting in enumerate(args.settings)
    }
    comparator_order = {
        comparator: index
        for index, comparator in enumerate(COMPARATORS)
    }

    results["_setting_order"] = results["setting"].map(setting_order)
    results["_comparator_order"] = results["comparator"].map(
        comparator_order
    )
    results = (
        results
        .sort_values(["_setting_order", "_comparator_order"])
        .drop(columns=["_setting_order", "_comparator_order"])
        .reset_index(drop=True)
    )

    paired_data = pd.concat(
        paired_frames,
        ignore_index=True,
        sort=False,
    )

    results_path = (
        output_dir / "component_ablation_paired_ari.csv"
    )
    paired_path = (
        output_dir / "component_ablation_paired_data.csv"
    )
    latex_path = (
        output_dir / "component_ablation_paired_ari_table.tex"
    )

    results.to_csv(results_path, index=False)
    paired_data.to_csv(paired_path, index=False)
    write_latex_table(results, latex_path)

    display_columns = [
        "setting",
        "comparator",
        "n_pairs",
        "reference_mean",
        "comparator_mean",
        "mean_difference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "cohens_dz",
        "rank_biserial",
        "p_value_raw",
        "p_value_holm",
        "wins",
        "ties",
        "losses",
    ]

    print("\n=== Primary paired ARI tests ===")
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nInterpretation convention:")
    print("  mean_difference > 0 favors Full-DIVI")
    print("  wins/ties/losses are counted from Full-DIVI's perspective")
    print("  Holm adjustment is across all six pre-specified ARI tests")

    print("\nOutput files:")
    print(results_path)
    print(paired_path)
    print(latex_path)


if __name__ == "__main__":
    main()
