#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paired statistical tests and effect sizes for DIVI experiments.

Required input columns
----------------------
experiment
scenario
seed
method

At least one metric column, for example:
ARI
NMI
feature_f1

Example
-------
python run_paired_tests.py \
    --input_csv /content/all_results_long.csv \
    --output_dir /content/paired_tests \
    --reference_method DIVI-Info \
    --metrics ARI NMI feature_f1 \
    --bootstrap_reps 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """
    Holm step-down adjusted p-values.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    adjusted = np.full(m, np.nan)
    valid = np.isfinite(p_values)

    if valid.sum() == 0:
        return adjusted

    idx_valid = np.where(valid)[0]
    p_valid = p_values[valid]

    order = np.argsort(p_valid)
    ordered_p = p_valid[order]

    raw_adjusted = (len(p_valid) - np.arange(len(p_valid))) * ordered_p
    monotone_adjusted = np.maximum.accumulate(raw_adjusted)
    monotone_adjusted = np.minimum(monotone_adjusted, 1.0)

    restored = np.empty(len(p_valid))
    restored[order] = monotone_adjusted
    adjusted[idx_valid] = restored

    return adjusted


def paired_cohens_dz(diff: np.ndarray) -> float:
    """
    Cohen's dz for paired data:
        mean(diff) / sd(diff)
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) < 2:
        return np.nan

    sd = np.std(diff, ddof=1)

    if sd == 0:
        if np.mean(diff) > 0:
            return np.inf
        if np.mean(diff) < 0:
            return -np.inf
        return 0.0

    return float(np.mean(diff) / sd)


def rank_biserial_from_differences(diff: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation.

    Positive value means the reference method tends to outperform
    the comparator.
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    abs_diff = np.abs(diff)
    ranks = pd.Series(abs_diff).rank(method="average").to_numpy()

    positive_sum = ranks[diff > 0].sum()
    negative_sum = ranks[diff < 0].sum()
    total = positive_sum + negative_sum

    if total == 0:
        return 0.0

    return float((positive_sum - negative_sum) / total)


def bootstrap_mean_ci(
    diff: np.ndarray,
    reps: int,
    confidence: float,
    seed: int,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for the mean paired difference.
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(diff)

    sampled = rng.choice(
        diff,
        size=(reps, n),
        replace=True,
    )
    boot_means = sampled.mean(axis=1)

    alpha = 1.0 - confidence
    lower = np.quantile(boot_means, alpha / 2.0)
    upper = np.quantile(boot_means, 1.0 - alpha / 2.0)

    return float(lower), float(upper)


def paired_analysis(
    ref: pd.Series,
    comp: pd.Series,
    bootstrap_reps: int,
    confidence: float,
    bootstrap_seed: int,
) -> Dict[str, float]:
    paired = pd.concat(
        [ref.rename("reference"), comp.rename("comparator")],
        axis=1,
    ).dropna()

    n = len(paired)

    if n == 0:
        return {
            "n_pairs": 0,
            "reference_mean": np.nan,
            "comparator_mean": np.nan,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "paired_t_p": np.nan,
            "wilcoxon_p": np.nan,
            "cohens_dz": np.nan,
            "rank_biserial": np.nan,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }

    diff = paired["reference"].to_numpy() - paired["comparator"].to_numpy()

    ci_lower, ci_upper = bootstrap_mean_ci(
        diff,
        reps=bootstrap_reps,
        confidence=confidence,
        seed=bootstrap_seed,
    )

    if n >= 2:
        t_result = ttest_rel(
            paired["reference"],
            paired["comparator"],
            nan_policy="omit",
        )
        paired_t_p = float(t_result.pvalue)
    else:
        paired_t_p = np.nan

    nonzero = diff[np.abs(diff) > 1e-15]

    if len(nonzero) == 0:
        wilcoxon_p = 1.0
    else:
        try:
            w_result = wilcoxon(
                diff,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_p = np.nan

    return {
        "n_pairs": int(n),
        "reference_mean": float(paired["reference"].mean()),
        "comparator_mean": float(paired["comparator"].mean()),
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "cohens_dz": paired_cohens_dz(diff),
        "rank_biserial": rank_biserial_from_differences(diff),
        "wins": int(np.sum(diff > 1e-15)),
        "ties": int(np.sum(np.abs(diff) <= 1e-15)),
        "losses": int(np.sum(diff < -1e-15)),
    }


def make_latex_table(results: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Paired comparisons of DIVI-Info with selected baselines. "
        r"Positive differences and effect sizes favor DIVI-Info. "
        r"Reported $p$-values are from two-sided Wilcoxon signed-rank tests "
        r"with Holm adjustment within each experiment and metric.}",
        r"\label{tab:paired_tests}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Experiment & Scenario & Comparator & Metric & "
        r"$\Delta$ mean & 95\% CI & $d_z$ & $r_{\mathrm{rb}}$ & "
        r"Holm $p$ \\",
        r"\midrule",
    ]

    for _, row in results.iterrows():
        ci = (
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            if np.isfinite(row["ci_lower"])
            else "--"
        )

        p = row["wilcoxon_p_holm"]
        if np.isnan(p):
            p_text = "--"
        elif p < 0.001:
            p_text = "$<.001$"
        else:
            p_text = f"{p:.3f}"

        dz = row["cohens_dz"]
        rb = row["rank_biserial"]

        lines.append(
            f"{row['experiment']} & "
            f"{row['scenario']} & "
            f"{row['comparator']} & "
            f"{row['metric']} & "
            f"{row['mean_difference']:.3f} & "
            f"{ci} & "
            f"{dz:.3f} & "
            f"{rb:.3f} & "
            f"{p_text} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--reference_method",
        default="DIVI-Info",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ARI", "NMI", "feature_f1"],
    )

    parser.add_argument(
        "--bootstrap_reps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required = {
        "experiment",
        "scenario",
        "seed",
        "method",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    records: List[Dict[str, object]] = []

    group_cols = ["experiment", "scenario"]

    for (experiment, scenario), group in df.groupby(group_cols):
        methods = sorted(group["method"].dropna().unique())

        if args.reference_method not in methods:
            continue

        reference = (
            group[group["method"] == args.reference_method]
            .set_index("seed")
        )

        comparators = [
            method for method in methods
            if method != args.reference_method
        ]

        for comparator in comparators:
            comparator_df = (
                group[group["method"] == comparator]
                .set_index("seed")
            )

            for metric in args.metrics:
                if metric not in group.columns:
                    continue

                result = paired_analysis(
                    ref=reference[metric],
                    comp=comparator_df[metric],
                    bootstrap_reps=args.bootstrap_reps,
                    confidence=args.confidence,
                    bootstrap_seed=args.bootstrap_seed,
                )

                records.append({
                    "experiment": experiment,
                    "scenario": scenario,
                    "reference": args.reference_method,
                    "comparator": comparator,
                    "metric": metric,
                    **result,
                })

    results = pd.DataFrame(records)

    if results.empty:
        raise RuntimeError(
            "No valid paired comparisons were found. "
            "Check method names, seed alignment, and metric columns."
        )

    results["wilcoxon_p_holm"] = np.nan
    results["paired_t_p_holm"] = np.nan

    # Holm correction within each experiment and metric.
    for _, idx in results.groupby(
        ["experiment", "metric"]
    ).groups.items():
        idx = list(idx)

        results.loc[idx, "wilcoxon_p_holm"] = holm_adjust(
            results.loc[idx, "wilcoxon_p"].to_numpy()
        )
        results.loc[idx, "paired_t_p_holm"] = holm_adjust(
            results.loc[idx, "paired_t_p"].to_numpy()
        )

    results = results.sort_values(
        ["experiment", "scenario", "metric", "comparator"]
    )

    results.to_csv(
        output_dir / "paired_tests_results.csv",
        index=False,
    )

    make_latex_table(
        results,
        output_dir / "paired_tests_table.tex",
    )

    config = vars(args)
    (output_dir / "paired_tests_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "experiment",
        "scenario",
        "comparator",
        "metric",
        "n_pairs",
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "cohens_dz",
        "rank_biserial",
        "wilcoxon_p",
        "wilcoxon_p_holm",
        "wins",
        "ties",
        "losses",
    ]

    print(results[display_cols].to_string(index=False))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
