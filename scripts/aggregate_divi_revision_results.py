#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import math
import numpy as np
import pandas as pd

from experiment_utils import ensure_dir

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


METHOD_LABELS = {
    "divi": "DIVI-Info",
    "kmeans_oracle": "KMeans",
    "gmm_oracle": "GMM",
    "spkm": "SPKM",
}

DATASET_LABELS = {
    "synthetic": "Synthetic",
    "isolet": "ISOLET",
    "20ng": "20NG",
}

FACTOR_LABELS = {
    "beta_mult": r"$\beta/N$ multiplier",
    "Tsplit": r"$T_{\mathrm{split}}$",
    "tau_mult": r"$\tau/\tau_0$",
    "lr": "Learning rate",
    "temp_end": "Final temperature",
}

RUNTIME_COLUMNS = [
    "dataset", "dataset_variant", "method", "runtime_axis", "runtime_value",
    "ari", "nmi", "acc", "final_K", "selected_dims_count",
    "wallclock_total_sec", "peak_memory_mb", "split_count",
]

SENS_COLUMNS = [
    "dataset", "dataset_variant", "method", "sensitivity_factor", "sensitivity_value",
    "ari", "nmi", "f1_feature", "acc", "final_K", "selected_dims_count",
    "wallclock_total_sec", "split_count",
]


def _safe_std(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1))


def _safe_mean(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) == 0:
        return float("nan")
    return float(vals.mean())


def _with_defaults(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _label_method(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def _label_dataset(dataset: str) -> str:
    return DATASET_LABELS.get(str(dataset), str(dataset))


def _label_factor(factor: str) -> str:
    return FACTOR_LABELS.get(str(factor), str(factor))


def _metric_str(mean_val: float, sd_val: float, digits: int = 3) -> str:
    if pd.isna(mean_val):
        return "--"
    if pd.isna(sd_val):
        sd_val = 0.0
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean_val)} ({fmt.format(sd_val)})"


def _count_str(mean_val: float, sd_val: float) -> str:
    if pd.isna(mean_val):
        return "--"
    if pd.isna(sd_val):
        sd_val = 0.0
    return f"{mean_val:.1f} ({sd_val:.1f})"


def _agg_summary(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    agg_spec: dict[str, list[str]] = {"experiment_id": ["count"]}
    for metric in metrics:
        if metric in df.columns:
            agg_spec[metric] = [_safe_mean, _safe_std]
    out = df.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    out.columns = [
        "_".join([c for c in tup if c]).replace("<lambda>", "") if isinstance(tup, tuple) else tup
        for tup in out.columns
    ]
    rename_map = {"experiment_id_count": "n_runs"}
    for metric in metrics:
        rename_map[f"{metric}__safe_mean"] = f"{metric}_mean"
        rename_map[f"{metric}__safe_std"] = f"{metric}_sd"
        rename_map[f"{metric}_safe_mean"] = f"{metric}_mean"
        rename_map[f"{metric}_safe_std"] = f"{metric}_sd"
    return out.rename(columns=rename_map)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _write_latex(df: pd.DataFrame, path: Path, caption: str, label: str, column_format: str | None = None) -> None:
    ensure_dir(path.parent)
    latex = df.to_latex(index=False, escape=False, column_format=column_format)
    wrapped = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}"
        "\\end{table}\n"
    )
    path.write_text(wrapped, encoding="utf-8")


def _make_runtime_real_table(runtime_summary: pd.DataFrame) -> pd.DataFrame:
    df = runtime_summary[runtime_summary["runtime_axis"] == "dataset"].copy()
    if df.empty:
        return df

    df["Dataset"] = df["dataset"].map(_label_dataset)
    df["Method"] = df["method"].map(_label_method)
    df["ARI"] = df.apply(lambda r: _metric_str(r.get("ari_mean"), r.get("ari_sd"), 3), axis=1)
    df["NMI"] = df.apply(lambda r: _metric_str(r.get("nmi_mean"), r.get("nmi_sd"), 3), axis=1)
    df["ACC"] = df.apply(lambda r: _metric_str(r.get("acc_mean"), r.get("acc_sd"), 3), axis=1)
    df["Runtime (s)"] = df.apply(lambda r: _metric_str(r.get("wallclock_total_sec_mean"), r.get("wallclock_total_sec_sd"), 2), axis=1)
    df["Final $K$"] = df.apply(lambda r: _count_str(r.get("final_K_mean"), r.get("final_K_sd")), axis=1)
    df["Selected dims"] = df.apply(lambda r: _count_str(r.get("selected_dims_count_mean"), r.get("selected_dims_count_sd")), axis=1)
    df["Splits"] = df.apply(lambda r: _count_str(r.get("split_count_mean"), r.get("split_count_sd")), axis=1)
    out = df[["Dataset", "Method", "ARI", "NMI", "ACC", "Runtime (s)", "Final $K$", "Selected dims", "Splits"]].copy()
    method_order = {"DIVI-Info": 0, "KMeans": 1, "GMM": 2, "SPKM": 3}
    out["_method_order"] = out["Method"].map(method_order).fillna(99)
    out["_dataset_order"] = out["Dataset"].map({"ISOLET": 0, "20NG": 1}).fillna(99)
    out = out.sort_values(["_dataset_order", "_method_order"]).drop(columns=["_method_order", "_dataset_order"])
    return out


def _make_runtime_scaling_table(runtime_summary: pd.DataFrame, axis: str) -> pd.DataFrame:
    df = runtime_summary[runtime_summary["runtime_axis"] == axis].copy()
    if df.empty:
        return df
    df["method_label"] = df["method"].map(_label_method)
    keep = [
        "dataset_variant", "method", "method_label", "runtime_axis", "runtime_value", "n_runs",
        "ari_mean", "ari_sd", "nmi_mean", "nmi_sd",
        "final_K_mean", "final_K_sd",
        "selected_dims_count_mean", "selected_dims_count_sd",
        "wallclock_total_sec_mean", "wallclock_total_sec_sd",
        "peak_memory_mb_mean", "peak_memory_mb_sd",
        "split_count_mean", "split_count_sd",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values(["method_label", "runtime_value"])


def _make_sensitivity_long_table(sens_summary: pd.DataFrame, factor: str, synthetic_only: bool | None = None) -> pd.DataFrame:
    df = sens_summary[sens_summary["sensitivity_factor"] == factor].copy()
    if synthetic_only is True:
        df = df[df["dataset"] == "synthetic"].copy()
    elif synthetic_only is False:
        df = df[df["dataset"] != "synthetic"].copy()
    if df.empty:
        return df

    df["dataset_label"] = np.where(
        df["dataset"] == "synthetic",
        df["dataset_variant"],
        df["dataset"].map(_label_dataset),
    )
    df["factor_label"] = df["sensitivity_factor"].map(_label_factor)
    keep = [
        "dataset", "dataset_variant", "dataset_label", "sensitivity_factor", "factor_label", "sensitivity_value", "n_runs",
        "ari_mean", "ari_sd", "nmi_mean", "nmi_sd", "f1_feature_mean", "f1_feature_sd",
        "final_K_mean", "final_K_sd", "selected_dims_count_mean", "selected_dims_count_sd",
        "wallclock_total_sec_mean", "wallclock_total_sec_sd", "split_count_mean", "split_count_sd",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values(["dataset_label", "sensitivity_value"])


def _make_sensitivity_display_table(sens_long: pd.DataFrame) -> pd.DataFrame:
    if sens_long.empty:
        return sens_long
    df = sens_long.copy()
    df["Sensitivity value"] = df["sensitivity_value"]
    df["ARI"] = df.apply(lambda r: _metric_str(r.get("ari_mean"), r.get("ari_sd"), 3), axis=1)
    df["NMI"] = df.apply(lambda r: _metric_str(r.get("nmi_mean"), r.get("nmi_sd"), 3), axis=1)
    if "f1_feature_mean" in df.columns:
        df["Feature F1"] = df.apply(lambda r: _metric_str(r.get("f1_feature_mean"), r.get("f1_feature_sd"), 3), axis=1)
    else:
        df["Feature F1"] = "--"
    df["Runtime (s)"] = df.apply(lambda r: _metric_str(r.get("wallclock_total_sec_mean"), r.get("wallclock_total_sec_sd"), 2), axis=1)
    df["Final $K$"] = df.apply(lambda r: _count_str(r.get("final_K_mean"), r.get("final_K_sd")), axis=1)
    df["Selected dims"] = df.apply(lambda r: _count_str(r.get("selected_dims_count_mean"), r.get("selected_dims_count_sd")), axis=1)
    df["Splits"] = df.apply(lambda r: _count_str(r.get("split_count_mean"), r.get("split_count_sd")), axis=1)
    return df[["dataset_label", "Sensitivity value", "ARI", "NMI", "Feature F1", "Runtime (s)", "Final $K$", "Selected dims", "Splits"]].rename(columns={"dataset_label": "Dataset"})


def _plot_runtime(fig_df: pd.DataFrame, axis: str, outdir: Path, formats: list[str]) -> None:
    if plt is None or fig_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method_label, sub in fig_df.groupby("method_label", dropna=False):
        sub = sub.sort_values("runtime_value")
        ax.errorbar(sub["runtime_value"], sub["wallclock_total_sec_mean"], yerr=sub["wallclock_total_sec_sd"], marker="o", label=method_label)
    ax.set_xlabel(axis)
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtime scaling with {axis}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ensure_dir(outdir)
    for fmt in formats:
        fig.savefig(outdir / f"runtime_{axis.lower()}scale.{fmt}", bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity(fig_df: pd.DataFrame, factor: str, outdir: Path, formats: list[str], metric: str = "ari_mean") -> None:
    if plt is None or fig_df.empty or metric not in fig_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for dataset_label, sub in fig_df.groupby("dataset_label", dropna=False):
        sub = sub.sort_values("sensitivity_value")
        err_col = metric.replace("_mean", "_sd")
        yerr = sub[err_col] if err_col in sub.columns else None
        ax.errorbar(sub["sensitivity_value"], sub[metric], yerr=yerr, marker="o", label=dataset_label)
    ax.set_xlabel(_label_factor(factor))
    ax.set_ylabel("ARI" if metric == "ari_mean" else metric)
    ax.set_title(f"Sensitivity: {factor}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ensure_dir(outdir)
    for fmt in formats:
        fig.savefig(outdir / f"sensitivity_{factor}_{metric.replace('_mean','')}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate DIVI revision runs into paper-ready tables/CSVs/LaTeX.")
    ap.add_argument("--root", type=str, default="outputs_revision")
    ap.add_argument("--outdir", type=str, default="outputs_revision/aggregated")
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--plot-formats", type=str, default="pdf,png")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = ensure_dir(args.outdir)
    figure_csv_dir = ensure_dir(outdir / "figure_csv")
    table_dir = ensure_dir(outdir / "paper_tables")
    latex_dir = ensure_dir(outdir / "latex")
    figure_dir = ensure_dir(outdir / "figures")

    run_files = sorted(root.rglob("runs.csv"))
    if not run_files:
        raise FileNotFoundError(f"No runs.csv found under {root}")

    dfs = []
    for path in run_files:
        df = pd.read_csv(path)
        df["source_runs_csv"] = str(path)
        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True, sort=False)
    master = _with_defaults(master, [
        "experiment_id", "dataset", "dataset_variant", "method", "split_name", "runtime_axis", "runtime_value",
        "sensitivity_factor", "sensitivity_value", "ari", "nmi", "acc", "f1_feature",
        "final_K", "selected_dims_count", "wallclock_total_sec", "peak_memory_mb", "split_count",
    ])
    _write_csv(master, outdir / "master_runs.csv")

    runtime = master[master["split_name"].astype(str).str.contains("runtime", na=False)].copy()
    sens = master[master["split_name"].astype(str).str.contains("sens|confirm", na=False)].copy()

    if not runtime.empty:
        runtime = _with_defaults(runtime, RUNTIME_COLUMNS)
        runtime_summary = _agg_summary(
            runtime,
            group_cols=["dataset", "dataset_variant", "method", "runtime_axis", "runtime_value"],
            metrics=["ari", "nmi", "acc", "final_K", "selected_dims_count", "wallclock_total_sec", "peak_memory_mb", "split_count"],
        )
        runtime_summary["dataset_label"] = runtime_summary["dataset"].map(_label_dataset)
        runtime_summary["method_label"] = runtime_summary["method"].map(_label_method)
        _write_csv(runtime_summary, outdir / "runtime_summary_long.csv")

        runtime_real_table = _make_runtime_real_table(runtime_summary)
        if not runtime_real_table.empty:
            _write_csv(runtime_real_table, table_dir / "table_runtime_real_main.csv")
            _write_latex(
                runtime_real_table,
                latex_dir / "table_runtime_real_main.tex",
                caption="Real-data runtime and clustering performance across methods.",
                label="tab:runtime-real-main",
                column_format="llccccccc",
            )

        for axis in ["D", "N"]:
            fig_df = _make_runtime_scaling_table(runtime_summary, axis)
            if fig_df.empty:
                continue
            suffix = "dscale" if axis == "D" else "nscale"
            _write_csv(fig_df, figure_csv_dir / f"fig_runtime_{suffix}.csv")
            # Also keep a paper-table version for appendix / supplement.
            pretty = fig_df.copy()
            pretty["ARI"] = pretty.apply(lambda r: _metric_str(r.get("ari_mean"), r.get("ari_sd"), 3), axis=1)
            pretty["NMI"] = pretty.apply(lambda r: _metric_str(r.get("nmi_mean"), r.get("nmi_sd"), 3), axis=1)
            pretty["Runtime (s)"] = pretty.apply(lambda r: _metric_str(r.get("wallclock_total_sec_mean"), r.get("wallclock_total_sec_sd"), 2), axis=1)
            pretty["Final $K$"] = pretty.apply(lambda r: _count_str(r.get("final_K_mean"), r.get("final_K_sd")), axis=1)
            pretty["Selected dims"] = pretty.apply(lambda r: _count_str(r.get("selected_dims_count_mean"), r.get("selected_dims_count_sd")), axis=1)
            pretty = pretty[["dataset_variant", "method_label", "runtime_value", "ARI", "NMI", "Runtime (s)", "Final $K$", "Selected dims"]]
            _write_csv(pretty, table_dir / f"table_runtime_{suffix}.csv")
            _write_latex(
                pretty,
                latex_dir / f"table_runtime_{suffix}.tex",
                caption=f"Synthetic runtime scaling summary for {axis}-axis variation.",
                label=f"tab:runtime-{suffix}",
                column_format="llrccccc",
            )
            if args.make_plots:
                _plot_runtime(fig_df, axis, figure_dir, [fmt for fmt in args.plot_formats.split(",") if fmt])

    if not sens.empty:
        sens = _with_defaults(sens, SENS_COLUMNS)
        sens_summary = _agg_summary(
            sens,
            group_cols=["dataset", "dataset_variant", "method", "sensitivity_factor", "sensitivity_value"],
            metrics=["ari", "nmi", "f1_feature", "acc", "final_K", "selected_dims_count", "wallclock_total_sec", "split_count"],
        )
        sens_summary["dataset_label"] = np.where(
            sens_summary["dataset"] == "synthetic",
            sens_summary["dataset_variant"],
            sens_summary["dataset"].map(_label_dataset),
        )
        sens_summary["method_label"] = sens_summary["method"].map(_label_method)
        sens_summary["factor_label"] = sens_summary["sensitivity_factor"].map(_label_factor)
        _write_csv(sens_summary, outdir / "sensitivity_summary_long.csv")

        for factor in [f for f in sens_summary["sensitivity_factor"].dropna().unique() if str(f) != "nan"]:
            for synthetic_only, suffix in [(True, "synthetic"), (False, "real")]:
                fig_df = _make_sensitivity_long_table(sens_summary, factor=factor, synthetic_only=synthetic_only)
                if fig_df.empty:
                    continue
                _write_csv(fig_df, figure_csv_dir / f"fig_sens_{factor}_{suffix}.csv")
                display_table = _make_sensitivity_display_table(fig_df)
                _write_csv(display_table, table_dir / f"table_sensitivity_{factor}_{suffix}.csv")
                _write_latex(
                    display_table,
                    latex_dir / f"table_sensitivity_{factor}_{suffix}.tex",
                    caption=f"Sensitivity summary for {factor} on {suffix} datasets.",
                    label=f"tab:sens-{factor}-{suffix}",
                    column_format="llccccccc",
                )
                if args.make_plots and synthetic_only:
                    _plot_sensitivity(fig_df, factor, figure_dir, [fmt for fmt in args.plot_formats.split(",") if fmt], metric="ari_mean")
                    if "f1_feature_mean" in fig_df.columns and not fig_df["f1_feature_mean"].isna().all():
                        _plot_sensitivity(fig_df, factor, figure_dir, [fmt for fmt in args.plot_formats.split(",") if fmt], metric="f1_feature_mean")

    manifest = pd.DataFrame(
        {
            "artifact_type": ["master_runs", "runtime_summary", "sensitivity_summary", "paper_tables", "latex", "figure_csv", "figures"],
            "path": [
                str(outdir / "master_runs.csv"),
                str(outdir / "runtime_summary_long.csv"),
                str(outdir / "sensitivity_summary_long.csv"),
                str(table_dir),
                str(latex_dir),
                str(figure_csv_dir),
                str(figure_dir),
            ],
        }
    )
    _write_csv(manifest, outdir / "artifact_manifest.csv")


if __name__ == "__main__":
    main()
