#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import run_gmm, run_kmeans, run_spkm
from data_utils import load_20ng_subset_embeddings, load_isolet_subset
from divi_core import DIVIClustering
from experiment_utils import (
    append_run_record,
    clustering_metrics,
    COMMON_COLUMNS,
    ensure_dir,
    get_device_info,
    get_git_commit,
    get_peak_memory_mb,
    load_yaml_config,
    make_experiment_id,
    safe_json,
    set_global_seed,
    summarize_results,
    utc_timestamp,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset", type=str, choices=["isolet", "20ng"], required=True)
    ap.add_argument("--outdir", type=str, default="outputs_revision/runtime_real")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--prior-mode", type=int, default=1)
    ap.add_argument("--beta-mult", type=float, default=1.0)
    ap.add_argument("--split-interval", type=int, default=80)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--temp-start", type=float, default=1.0)
    ap.add_argument("--temp-end", type=float, default=0.1)
    ap.add_argument("--spkm-l1-bound", type=float, default=4.0)
    ap.add_argument("--isolet-classes", type=int, default=5)
    ap.add_argument("--ng-max-docs", type=int, default=2000)
    ap.add_argument("--ng-data-home", type=str, default=None)
    ap.add_argument("--ng-model-name", type=str, default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    for k, v in cfg.items():
        if hasattr(args, k.replace("-", "_")):
            setattr(args, k.replace("-", "_"), v)

    outdir = ensure_dir(Path(args.outdir) / args.dataset)
    runs_csv = outdir / "runs.csv"
    columns = COMMON_COLUMNS + ["runtime_axis", "runtime_value", "baseline_timeout_sec", "baseline_converged_flag"]
    device_type, device_name = get_device_info()
    git_commit = get_git_commit()

    if args.dataset == "isolet":
        data = load_isolet_subset(target_classes=args.isolet_classes, standardize=True)
        K_true = args.isolet_classes
        split_name = "runtime_isolet"
    else:
        data = load_20ng_subset_embeddings(
            max_docs=args.ng_max_docs,
            data_home=args.ng_data_home,
            model_name=args.ng_model_name,
        )
        K_true = len(np.unique(data["y"]))
        split_name = "runtime_20ng"

    X = data["X"]
    y_true = data["y"]
    dataset_variant = data["dataset"]
    N, D = X.shape

    for run_seed in range(args.n_runs):
        set_global_seed(run_seed)
        divi = DIVIClustering(
            split_threshold=None,
            split_interval=args.split_interval,
            max_epochs=args.max_epochs,
            lr=args.lr,
            beta_mult=args.beta_mult,
            temperature_start=args.temp_start,
            temperature_end=args.temp_end,
            verbose=False,
        )
        divi.fit(X, use_prior=args.prior_mode)
        y_pred = divi.predict(X)
        metrics = clustering_metrics(y_true, y_pred)

        record = {
            "experiment_id": make_experiment_id("runtime_real", dataset_variant, split_name, run_seed, "divi"),
            "script_name": Path(__file__).name,
            "timestamp": utc_timestamp(),
            "git_commit": git_commit,
            "hostname": device_name,
            "device_type": device_type,
            "device_name": device_name,
            "dataset": args.dataset,
            "dataset_variant": dataset_variant,
            "split_name": split_name,
            "N": N,
            "D": D,
            "K_true": K_true,
            "informative_ratio": np.nan,
            "noise_ratio": np.nan,
            "noise_sigma": np.nan,
            "data_seed": np.nan,
            "run_seed": run_seed,
            "method": "divi",
            "prior_mode": args.prior_mode,
            "baseline_name": "divi",
            "beta": divi.fit_summary_["beta"],
            "beta_mult": args.beta_mult,
            "Tsplit": args.split_interval,
            "tau_mode": "auto_entropy",
            "tau": divi.split_threshold,
            "tau_mult": 1.0,
            "lr": args.lr,
            "temp_start": args.temp_start,
            "temp_end": args.temp_end,
            "max_epochs": args.max_epochs,
            "epochs_completed": divi.fit_summary_["epochs_completed"],
            "final_K": divi.fit_summary_["final_K"],
            "split_count": divi.fit_summary_["split_count"],
            "first_split_epoch": divi.fit_summary_["first_split_epoch"],
            "last_split_epoch": divi.fit_summary_["last_split_epoch"],
            "split_epochs_json": safe_json(divi.fit_summary_["split_epochs"]),
            "ari": metrics["ari"],
            "nmi": metrics["nmi"],
            "f1_feature": np.nan,
            "acc": metrics["acc"],
            "selected_dims_count": divi.fit_summary_["selected_dims_count"],
            "selected_dims_ratio": divi.fit_summary_["selected_dims_ratio"],
            "mean_phi": divi.fit_summary_["mean_phi"],
            "median_phi": divi.fit_summary_["median_phi"],
            "objective_final": divi.fit_summary_["objective_final"],
            "objective_best": divi.fit_summary_["objective_best"],
            "objective_gap": divi.fit_summary_["objective_gap"],
            "nll_final": divi.fit_summary_["nll_final"],
            "kl_final": divi.fit_summary_["kl_final"],
            "wallclock_total_sec": divi.fit_summary_["wallclock_total_sec"],
            "wallclock_stepA_sec": divi.fit_summary_["wallclock_stepA_sec"],
            "wallclock_train_sec": divi.fit_summary_["wallclock_train_sec"],
            "wallclock_split_diag_sec": divi.fit_summary_["wallclock_split_diag_sec"],
            "wallclock_post_sec": divi.fit_summary_["wallclock_post_sec"],
            "time_per_epoch_sec": divi.fit_summary_["time_per_epoch_sec"],
            "peak_memory_mb": get_peak_memory_mb(),
            "status": "ok",
            "timeout_flag": False,
            "error_message": "",
            "artifact_dir": "",
            "runtime_axis": "dataset",
            "runtime_value": args.dataset,
            "baseline_timeout_sec": np.nan,
            "baseline_converged_flag": True,
        }
        append_run_record(runs_csv, record, columns=columns)

        for baseline_name, runner in {
            "kmeans_oracle": lambda: run_kmeans(X, K_true, run_seed),
            "gmm_oracle": lambda: run_gmm(X, K_true, run_seed),
            "spkm": lambda: run_spkm(X, K_true, args.spkm_l1_bound),
        }.items():
            t0 = pd.Timestamp.utcnow()
            result = runner()
            elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
            metrics_b = clustering_metrics(y_true, result.labels)
            selected_count = int(result.selected_mask.sum()) if result.selected_mask is not None else np.nan
            selected_ratio = float(result.selected_mask.mean()) if result.selected_mask is not None else np.nan
            record_b = {
                "experiment_id": make_experiment_id("runtime_real", dataset_variant, split_name, run_seed, baseline_name),
                "script_name": Path(__file__).name,
                "timestamp": utc_timestamp(),
                "git_commit": git_commit,
                "hostname": device_name,
                "device_type": device_type,
                "device_name": device_name,
                "dataset": args.dataset,
                "dataset_variant": dataset_variant,
                "split_name": split_name,
                "N": N,
                "D": D,
                "K_true": K_true,
                "informative_ratio": np.nan,
                "noise_ratio": np.nan,
                "noise_sigma": np.nan,
                "data_seed": np.nan,
                "run_seed": run_seed,
                "method": baseline_name,
                "prior_mode": np.nan,
                "baseline_name": baseline_name,
                "beta": np.nan,
                "beta_mult": np.nan,
                "Tsplit": np.nan,
                "tau_mode": "",
                "tau": np.nan,
                "tau_mult": np.nan,
                "lr": np.nan,
                "temp_start": np.nan,
                "temp_end": np.nan,
                "max_epochs": np.nan,
                "epochs_completed": np.nan,
                "final_K": K_true,
                "split_count": 0,
                "first_split_epoch": np.nan,
                "last_split_epoch": np.nan,
                "split_epochs_json": "[]",
                "ari": metrics_b["ari"],
                "nmi": metrics_b["nmi"],
                "f1_feature": np.nan,
                "acc": metrics_b["acc"],
                "selected_dims_count": selected_count,
                "selected_dims_ratio": selected_ratio,
                "mean_phi": np.nan,
                "median_phi": np.nan,
                "objective_final": np.nan,
                "objective_best": np.nan,
                "objective_gap": np.nan,
                "nll_final": np.nan,
                "kl_final": np.nan,
                "wallclock_total_sec": elapsed,
                "wallclock_stepA_sec": np.nan,
                "wallclock_train_sec": elapsed,
                "wallclock_split_diag_sec": np.nan,
                "wallclock_post_sec": np.nan,
                "time_per_epoch_sec": np.nan,
                "peak_memory_mb": get_peak_memory_mb(),
                "status": "ok",
                "timeout_flag": False,
                "error_message": "",
                "artifact_dir": "",
                "runtime_axis": "dataset",
                "runtime_value": args.dataset,
                "baseline_timeout_sec": np.nan,
                "baseline_converged_flag": True,
            }
            append_run_record(runs_csv, record_b, columns=columns)

    summarize_results(
        runs_csv=runs_csv,
        group_cols=["dataset", "method", "runtime_axis", "runtime_value"],
        out_csv=outdir / "summary.csv",
    )


if __name__ == "__main__":
    main()
