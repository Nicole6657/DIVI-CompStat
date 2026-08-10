#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import run_gmm, run_kmeans, run_spkm
from data_utils import generate_synthetic_data_sweet_spot, standardize_features
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


def parse_int_list(x: str) -> list[int]:
    return [int(v) for v in x.split(",") if v.strip()]


def run_divi(X, args, run_seed, dataset_variant, split_name):
    set_global_seed(run_seed)
    model = DIVIClustering(
        split_threshold=None,
        split_interval=args.split_interval,
        max_epochs=args.max_epochs,
        lr=args.lr,
        beta_mult=args.beta_mult,
        temperature_start=args.temp_start,
        temperature_end=args.temp_end,
        verbose=False,
    )
    model.fit(X, use_prior=args.prior_mode)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs_revision/runtime_synth")
    ap.add_argument("--n-list", type=str, default="200,500,1000,2000,5000")
    ap.add_argument("--d-list", type=str, default="100,500,1000,2000,5000")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--data-seed-d", type=int, default=20260325)
    ap.add_argument("--data-seed-n", type=int, default=20260326)
    ap.add_argument("--fixed-n", type=int, default=1000)
    ap.add_argument("--fixed-d", type=int, default=1000)
    ap.add_argument("--K-true", type=int, default=3)
    ap.add_argument("--D-signal", type=int, default=10)
    ap.add_argument("--noise-scale", type=float, default=3.0)
    ap.add_argument("--prior-mode", type=int, default=1)
    ap.add_argument("--beta-mult", type=float, default=1.0)
    ap.add_argument("--split-interval", type=int, default=80)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--temp-start", type=float, default=1.0)
    ap.add_argument("--temp-end", type=float, default=0.1)
    ap.add_argument("--spkm-l1-bound", type=float, default=4.0)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    for k, v in cfg.items():
        if hasattr(args, k.replace("-", "_")):
            setattr(args, k.replace("-", "_"), v)

    outdir = ensure_dir(args.outdir)
    runs_csv = outdir / "runs.csv"
    device_type, device_name = get_device_info()
    git_commit = get_git_commit()

    n_list = parse_int_list(args.n_list)
    d_list = parse_int_list(args.d_list)

    columns = COMMON_COLUMNS + ["runtime_axis", "runtime_value", "baseline_timeout_sec", "baseline_converged_flag"]

    for axis, values in [("D", d_list), ("N", n_list)]:
        for value in values:
            if axis == "D":
                N = args.fixed_n
                D = value
                data_seed = args.data_seed_d + value
            else:
                N = value
                D = args.fixed_d
                data_seed = args.data_seed_n + value

            X_raw, y_true = generate_synthetic_data_sweet_spot(
                N=N,
                D=D,
                n_signal=args.D_signal,
                noise_scale=args.noise_scale,
                random_state=data_seed,
            )
            X = standardize_features(X_raw)
            dataset_variant = f"synthetic_n{N}_d{D}_noise{args.noise_scale}"

            for run_seed in range(args.n_runs):
                split_name = f"runtime_{axis.lower()}scale"
                # DIVI
                divi = run_divi(X, args, run_seed, dataset_variant, split_name)
                y_pred = divi.predict(X)
                metrics = clustering_metrics(y_true, y_pred)
                phi = divi.get_phi()
                record = {
                    "experiment_id": make_experiment_id("runtime_synth", dataset_variant, split_name, run_seed, "divi"),
                    "script_name": Path(__file__).name,
                    "timestamp": utc_timestamp(),
                    "git_commit": git_commit,
                    "hostname": device_name,
                    "device_type": device_type,
                    "device_name": device_name,
                    "dataset": "synthetic",
                    "dataset_variant": dataset_variant,
                    "split_name": split_name,
                    "N": N,
                    "D": D,
                    "K_true": args.K_true,
                    "informative_ratio": args.D_signal / D,
                    "noise_ratio": 1 - args.D_signal / D,
                    "noise_sigma": args.noise_scale,
                    "data_seed": data_seed,
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
                    "runtime_axis": axis,
                    "runtime_value": value,
                    "baseline_timeout_sec": np.nan,
                    "baseline_converged_flag": True,
                }
                append_run_record(runs_csv, record, columns=columns)

                # KMeans
                for baseline_name, runner in {
                    "kmeans_oracle": lambda: run_kmeans(X, args.K_true, run_seed),
                    "gmm_oracle": lambda: run_gmm(X, args.K_true, run_seed),
                    "spkm": lambda: run_spkm(X, args.K_true, args.spkm_l1_bound),
                }.items():
                    t0 = pd.Timestamp.utcnow()
                    result = runner()
                    elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
                    metrics_b = clustering_metrics(y_true, result.labels)
                    selected_count = int(result.selected_mask.sum()) if result.selected_mask is not None else np.nan
                    selected_ratio = (
                        float(result.selected_mask.mean()) if result.selected_mask is not None else np.nan
                    )
                    record_b = {
                        "experiment_id": make_experiment_id("runtime_synth", dataset_variant, split_name, run_seed, baseline_name),
                        "script_name": Path(__file__).name,
                        "timestamp": utc_timestamp(),
                        "git_commit": git_commit,
                        "hostname": device_name,
                        "device_type": device_type,
                        "device_name": device_name,
                        "dataset": "synthetic",
                        "dataset_variant": dataset_variant,
                        "split_name": split_name,
                        "N": N,
                        "D": D,
                        "K_true": args.K_true,
                        "informative_ratio": args.D_signal / D,
                        "noise_ratio": 1 - args.D_signal / D,
                        "noise_sigma": args.noise_scale,
                        "data_seed": data_seed,
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
                        "final_K": args.K_true,
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
                        "runtime_axis": axis,
                        "runtime_value": value,
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
