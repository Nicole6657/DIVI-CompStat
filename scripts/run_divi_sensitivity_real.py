#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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


def parse_float_list(x: str) -> list[float]:
    return [float(v) for v in x.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset", type=str, choices=["isolet", "20ng"], required=True)
    ap.add_argument("--outdir", type=str, default="outputs_revision/sensitivity_real")
    ap.add_argument("--factor", choices=["beta_mult", "Tsplit"], required=True)
    ap.add_argument("--values", type=str, required=True)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--prior-mode", type=int, default=1)
    ap.add_argument("--beta-mult", type=float, default=1.0)
    ap.add_argument("--split-interval", type=int, default=80)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--temp-start", type=float, default=1.0)
    ap.add_argument("--temp-end", type=float, default=0.1)
    ap.add_argument("--isolet-classes", type=int, default=5)
    ap.add_argument("--ng-max-docs", type=int, default=2000)
    ap.add_argument("--ng-data-home", type=str, default=None)
    ap.add_argument("--ng-model-name", type=str, default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    for k, v in cfg.items():
        if hasattr(args, k.replace("-", "_")):
            setattr(args, k.replace("-", "_"), v)

    outdir = ensure_dir(Path(args.outdir) / args.dataset / args.factor)
    runs_csv = outdir / "runs.csv"
    columns = COMMON_COLUMNS + ["sensitivity_factor", "sensitivity_value", "default_reference_value", "fixed_data_flag", "confirmatory_flag", "realdata_metric_note"]
    device_type, device_name = get_device_info()
    git_commit = get_git_commit()

    if args.dataset == "isolet":
        data = load_isolet_subset(target_classes=args.isolet_classes, standardize=True)
        split_name = "confirm_isolet"
    else:
        data = load_20ng_subset_embeddings(
            max_docs=args.ng_max_docs,
            data_home=args.ng_data_home,
            model_name=args.ng_model_name,
        )
        split_name = "confirm_20ng"

    X = data["X"]
    y_true = data["y"]
    dataset_variant = data["dataset"]
    N, D = X.shape
    K_true = len(np.unique(y_true))
    values = parse_float_list(args.values)

    for sens_value in values:
        for run_seed in range(args.n_runs):
            kwargs = dict(
                split_threshold=None,
                split_interval=args.split_interval,
                max_epochs=args.max_epochs,
                lr=args.lr,
                beta_mult=args.beta_mult,
                temperature_start=args.temp_start,
                temperature_end=args.temp_end,
                verbose=False,
            )
            if args.factor == "beta_mult":
                kwargs["beta_mult"] = sens_value
            else:
                kwargs["split_interval"] = int(round(sens_value))

            set_global_seed(run_seed)
            model = DIVIClustering(**kwargs)
            model.fit(X, use_prior=args.prior_mode)
            y_pred = model.predict(X)
            metrics = clustering_metrics(y_true, y_pred)

            default_ref = args.beta_mult if args.factor == "beta_mult" else args.split_interval
            record = {
                "experiment_id": make_experiment_id("sens_real", dataset_variant, args.factor, run_seed, "divi"),
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
                "beta": model.fit_summary_["beta"],
                "beta_mult": kwargs["beta_mult"],
                "Tsplit": kwargs["split_interval"],
                "tau_mode": "auto_entropy",
                "tau": model.split_threshold,
                "tau_mult": 1.0,
                "lr": kwargs["lr"],
                "temp_start": args.temp_start,
                "temp_end": kwargs["temperature_end"],
                "max_epochs": args.max_epochs,
                "epochs_completed": model.fit_summary_["epochs_completed"],
                "final_K": model.fit_summary_["final_K"],
                "split_count": model.fit_summary_["split_count"],
                "first_split_epoch": model.fit_summary_["first_split_epoch"],
                "last_split_epoch": model.fit_summary_["last_split_epoch"],
                "split_epochs_json": safe_json(model.fit_summary_["split_epochs"]),
                "ari": metrics["ari"],
                "nmi": metrics["nmi"],
                "f1_feature": np.nan,
                "acc": metrics["acc"],
                "selected_dims_count": model.fit_summary_["selected_dims_count"],
                "selected_dims_ratio": model.fit_summary_["selected_dims_ratio"],
                "mean_phi": model.fit_summary_["mean_phi"],
                "median_phi": model.fit_summary_["median_phi"],
                "objective_final": model.fit_summary_["objective_final"],
                "objective_best": model.fit_summary_["objective_best"],
                "objective_gap": model.fit_summary_["objective_gap"],
                "nll_final": model.fit_summary_["nll_final"],
                "kl_final": model.fit_summary_["kl_final"],
                "wallclock_total_sec": model.fit_summary_["wallclock_total_sec"],
                "wallclock_stepA_sec": model.fit_summary_["wallclock_stepA_sec"],
                "wallclock_train_sec": model.fit_summary_["wallclock_train_sec"],
                "wallclock_split_diag_sec": model.fit_summary_["wallclock_split_diag_sec"],
                "wallclock_post_sec": model.fit_summary_["wallclock_post_sec"],
                "time_per_epoch_sec": model.fit_summary_["time_per_epoch_sec"],
                "peak_memory_mb": get_peak_memory_mb(),
                "status": "ok",
                "timeout_flag": False,
                "error_message": "",
                "artifact_dir": "",
                "sensitivity_factor": args.factor,
                "sensitivity_value": sens_value,
                "default_reference_value": default_ref,
                "fixed_data_flag": True,
                "confirmatory_flag": True,
                "realdata_metric_note": "real-data confirmatory sensitivity; no feature-F1",
            }
            append_run_record(runs_csv, record, columns=columns)

    summarize_results(
        runs_csv=runs_csv,
        group_cols=["dataset_variant", "sensitivity_factor", "sensitivity_value"],
        out_csv=outdir / "summary.csv",
    )


if __name__ == "__main__":
    main()
