from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from divi_core import DIVIClustering
from experiment_utils import (
    append_run_record,
    clustering_metrics,
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
from robust_synth_utils import SCENARIO_GENERATORS



def feature_f1_from_truth_mask(truth_mask: np.ndarray, phi_probs: np.ndarray, threshold: float = 0.5) -> float:
    truth_mask = np.asarray(truth_mask, dtype=int)
    pred_mask = (np.asarray(phi_probs) >= threshold).astype(int)

    tp = int(np.sum((truth_mask == 1) & (pred_mask == 1)))
    fp = int(np.sum((truth_mask == 0) & (pred_mask == 1)))
    fn = int(np.sum((truth_mask == 1) & (pred_mask == 0)))

    if tp == 0 and fp == 0 and fn == 0:
        return float("nan")
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DIVI misspecification robustness experiments on synthetic data."
    )
    parser.add_argument("--config", type=str, default="/mnt/data/defaults_divi.yaml")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["heavy_tail_signal", "correlated_noise"],
        choices=sorted(SCENARIO_GENERATORS.keys()),
        help="One or more misspecified synthetic scenarios.",
    )
    parser.add_argument("--N-list", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--D", type=int, default=100)
    parser.add_argument("--n-signal", type=int, default=10)
    parser.add_argument("--K-true", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--data-seed-offset", type=int, default=1000)
    parser.add_argument("--run-seed-offset", type=int, default=2000)
    parser.add_argument("--prior-modes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--phi-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="./outputs/misspec_robustness")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Scenario-specific knobs
    parser.add_argument("--signal-df", type=float, default=5.0)
    parser.add_argument("--signal-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=3.0)
    parser.add_argument("--noise-rho", type=float, default=0.6)
    parser.add_argument("--noise-block-size", type=int, default=10)
    parser.add_argument("--latent-signal-dim", type=int, default=3)
    return parser.parse_args()



def scenario_kwargs(args: argparse.Namespace, scenario: str, N: int, data_seed: int) -> Dict[str, Any]:
    common = {
        "N": N,
        "D": args.D,
        "n_signal": args.n_signal,
        "K": args.K_true,
        "signal_scale": args.signal_scale,
        "noise_scale": args.noise_scale,
        "standardize": True,
        "random_state": data_seed,
    }
    if scenario == "heavy_tail_signal":
        common.update({"signal_df": args.signal_df})
    elif scenario == "correlated_noise":
        common.update({"noise_rho": args.noise_rho, "noise_block_size": args.noise_block_size})
    elif scenario == "rotated_signal":
        common.update({"latent_signal_dim": args.latent_signal_dim})
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return common



def run_divi_once(
    X: np.ndarray,
    y: np.ndarray,
    truth_mask: np.ndarray,
    prior_mode: int,
    run_seed: int,
    cfg: Dict[str, Any],
    verbose: bool,
) -> Dict[str, Any]:
    set_global_seed(run_seed)
    model = DIVIClustering(
        split_interval=int(cfg.get("split_interval", 80)),
        max_epochs=int(cfg.get("max_epochs", 300)),
        lr=float(cfg.get("lr", 0.01)),
        beta_mult=float(cfg.get("beta_mult", 1.0)),
        temperature_start=float(cfg.get("temp_start", 1.0)),
        temperature_end=float(cfg.get("temp_end", 0.1)),
        verbose=verbose,
    )
    model.fit(X, use_prior=int(prior_mode))

    y_pred = model.predict(X)
    phi = model.get_phi()
    met = clustering_metrics(y, y_pred)
    f1_feature = feature_f1_from_truth_mask(truth_mask, phi, threshold=0.5)

    out = {
        **met,
        "f1_feature": f1_feature,
        "selected_dims_count": int(np.sum(phi >= 0.5)),
        "selected_dims_ratio": float(np.mean(phi >= 0.5)),
        "mean_phi": float(np.mean(phi)),
        "median_phi": float(np.median(phi)),
        "peak_memory_mb": get_peak_memory_mb(),
        **model.fit_summary_,
    }
    return out



def run_baselines_once(X: np.ndarray, y: np.ndarray, K_true: int, run_seed: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    km = KMeans(n_clusters=K_true, n_init=20, random_state=run_seed)
    y_km = km.fit_predict(X)
    met_km = clustering_metrics(y, y_km)
    records.append(
        {
            "method": "kmeans_oracle",
            "baseline_name": "KMeans",
            "prior_mode": np.nan,
            "ari": met_km["ari"],
            "nmi": met_km["nmi"],
            "acc": met_km["acc"],
            "f1_feature": np.nan,
            "final_K": int(K_true),
            "selected_dims_count": np.nan,
            "selected_dims_ratio": np.nan,
            "mean_phi": np.nan,
            "median_phi": np.nan,
            "split_count": 0,
            "peak_memory_mb": get_peak_memory_mb(),
            "status": "ok",
        }
    )

    gmm = GaussianMixture(n_components=K_true, covariance_type="diag", random_state=run_seed, reg_covar=1e-6)
    y_gmm = gmm.fit_predict(X)
    met_gmm = clustering_metrics(y, y_gmm)
    records.append(
        {
            "method": "gmm_oracle",
            "baseline_name": "GaussianMixtureDiag",
            "prior_mode": np.nan,
            "ari": met_gmm["ari"],
            "nmi": met_gmm["nmi"],
            "acc": met_gmm["acc"],
            "f1_feature": np.nan,
            "final_K": int(K_true),
            "selected_dims_count": np.nan,
            "selected_dims_ratio": np.nan,
            "mean_phi": np.nan,
            "median_phi": np.nan,
            "split_count": 0,
            "peak_memory_mb": get_peak_memory_mb(),
            "status": "ok",
        }
    )
    return records



def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    output_dir = ensure_dir(args.output_dir)
    runs_csv = output_dir / "runs.csv"
    summary_csv = output_dir / "summary_by_scenario_method.csv"

    device_type, device_name = get_device_info()
    script_name = Path(__file__).name
    git_commit = get_git_commit(cwd=str(Path(__file__).resolve().parent))

    for scenario in args.scenarios:
        generator = SCENARIO_GENERATORS[scenario]

        for N in args.N_list:
            for run_idx in range(args.n_runs):
                data_seed = args.data_seed_offset + 10000 * hash(scenario) % 1000 + run_idx + 100 * N
                run_seed = args.run_seed_offset + run_idx

                ds_kwargs = scenario_kwargs(args, scenario=scenario, N=N, data_seed=data_seed)
                ds = generator(**ds_kwargs)
                X = ds["X"]
                y = ds["y"]
                truth_mask = ds["truth_mask"]
                dataset = ds["dataset"]
                dataset_variant = ds["dataset_variant"]
                K_true = int(ds["K_true"])

                common = {
                    "script_name": script_name,
                    "timestamp": utc_timestamp(),
                    "git_commit": git_commit,
                    "hostname": "local",
                    "device_type": device_type,
                    "device_name": device_name,
                    "dataset": dataset,
                    "dataset_variant": dataset_variant,
                    "split_name": scenario,
                    "N": int(X.shape[0]),
                    "D": int(X.shape[1]),
                    "K_true": K_true,
                    "informative_ratio": float(np.mean(truth_mask)),
                    "noise_ratio": float(1.0 - np.mean(truth_mask)),
                    "noise_sigma": float(args.noise_scale),
                    "data_seed": int(data_seed),
                    "run_seed": int(run_seed),
                    "Tsplit": int(cfg.get("split_interval", 80)),
                    "tau_mode": "auto",
                    "tau": float(DIVIClustering.auto_split_threshold(D=int(X.shape[1]), sigma2=1.0)),
                    "tau_mult": 1.0,
                    "lr": float(cfg.get("lr", 0.01)),
                    "temp_start": float(cfg.get("temp_start", 1.0)),
                    "temp_end": float(cfg.get("temp_end", 0.1)),
                    "max_epochs": int(cfg.get("max_epochs", 300)),
                    "artifact_dir": str(output_dir),
                }

                for prior_mode in args.prior_modes:
                    record: Dict[str, Any] = {
                        **common,
                        "method": "DIVI",
                        "baseline_name": "DIVI",
                        "prior_mode": int(prior_mode),
                        "status": "ok",
                        "timeout_flag": 0,
                        "error_message": "",
                    }
                    record["experiment_id"] = make_experiment_id(
                        prefix=f"misspec{('_' + args.tag) if args.tag else ''}",
                        dataset=dataset_variant,
                        split_name=scenario,
                        run_seed=run_seed,
                        method=f"DIVI_p{prior_mode}",
                    )

                    try:
                        result = run_divi_once(
                            X=X,
                            y=y,
                            truth_mask=truth_mask,
                            prior_mode=prior_mode,
                            run_seed=run_seed,
                            cfg=cfg,
                            verbose=args.verbose,
                        )
                        record.update(result)
                        record["beta"] = result.get("beta", cfg.get("beta_mult", 1.0) * len(X))
                        record["split_epochs_json"] = safe_json(result.get("split_epochs", []))
                    except Exception as e:
                        record.update(
                            {
                                "status": "error",
                                "timeout_flag": 0,
                                "error_message": f"{type(e).__name__}: {e}",
                                "split_epochs_json": "[]",
                                "peak_memory_mb": get_peak_memory_mb(),
                            }
                        )
                        if args.verbose:
                            traceback.print_exc()

                    append_run_record(runs_csv, record)
                    print(
                        f"[DIVI] scenario={scenario:>18s} N={N:<4d} run={run_idx:<2d} prior={prior_mode} "
                        f"status={record['status']} ari={record.get('ari', float('nan')):.3f} "
                        f"f1={record.get('f1_feature', float('nan')):.3f} K={record.get('final_K', 'NA')}"
                    )

                if args.include_baselines:
                    baseline_records = run_baselines_once(X=X, y=y, K_true=K_true, run_seed=run_seed)
                    for base in baseline_records:
                        record = {
                            **common,
                            **base,
                            "experiment_id": make_experiment_id(
                                prefix=f"misspec{('_' + args.tag) if args.tag else ''}",
                                dataset=dataset_variant,
                                split_name=scenario,
                                run_seed=run_seed,
                                method=base["method"],
                            ),
                            "beta": np.nan,
                            "beta_mult": np.nan,
                            "epochs_completed": np.nan,
                            "first_split_epoch": np.nan,
                            "last_split_epoch": np.nan,
                            "split_epochs_json": "[]",
                            "objective_final": np.nan,
                            "objective_best": np.nan,
                            "objective_gap": np.nan,
                            "nll_final": np.nan,
                            "kl_final": np.nan,
                            "wallclock_total_sec": np.nan,
                            "wallclock_stepA_sec": np.nan,
                            "wallclock_train_sec": np.nan,
                            "wallclock_split_diag_sec": np.nan,
                            "wallclock_post_sec": np.nan,
                            "time_per_epoch_sec": np.nan,
                            "artifact_dir": str(output_dir),
                            "error_message": "",
                            "timeout_flag": 0,
                        }
                        append_run_record(runs_csv, record)
                        print(
                            f"[{base['baseline_name']}] scenario={scenario:>18s} N={N:<4d} run={run_idx:<2d} "
                            f"ARI={record['ari']:.3f}"
                        )

    summary = summarize_results(
        runs_csv=runs_csv,
        group_cols=["dataset", "dataset_variant", "split_name", "method", "prior_mode", "N"],
        out_csv=summary_csv,
    )
    print("\nSaved:")
    print(f"  runs   -> {runs_csv}")
    print(f"  summary-> {summary_csv}")
    print("\nSummary preview:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
