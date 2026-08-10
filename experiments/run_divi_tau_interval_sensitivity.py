from __future__ import annotations

import argparse
import socket
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

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


def feature_f1_from_truth_mask(
    truth_mask: np.ndarray,
    phi_probs: np.ndarray,
    threshold: float = 0.5,
) -> float:
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
        description="Sensitivity analysis for DIVI over tau_mult x split_interval."
    )
    parser.add_argument("--config", type=str, default="/mnt/data/defaults_divi.yaml")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["heavy_tail_signal", "correlated_noise"],
        choices=sorted(SCENARIO_GENERATORS.keys()),
    )
    parser.add_argument("--N-list", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--D", type=int, default=100)
    parser.add_argument("--n-signal", type=int, default=10)
    parser.add_argument("--K-true", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--data-seed-offset", type=int, default=1000)
    parser.add_argument("--run-seed-offset", type=int, default=1)
    parser.add_argument("--prior-modes", nargs="+", type=int, default=[1])
    parser.add_argument("--phi-threshold", type=float, default=0.5)

    parser.add_argument("--tau-mults", nargs="+", type=float, default=[1.0, 1.05, 1.1, 1.2])
    parser.add_argument("--split-intervals", nargs="+", type=int, default=[80, 120, 160])
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--beta-mult", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--temp-start", type=float, default=None)
    parser.add_argument("--temp-end", type=float, default=None)
    parser.add_argument("--prior-logvar0", type=float, default=2.197)

    parser.add_argument("--output-dir", type=str, default="./outputs/tau_interval_sensitivity")
    parser.add_argument("--tag", type=str, default="")
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


def build_model(
    cfg: Dict[str, Any],
    split_interval: int,
    split_threshold: float,
    verbose: bool,
    args: argparse.Namespace,
) -> DIVIClustering:
    return DIVIClustering(
        split_threshold=float(split_threshold),
        split_interval=int(split_interval),
        max_epochs=int(args.max_epochs if args.max_epochs is not None else cfg.get("max_epochs", 300)),
        lr=float(args.lr if args.lr is not None else cfg.get("lr", 0.01)),
        beta_mult=float(args.beta_mult if args.beta_mult is not None else cfg.get("beta_mult", 1.0)),
        temperature_start=float(args.temp_start if args.temp_start is not None else cfg.get("temp_start", 1.0)),
        temperature_end=float(args.temp_end if args.temp_end is not None else cfg.get("temp_end", 0.1)),
        prior_logvar0=float(args.prior_logvar0),
        verbose=verbose,
    )


def run_divi_once(
    X: np.ndarray,
    y: np.ndarray,
    truth_mask: np.ndarray,
    K_true: int,
    prior_mode: int,
    run_seed: int,
    cfg: Dict[str, Any],
    tau_mult: float,
    split_interval: int,
    phi_threshold: float,
    verbose: bool,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    set_global_seed(run_seed)
    tau_auto = DIVIClustering.auto_split_threshold(D=int(X.shape[1]), sigma2=1.0)
    tau = float(tau_mult) * float(tau_auto)

    model = build_model(
        cfg=cfg,
        split_interval=split_interval,
        split_threshold=tau,
        verbose=verbose,
        args=args,
    )
    model.fit(X, use_prior=int(prior_mode))

    y_pred = model.predict(X)
    phi = model.get_phi()
    met = clustering_metrics(y, y_pred)
    f1_feature = feature_f1_from_truth_mask(truth_mask, phi, threshold=phi_threshold)

    final_K = int(model.fit_summary_.get("final_K", np.nan))
    delta_K = final_K - int(K_true)
    out = {
        **met,
        "f1_feature": float(f1_feature),
        "selected_dims_count": int(np.sum(phi >= phi_threshold)),
        "selected_dims_ratio": float(np.mean(phi >= phi_threshold)),
        "mean_phi": float(np.mean(phi)),
        "median_phi": float(np.median(phi)),
        "tau_auto": float(tau_auto),
        "tau": float(tau),
        "tau_mult": float(tau_mult),
        "delta_K": int(delta_K),
        "oversplit_flag": int(delta_K > 0),
        "exactK_flag": int(delta_K == 0),
        "undersplit_flag": int(delta_K < 0),
        "peak_memory_mb": get_peak_memory_mb(),
        **model.fit_summary_,
    }
    return out


def write_sensitivity_summary(runs_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(runs_csv)
    if df.empty:
        raise ValueError(f"No runs found in {runs_csv}")

    for col in ["tau_mult", "Tsplit", "prior_mode", "N"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = [
        c for c in ["dataset", "dataset_variant", "split_name", "N", "prior_mode", "tau_mult", "Tsplit"]
        if c in df.columns
    ]

    metrics = [
        "ari",
        "nmi",
        "acc",
        "f1_feature",
        "final_K",
        "delta_K",
        "oversplit_flag",
        "exactK_flag",
        "undersplit_flag",
        "split_count",
        "selected_dims_count",
        "selected_dims_ratio",
        "mean_phi",
        "median_phi",
        "wallclock_total_sec",
    ]
    metrics = [c for c in metrics if c in df.columns]

    agg_spec: Dict[str, List[str]] = {}
    for c in metrics:
        if c in {"oversplit_flag", "exactK_flag", "undersplit_flag"}:
            agg_spec[c] = ["mean", "sum"]
        else:
            agg_spec[c] = ["mean", "std", "median"]

    summary = df.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    summary.columns = [
        "__".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns.values
    ]

    rename_map = {
        "oversplit_flag__mean": "oversplit_rate",
        "oversplit_flag__sum": "oversplit_count",
        "exactK_flag__mean": "exactK_rate",
        "exactK_flag__sum": "exactK_count",
        "undersplit_flag__mean": "undersplit_rate",
        "undersplit_flag__sum": "undersplit_count",
    }
    summary = summary.rename(columns=rename_map)
    ensure_dir(out_csv.parent)
    summary.to_csv(out_csv, index=False)
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    output_dir = ensure_dir(args.output_dir)
    runs_csv = output_dir / "runs.csv"
    summary_csv = output_dir / "summary_tau_interval.csv"
    compact_csv = output_dir / "summary_tau_interval_compact.csv"

    device_type, device_name = get_device_info()
    script_name = Path(__file__).name
    git_commit = get_git_commit(cwd=str(Path(__file__).resolve().parent))
    hostname = socket.gethostname()

    base_cfg = {
        "max_epochs": int(args.max_epochs if args.max_epochs is not None else cfg.get("max_epochs", 300)),
        "beta_mult": float(args.beta_mult if args.beta_mult is not None else cfg.get("beta_mult", 1.0)),
        "lr": float(args.lr if args.lr is not None else cfg.get("lr", 0.01)),
        "temp_start": float(args.temp_start if args.temp_start is not None else cfg.get("temp_start", 1.0)),
        "temp_end": float(args.temp_end if args.temp_end is not None else cfg.get("temp_end", 0.1)),
    }

    for scenario in args.scenarios:
        generator = SCENARIO_GENERATORS[scenario]

        for N in args.N_list:
            for run_idx in range(args.n_runs):
                data_seed = args.data_seed_offset + 10000 * (abs(hash(scenario)) % 1000) + run_idx + 100 * N
                ds_kwargs = scenario_kwargs(args, scenario=scenario, N=N, data_seed=data_seed)
                ds = generator(**ds_kwargs)
                X = ds["X"]
                y = ds["y"]
                truth_mask = ds["truth_mask"]
                dataset = ds["dataset"]
                dataset_variant = ds["dataset_variant"]
                K_true = int(ds["K_true"])

                for prior_mode in args.prior_modes:
                    for tau_mult in args.tau_mults:
                        for split_interval in args.split_intervals:
                            run_seed = (
                                args.run_seed_offset
                                + run_idx
                                + 1000 * int(prior_mode)
                                + 10000 * int(split_interval)
                                + int(round(100 * tau_mult))
                            )

                            tau_auto = DIVIClustering.auto_split_threshold(D=int(X.shape[1]), sigma2=1.0)
                            tau = float(tau_mult) * float(tau_auto)
                            common = {
                                "script_name": script_name,
                                "timestamp": utc_timestamp(),
                                "git_commit": git_commit,
                                "hostname": hostname,
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
                                "method": "DIVI",
                                "prior_mode": int(prior_mode),
                                "baseline_name": "DIVI",
                                "beta_mult": base_cfg["beta_mult"],
                                "Tsplit": int(split_interval),
                                "tau_mode": "auto_mult",
                                "tau": float(tau),
                                "tau_mult": float(tau_mult),
                                "lr": base_cfg["lr"],
                                "temp_start": base_cfg["temp_start"],
                                "temp_end": base_cfg["temp_end"],
                                "max_epochs": base_cfg["max_epochs"],
                                "artifact_dir": str(output_dir),
                                "status": "ok",
                                "timeout_flag": 0,
                                "error_message": "",
                            }
                            record: Dict[str, Any] = dict(common)
                            record["experiment_id"] = make_experiment_id(
                                prefix=f"tau_interval{('_' + args.tag) if args.tag else ''}",
                                dataset=dataset_variant,
                                split_name=f"{scenario}_tau{tau_mult:g}_Ts{split_interval}",
                                run_seed=run_seed,
                                method=f"DIVI_p{prior_mode}",
                            )

                            try:
                                result = run_divi_once(
                                    X=X,
                                    y=y,
                                    truth_mask=truth_mask,
                                    K_true=K_true,
                                    prior_mode=prior_mode,
                                    run_seed=run_seed,
                                    cfg=cfg,
                                    tau_mult=tau_mult,
                                    split_interval=split_interval,
                                    phi_threshold=args.phi_threshold,
                                    verbose=args.verbose,
                                    args=args,
                                )
                                record.update(result)
                                record["beta"] = result.get("beta", base_cfg["beta_mult"] * len(X))
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
                                f"[DIVI] scenario={scenario:>18s} N={N:<4d} run={run_idx:<2d} "
                                f"prior={prior_mode} tau_mult={tau_mult:<4.2f} Tsplit={split_interval:<3d} "
                                f"status={record['status']} ari={record.get('ari', float('nan')):.3f} "
                                f"f1={record.get('f1_feature', float('nan')):.3f} "
                                f"K={record.get('final_K', 'NA')} over={record.get('oversplit_flag', 'NA')}"
                            )

    summarize_results(
        runs_csv=runs_csv,
        group_cols=["dataset", "dataset_variant", "split_name", "N", "prior_mode", "tau_mult", "Tsplit"],
        out_csv=compact_csv,
    )
    write_sensitivity_summary(runs_csv=runs_csv, out_csv=summary_csv)
    print(f"Saved run-level records to: {runs_csv}")
    print(f"Saved compact summary to:   {compact_csv}")
    print(f"Saved sensitivity summary to: {summary_csv}")


if __name__ == "__main__":
    main()
