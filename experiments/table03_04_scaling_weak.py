#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary experiments for DIVI (MLWA revision).

This script implements the minimal high-impact empirical supplement:
  1. Added baselines:
     - PCA + K-means with oracle K
     - PCA + diagonal-covariance GMM with oracle K
  2. Added simulated settings:
     - High-dimensional scaling: D in {100, 500, 1000, 2000}
     - Weak-signal regime: Delta in {1.0, 1.5, 2.0}

The script is designed to work with the uploaded DIVI_HAR.py implementation,
which exposes DIVIClustering.

Example:
    python run_divi_mlwa_supplement.py \
        --divi_path /mnt/data/DIVI_HAR.py \
        --output_dir /mnt/data/divi_mlwa_supplement_results \
        --seeds 1 2 3 4 5 6 7 8 9 10

Quick smoke test:
    python run_divi_mlwa_supplement.py --quick
"""

from __future__ import annotations

import os

# Limit numerical-library threads to avoid excessive oversubscription on high-D simulations.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import importlib.util
import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_divi_class(divi_path: str):
    """Dynamically import DIVIClustering from a Python source file."""
    divi_path = str(Path(divi_path).resolve())
    if not os.path.exists(divi_path):
        raise FileNotFoundError(f"DIVI source file not found: {divi_path}")

    spec = importlib.util.spec_from_file_location("divi_impl", divi_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {divi_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "DIVIClustering"):
        raise AttributeError(f"{divi_path} does not define DIVIClustering")
    return module.DIVIClustering


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def balanced_labels(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    counts = [n // k] * k
    counts[-1] += n - sum(counts)
    y = np.concatenate([np.full(c, kk, dtype=int) for kk, c in enumerate(counts)])
    rng.shuffle(y)
    return y


def standardize_full_data(X: np.ndarray) -> np.ndarray:
    """Standardize all features before fitting, as described in the paper."""
    return StandardScaler().fit_transform(X)


def feature_f1_from_scores(
    scores: np.ndarray,
    informative_idx: np.ndarray,
    threshold: float = 0.5,
    top_m: Optional[int] = None,
) -> Tuple[float, int]:
    """Return feature F1 and selected count.

    If top_m is supplied, select exactly top_m features by score.
    Otherwise select scores >= threshold.
    """
    d = scores.shape[0]
    true_mask = np.zeros(d, dtype=int)
    true_mask[informative_idx] = 1

    pred_mask = np.zeros(d, dtype=int)
    if top_m is None:
        pred_mask[scores >= threshold] = 1
    else:
        top_m = int(min(max(1, top_m), d))
        top_idx = np.argsort(scores)[-top_m:]
        pred_mask[top_idx] = 1

    # zero_division=0 avoids undefined precision/recall warnings when none selected.
    return float(f1_score(true_mask, pred_mask, zero_division=0)), int(pred_mask.sum())


# -----------------------------------------------------------------------------
# Synthetic data generators
# -----------------------------------------------------------------------------

def generate_sparse_gaussian_mixture(
    n: int,
    d: int,
    k: int = 3,
    d_info: int = 10,
    delta: float = 2.0,
    signal_sd: float = 1.0,
    noise_sd: float = 3.0,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sparse-signal Gaussian mixture data.

    Informative dimensions have cluster means -delta, 0, +delta for K=3.
    Nuisance dimensions are independent N(0, noise_sd^2).

    Returns
    -------
    X : ndarray, shape (n, d)
    y : ndarray, shape (n,)
    informative_idx : ndarray of informative feature indices
    """
    if k != 3:
        raise ValueError("This generator currently uses means (-Delta, 0, Delta) and expects k=3.")
    if d_info > d:
        raise ValueError("d_info cannot exceed d.")

    rng = np.random.default_rng(seed)
    y = balanced_labels(n, k, rng)
    X = rng.normal(loc=0.0, scale=noise_sd, size=(n, d))

    means = np.array([-delta, 0.0, delta], dtype=float)
    for kk in range(k):
        idx = y == kk
        X[idx, :d_info] = rng.normal(loc=means[kk], scale=signal_sd, size=(idx.sum(), d_info))

    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]
    informative_idx = np.arange(d_info, dtype=int)
    return X, y, informative_idx


@dataclass(frozen=True)
class Scenario:
    scenario: str
    n: int
    d: int
    k: int
    d_info: int
    delta: float
    noise_sd: float = 3.0


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    scenarios: List[Scenario] = []

    if args.quick:
        high_d_list = [100, 500]
        weak_delta_list = [1.0, 2.0]
    else:
        high_d_list = args.high_d_values
        weak_delta_list = args.weak_delta_values

    for d in high_d_list:
        scenarios.append(
            Scenario(
                scenario="high_dim_scaling",
                n=args.high_n,
                d=int(d),
                k=3,
                d_info=args.d_info,
                delta=args.high_delta,
                noise_sd=args.noise_sd,
            )
        )

    for delta in weak_delta_list:
        scenarios.append(
            Scenario(
                scenario="weak_signal",
                n=args.weak_n,
                d=args.weak_d,
                k=3,
                d_info=args.d_info,
                delta=float(delta),
                noise_sd=args.noise_sd,
            )
        )

    return scenarios


# -----------------------------------------------------------------------------
# Baselines and DIVI fitting
# -----------------------------------------------------------------------------

def run_kmeans(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> Dict[str, float]:
    t0 = time.perf_counter()
    labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(X)
    runtime = time.perf_counter() - t0
    return {
        "ARI": adjusted_rand_score(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "runtime_sec": runtime,
        "final_K": k,
        "active_dims": np.nan,
        "feature_f1_thresh": np.nan,
        "feature_f1_topd": np.nan,
    }


def run_diag_gmm(X: np.ndarray, y: np.ndarray, k: int, seed: int, reg_covar: float) -> Dict[str, float]:
    t0 = time.perf_counter()
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        reg_covar=reg_covar,
        random_state=seed,
        n_init=5,
        max_iter=300,
    )
    labels = gmm.fit_predict(X)
    runtime = time.perf_counter() - t0
    return {
        "ARI": adjusted_rand_score(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "runtime_sec": runtime,
        "final_K": k,
        "active_dims": np.nan,
        "feature_f1_thresh": np.nan,
        "feature_f1_topd": np.nan,
    }


def pca_dimension(n: int, d: int, requested_q: int) -> int:
    return int(min(requested_q, n - 1, d))


def run_pca_kmeans(X: np.ndarray, y: np.ndarray, k: int, seed: int, q: int) -> Dict[str, float]:
    t0 = time.perf_counter()
    q_eff = pca_dimension(X.shape[0], X.shape[1], q)
    pipe = make_pipeline(
        PCA(n_components=q_eff, random_state=seed),
        KMeans(n_clusters=k, random_state=seed, n_init=20),
    )
    labels = pipe.fit_predict(X)
    runtime = time.perf_counter() - t0
    return {
        "ARI": adjusted_rand_score(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "runtime_sec": runtime,
        "final_K": k,
        "active_dims": q_eff,
        "feature_f1_thresh": np.nan,
        "feature_f1_topd": np.nan,
    }


def run_pca_diag_gmm(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int,
    q: int,
    reg_covar: float,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    q_eff = pca_dimension(X.shape[0], X.shape[1], q)
    Z = PCA(n_components=q_eff, random_state=seed).fit_transform(X)
    labels = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        reg_covar=reg_covar,
        random_state=seed,
        n_init=5,
        max_iter=300,
    ).fit_predict(Z)
    runtime = time.perf_counter() - t0
    return {
        "ARI": adjusted_rand_score(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "runtime_sec": runtime,
        "final_K": k,
        "active_dims": q_eff,
        "feature_f1_thresh": np.nan,
        "feature_f1_topd": np.nan,
    }


def run_divi(
    X: np.ndarray,
    y: np.ndarray,
    informative_idx: np.ndarray,
    k: int,
    seed: int,
    DIVIClustering,
    args: argparse.Namespace,
    use_prior: int = 1,
) -> Dict[str, float]:
    set_seed(seed)
    t0 = time.perf_counter()

    divi = DIVIClustering(
        split_threshold=args.divi_split_threshold,
        split_interval=args.divi_split_interval,
        max_epochs=args.divi_epochs,
        lr=args.divi_lr,
        max_components=k if args.divi_cap_at_true_k else None,
        beta_mult=args.divi_beta_mult,
        rough_k=k,
        verbose=args.divi_verbose,
    )
    labels = divi.fit_predict(X, use_prior=use_prior)
    runtime = time.perf_counter() - t0

    scores = divi.get_feature_relevance()
    f1_thresh, active_thresh = feature_f1_from_scores(
        scores=scores,
        informative_idx=informative_idx,
        threshold=args.feature_threshold,
        top_m=None,
    )
    f1_topd, active_topd = feature_f1_from_scores(
        scores=scores,
        informative_idx=informative_idx,
        top_m=len(informative_idx),
    )

    return {
        "ARI": adjusted_rand_score(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "runtime_sec": runtime,
        "final_K": int(divi.model.K),
        "active_dims": int(active_thresh),
        "feature_f1_thresh": f1_thresh,
        "feature_f1_topd": f1_topd,
    }


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_one(
    scenario: Scenario,
    seed: int,
    DIVIClustering,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    set_seed(seed)
    X_raw, y, informative_idx = generate_sparse_gaussian_mixture(
        n=scenario.n,
        d=scenario.d,
        k=scenario.k,
        d_info=scenario.d_info,
        delta=scenario.delta,
        noise_sd=scenario.noise_sd,
        seed=seed,
    )
    X = standardize_full_data(X_raw)

    rows: List[Dict[str, object]] = []

    methods = []
    if "divi_info" in args.methods:
        methods.append(("DIVI-Info", lambda: run_divi(X, y, informative_idx, scenario.k, seed, DIVIClustering, args, use_prior=1)))
    if "divi_noninfo" in args.methods:
        methods.append(("DIVI-NonInfo", lambda: run_divi(X, y, informative_idx, scenario.k, seed, DIVIClustering, args, use_prior=2)))
    if "kmeans" in args.methods:
        methods.append(("K-means", lambda: run_kmeans(X, y, scenario.k, seed)))
    if "gmm" in args.methods:
        methods.append(("diag-GMM", lambda: run_diag_gmm(X, y, scenario.k, seed, args.gmm_reg_covar)))
    if "pca_kmeans" in args.methods:
        methods.append((f"PCA({args.pca_q})+K-means", lambda: run_pca_kmeans(X, y, scenario.k, seed, args.pca_q)))
    if "pca_gmm" in args.methods:
        methods.append((f"PCA({args.pca_q})+diag-GMM", lambda: run_pca_diag_gmm(X, y, scenario.k, seed, args.pca_q, args.gmm_reg_covar)))

    for method_name, fn in methods:
        base = {
            **asdict(scenario),
            "seed": seed,
            "method": method_name,
            "status": "ok",
            "error": "",
        }
        try:
            metrics = fn()
            row = {**base, **metrics}
        except Exception as exc:
            row = {
                **base,
                "status": "failed",
                "error": repr(exc),
                "ARI": np.nan,
                "NMI": np.nan,
                "runtime_sec": np.nan,
                "final_K": np.nan,
                "active_dims": np.nan,
                "feature_f1_thresh": np.nan,
                "feature_f1_topd": np.nan,
            }
            if args.print_errors:
                print(f"[ERROR] {scenario.scenario} seed={seed} method={method_name}: {exc}")
                traceback.print_exc()
        rows.append(row)

    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ARI",
        "NMI",
        "runtime_sec",
        "final_K",
        "active_dims",
        "feature_f1_thresh",
        "feature_f1_topd",
    ]
    group_cols = ["scenario", "n", "d", "d_info", "delta", "noise_sd", "method"]

    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join([str(c) for c in col if c != ""]).rstrip("_") for col in summary.columns.values]
    return summary


def write_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    """Write a compact LaTeX table for ARI/NMI/F1/runtime summaries."""
    keep_cols = [
        "scenario", "d", "delta", "method",
        "ARI_mean", "ARI_std",
        "NMI_mean", "NMI_std",
        "feature_f1_thresh_mean", "feature_f1_thresh_std",
        "runtime_sec_mean", "runtime_sec_std",
        "final_K_mean", "active_dims_mean",
    ]
    tab = summary[[c for c in keep_cols if c in summary.columns]].copy()

    def ms(mean, std, digits=3):
        if pd.isna(mean):
            return "--"
        if pd.isna(std):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ({std:.{digits}f})"

    rows = []
    for _, r in tab.iterrows():
        rows.append({
            "Scenario": r.get("scenario"),
            "D": int(r.get("d")),
            "Delta": float(r.get("delta")),
            "Method": r.get("method"),
            "ARI": ms(r.get("ARI_mean"), r.get("ARI_std")),
            "NMI": ms(r.get("NMI_mean"), r.get("NMI_std")),
            "Feature F1": ms(r.get("feature_f1_thresh_mean"), r.get("feature_f1_thresh_std")),
            "Runtime": ms(r.get("runtime_sec_mean"), r.get("runtime_sec_std"), digits=2),
            "Final K": "--" if pd.isna(r.get("final_K_mean")) else f"{r.get('final_K_mean'):.2f}",
            "Active dim": "--" if pd.isna(r.get("active_dims_mean")) else f"{r.get('active_dims_mean'):.1f}",
        })
    latex_df = pd.DataFrame(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DIVI MLWA supplementary simulations.")

    parser.add_argument("--divi_path", type=str, default="/mnt/data/DIVI_HAR.py",
                        help="Path to DIVI_HAR.py containing DIVIClustering.")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/divi_mlwa_supplement_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")

    parser.add_argument("--methods", type=str, nargs="+",
                        default=["divi_info", "kmeans", "gmm", "pca_kmeans", "pca_gmm"],
                        choices=["divi_info", "divi_noninfo", "kmeans", "gmm", "pca_kmeans", "pca_gmm"],
                        help="Methods to run.")

    # Scenario settings.
    parser.add_argument("--d_info", type=int, default=10)
    parser.add_argument("--noise_sd", type=float, default=3.0)
    parser.add_argument("--high_n", type=int, default=600)
    parser.add_argument("--high_delta", type=float, default=2.0)
    parser.add_argument("--high_d_values", type=int, nargs="+", default=[100, 500, 1000, 2000])
    parser.add_argument("--weak_n", type=int, default=200)
    parser.add_argument("--weak_d", type=int, default=500)
    parser.add_argument("--weak_delta_values", type=float, nargs="+", default=[1.0, 1.5, 2.0])

    # Baseline settings.
    parser.add_argument("--pca_q", type=int, default=50)
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-6)

    # DIVI settings aligned with DIVI_HAR.py.
    parser.add_argument("--divi_epochs", type=int, default=300)
    parser.add_argument("--divi_lr", type=float, default=0.05)
    parser.add_argument("--divi_split_interval", type=int, default=60)
    parser.add_argument("--divi_split_threshold", type=float, default=22.0)
    parser.add_argument("--divi_beta_mult", type=float, default=1.0)
    parser.add_argument("--divi_cap_at_true_k", action=argparse.BooleanOptionalAction, default=True,
                        help="Use max_components=true K for synthetic experiments.")
    parser.add_argument("--feature_threshold", type=float, default=0.5)
    parser.add_argument("--divi_verbose", action="store_true")
    parser.add_argument("--num_threads", type=int, default=1, help="Torch CPU thread count.")
    parser.add_argument("--print_errors", action="store_true")

    args = parser.parse_args()
    if args.quick:
        args.seeds = args.seeds[:2]
        args.divi_epochs = min(args.divi_epochs, 120)
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    out_dir = ensure_dir(args.output_dir)

    DIVIClustering = load_divi_class(args.divi_path)
    scenarios = build_scenarios(args)

    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    all_rows: List[Dict[str, object]] = []
    total_jobs = len(scenarios) * len(args.seeds)
    job = 0

    print(f"Output directory: {out_dir}")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print(f"Number of scenario x seed jobs: {total_jobs}")

    for scenario in scenarios:
        for seed in args.seeds:
            job += 1
            print(
                f"[{job}/{total_jobs}] scenario={scenario.scenario}, "
                f"N={scenario.n}, D={scenario.d}, delta={scenario.delta}, seed={seed}"
            )
            rows = run_one(scenario, seed, DIVIClustering, args)
            all_rows.extend(rows)

            # Incremental checkpoint.
            pd.DataFrame(all_rows).to_csv(out_dir / "per_run_results_checkpoint.csv", index=False)

    df = pd.DataFrame(all_rows)
    per_run_path = out_dir / "per_run_results.csv"
    summary_path = out_dir / "summary_results.csv"
    latex_path = out_dir / "summary_table.tex"

    df.to_csv(per_run_path, index=False)
    summary = summarize_results(df)
    summary.to_csv(summary_path, index=False)
    write_latex_table(summary, latex_path)

    print("\nCompleted.")
    print(f"Per-run results: {per_run_path}")
    print(f"Summary results: {summary_path}")
    print(f"LaTeX table: {latex_path}")
    print("\nSummary preview:")
    with pd.option_context("display.max_rows", 50, "display.max_columns", 50, "display.width", 200):
        print(summary)


if __name__ == "__main__":
    main()
