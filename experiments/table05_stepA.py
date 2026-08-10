#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step A prior-quality diagnostics for DIVI.

This experiment quantifies:
1. quality of the rough K-means partition used by Step A;
2. quality of the Step A prior probabilities rho_j;
3. improvement from rough K-means to the final DIVI partition;
4. improvement from the Step A prior to the final variational feature gates.

Default matched benchmark:
    K = 3
    D = 100
    d_info = 10
    N in {200, 1000}
    informative coordinates: Gaussian with cluster means (-delta, 0, +delta)
    nuisance coordinates: Gaussian N(0, noise_sd^2)
    delta = 2.0
    noise_sd = 3.0

Outputs
-------
stepA_prior_diagnostics_raw.csv
stepA_prior_diagnostics_summary.csv
stepA_prior_diagnostics_correlations.csv
stepA_prior_diagnostics_table.tex
stepA_prior_diagnostics_config.json

Quick test
----------
python run_stepA_prior_diagnostics.py \
  --divi_path /content/divi_core_fixedk.py \
  --output_dir /content/stepA_test \
  --quick

Full experiment
---------------
python run_stepA_prior_diagnostics.py \
  --divi_path /content/divi_core_fixedk.py \
  --output_dir /content/drive/MyDrive/stepA_prior_diagnostics \
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_divi_class(path: str):
    path = str(Path(path).resolve())
    if not Path(path).exists():
        raise FileNotFoundError(f"DIVI source not found: {path}")

    spec = importlib.util.spec_from_file_location("divi_impl_stepa", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import DIVI module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "DIVIClustering"):
        raise AttributeError(f"{path} does not define DIVIClustering")

    return module.DIVIClustering


def balanced_labels(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.full(k, n // k, dtype=int)
    counts[: n % k] += 1
    y = np.concatenate([
        np.full(counts[j], j, dtype=int) for j in range(k)
    ])
    rng.shuffle(y)
    return y


def generate_matched_data(
    n: int,
    d: int,
    d_info: int,
    k: int,
    delta: float,
    signal_sd: float,
    noise_sd: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a balanced Gaussian mixture with informative and nuisance features.

    For K=3 the informative means are (-delta, 0, +delta).
    For general K they are equally spaced on [-delta, +delta].
    """
    if d_info <= 0 or d_info >= d:
        raise ValueError("Require 0 < d_info < d")
    if k < 2:
        raise ValueError("k must be at least 2")

    rng = np.random.default_rng(seed)
    y = balanced_labels(n, k, rng)

    X = rng.normal(0.0, noise_sd, size=(n, d))
    support = np.arange(d_info, dtype=int)

    centers = np.linspace(-delta, delta, k)

    for cluster in range(k):
        rows = y == cluster
        X[np.ix_(rows, support)] = rng.normal(
            loc=centers[cluster],
            scale=signal_sd,
            size=(rows.sum(), d_info),
        )

    # Shuffle rows after generation.
    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]

    return X, y, support


def feature_metrics(
    scores: np.ndarray,
    support: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    d = scores.shape[0]

    truth = np.zeros(d, dtype=int)
    truth[support] = 1

    selected = (scores >= threshold).astype(int)

    top_idx = np.argsort(scores)[-len(support):]
    selected_top = np.zeros(d, dtype=int)
    selected_top[top_idx] = 1

    noise_idx = np.setdiff1d(np.arange(d), support)

    return {
        "f1_thresh": float(f1_score(truth, selected, zero_division=0)),
        "f1_top": float(f1_score(truth, selected_top, zero_division=0)),
        "auroc": float(roc_auc_score(truth, scores)),
        "auprc": float(average_precision_score(truth, scores)),
        "selected_dims": int(selected.sum()),
        "signal_mean": float(scores[support].mean()),
        "noise_mean": float(scores[noise_idx].mean()),
        "signal_noise_gap": float(
            scores[support].mean() - scores[noise_idx].mean()
        ),
    }


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    if method == "pearson":
        r, p = pearsonr(x, y)
    elif method == "spearman":
        r, p = spearmanr(x, y)
    else:
        raise ValueError(method)

    return float(r), float(p)


def run_one(
    DIVIClustering,
    n: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    set_seed(seed)

    X, y, support = generate_matched_data(
        n=n,
        d=args.d,
        d_info=args.d_info,
        k=args.k,
        delta=args.delta,
        signal_sd=args.signal_sd,
        noise_sd=args.noise_sd,
        seed=seed,
    )

    # Apply the same whole-dataset standardization used in the synthetic study.
    X = StandardScaler().fit_transform(X).astype(np.float32)

    # The same random_state is used by rough K-means and Step A.
    rough = KMeans(
        n_clusters=args.k,
        random_state=seed,
        n_init=10,
    )
    rough_labels = rough.fit_predict(X)

    rough_ari = adjusted_rand_score(y, rough_labels)
    rough_nmi = normalized_mutual_info_score(y, rough_labels)

    divi = DIVIClustering(
        split_threshold=None,
        split_interval=args.split_interval,
        max_epochs=args.max_epochs,
        lr=args.lr,
        beta_mult=args.beta_mult,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        prior_logvar0=args.prior_logvar0,
        verbose=args.verbose,
        allow_split=False,
        init_num_components=args.k,
        init_method="kmeans",
        random_state=seed,
    )

    # Evaluate Step A before optimization.
    t0 = time.perf_counter()
    rho = (
        divi._step_a_calculate_prior(
            X,
            mode=1,
            rough_k=args.k,
        )
        .detach()
        .cpu()
        .numpy()
    )
    step_a_eval_sec = time.perf_counter() - t0

    prior_metrics = feature_metrics(
        rho,
        support=support,
        threshold=args.threshold,
    )

    # Fit informative-prior DIVI.
    t1 = time.perf_counter()
    divi.fit(X, use_prior=1)
    fit_sec = time.perf_counter() - t1

    final_labels = divi.predict(X)
    final_scores = divi.get_phi()

    final_ari = adjusted_rand_score(y, final_labels)
    final_nmi = normalized_mutual_info_score(y, final_labels)

    final_metrics = feature_metrics(
        final_scores,
        support=support,
        threshold=args.threshold,
    )

    return {
        "n": n,
        "seed": seed,
        "d": args.d,
        "d_info": args.d_info,
        "k": args.k,
        "delta": args.delta,
        "rough_ari": float(rough_ari),
        "rough_nmi": float(rough_nmi),
        "prior_f1_thresh": prior_metrics["f1_thresh"],
        "prior_f1_top": prior_metrics["f1_top"],
        "prior_auroc": prior_metrics["auroc"],
        "prior_auprc": prior_metrics["auprc"],
        "prior_selected_dims": prior_metrics["selected_dims"],
        "prior_signal_mean": prior_metrics["signal_mean"],
        "prior_noise_mean": prior_metrics["noise_mean"],
        "prior_signal_noise_gap": prior_metrics["signal_noise_gap"],
        "final_ari": float(final_ari),
        "final_nmi": float(final_nmi),
        "final_f1_thresh": final_metrics["f1_thresh"],
        "final_f1_top": final_metrics["f1_top"],
        "final_auroc": final_metrics["auroc"],
        "final_auprc": final_metrics["auprc"],
        "final_selected_dims": final_metrics["selected_dims"],
        "final_signal_mean": final_metrics["signal_mean"],
        "final_noise_mean": final_metrics["noise_mean"],
        "final_signal_noise_gap": final_metrics["signal_noise_gap"],
        "delta_ari": float(final_ari - rough_ari),
        "delta_nmi": float(final_nmi - rough_nmi),
        "delta_f1_thresh": float(
            final_metrics["f1_thresh"] - prior_metrics["f1_thresh"]
        ),
        "delta_f1_top": float(
            final_metrics["f1_top"] - prior_metrics["f1_top"]
        ),
        "delta_auprc": float(
            final_metrics["auprc"] - prior_metrics["auprc"]
        ),
        "step_a_eval_sec": float(step_a_eval_sec),
        "fit_sec": float(fit_sec),
        "final_K": int(divi.model.K),
        "status": "ok",
        "error": "",
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rough_ari", "rough_nmi",
        "prior_f1_thresh", "prior_f1_top",
        "prior_auroc", "prior_auprc",
        "prior_selected_dims",
        "prior_signal_mean", "prior_noise_mean",
        "prior_signal_noise_gap",
        "final_ari", "final_nmi",
        "final_f1_thresh", "final_f1_top",
        "final_auroc", "final_auprc",
        "final_selected_dims",
        "final_signal_mean", "final_noise_mean",
        "final_signal_noise_gap",
        "delta_ari", "delta_nmi",
        "delta_f1_thresh", "delta_f1_top", "delta_auprc",
        "step_a_eval_sec", "fit_sec", "final_K",
    ]

    summary = (
        raw.groupby(["n", "d", "d_info", "delta"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in col if str(x) != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary


def compute_correlations(raw: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("rough_ari", "final_ari"),
        ("prior_auprc", "final_ari"),
        ("prior_auprc", "final_f1_thresh"),
        ("prior_f1_top", "final_f1_top"),
        ("rough_ari", "delta_ari"),
        ("prior_auprc", "delta_auprc"),
    ]

    rows: List[Dict[str, object]] = []

    for n, sub in raw.groupby("n"):
        for x_name, y_name in pairs:
            for method in ["pearson", "spearman"]:
                r, p = safe_corr(sub[x_name], sub[y_name], method)
                rows.append({
                    "n": int(n),
                    "x": x_name,
                    "y": y_name,
                    "method": method,
                    "correlation": r,
                    "p_value": p,
                    "replicates": int(len(sub)),
                })

    return pd.DataFrame(rows)


def make_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Diagnostic quality of the data-informed Step A prior. "
        r"Values are mean (standard deviation) over independently generated datasets.}",
        r"\label{tab:stepA_prior_diagnostics}",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"$N$ & Rough ARI & Prior AUPRC & Prior top-$d_0$ F1 "
        r"& Final ARI & Final F1 & $\Delta$ARI & $\Delta$F1 \\",
        r"\midrule",
    ]

    for _, row in summary.sort_values("n").iterrows():
        def fmt(metric: str) -> str:
            mean = row.get(f"{metric}_mean", np.nan)
            sd = row.get(f"{metric}_std", np.nan)
            if pd.isna(mean):
                return "--"
            if pd.isna(sd):
                return f"{mean:.3f}"
            return f"{mean:.3f} ({sd:.3f})"

        rows.append(
            f"{int(row['n'])} & "
            f"{fmt('rough_ari')} & "
            f"{fmt('prior_auprc')} & "
            f"{fmt('prior_f1_top')} & "
            f"{fmt('final_ari')} & "
            f"{fmt('final_f1_thresh')} & "
            f"{fmt('delta_ari')} & "
            f"{fmt('delta_f1_thresh')} \\\\"
        )

    rows.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.write_text("\n".join(rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--divi_path",
        type=str,
        default="/content/divi_core_fixedk.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/content/stepA_prior_diagnostics",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 21)),
    )
    p.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[200, 1000],
    )
    p.add_argument("--quick", action="store_true")

    p.add_argument("--d", type=int, default=100)
    p.add_argument("--d_info", type=int, default=10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--signal_sd", type=float, default=1.0)
    p.add_argument("--noise_sd", type=float, default=3.0)

    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--beta_mult", type=float, default=1.0)
    p.add_argument("--temperature_start", type=float, default=1.0)
    p.add_argument("--temperature_end", type=float, default=0.1)
    p.add_argument("--prior_logvar0", type=float, default=2.197)
    p.add_argument("--split_interval", type=int, default=120)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [1, 2] if args.quick else args.seeds
    sample_sizes = [200] if args.quick else args.sample_sizes

    DIVIClustering = load_divi_class(args.divi_path)

    config = vars(args).copy()
    config["effective_seeds"] = seeds
    config["effective_sample_sizes"] = sample_sizes
    config["prior_mode"] = 1
    config["rough_k"] = args.k

    (output_dir / "stepA_prior_diagnostics_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    records: List[Dict[str, object]] = []

    for n in sample_sizes:
        for seed in seeds:
            print(
                f"[run] n={n} d={args.d} d_info={args.d_info} "
                f"delta={args.delta} seed={seed}",
                flush=True,
            )

            try:
                result = run_one(
                    DIVIClustering=DIVIClustering,
                    n=n,
                    seed=seed,
                    args=args,
                )
                records.append(result)

            except Exception as exc:
                print(f"[ERROR] {exc!r}", flush=True)
                traceback.print_exc()
                records.append({
                    "n": n,
                    "seed": seed,
                    "d": args.d,
                    "d_info": args.d_info,
                    "k": args.k,
                    "delta": args.delta,
                    "status": "failed",
                    "error": repr(exc),
                })

    raw = pd.DataFrame(records)
    raw_path = output_dir / "stepA_prior_diagnostics_raw.csv"
    raw.to_csv(raw_path, index=False)

    ok = raw[raw["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError(
            "All runs failed. Inspect stepA_prior_diagnostics_raw.csv."
        )

    summary = summarize(ok)
    summary_path = output_dir / "stepA_prior_diagnostics_summary.csv"
    summary.to_csv(summary_path, index=False)

    correlations = compute_correlations(ok)
    corr_path = output_dir / "stepA_prior_diagnostics_correlations.csv"
    correlations.to_csv(corr_path, index=False)

    latex_path = output_dir / "stepA_prior_diagnostics_table.tex"
    make_latex_table(summary, latex_path)

    display_cols = [
        c for c in [
            "n",
            "rough_ari_mean", "rough_ari_std",
            "prior_auprc_mean", "prior_auprc_std",
            "prior_f1_top_mean", "prior_f1_top_std",
            "prior_f1_thresh_mean", "prior_f1_thresh_std",
            "final_ari_mean", "final_ari_std",
            "final_f1_thresh_mean", "final_f1_thresh_std",
            "final_f1_top_mean", "final_f1_top_std",
            "delta_ari_mean", "delta_ari_std",
            "delta_f1_thresh_mean", "delta_f1_thresh_std",
            "prior_selected_dims_mean",
            "final_selected_dims_mean",
            "final_K_mean",
        ]
        if c in summary.columns
    ]

    print("\n[done]")
    print(summary[display_cols].to_string(index=False))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
