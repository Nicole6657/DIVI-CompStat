#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run sparse K-means (SPKM; Witten & Tibshirani) on DIVI misspecification settings.

This script reproduces the two robustness settings used in the DIVI manuscript:
  1) heavy-tailed informative signal (Student-t df=5, unit variance, shifted means)
  2) block-correlated Gaussian nuisance features (rho=0.6, block size=10)

SPKM is fitted with the CRAN R package ``sparcl``.  Its L1 weight bound is selected
by the package's permutation procedure on each independently generated data set.
The true K=3 is supplied, matching the manuscript protocol for external baselines.

Outputs
-------
  spkm_misspec_raw.csv
  spkm_misspec_summary.csv
  spkm_misspec_table.tex
  run_config.json

Example (Colab/Linux)
---------------------
  python run_spkm_misspecification.py \
      --output_dir /content/spkm_misspec \
      --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
      --install_sparcl

Quick smoke test
----------------
  python run_spkm_misspecification.py --quick --install_sparcl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Avoid thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def balanced_labels(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.full(k, n // k, dtype=int)
    counts[: n - counts.sum()] += 1
    y = np.concatenate([np.full(c, kk, dtype=int) for kk, c in enumerate(counts)])
    rng.shuffle(y)
    return y


def generate_heavy_tailed_signal(
    n: int,
    d: int = 100,
    d_info: int = 10,
    delta: float = 2.0,
    signal_df: float = 5.0,
    noise_sd: float = 3.0,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heavy-tailed informative coordinates; independent Gaussian nuisance coordinates.

    A t_df variable has variance df/(df-2).  Multiplication by sqrt((df-2)/df)
    standardizes it to unit variance before adding cluster means (-delta, 0, delta).
    """
    if d_info > d:
        raise ValueError("d_info must not exceed d")
    rng = np.random.default_rng(seed)
    y = balanced_labels(n, 3, rng)
    X = rng.normal(0.0, noise_sd, size=(n, d))
    means = np.array([-delta, 0.0, delta])
    t_scale = np.sqrt((signal_df - 2.0) / signal_df)
    for k in range(3):
        idx = y == k
        X[idx, :d_info] = means[k] + t_scale * rng.standard_t(
            df=signal_df, size=(idx.sum(), d_info)
        )
    perm = rng.permutation(n)
    return X[perm], y[perm], np.arange(d_info, dtype=int)


def generate_correlated_noise(
    n: int,
    d: int = 100,
    d_info: int = 10,
    delta: float = 2.0,
    signal_sd: float = 1.0,
    noise_sd: float = 3.0,
    rho: float = 0.6,
    block_size: int = 10,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian informative coordinates and block-correlated Gaussian nuisance features."""
    if d_info > d:
        raise ValueError("d_info must not exceed d")
    if not (-1.0 / max(1, block_size - 1) < rho < 1.0):
        raise ValueError("rho is outside the positive-definite equicorrelation range")

    rng = np.random.default_rng(seed)
    y = balanced_labels(n, 3, rng)
    X = np.empty((n, d), dtype=float)
    means = np.array([-delta, 0.0, delta])

    for k in range(3):
        idx = y == k
        X[idx, :d_info] = rng.normal(means[k], signal_sd, size=(idx.sum(), d_info))

    p_noise = d - d_info
    start = d_info
    while start < d:
        b = min(block_size, d - start)
        cov = (noise_sd ** 2) * ((1.0 - rho) * np.eye(b) + rho * np.ones((b, b)))
        X[:, start : start + b] = rng.multivariate_normal(np.zeros(b), cov, size=n)
        start += b

    perm = rng.permutation(n)
    return X[perm], y[perm], np.arange(d_info, dtype=int)


R_HELPER = r'''
suppressPackageStartupMessages(library(sparcl))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 8) {
  stop("Expected 8 arguments: x_csv y_csv out_csv K nperms grid_size min_w seed")
}

x_csv    <- args[[1]]
y_csv    <- args[[2]]
out_csv  <- args[[3]]
K        <- as.integer(args[[4]])
nperms   <- as.integer(args[[5]])
grid_n   <- as.integer(args[[6]])
min_w    <- as.numeric(args[[7]])
seed     <- as.integer(args[[8]])

set.seed(seed)
x <- as.matrix(read.csv(x_csv, header=FALSE, check.names=FALSE))
y <- scan(y_csv, quiet=TRUE)
p <- ncol(x)

# In sparse k-means, ||w||_2 <= 1 implies 1 <= ||w||_1 <= sqrt(p).
# Avoid the exact lower boundary because an excessively sparse one-feature solution
# can be numerically unstable in permutation tuning.
max_w <- sqrt(p)
if (min_w >= max_w) min_w <- max(1.01, 0.5 * max_w)
wbounds <- seq(min_w, max_w, length.out=grid_n)

ptm <- proc.time()[[3]]
perm_fit <- KMeansSparseCluster.permute(
  x=x,
  K=K,
  wbounds=wbounds,
  nperms=nperms
)
bestw <- perm_fit$bestw
fit <- KMeansSparseCluster(x=x, K=K, wbounds=bestw)
runtime <- proc.time()[[3]] - ptm

# KMeansSparseCluster() can return either a direct fitted object or an
# unnamed one-element list, depending on how wbounds is represented.
# Recursively locate the first object containing a p-vector named `ws`.
find_spkm_fit <- function(obj, p, depth=0L) {
  if (depth > 5L) return(NULL)
  if (is.list(obj) && !is.null(obj$ws) && length(obj$ws) == p) return(obj)
  if (is.list(obj)) {
    for (ii in seq_along(obj)) {
      ans <- find_spkm_fit(obj[[ii]], p, depth + 1L)
      if (!is.null(ans)) return(ans)
    }
  }
  return(NULL)
}
fit_core <- find_spkm_fit(fit, p)
if (is.null(fit_core)) {
  stop(paste0(
    "Could not locate SPKM fit with a length-p `ws` vector. ",
    "top-level class=", paste(class(fit), collapse="/"),
    "; length=", length(fit),
    "; names=", paste(names(fit), collapse=","),
    "; structure=", paste(capture.output(str(fit, max.level=2)), collapse=" | ")
  ))
}
weights <- as.numeric(fit_core$ws)
weights[!is.finite(weights)] <- 0
weights <- pmax(weights, 0)
if (sum(weights) <= 0) stop("All SPKM feature weights are zero")

# Prefer the clustering returned by sparcl when it can be converted safely.
clusters <- NULL
Cs <- fit_core$Cs
if (!is.null(Cs)) {
  if (is.atomic(Cs) && length(Cs) == nrow(x)) {
    clusters <- as.integer(Cs)
  } else if (is.list(Cs) && length(Cs) == K) {
    tmp <- integer(nrow(x))
    for (kk in seq_along(Cs)) {
      idx <- as.integer(Cs[[kk]])
      idx <- idx[is.finite(idx) & idx >= 1L & idx <= nrow(x)]
      if (length(idx)) tmp[idx] <- kk
    }
    if (all(tmp > 0L)) clusters <- tmp
  }
}

# Fallback: optimize the conditional assignment for the learned SPKM weights
# by k-means after x_ij -> sqrt(w_j) x_ij. This has the same weighted
# within-cluster sum-of-squares objective for fixed w.
if (is.null(clusters)) {
  x_weighted <- sweep(x, 2L, sqrt(weights), FUN="*")
  set.seed(seed + 104729L)
  km <- stats::kmeans(x_weighted, centers=K, nstart=50L, iter.max=100L)
  clusters <- as.integer(km$cluster)
}
if (length(clusters) != nrow(x) || anyNA(clusters)) {
  stop(sprintf("Weighted k-means did not return valid labels: length=%d", length(clusters)))
}

# Return one row per feature; cluster labels are written as a semicolon-separated field.
out <- data.frame(
  feature=seq_len(p)-1L,
  weight=weights,
  bestw=rep(bestw, p),
  runtime_sec=rep(runtime, p),
  clusters=rep(paste(clusters, collapse=";"), p),
  stringsAsFactors=FALSE
)
write.csv(out, out_csv, row.names=FALSE)
'''


def ensure_r_and_sparcl(install_sparcl: bool) -> None:
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript was not found. In Colab, install R first, e.g. !apt-get -qq update && !apt-get -qq install -y r-base"
        )
    check = subprocess.run(
        ["Rscript", "-e", "quit(status=ifelse(requireNamespace('sparcl', quietly=TRUE),0,1))"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return
    if not install_sparcl:
        raise RuntimeError(
            "R package 'sparcl' is not installed. Re-run with --install_sparcl or install.packages('sparcl')."
        )
    cmd = [
        "Rscript",
        "-e",
        "install.packages('sparcl', repos='https://cloud.r-project.org')",
    ]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError("Automatic installation of R package 'sparcl' failed")


def run_spkm(
    X: np.ndarray,
    y: np.ndarray,
    informative_idx: np.ndarray,
    seed: int,
    nperms: int,
    grid_size: int,
    min_w: float,
    weight_threshold: float,
    helper_path: Path,
) -> Dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="spkm_divi_") as td:
        td = Path(td)
        x_csv = td / "x.csv"
        y_csv = td / "y.txt"
        out_csv = td / "out.csv"
        np.savetxt(x_csv, X, delimiter=",", fmt="%.10g")
        np.savetxt(y_csv, y, fmt="%d")

        cmd = [
            "Rscript",
            str(helper_path),
            str(x_csv),
            str(y_csv),
            str(out_csv),
            "3",
            str(nperms),
            str(grid_size),
            str(min_w),
            str(seed),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "SPKM R call failed.\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr
            )
        ans = pd.read_csv(out_csv)

    clusters = np.fromstring(str(ans.loc[0, "clusters"]), sep=";", dtype=int)
    weights = ans["weight"].to_numpy(float)
    bestw = float(ans.loc[0, "bestw"])
    runtime = float(ans.loc[0, "runtime_sec"])

    true_mask = np.zeros(X.shape[1], dtype=int)
    true_mask[informative_idx] = 1
    selected = (weights > weight_threshold).astype(int)
    topd = np.zeros_like(selected)
    topd[np.argsort(weights)[-len(informative_idx):]] = 1

    return {
        "ARI": adjusted_rand_score(y, clusters),
        "NMI": normalized_mutual_info_score(y, clusters),
        "feature_f1_thresh": f1_score(true_mask, selected, zero_division=0),
        "feature_f1_topd": f1_score(true_mask, topd, zero_division=0),
        "active_dims": int(selected.sum()),
        "bestw": bestw,
        "runtime_sec": runtime,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ARI", "NMI", "feature_f1_thresh", "feature_f1_topd",
        "active_dims", "bestw", "runtime_sec"
    ]
    s = df.groupby(["scenario", "n"], sort=True)[metrics].agg(["mean", "std", "count"]).reset_index()
    s.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in s.columns]
    return s


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    def ms(m: float, s: float, digits: int = 3) -> str:
        return f"{m:.{digits}f} ({s:.{digits}f})"

    rows: List[Dict[str, object]] = []
    names = {
        "heavy_tailed_signal": "Heavy-tailed signal",
        "correlated_noise": "Correlated noise",
    }
    for _, r in summary.iterrows():
        rows.append({
            "Scenario": names.get(r["scenario"], r["scenario"]),
            "$N$": int(r["n"]),
            "Method": "SPKM",
            "ARI": ms(r["ARI_mean"], r["ARI_std"]),
            "NMI": ms(r["NMI_mean"], r["NMI_std"]),
            "Feature F1": ms(r["feature_f1_thresh_mean"], r["feature_f1_thresh_std"]),
            "Active dims": ms(r["active_dims_mean"], r["active_dims_std"], 1),
            "bestw": ms(r["bestw_mean"], r["bestw_std"], 2),
        })
    with path.open("w", encoding="utf-8") as f:
        f.write(pd.DataFrame(rows).to_latex(index=False, escape=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add SPKM to DIVI misspecification experiments")
    p.add_argument("--output_dir", default="/mnt/data/spkm_misspecification_results")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    p.add_argument("--quick", action="store_true", help="Use seeds 1-2 and nperms=2 for a smoke test")
    p.add_argument("--install_sparcl", action="store_true")
    p.add_argument("--nperms", type=int, default=10,
                   help="Permutation replicates for selecting the SPKM L1 bound")
    p.add_argument("--grid_size", type=int, default=20,
                   help="Number of candidate L1 bounds between min_w and sqrt(D)")
    p.add_argument("--min_w", type=float, default=1.1)
    p.add_argument("--weight_threshold", type=float, default=1e-6,
                   help="A feature is active when its SPKM weight exceeds this value")
    p.add_argument("--d", type=int, default=100)
    p.add_argument("--d_info", type=int, default=10)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--noise_sd", type=float, default=3.0)
    p.add_argument("--signal_df", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--block_size", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_r_and_sparcl(args.install_sparcl)

    helper_path = out_dir / "_run_sparcl_spkm.R"
    helper_path.write_text(R_HELPER, encoding="utf-8")

    seeds = [1, 2] if args.quick else args.seeds
    nperms = 2 if args.quick else args.nperms
    rows: List[Dict[str, object]] = []

    for scenario in ["heavy_tailed_signal", "correlated_noise"]:
        for n in [200, 1000]:
            for seed in seeds:
                print(f"[run] scenario={scenario} n={n} seed={seed}", flush=True)
                if scenario == "heavy_tailed_signal":
                    X_raw, y, info = generate_heavy_tailed_signal(
                        n=n, d=args.d, d_info=args.d_info, delta=args.delta,
                        signal_df=args.signal_df, noise_sd=args.noise_sd, seed=seed,
                    )
                else:
                    X_raw, y, info = generate_correlated_noise(
                        n=n, d=args.d, d_info=args.d_info, delta=args.delta,
                        noise_sd=args.noise_sd, rho=args.rho,
                        block_size=args.block_size, seed=seed,
                    )

                # Match manuscript preprocessing: standardize each feature before fitting.
                X = StandardScaler().fit_transform(X_raw)
                t0 = time.perf_counter()
                try:
                    metrics = run_spkm(
                        X=X, y=y, informative_idx=info, seed=seed,
                        nperms=nperms, grid_size=args.grid_size, min_w=args.min_w,
                        weight_threshold=args.weight_threshold, helper_path=helper_path,
                    )
                    status, error = "ok", ""
                except Exception as exc:
                    metrics = {k: np.nan for k in [
                        "ARI", "NMI", "feature_f1_thresh", "feature_f1_topd",
                        "active_dims", "bestw", "runtime_sec"
                    ]}
                    status, error = "failed", repr(exc)
                    print(f"[ERROR] {error}", flush=True)
                metrics["wall_sec_python"] = time.perf_counter() - t0
                rows.append({
                    "scenario": scenario,
                    "n": n,
                    "d": args.d,
                    "d_info": args.d_info,
                    "delta": args.delta,
                    "noise_sd": args.noise_sd,
                    "signal_df": args.signal_df if scenario == "heavy_tailed_signal" else np.nan,
                    "rho": args.rho if scenario == "correlated_noise" else np.nan,
                    "block_size": args.block_size if scenario == "correlated_noise" else np.nan,
                    "seed": seed,
                    "method": "SPKM",
                    "status": status,
                    "error": error,
                    **metrics,
                })
                pd.DataFrame(rows).to_csv(out_dir / "spkm_misspec_raw.csv", index=False)

    raw = pd.DataFrame(rows)
    ok = raw.loc[raw["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("All SPKM runs failed; inspect spkm_misspec_raw.csv and the printed R error")
    summary = summarize(ok)
    summary.to_csv(out_dir / "spkm_misspec_summary.csv", index=False)
    write_latex(summary, out_dir / "spkm_misspec_table.tex")

    config = vars(args).copy()
    config["effective_seeds"] = seeds
    config["effective_nperms"] = nperms
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\n[done]")
    print(summary.to_string(index=False))
    print(f"\nResults written to: {out_dir}")


if __name__ == "__main__":
    main()
