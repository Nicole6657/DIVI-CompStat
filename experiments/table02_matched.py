from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.preprocessing import StandardScaler

from divi_core import DIVIClustering


# =========================================================
# Synthetic data generator aligned with current paper
# =========================================================

def generate_matched_synthetic(
    N: int,
    D: int = 100,
    K: int = 3,
    n_signal: int = 10,
    signal_means: Tuple[float, float, float] = (-2.0, 0.0, 2.0),
    noise_sigma: float = 3.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if K != 3:
        raise ValueError("This helper is written for K=3 to match the paper.")

    sizes = [N // 3, N // 3, N - 2 * (N // 3)]
    X_list, y_list = [], []

    for k, nk in enumerate(sizes):
        X_sig = rng.normal(
            loc=signal_means[k],
            scale=1.0,
            size=(nk, n_signal),
        )
        X_noise = rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=(nk, D - n_signal),
        )
        Xk = np.hstack([X_sig, X_noise])
        X_list.append(Xk)
        y_list.append(np.full(nk, k, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    perm = rng.permutation(N)
    X = X[perm]
    y = y[perm]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    support = np.zeros(D, dtype=int)
    support[:n_signal] = 1
    return X_scaled, y, support


# =========================================================
# Feature F1
# =========================================================

def feature_f1_from_mask(selected: np.ndarray, truth: np.ndarray) -> float:
    return f1_score(truth.astype(int), selected.astype(int), zero_division=0)


def feature_f1_from_phi(phi_probs: np.ndarray, n_signal: int = 10, threshold: float = 0.5) -> float:
    D = len(phi_probs)
    truth = np.zeros(D, dtype=int)
    truth[:n_signal] = 1
    selected = (phi_probs >= threshold).astype(int)
    return feature_f1_from_mask(selected, truth)


# =========================================================
# Baselines
# =========================================================

def run_kmeans_oracle(X: np.ndarray, K: int, seed: int) -> Dict[str, object]:
    model = KMeans(n_clusters=K, random_state=seed, n_init=20)
    labels = model.fit_predict(X)
    return {"labels": labels, "K_final": K}


def run_gmm_diag_oracle(X: np.ndarray, K: int, seed: int) -> Dict[str, object]:
    model = GaussianMixture(
        n_components=K,
        covariance_type="diag",
        random_state=seed,
        n_init=5,
        reg_covar=1e-6,
        max_iter=500,
    )
    labels = model.fit(X).predict(X)
    return {"labels": labels, "K_final": K}


# =========================================================
# SPKM via R/sparcl
# =========================================================

def _which_rscript() -> str:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "找不到 Rscript。請先安裝 R，並確認 Rscript 在 PATH 中。"
        )
    return rscript


def run_spkm_sparcl(
    X: np.ndarray,
    K: int,
    seed: int,
    nperms: int = 25,
    wbound_min: float = 1.1,
    wbound_max: float | None = None,
    wbound_len: int = 20,
) -> Dict[str, object]:
    """
    Run Sparse K-Means via R package sparcl.
    Uses KMeansSparseCluster.permute to tune wbound, then KMeansSparseCluster to fit.
    """
    rscript = _which_rscript()

    if wbound_max is None:
        wbound_max = float(np.sqrt(X.shape[1]))

    with tempfile.TemporaryDirectory() as td:
        x_path = os.path.join(td, "X.csv")
        out_path = os.path.join(td, "spkm_out.json")
        r_path = os.path.join(td, "run_spkm.R")

        pd.DataFrame(X).to_csv(x_path, index=False, header=False)

        r_code = f"""
suppressWarnings(suppressMessages(library(sparcl)))
suppressWarnings(suppressMessages(library(jsonlite)))

X <- as.matrix(read.csv("{x_path}", header=FALSE))
set.seed({seed})

wbounds <- seq({wbound_min}, {wbound_max}, length.out={wbound_len})

perm.out <- KMeansSparseCluster.permute(
  X,
  K={K},
  nperms={nperms},
  wbounds=wbounds,
  silent=TRUE
)

best.idx <- which.max(perm.out$Gap)
best.w <- perm.out$wbounds[best.idx]

fit <- KMeansSparseCluster(
  X,
  K={K},
  wbounds=best.w,
  nstart=20,
  silent=TRUE
)

# fit is a list; cluster assignments and weights are in fit[[1]]
res <- list(
  labels = as.integer(fit[[1]]$Cs),
  weights = as.numeric(fit[[1]]$ws),
  best_w = as.numeric(best.w),
  K_final = {K}
)

write(toJSON(res, auto_unbox=TRUE), "{out_path}")
"""

        with open(r_path, "w", encoding="utf-8") as f:
            f.write(r_code)

        proc = subprocess.run(
            [rscript, r_path],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "R/sparcl 執行失敗。\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )

        if not os.path.exists(out_path):
            raise RuntimeError("R/sparcl 沒有產出結果檔。")

        with open(out_path, "r", encoding="utf-8") as f:
            res = json.load(f)

        labels = np.asarray(res["labels"], dtype=int) - 1  # R is 1-based
        weights = np.asarray(res["weights"], dtype=float)
        selected_support = (weights > 1e-8).astype(int)

        return {
            "labels": labels,
            "K_final": int(res["K_final"]),
            "selected_support": selected_support,
            "weights": weights,
            "best_w": float(res["best_w"]),
        }


# =========================================================
# DIVI current protocol + split_interval=120
# =========================================================

PRIOR_MODES = {
    1: "DIVI-Info",
    2: "DIVI-NonInfo",
    3: "DIVI-Random",
}


def run_divi_current120(
    X: np.ndarray,
    seed: int,
    prior_mode: int,
    max_epochs: int = 300,
    split_interval: int = 120,
    lr: float = 0.01,
    beta_mult: float = 1.0,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    prior_logvar0: float = 2.197,
    verbose: bool = False,
) -> Dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = DIVIClustering(
        split_threshold=None,
        split_interval=split_interval,
        max_epochs=max_epochs,
        lr=lr,
        beta_mult=beta_mult,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        prior_logvar0=prior_logvar0,
        verbose=verbose,
    )
    model.fit(X, use_prior=prior_mode)

    labels = model.predict(X)
    phi = model.get_phi()
    selected = (phi >= 0.5).astype(int)

    return {
        "labels": labels,
        "K_final": int(model.fit_summary_["final_K"]),
        "selected_support": selected,
        "mean_phi": float(np.mean(phi)),
        "split_count": int(model.fit_summary_["split_count"]),
        "split_epochs": model.get_split_epochs(),
    }


# =========================================================
# Evaluation helpers
# =========================================================

def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return float(ari), float(nmi)


def summarize(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def fmt_mean_sd(m: float, s: float, digits: int = 3) -> str:
    if np.isnan(m):
        return "--"
    return f"{m:.{digits}f} ({s:.{digits}f})"


# =========================================================
# Main experiment
# =========================================================

def run_experiment(
    N_list: List[int],
    n_trials: int,
    master_seed: int,
    spkm_nperms: int,
    spkm_wbound_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 99999, size=n_trials)

    all_rows: List[Dict[str, object]] = []

    for N in N_list:
        for trial_idx, seed in enumerate(seeds, start=1):
            X, y_true, support_true = generate_matched_synthetic(
                N=N,
                D=100,
                K=3,
                n_signal=10,
                signal_means=(-2.0, 0.0, 2.0),
                noise_sigma=3.0,
                seed=int(seed),
            )

            # K-Means
            out = run_kmeans_oracle(X, K=3, seed=int(seed))
            ari, nmi = evaluate_clustering(y_true, out["labels"])
            all_rows.append({
                "N": N,
                "Trial": trial_idx,
                "Seed": int(seed),
                "Method": "K-Means",
                "Prior": "N/A",
                "FinalK": out["K_final"],
                "ARI": ari,
                "NMI": nmi,
                "F1": np.nan,
            })

            # diagonal GMM
            out = run_gmm_diag_oracle(X, K=3, seed=int(seed))
            ari, nmi = evaluate_clustering(y_true, out["labels"])
            all_rows.append({
                "N": N,
                "Trial": trial_idx,
                "Seed": int(seed),
                "Method": "GMM",
                "Prior": "N/A",
                "FinalK": out["K_final"],
                "ARI": ari,
                "NMI": nmi,
                "F1": np.nan,
            })

            # SPKM via sparcl
            out = run_spkm_sparcl(
                X=X,
                K=3,
                seed=int(seed),
                nperms=spkm_nperms,
                wbound_len=spkm_wbound_len,
            )
            ari, nmi = evaluate_clustering(y_true, out["labels"])
            f1 = feature_f1_from_mask(out["selected_support"], support_true)
            all_rows.append({
                "N": N,
                "Trial": trial_idx,
                "Seed": int(seed),
                "Method": "SPKM",
                "Prior": "N/A",
                "FinalK": out["K_final"],
                "ARI": ari,
                "NMI": nmi,
                "F1": f1,
            })

            # DIVI modes
            for prior_mode, method_name in PRIOR_MODES.items():
                out = run_divi_current120(
                    X=X,
                    seed=int(seed),
                    prior_mode=prior_mode,
                    max_epochs=300,
                    split_interval=120,
                    lr=0.01,
                    beta_mult=1.0,
                    temperature_start=1.0,
                    temperature_end=0.1,
                    prior_logvar0=2.197,
                    verbose=False,
                )
                ari, nmi = evaluate_clustering(y_true, out["labels"])
                f1 = feature_f1_from_mask(out["selected_support"], support_true)

                all_rows.append({
                    "N": N,
                    "Trial": trial_idx,
                    "Seed": int(seed),
                    "Method": method_name,
                    "Prior": method_name,
                    "FinalK": out["K_final"],
                    "ARI": ari,
                    "NMI": nmi,
                    "F1": f1,
                })

    raw_df = pd.DataFrame(all_rows)

    summary_rows: List[Dict[str, object]] = []
    for (N, method), g in raw_df.groupby(["N", "Method"], sort=False):
        k_m, k_s = summarize(g["FinalK"].tolist())
        ari_m, ari_s = summarize(g["ARI"].tolist())
        nmi_m, nmi_s = summarize(g["NMI"].tolist())

        if g["F1"].notna().any():
            f1_m, f1_s = summarize(g["F1"].dropna().tolist())
        else:
            f1_m, f1_s = np.nan, np.nan

        summary_rows.append({
            "N": N,
            "Method": method,
            "K_mean": k_m,
            "K_sd": k_s,
            "ARI_mean": ari_m,
            "ARI_sd": ari_s,
            "NMI_mean": nmi_m,
            "NMI_sd": nmi_s,
            "F1_mean": f1_m,
            "F1_sd": f1_s,
        })

    summary_df = pd.DataFrame(summary_rows)

    methods_order = ["K-Means", "GMM", "SPKM", "DIVI-Info", "DIVI-NonInfo", "DIVI-Random"]
    paper_rows: List[Dict[str, str]] = []
    for method in methods_order:
        row = {"Method": method}
        for N in N_list:
            g = summary_df[(summary_df["Method"] == method) & (summary_df["N"] == N)]
            if len(g) == 0:
                continue
            g = g.iloc[0]
            row[f"K_{N}"] = fmt_mean_sd(g["K_mean"], g["K_sd"])
            row[f"ARI_{N}"] = fmt_mean_sd(g["ARI_mean"], g["ARI_sd"])
            row[f"NMI_{N}"] = fmt_mean_sd(g["NMI_mean"], g["NMI_sd"])
            row[f"F1_{N}"] = fmt_mean_sd(g["F1_mean"], g["F1_sd"]) if not np.isnan(g["F1_mean"]) else "--"
        paper_rows.append(row)

    paper_df = pd.DataFrame(paper_rows)
    return raw_df, summary_df, paper_df


# =========================================================
# LaTeX rendering
# =========================================================

def make_two_line_latex_table(paper_df: pd.DataFrame, caption: str, label: str) -> str:
    def split_mean_sd(cell: str) -> Tuple[str, str]:
        if cell == "--":
            return "--", "--"
        mean, sd = cell.split(" ")
        return mean, sd

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.96}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{4}{c}{$N=200$ (Small-$n$)} & \multicolumn{4}{c}{$N=1000$ (Large-$n$)} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"Method & $K$ & ARI & NMI & F1 & $K$ & ARI & NMI & F1 \\")
    lines.append(r"\midrule")

    for _, row in paper_df.iterrows():
        def strip0(x: str) -> str:
            if x == "--":
                return x
            return "." + x[2:] if x.startswith("0.") else x

        def paren_strip0(x: str) -> str:
            if x == "--":
                return x
            x = x.strip("()")
            x = "." + x[2:] if x.startswith("0.") else x
            return f"({x})"

        k200_m, k200_s = split_mean_sd(row.get("K_200", "--"))
        ari200_m, ari200_s = split_mean_sd(row.get("ARI_200", "--"))
        nmi200_m, nmi200_s = split_mean_sd(row.get("NMI_200", "--"))
        f1200_m, f1200_s = split_mean_sd(row.get("F1_200", "--"))

        k1000_m, k1000_s = split_mean_sd(row.get("K_1000", "--"))
        ari1000_m, ari1000_s = split_mean_sd(row.get("ARI_1000", "--"))
        nmi1000_m, nmi1000_s = split_mean_sd(row.get("NMI_1000", "--"))
        f11000_m, f11000_s = split_mean_sd(row.get("F1_1000", "--"))

        lines.append(
            f"{row['Method']} & "
            f"{strip0(k200_m)} & {strip0(ari200_m)} & {strip0(nmi200_m)} & {strip0(f1200_m)} & "
            f"{strip0(k1000_m)} & {strip0(ari1000_m)} & {strip0(nmi1000_m)} & {strip0(f11000_m)} \\\\"
        )
        lines.append(
            f"& {paren_strip0(k200_s)} & {paren_strip0(ari200_s)} & {paren_strip0(nmi200_s)} & {paren_strip0(f1200_s)} & "
            f"{paren_strip0(k1000_s)} & {paren_strip0(ari1000_s)} & {paren_strip0(nmi1000_s)} & {paren_strip0(f11000_s)} \\\\"
        )
        lines.append(r"\addlinespace[1pt]")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 1 with current DIVI protocol and SPKM via R/sparcl.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=20260406)
    parser.add_argument("--spkm-nperms", type=int, default=25)
    parser.add_argument("--spkm-wbound-len", type=int, default=20)
    parser.add_argument("--output-prefix", type=str, default="table1_current120_sparcl")
    args = parser.parse_args()

    raw_df, summary_df, paper_df = run_experiment(
        N_list=[200, 1000],
        n_trials=args.n_trials,
        master_seed=args.master_seed,
        spkm_nperms=args.spkm_nperms,
        spkm_wbound_len=args.spkm_wbound_len,
    )

    raw_path = f"{args.output_prefix}_raw.csv"
    summary_path = f"{args.output_prefix}_summary.csv"
    paper_path = f"{args.output_prefix}_paper_like.csv"
    tex_path = f"{args.output_prefix}.tex"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    paper_df.to_csv(paper_path, index=False)

    latex = make_two_line_latex_table(
        paper_df=paper_df,
        caption=(
            "Clustering performance under high-dimensional noise. Results are averaged "
            "over 20 independently generated synthetic datasets with $D=100$ and "
            "$90\\%$ irrelevant dimensions. For DIVI, we use the current implementation "
            "protocol with the sensitivity-selected split interval "
            "$T_{\\mathrm{split}}=120$ in order to avoid the mild over-splitting "
            "observed under the default split schedule in this split-only formulation."
        ),
        label="tab:synthetic_main",
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print("\n=== Summary ===")
    print(summary_df.sort_values(["N", "Method"]).to_string(index=False))

    print("\n=== Paper-like layout ===")
    print(paper_df.to_string(index=False))

    print(f"\nSaved:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {paper_path}")
    print(f"  {tex_path}")


if __name__ == "__main__":
    main()