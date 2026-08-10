#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR-DIVI cluster-specific relevance simulation (fixed K).

This runner uses hr_divi_core_fixedk_stable.py and the same matched-separation
design used for the standard DIVI cluster-specific relevance experiment.

Main inferential targets
------------------------
1. Partition recovery:
   ARI and NMI.

2. Global screening:
   Does g_j recover the union S1 ∪ S2 ∪ S3?

3. Cluster-specific screening:
   After matching estimated component labels to the true labels by the
   Hungarian algorithm, does effective relevance
       w_kj = g_j r_kj
   recover the cluster-specific support S_k?

The cluster-specific design is:
    S1: cluster 1 versus the other two clusters
    S2: cluster 2 versus the other two clusters
    S3: cluster 3 versus the other two clusters

The local contrast is rescaled by sqrt(3/4) so that aggregate pairwise
separation matches the global-support control.

Example
-------
python run_hrdivi_cluster_specific_relevance.py \
  --hrdivi_path /content/hr_divi_core_fixedk_stable.py \
  --output_dir /content/hrdivi_cluster_specific_delta10 \
  --delta 1.0 \
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

Quick test
----------
python run_hrdivi_cluster_specific_relevance.py \
  --hrdivi_path /content/hr_divi_core_fixedk_stable.py \
  --output_dir /content/hrdivi_cluster_specific_test \
  --delta 1.0 \
  --quick
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
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
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


def load_hrdivi_class(path: str):
    path = str(Path(path).resolve())
    if not Path(path).exists():
        raise FileNotFoundError(f"HR-DIVI source not found: {path}")

    spec = importlib.util.spec_from_file_location("hrdivi_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import HR-DIVI module from {path}")

    module = importlib.util.module_from_spec(spec)

    # Register the dynamically created module before execution.
    # Python 3.12 dataclasses inspect sys.modules[cls.__module__]
    # while processing @dataclass; without this registration,
    # sys.modules.get(cls.__module__) returns None.
    import sys
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    if not hasattr(module, "HRDIVIClustering"):
        raise AttributeError(f"{path} does not define HRDIVIClustering")

    return module.HRDIVIClustering


def balanced_labels(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.full(k, n // k, dtype=int)
    counts[: n % k] += 1
    y = np.concatenate([
        np.full(counts[j], j, dtype=int) for j in range(k)
    ])
    rng.shuffle(y)
    return y


def generate_data(
    design: str,
    n: int,
    d: int,
    block_size: int,
    delta: float,
    signal_sd: float,
    noise_sd: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Generate matched-separation global and cluster-specific designs.

    Returns
    -------
    X : (n, d)
    y : (n,)
    supports : dict with S1, S2, S3, union
    local_truth : (3, d) binary cluster-specific support matrix
    """
    if d < 3 * block_size:
        raise ValueError("d must be at least 3 * block_size")

    rng = np.random.default_rng(seed)
    y = balanced_labels(n, 3, rng)

    # Start all variables as high-variance nuisance variables.
    X = rng.normal(0.0, noise_sd, size=(n, d))

    s1 = np.arange(0, block_size)
    s2 = np.arange(block_size, 2 * block_size)
    s3 = np.arange(2 * block_size, 3 * block_size)
    union = np.concatenate([s1, s2, s3])

    if design == "global_support":
        # Every informative feature is globally discriminative.
        # Cyclic blocks equalize aggregate separation across cluster pairs.
        patterns = {
            "S1": np.array([-delta, 0.0, +delta]),
            "S2": np.array([0.0, +delta, -delta]),
            "S3": np.array([+delta, -delta, 0.0]),
        }

        # In the global-support control, all 15 features are relevant to
        # every component.
        local_truth = np.zeros((3, d), dtype=int)
        local_truth[:, union] = 1

    elif design == "cluster_specific":
        # One-cluster-versus-rest support.
        # sqrt(3/4) matches aggregate pairwise separation to global control.
        local_delta = np.sqrt(3.0 / 4.0) * delta
        patterns = {
            "S1": np.array([+local_delta, -local_delta, -local_delta]),
            "S2": np.array([-local_delta, +local_delta, -local_delta]),
            "S3": np.array([-local_delta, -local_delta, +local_delta]),
        }

        local_truth = np.zeros((3, d), dtype=int)
        local_truth[0, s1] = 1
        local_truth[1, s2] = 1
        local_truth[2, s3] = 1

    else:
        raise ValueError(f"Unknown design: {design}")

    blocks = {"S1": s1, "S2": s2, "S3": s3}
    for block_name, idx in blocks.items():
        pattern = patterns[block_name]
        for k in range(3):
            rows = y == k
            X[np.ix_(rows, idx)] = rng.normal(
                loc=pattern[k],
                scale=signal_sd,
                size=(rows.sum(), len(idx)),
            )

    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]

    supports = {
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "union": union,
    }
    return X, y, supports, local_truth


def match_components(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 3,
) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    """
    Match estimated component labels to true labels by maximizing overlap.

    Returns
    -------
    pred_to_true : array, pred_to_true[predicted_component] = true_cluster
    mapping : dict
    contingency : true-by-predicted count matrix
    """
    contingency = np.zeros((k, k), dtype=int)
    for t in range(k):
        for p in range(k):
            contingency[t, p] = np.sum((y_true == t) & (y_pred == p))

    true_rows, pred_cols = linear_sum_assignment(-contingency)

    pred_to_true = np.full(k, -1, dtype=int)
    for t, p in zip(true_rows, pred_cols):
        pred_to_true[p] = t

    if np.any(pred_to_true < 0):
        raise RuntimeError(
            f"Incomplete label matching: pred_to_true={pred_to_true.tolist()}"
        )

    mapping = {int(p): int(pred_to_true[p]) for p in range(k)}
    return pred_to_true, mapping, contingency


def reorder_component_matrix(
    matrix_pred_order: np.ndarray,
    pred_to_true: np.ndarray,
) -> np.ndarray:
    """
    Reorder a K x D matrix from predicted-component order to true-cluster order.
    """
    k, d = matrix_pred_order.shape
    matrix_true_order = np.empty((k, d), dtype=float)

    for pred_k in range(k):
        true_k = int(pred_to_true[pred_k])
        matrix_true_order[true_k] = matrix_pred_order[pred_k]

    return matrix_true_order


def binary_score_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    top_m: int,
) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)

    pred = (scores >= threshold).astype(int)

    top_idx = np.argsort(scores)[-top_m:]
    pred_top = np.zeros_like(truth)
    pred_top[top_idx] = 1

    result = {
        "f1_thresh": float(f1_score(truth, pred, zero_division=0)),
        "f1_topm": float(f1_score(truth, pred_top, zero_division=0)),
        "selected_count": int(pred.sum()),
    }

    if np.unique(truth).size == 2:
        result["auroc"] = float(roc_auc_score(truth, scores))
        result["auprc"] = float(average_precision_score(truth, scores))
    else:
        result["auroc"] = np.nan
        result["auprc"] = np.nan

    return result


def evaluate_relevance(
    g: np.ndarray,
    r_true_order: np.ndarray,
    w_true_order: np.ndarray,
    supports: Dict[str, np.ndarray],
    local_truth: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    d = len(g)
    union = supports["union"]

    union_truth = np.zeros(d, dtype=int)
    union_truth[union] = 1

    # Global gate must recover the union of all locally relevant variables.
    global_metrics = binary_score_metrics(
        union_truth,
        g,
        threshold=threshold,
        top_m=len(union),
    )

    out = {
        "global_union_f1_thresh": global_metrics["f1_thresh"],
        "global_union_f1_top": global_metrics["f1_topm"],
        "global_selected_dims": global_metrics["selected_count"],
        "global_union_auroc": global_metrics["auroc"],
        "global_union_auprc": global_metrics["auprc"],
        "global_signal_mean": float(g[union].mean()),
        "global_noise_mean": float(
            g[np.setdiff1d(np.arange(d), union)].mean()
        ),
    }

    # Cluster-specific target: effective relevance w_kj.
    local_f1_thresh: List[float] = []
    local_f1_top: List[float] = []
    local_auroc: List[float] = []
    local_auprc: List[float] = []
    local_selected: List[int] = []
    local_signal_mean: List[float] = []
    local_noise_mean: List[float] = []

    # Also evaluate r separately as a diagnostic.
    r_f1_thresh: List[float] = []
    r_f1_top: List[float] = []

    for k in range(3):
        truth_k = local_truth[k]
        m_k = int(truth_k.sum())

        w_metrics = binary_score_metrics(
            truth_k,
            w_true_order[k],
            threshold=threshold,
            top_m=m_k,
        )
        r_metrics = binary_score_metrics(
            truth_k,
            r_true_order[k],
            threshold=threshold,
            top_m=m_k,
        )

        local_f1_thresh.append(w_metrics["f1_thresh"])
        local_f1_top.append(w_metrics["f1_topm"])
        local_auroc.append(w_metrics["auroc"])
        local_auprc.append(w_metrics["auprc"])
        local_selected.append(w_metrics["selected_count"])
        r_f1_thresh.append(r_metrics["f1_thresh"])
        r_f1_top.append(r_metrics["f1_topm"])

        signal_idx = np.flatnonzero(truth_k == 1)
        noise_idx = np.flatnonzero(truth_k == 0)

        local_signal_mean.append(float(w_true_order[k, signal_idx].mean()))
        local_noise_mean.append(float(w_true_order[k, noise_idx].mean()))

        out[f"local_f1_thresh_k{k+1}"] = w_metrics["f1_thresh"]
        out[f"local_f1_top_k{k+1}"] = w_metrics["f1_topm"]
        out[f"local_selected_k{k+1}"] = w_metrics["selected_count"]
        out[f"local_auroc_k{k+1}"] = w_metrics["auroc"]
        out[f"local_auprc_k{k+1}"] = w_metrics["auprc"]
        out[f"r_f1_thresh_k{k+1}"] = r_metrics["f1_thresh"]
        out[f"r_f1_top_k{k+1}"] = r_metrics["f1_topm"]

    out.update({
        "local_macro_f1_thresh": float(np.mean(local_f1_thresh)),
        "local_macro_f1_top": float(np.mean(local_f1_top)),
        "local_macro_auroc": float(np.mean(local_auroc)),
        "local_macro_auprc": float(np.mean(local_auprc)),
        "local_selected_mean": float(np.mean(local_selected)),
        "local_signal_mean": float(np.mean(local_signal_mean)),
        "local_noise_mean": float(np.mean(local_noise_mean)),
        "local_signal_noise_gap": float(
            np.mean(local_signal_mean) - np.mean(local_noise_mean)
        ),
        "r_macro_f1_thresh": float(np.mean(r_f1_thresh)),
        "r_macro_f1_top": float(np.mean(r_f1_top)),
    })

    return out


def run_hrdivi(
    HRDIVIClustering,
    X: np.ndarray,
    y: np.ndarray,
    supports: Dict[str, np.ndarray],
    local_truth: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    set_seed(seed)

    model = HRDIVIClustering(
        max_epochs=args.max_epochs,
        lr=args.lr,
        beta_g_mult=args.beta_g_mult,
        beta_r_mult=args.beta_r_mult,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        prior_logvar0=args.prior_logvar0,
        init_num_components=3,
        init_method="kmeans",
        random_state=seed,
        targeted_birth_enabled=False,
        verbose=args.verbose,
        grad_clip_norm=args.grad_clip_norm,
    )

    t0 = time.perf_counter()
    model.fit(X, use_prior=args.prior_mode)
    elapsed = time.perf_counter() - t0

    y_pred = model.predict(X)
    g = model.get_global_relevance()
    r_pred_order = model.get_cluster_relevance()
    w_pred_order = model.get_effective_relevance()

    pred_to_true, mapping, contingency = match_components(y, y_pred, k=3)
    r_true_order = reorder_component_matrix(r_pred_order, pred_to_true)
    w_true_order = reorder_component_matrix(w_pred_order, pred_to_true)

    result: Dict[str, object] = {
        "method": "HR-DIVI",
        "ARI": float(adjusted_rand_score(y, y_pred)),
        "NMI": float(normalized_mutual_info_score(y, y_pred)),
        "runtime_sec": float(elapsed),
        "final_K": int(model.model.K),
        "label_mapping": json.dumps(mapping, sort_keys=True),
        "contingency": json.dumps(contingency.tolist()),
        "objective_final": float(model.fit_summary_.get("objective_final", np.nan)),
    }

    result.update(
        evaluate_relevance(
            g=g,
            r_true_order=r_true_order,
            w_true_order=w_true_order,
            supports=supports,
            local_truth=local_truth,
            threshold=args.threshold,
        )
    )

    return result


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["design", "delta", "method", "n", "d", "d_info"]
    excluded = set(id_cols + [
        "seed", "status", "error", "label_mapping", "contingency"
    ])

    numeric_cols = [
        c for c in raw.select_dtypes(include=[np.number]).columns
        if c not in excluded
    ]

    summary = (
        raw.groupby(id_cols, dropna=False)[numeric_cols]
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


def make_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows: List[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{HR-DIVI under global and cluster-specific relevance structures. "
        r"Values are mean (standard deviation) over independently generated datasets. "
        r"Global F1 evaluates recovery of the union support using $g_j$; local F1 "
        r"evaluates recovery of cluster-specific supports using $w_{kj}=g_jr_{kj}$.}",
        r"\label{tab:hrdivi_cluster_specific}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Design & ARI & NMI & Global F1 & Global top-$|S|$ F1 & Local F1 & Local top-$|S_k|$ F1 \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        def fmt(metric: str) -> str:
            mean = row.get(f"{metric}_mean", np.nan)
            sd = row.get(f"{metric}_std", np.nan)
            if pd.isna(mean):
                return "--"
            if pd.isna(sd):
                return f"{mean:.3f}"
            return f"{mean:.3f} ({sd:.3f})"

        label = str(row["design"]).replace("_", " ")
        rows.append(
            f"{label} & {fmt('ARI')} & {fmt('NMI')} & "
            f"{fmt('global_union_f1_thresh')} & {fmt('global_union_f1_top')} & "
            f"{fmt('local_macro_f1_thresh')} & {fmt('local_macro_f1_top')} \\\\"
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
        "--hrdivi_path",
        type=str,
        default="/content/hr_divi_core_fixedk_stable.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/content/hrdivi_cluster_specific_results",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 21)),
    )
    p.add_argument("--quick", action="store_true")

    # Same data setting as the DIVI experiment.
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--d", type=int, default=200)
    p.add_argument("--block_size", type=int, default=5)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--signal_sd", type=float, default=1.0)
    p.add_argument("--noise_sd", type=float, default=3.0)

    # HR-DIVI settings.
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--beta_g_mult", type=float, default=1.0)
    p.add_argument("--beta_r_mult", type=float, default=1.0)
    p.add_argument("--temperature_start", type=float, default=1.0)
    p.add_argument("--temperature_end", type=float, default=0.1)
    p.add_argument("--prior_logvar0", type=float, default=2.197)
    p.add_argument("--prior_mode", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--grad_clip_norm", type=float, default=5.0)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [1, 2] if args.quick else args.seeds
    HRDIVIClustering = load_hrdivi_class(args.hrdivi_path)

    config = vars(args).copy()
    config["effective_seeds"] = seeds
    config["K_fixed"] = 3

    (output_dir / "hrdivi_cluster_specific_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    records: List[Dict[str, object]] = []

    for design in ["global_support", "cluster_specific"]:
        for seed in seeds:
            print(
                f"[run] design={design} delta={args.delta} "
                f"n={args.n} d={args.d} seed={seed}",
                flush=True,
            )

            base = {
                "design": design,
                "delta": args.delta,
                "seed": seed,
                "n": args.n,
                "d": args.d,
                "d_info": 3 * args.block_size,
                "status": "ok",
                "error": "",
            }

            try:
                X, y, supports, local_truth = generate_data(
                    design=design,
                    n=args.n,
                    d=args.d,
                    block_size=args.block_size,
                    delta=args.delta,
                    signal_sd=args.signal_sd,
                    noise_sd=args.noise_sd,
                    seed=seed,
                )

                X = StandardScaler().fit_transform(X).astype(np.float32)

                result = run_hrdivi(
                    HRDIVIClustering=HRDIVIClustering,
                    X=X,
                    y=y,
                    supports=supports,
                    local_truth=local_truth,
                    seed=seed,
                    args=args,
                )

                records.append({**base, **result})

            except Exception as exc:
                print(f"[ERROR] {exc!r}", flush=True)
                traceback.print_exc()

                records.append({
                    **base,
                    "method": "HR-DIVI",
                    "status": "failed",
                    "error": repr(exc),
                })

    raw = pd.DataFrame(records)
    raw_path = output_dir / "hrdivi_cluster_specific_raw.csv"
    raw.to_csv(raw_path, index=False)

    ok = raw[raw["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError(
            "All HR-DIVI runs failed. Inspect hrdivi_cluster_specific_raw.csv."
        )

    summary = summarize(ok)
    summary_path = output_dir / "hrdivi_cluster_specific_summary.csv"
    summary.to_csv(summary_path, index=False)

    latex_path = output_dir / "hrdivi_cluster_specific_table.tex"
    make_latex_table(summary, latex_path)

    display_cols = [
        c for c in [
            "design",
            "delta",
            "ARI_mean",
            "ARI_std",
            "NMI_mean",
            "NMI_std",
            "global_union_f1_thresh_mean",
            "global_union_f1_thresh_std",
            "global_union_f1_top_mean",
            "global_union_f1_top_std",
            "local_macro_f1_thresh_mean",
            "local_macro_f1_thresh_std",
            "local_macro_f1_top_mean",
            "local_macro_f1_top_std",
            "global_selected_dims_mean",
            "local_selected_mean_mean",
            "final_K_mean",
        ]
        if c in summary.columns
    ]

    print("\n[done]")
    print(summary[display_cols].to_string(index=False))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
