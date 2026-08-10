#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split-safeguard ablation for the DIVI revision.

Compares four split mechanisms on the matched sparse Gaussian benchmark:

1. original
   Original irreversible split rule from divi_mlwa.py.
2. burnin_only
   Burn-in is active; an eligible proposal is accepted immediately.
3. burnin_persistence
   Burn-in and persistence are active; an eligible proposal is accepted immediately.
4. full_safeguard
   Burn-in, persistence, minimum cluster size, and matched-budget objective
   acceptance are all active.

The script reports clustering accuracy, feature recovery, final K,
over-/under-splitting rates, split counts, and runtime. Each method receives
exactly the same generated dataset for a given (N, seed) pair.

Example
-------
python run_split_safeguard_ablation.py \
    --original_divi /content/divi_mlwa.py \
    --safeguarded_divi /content/divi_mlwa_safeguarded.py \
    --output_dir /content/drive/MyDrive/split_safeguard_ablation \
    --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

Quick smoke test
----------------
python run_split_safeguard_ablation.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def load_module(path: str | Path, module_name: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Python source not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_labels(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    counts = [n // k] * k
    counts[-1] += n - sum(counts)
    y = np.concatenate([np.full(c, g, dtype=int) for g, c in enumerate(counts)])
    rng.shuffle(y)
    return y


def generate_sparse_gaussian_mixture(
    n: int,
    d: int = 100,
    k: int = 3,
    d_info: int = 10,
    delta: float = 2.0,
    signal_sd: float = 1.0,
    noise_sd: float = 3.0,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k != 3:
        raise ValueError("This generator assumes K=3 with means (-delta, 0, +delta).")
    if not 0 < d_info <= d:
        raise ValueError("Require 0 < d_info <= d.")

    rng = np.random.default_rng(seed)
    y = balanced_labels(n, k, rng)
    X = rng.normal(0.0, noise_sd, size=(n, d))
    means = np.array([-delta, 0.0, delta], dtype=float)

    for g in range(k):
        mask = y == g
        X[mask, :d_info] = rng.normal(
            loc=means[g], scale=signal_sd, size=(int(mask.sum()), d_info)
        )

    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]
    informative = np.arange(d_info, dtype=int)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, y, informative


def feature_scores(model: Any) -> np.ndarray:
    if hasattr(model, "get_feature_probabilities"):
        values = model.get_feature_probabilities()
        return np.asarray(values, dtype=float).reshape(-1)
    if getattr(model, "model", None) is None:
        raise RuntimeError("Fitted DIVI object has no internal model.")
    with torch.no_grad():
        return (
            torch.sigmoid(model.model.phi_logits)
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .reshape(-1)
        )


def predict_labels(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=int)
    if getattr(model, "model", None) is None:
        raise RuntimeError("Fitted DIVI object has no internal model.")
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        try:
            _, _, log_scores = model.model(X_tensor, sample_phi=False)
        except TypeError:
            _, _, log_scores = model.model(X_tensor)
    return torch.argmax(log_scores, dim=1).cpu().numpy().astype(int)


def feature_metrics(scores: np.ndarray, informative: np.ndarray) -> Dict[str, float]:
    d = len(scores)
    true_mask = np.zeros(d, dtype=int)
    true_mask[informative] = 1

    threshold_mask = (scores >= 0.5).astype(int)
    topd_mask = np.zeros(d, dtype=int)
    topd_idx = np.argsort(scores)[-len(informative):]
    topd_mask[topd_idx] = 1

    return {
        "active_dims": int(threshold_mask.sum()),
        "feature_f1_thresh": float(f1_score(true_mask, threshold_mask, zero_division=0)),
        "feature_f1_topd": float(f1_score(true_mask, topd_mask, zero_division=0)),
    }


def original_split_count(model: Any) -> int:
    k_path = list(getattr(model, "history", {}).get("k", []))
    if not k_path:
        return 0
    return int(sum(int(k_path[i] > k_path[i - 1]) for i in range(1, len(k_path))))


class AcceptAllEligibleSafeguarded:
    """Mixin-like helper that overrides only proposal acceptance.

    It is attached dynamically below to preserve the safeguarded implementation's
    burn-in, persistence, and minimum-size logic while bypassing its objective
    acceptance rule. This creates clean burn-in and persistence ablations.
    """

    def _propose_and_evaluate_split(self, X_tensor, split_idx):
        if self.model is None:
            raise RuntimeError("Model has not been initialized.")
        old_model = self.model
        split_model = self._expand_model(old_model, split_idx)
        self.model = split_model
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        event = {
            "split_idx": int(split_idx),
            "old_K": int(old_model.K),
            "proposed_K": int(split_model.K),
            "null_loss": np.nan,
            "split_loss": np.nan,
            "improvement": np.nan,
            "improvement_per_sample": np.nan,
            "accept_tol_per_sample": np.nan,
            "accepted": True,
            "acceptance_mode": "accept_all_eligible",
        }
        return optimizer, event


def build_accept_all_class(base_class):
    return type(
        "AcceptAllEligibleDIVI",
        (AcceptAllEligibleSafeguarded, base_class),
        {},
    )


def run_one(
    method_name: str,
    original_class,
    safeguarded_class,
    accept_all_class,
    X: np.ndarray,
    y: np.ndarray,
    informative: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    set_seed(seed)

    common = dict(
        split_threshold=None,
        split_interval=args.split_interval,
        max_epochs=args.max_epochs,
        lr=args.lr,
        max_components=args.max_components,
        beta_mult=args.beta_mult,
        rough_k=args.rough_k,
        verbose=args.verbose,
    )

    if method_name == "original":
        model = original_class(**common)
    elif method_name == "burnin_only":
        model = accept_all_class(
            **common,
            split_burn_in=args.split_burn_in,
            split_persistence=1,
            split_warmup_epochs=args.split_warmup_epochs,
            split_accept_tol_per_sample=0.0,
            min_cluster_size_for_split=2,
        )
    elif method_name == "burnin_persistence":
        model = accept_all_class(
            **common,
            split_burn_in=args.split_burn_in,
            split_persistence=args.split_persistence,
            split_warmup_epochs=args.split_warmup_epochs,
            split_accept_tol_per_sample=0.0,
            min_cluster_size_for_split=2,
        )
    elif method_name == "full_safeguard":
        model = safeguarded_class(
            **common,
            split_burn_in=args.split_burn_in,
            split_persistence=args.split_persistence,
            split_warmup_epochs=args.split_warmup_epochs,
            split_accept_tol_per_sample=args.split_accept_tol_per_sample,
            min_cluster_size_for_split=args.min_cluster_size_for_split,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    started = time.perf_counter()
    model.fit(X, use_prior=args.use_prior)
    runtime = time.perf_counter() - started

    labels = predict_labels(model, X)
    scores = feature_scores(model)
    fm = feature_metrics(scores, informative)
    final_k = int(model.model.K)

    if method_name == "original":
        initial_k = 1
        accepted = final_k - initial_k
        proposed = accepted
        rejected = 0
    else:
        events = list(getattr(model, "split_history", []))
        proposed = len(events)
        accepted = int(sum(bool(e.get("accepted", False)) for e in events))
        rejected = proposed - accepted

    return {
        "method": method_name,
        "status": "ok",
        "error": "",
        "ARI": float(adjusted_rand_score(y, labels)),
        "NMI": float(normalized_mutual_info_score(y, labels)),
        "runtime_sec": float(runtime),
        "final_K": final_k,
        "over_split": int(final_k > args.k),
        "under_split": int(final_k < args.k),
        "correct_K": int(final_k == args.k),
        "split_proposals": int(proposed),
        "accepted_splits": int(accepted),
        "rejected_splits": int(rejected),
        **fm,
        "split_history_json": json.dumps(
            getattr(model, "split_history", []), default=float
        ),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ARI",
        "NMI",
        "runtime_sec",
        "final_K",
        "over_split",
        "under_split",
        "correct_K",
        "split_proposals",
        "accepted_splits",
        "rejected_splits",
        "active_dims",
        "feature_f1_thresh",
        "feature_f1_topd",
    ]
    ok = raw.loc[raw["status"] == "ok"].copy()
    rows: List[Dict[str, Any]] = []

    for keys, group in ok.groupby(["n", "method"], sort=True):
        n, method = keys
        row: Dict[str, Any] = {
            "n": int(n),
            "method": method,
            "successful_runs": int(len(group)),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["n", "method"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_divi", default="/content/divi_mlwa.py")
    parser.add_argument(
        "--safeguarded_divi", default="/content/divi_mlwa_safeguarded.py"
    )
    parser.add_argument(
        "--output_dir",
        default="/content/drive/MyDrive/split_safeguard_ablation",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--n_values", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--d_info", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--signal_sd", type=float, default=1.0)
    parser.add_argument("--noise_sd", type=float, default=3.0)

    parser.add_argument("--split_interval", type=int, default=60)
    parser.add_argument("--split_burn_in", type=int, default=120)
    parser.add_argument("--split_persistence", type=int, default=2)
    parser.add_argument("--split_warmup_epochs", type=int, default=20)
    parser.add_argument("--split_accept_tol_per_sample", type=float, default=1e-3)
    parser.add_argument("--min_cluster_size_for_split", type=int, default=25)
    parser.add_argument("--max_components", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta_mult", type=float, default=1.0)
    parser.add_argument("--rough_k", type=int, default=3)
    parser.add_argument("--use_prior", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [1, 2]
        args.n_values = [200]
        args.max_epochs = min(args.max_epochs, 180)
        args.split_warmup_epochs = min(args.split_warmup_epochs, 5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_module = load_module(args.original_divi, "divi_original_impl")
    safeguarded_module = load_module(args.safeguarded_divi, "divi_safeguarded_impl")
    original_class = original_module.DIVIClustering
    safeguarded_class = safeguarded_module.DIVIClustering
    accept_all_class = build_accept_all_class(safeguarded_class)

    methods = [
        "original",
        "burnin_only",
        "burnin_persistence",
        "full_safeguard",
    ]

    raw_path = output_dir / "split_safeguard_raw.csv"
    rows: List[Dict[str, Any]] = []

    for n in args.n_values:
        for seed in args.seeds:
            X, y, informative = generate_sparse_gaussian_mixture(
                n=n,
                d=args.d,
                k=args.k,
                d_info=args.d_info,
                delta=args.delta,
                signal_sd=args.signal_sd,
                noise_sd=args.noise_sd,
                seed=seed,
            )

            for method in methods:
                print(f"[run] N={n}, seed={seed}, method={method}", flush=True)
                base = {
                    "scenario": "matched_sparse_gaussian",
                    "n": int(n),
                    "d": int(args.d),
                    "d_info": int(args.d_info),
                    "k_true": int(args.k),
                    "delta": float(args.delta),
                    "noise_sd": float(args.noise_sd),
                    "seed": int(seed),
                    "split_interval": int(args.split_interval),
                    "split_burn_in": int(args.split_burn_in),
                    "split_persistence": int(args.split_persistence),
                    "split_warmup_epochs": int(args.split_warmup_epochs),
                    "split_accept_tol_per_sample": float(
                        args.split_accept_tol_per_sample
                    ),
                    "min_cluster_size_for_split": int(
                        args.min_cluster_size_for_split
                    ),
                }
                try:
                    result = run_one(
                        method,
                        original_class,
                        safeguarded_class,
                        accept_all_class,
                        X,
                        y,
                        informative,
                        seed,
                        args,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    result = {
                        "method": method,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append({**base, **result})
                pd.DataFrame(rows).to_csv(raw_path, index=False)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    summary.to_csv(output_dir / "split_safeguard_summary.csv", index=False)

    config = vars(args).copy()
    config["original_divi"] = str(Path(args.original_divi).resolve())
    config["safeguarded_divi"] = str(Path(args.safeguarded_divi).resolve())
    (output_dir / "split_safeguard_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {output_dir / 'split_safeguard_summary.csv'}")


if __name__ == "__main__":
    main()
