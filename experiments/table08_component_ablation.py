#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIVI component ablation for the Computational Statistics revision.

Variants
--------
1. Full-DIVI
   Informative Step-A prior + feature gating + safeguarded adaptive splitting.
2. No-StepA
   Uniform prior rho_j=0.5 + feature gating + safeguarded adaptive splitting.
3. No-Gating
   Informative Step-A prior + all feature gates fixed to one +
   safeguarded adaptive splitting.
4. Fixed-K
   Informative Step-A prior + feature gating + oracle fixed K.
5. Unsafe-Split
   Original irreversible split rule.
6. No-StepA-FixedK
   Uniform prior + feature gating + oracle fixed K.

Experiments
-----------
- matched sparse Gaussian mixture;
- correlated-noise misspecification.

Every variant receives exactly the same generated dataset for a given
(setting, seed) pair.

Outputs
-------
component_ablation_raw.csv
component_ablation_summary.csv
component_ablation_config.json
component_ablation_failures.csv

Example
-------
python run_component_ablation.py \
  --original_divi /content/divi_mlwa.py \
  --safeguarded_divi /content/divi_mlwa_safeguarded.py \
  --fixedk_divi /content/divi_core_fixedk.py \
  --output_dir /content/drive/MyDrive/component_ablation \
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

Quick smoke test
----------------
python run_component_ablation.py \
  --original_divi /content/divi_mlwa.py \
  --safeguarded_divi /content/divi_mlwa_safeguarded.py \
  --fixedk_divi /content/divi_core_fixedk.py \
  --output_dir /content/component_ablation_test \
  --quick
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler


VARIANTS = [
    "Full-DIVI",
    "No-StepA",
    "No-Gating",
    "Fixed-K",
    "Unsafe-Split",
    "No-StepA-FixedK",
]


def load_module(path: str | Path, module_name: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Python source not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)

    # Register the module before execution. This is required by
    # dataclasses (and some typing utilities) because they resolve
    # cls.__module__ through sys.modules during class decoration.
    import sys
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Avoid leaving a partially initialized module behind.
        sys.modules.pop(module_name, None)
        raise

    return module


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_labels(
    n: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = [n // k] * k
    counts[-1] += n - sum(counts)
    y = np.concatenate([
        np.full(count, group, dtype=int)
        for group, count in enumerate(counts)
    ])
    rng.shuffle(y)
    return y


def _mean_pattern(delta: float, k: int) -> np.ndarray:
    if k != 3:
        raise ValueError(
            "This experiment currently uses K=3 with means (-delta, 0, +delta)."
        )
    return np.asarray([-delta, 0.0, delta], dtype=float)


def generate_matched(
    n: int,
    d: int,
    k: int,
    d_info: int,
    delta: float,
    signal_sd: float,
    noise_sd: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = balanced_labels(n, k, rng)
    X = rng.normal(0.0, noise_sd, size=(n, d))
    means = _mean_pattern(delta, k)

    for group in range(k):
        mask = y == group
        X[mask, :d_info] = rng.normal(
            loc=means[group],
            scale=signal_sd,
            size=(int(mask.sum()), d_info),
        )

    permutation = rng.permutation(n)
    X = X[permutation]
    y = y[permutation]

    X = StandardScaler().fit_transform(X).astype(np.float32)
    informative = np.arange(d_info, dtype=int)
    return X, y, informative


def toeplitz_block_cov(block_size: int, rho: float) -> np.ndarray:
    idx = np.arange(block_size)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_correlated_noise(
    n: int,
    d: int,
    k: int,
    d_info: int,
    delta: float,
    signal_sd: float,
    noise_sd: float,
    rho: float,
    block_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d_info >= d:
        raise ValueError("Correlated-noise setting requires nuisance dimensions.")

    rng = np.random.default_rng(seed)
    y = balanced_labels(n, k, rng)
    X = np.empty((n, d), dtype=float)
    means = _mean_pattern(delta, k)

    for group in range(k):
        mask = y == group
        X[mask, :d_info] = rng.normal(
            loc=means[group],
            scale=signal_sd,
            size=(int(mask.sum()), d_info),
        )

    nuisance_dim = d - d_info
    cursor = 0
    while cursor < nuisance_dim:
        width = min(block_size, nuisance_dim - cursor)
        cov = (noise_sd ** 2) * toeplitz_block_cov(width, rho)
        block = rng.multivariate_normal(
            mean=np.zeros(width),
            cov=cov,
            size=n,
        )
        X[:, d_info + cursor : d_info + cursor + width] = block
        cursor += width

    permutation = rng.permutation(n)
    X = X[permutation]
    y = y[permutation]

    X = StandardScaler().fit_transform(X).astype(np.float32)
    informative = np.arange(d_info, dtype=int)
    return X, y, informative


def feature_scores(model: Any) -> np.ndarray:
    if hasattr(model, "get_feature_probabilities"):
        values = model.get_feature_probabilities()
        return np.asarray(values, dtype=float).reshape(-1)

    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "phi_logits"):
        raise RuntimeError("Could not extract feature-gate probabilities.")

    with torch.no_grad():
        return (
            torch.sigmoid(inner.phi_logits)
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .reshape(-1)
        )


def predict_labels(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict hard labels across the heterogeneous DIVI implementations.

    The safeguarded implementation accepts ``sample_phi=False`` in ``forward``;
    the original MLWA implementation exposes only ``forward(X)``; and some
    fixed-K implementations provide a wrapper-level ``predict`` method.
    """
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=int)

    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Fitted model has no internal mixture model.")

    X_tensor = torch.as_tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = None

        # Safeguarded DIVI: deterministic posterior-mean gates.
        try:
            output = inner(X_tensor, sample_phi=False)
        except TypeError:
            # Original DIVI: forward accepts X only.
            output = inner(X_tensor)

        if isinstance(output, dict):
            for key in ("log_p_x_given_z", "log_scores", "component_log_prob"):
                if key in output:
                    log_scores = output[key]
                    break
            else:
                raise RuntimeError(
                    "Could not find component log scores in model output dictionary."
                )
        elif isinstance(output, (tuple, list)):
            if len(output) < 3:
                raise RuntimeError(
                    "Model forward output must contain component log scores as "
                    "its third element."
                )
            log_scores = output[2]
        elif torch.is_tensor(output) and output.ndim == 2:
            log_scores = output
        else:
            raise RuntimeError(
                "Unsupported model forward output while predicting labels: "
                f"{type(output)!r}."
            )

    return (
        torch.argmax(log_scores, dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )


def feature_metrics(
    scores: np.ndarray,
    informative: np.ndarray,
) -> Dict[str, float]:
    d = len(scores)
    true_mask = np.zeros(d, dtype=int)
    true_mask[informative] = 1

    threshold_mask = (scores >= 0.5).astype(int)

    topd_mask = np.zeros(d, dtype=int)
    top_idx = np.argsort(scores)[-len(informative):]
    topd_mask[top_idx] = 1

    return {
        "active_dims": int(threshold_mask.sum()),
        "feature_f1_thresh": float(
            f1_score(true_mask, threshold_mask, zero_division=0)
        ),
        "feature_f1_topd": float(
            f1_score(true_mask, topd_mask, zero_division=0)
        ),
    }


@contextlib.contextmanager
def force_all_gates_on(
    module,
) -> Iterator[None]:
    """Temporarily replace safeguarded DiagnosableGMM.forward.

    The replacement fixes phi_j=1 for every feature and removes the gate KL.
    This yields an adaptive-mixture ablation with no feature gating while
    retaining the same component model and split safeguards.
    """
    cls = module.DiagnosableGMM
    original_forward = cls.forward

    def no_gating_forward(self, X, sample_phi=True):
        del sample_phi
        x_exp = X.unsqueeze(1)
        mu_exp = self.q_mu.unsqueeze(0)
        logvar_exp = self.q_logvar.unsqueeze(0)

        log_prob_cluster = -0.5 * (
            np.log(2.0 * np.pi)
            + logvar_exp
            + (x_exp - mu_exp) ** 2 / torch.exp(logvar_exp)
        )
        log_p_x_given_z = log_prob_cluster.sum(dim=2)

        pi = torch.softmax(self.pi_logits, dim=0)
        log_joint = (
            log_p_x_given_z
            + torch.log(pi + 1e-9).unsqueeze(0)
        )
        log_likelihood = torch.logsumexp(log_joint, dim=1).sum()
        loss = -log_likelihood

        q_phi = torch.ones(
            self.D,
            device=X.device,
            dtype=X.dtype,
        )
        return loss, q_phi, log_p_x_given_z

    cls.forward = no_gating_forward
    try:
        yield
    finally:
        cls.forward = original_forward


def count_split_events(model: Any) -> Dict[str, int]:
    events = list(getattr(model, "split_history", []) or [])

    if events:
        accepted = sum(bool(event.get("accepted", True)) for event in events)
        rejected = sum(not bool(event.get("accepted", True)) for event in events)
        return {
            "split_proposals": len(events),
            "accepted_splits": accepted,
            "rejected_splits": rejected,
        }

    k_path = list(getattr(model, "history", {}).get("k", []))
    accepted = 0
    if len(k_path) >= 2:
        accepted = sum(
            int(k_path[i] > k_path[i - 1])
            for i in range(1, len(k_path))
        )

    return {
        "split_proposals": accepted,
        "accepted_splits": accepted,
        "rejected_splits": 0,
    }


def build_model(
    variant: str,
    original_module,
    safeguarded_module,
    fixedk_module,
    args: argparse.Namespace,
):
    adaptive_common = dict(
        split_threshold=None,
        split_interval=args.split_interval,
        max_epochs=args.max_epochs,
        lr=args.lr,
        max_components=args.max_components,
        beta_mult=args.beta_mult,
        rough_k=args.rough_k,
        verbose=args.verbose,
    )

    safeguard_common = dict(
        split_burn_in=args.split_burn_in,
        split_persistence=args.split_persistence,
        split_warmup_epochs=args.split_warmup_epochs,
        split_accept_tol_per_sample=args.split_accept_tol_per_sample,
        min_cluster_size_for_split=args.min_cluster_size_for_split,
    )

    fixed_common = dict(
        split_threshold=None,
        split_interval=args.split_interval,
        max_epochs=args.max_epochs,
        lr=args.lr,
        beta_mult=args.beta_mult,
        temperature_start=1.0,
        temperature_end=0.1,
        verbose=args.verbose,
        allow_split=False,
        init_num_components=args.k,
        init_method="kmeans",
        random_state=args.model_random_state,
    )

    if variant in {"Full-DIVI", "No-StepA", "No-Gating"}:
        return safeguarded_module.DIVIClustering(
            **adaptive_common,
            **safeguard_common,
        )

    if variant in {"Fixed-K", "No-StepA-FixedK"}:
        return fixedk_module.DIVIClustering(**fixed_common)

    if variant == "Unsafe-Split":
        return original_module.DIVIClustering(**adaptive_common)

    raise ValueError(f"Unknown component-ablation variant: {variant}")


def prior_mode_for_variant(variant: str) -> int:
    if variant in {"No-StepA", "No-StepA-FixedK"}:
        return 2
    return 1


def run_one(
    setting: str,
    variant: str,
    X: np.ndarray,
    y: np.ndarray,
    informative: np.ndarray,
    seed: int,
    original_module,
    safeguarded_module,
    fixedk_module,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    set_seed(seed)
    start = time.perf_counter()

    model = build_model(
        variant,
        original_module,
        safeguarded_module,
        fixedk_module,
        args,
    )
    use_prior = prior_mode_for_variant(variant)

    if variant == "No-Gating":
        with force_all_gates_on(safeguarded_module):
            model.fit(X, use_prior=use_prior)
    else:
        model.fit(X, use_prior=use_prior)

    runtime = time.perf_counter() - start
    labels = predict_labels(model, X)
    final_k = int(getattr(model.model, "K"))

    if variant == "No-Gating":
        scores = np.ones(X.shape[1], dtype=float)
        feature_result = {
            "active_dims": int(X.shape[1]),
            "feature_f1_thresh": np.nan,
            "feature_f1_topd": np.nan,
        }
    else:
        scores = feature_scores(model)
        feature_result = feature_metrics(scores, informative)

    split_result = count_split_events(model)

    return {
        "setting": setting,
        "variant": variant,
        "seed": int(seed),
        "n": int(len(y)),
        "d": int(X.shape[1]),
        "d_info": int(len(informative)),
        "k_true": int(args.k),
        "status": "ok",
        "error": "",
        "ARI": float(adjusted_rand_score(y, labels)),
        "NMI": float(normalized_mutual_info_score(y, labels)),
        "runtime_sec": float(runtime),
        "final_K": final_k,
        "correct_K": int(final_k == args.k),
        "over_split": int(final_k > args.k),
        "under_split": int(final_k < args.k),
        **split_result,
        **feature_result,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    successful = raw.loc[raw["status"] == "ok"].copy()

    numeric_metrics = [
        "ARI",
        "NMI",
        "runtime_sec",
        "final_K",
        "correct_K",
        "over_split",
        "under_split",
        "split_proposals",
        "accepted_splits",
        "rejected_splits",
        "active_dims",
        "feature_f1_thresh",
        "feature_f1_topd",
    ]

    rows: List[Dict[str, Any]] = []

    for (setting, variant), group in successful.groupby(
        ["setting", "variant"],
        sort=False,
    ):
        row: Dict[str, Any] = {
            "setting": setting,
            "variant": variant,
            "successful_runs": int(len(group)),
        }

        for metric in numeric_metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            row[f"{metric}_mean"] = (
                float(values.mean()) if len(values) else np.nan
            )
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_count"] = int(len(values))

        rows.append(row)

    summary = pd.DataFrame(rows)

    order = {name: i for i, name in enumerate(VARIANTS)}
    summary["_variant_order"] = summary["variant"].map(order)
    summary = (
        summary.sort_values(["setting", "_variant_order"])
        .drop(columns="_variant_order")
        .reset_index(drop=True)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_divi",
        type=str,
        default="/content/divi_mlwa.py",
    )
    parser.add_argument(
        "--safeguarded_divi",
        type=str,
        default="/content/divi_mlwa_safeguarded.py",
    )
    parser.add_argument(
        "--fixedk_divi",
        type=str,
        default="/content/divi_core_fixedk.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/component_ablation",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 21)),
    )
    parser.add_argument(
        "--settings",
        type=str,
        nargs="+",
        default=["matched", "correlated_noise"],
        choices=["matched", "correlated_noise"],
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=VARIANTS,
        choices=VARIANTS,
    )

    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d_info", type=int, default=10)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--signal_sd", type=float, default=1.0)
    parser.add_argument("--noise_sd", type=float, default=3.0)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--block_size", type=int, default=10)

    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta_mult", type=float, default=1.0)
    parser.add_argument("--rough_k", type=int, default=3)
    parser.add_argument("--split_interval", type=int, default=60)
    parser.add_argument("--max_components", type=int, default=8)

    parser.add_argument("--split_burn_in", type=int, default=120)
    parser.add_argument("--split_persistence", type=int, default=2)
    parser.add_argument("--split_warmup_epochs", type=int, default=20)
    parser.add_argument(
        "--split_accept_tol_per_sample",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--min_cluster_size_for_split",
        type=int,
        default=25,
    )

    parser.add_argument("--model_random_state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n < args.k:
        raise ValueError("n must be at least k.")
    if not 0 < args.d_info <= args.d:
        raise ValueError("Require 0 < d_info <= d.")
    if args.max_components < args.k:
        raise ValueError("max_components must be at least k.")
    if args.rho < 0 or args.rho >= 1:
        raise ValueError("rho must satisfy 0 <= rho < 1.")
    if args.block_size < 1:
        raise ValueError("block_size must be positive.")

    if args.quick:
        args.seeds = [1, 2]
        args.settings = ["matched"]
        args.variants = [
            "Full-DIVI",
            "No-StepA",
            "No-Gating",
            "Fixed-K",
            "Unsafe-Split",
            "No-StepA-FixedK",
        ]
        args.max_epochs = min(args.max_epochs, 80)
        args.split_burn_in = min(args.split_burn_in, 20)
        args.split_interval = min(args.split_interval, 20)
        args.split_warmup_epochs = min(args.split_warmup_epochs, 5)


def make_dataset(
    setting: str,
    seed: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = dict(
        n=args.n,
        d=args.d,
        k=args.k,
        d_info=args.d_info,
        delta=args.delta,
        signal_sd=args.signal_sd,
        noise_sd=args.noise_sd,
        seed=seed,
    )

    if setting == "matched":
        return generate_matched(**common)

    if setting == "correlated_noise":
        return generate_correlated_noise(
            **common,
            rho=args.rho,
            block_size=args.block_size,
        )

    raise ValueError(f"Unknown setting: {setting}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "component_ablation_raw.csv"
    summary_path = output_dir / "component_ablation_summary.csv"
    config_path = output_dir / "component_ablation_config.json"
    failures_path = output_dir / "component_ablation_failures.csv"

    original_module = load_module(
        args.original_divi,
        "divi_mlwa_original_component_ablation",
    )
    safeguarded_module = load_module(
        args.safeguarded_divi,
        "divi_mlwa_safeguarded_component_ablation",
    )
    fixedk_module = load_module(
        args.fixedk_divi,
        "divi_fixedk_component_ablation",
    )

    existing = pd.DataFrame()
    completed = set()

    if args.resume and raw_path.exists():
        existing = pd.read_csv(raw_path)
        ok = existing.loc[existing["status"] == "ok"]
        completed = set(
            zip(
                ok["setting"].astype(str),
                ok["variant"].astype(str),
                ok["seed"].astype(int),
            )
        )
        print(f"[resume] Loaded {len(existing)} existing rows.")

    config = vars(args).copy()
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    new_rows: List[Dict[str, Any]] = []

    total = (
        len(args.settings)
        * len(args.variants)
        * len(args.seeds)
    )
    counter = 0

    for setting in args.settings:
        for seed in args.seeds:
            X, y, informative = make_dataset(setting, seed, args)

            for variant in args.variants:
                counter += 1
                key = (setting, variant, int(seed))

                if key in completed:
                    print(
                        f"[{counter}/{total}] skip "
                        f"setting={setting}, variant={variant}, seed={seed}"
                    )
                    continue

                print(
                    f"[{counter}/{total}] "
                    f"setting={setting}, variant={variant}, seed={seed}"
                )

                try:
                    result = run_one(
                        setting=setting,
                        variant=variant,
                        X=X,
                        y=y,
                        informative=informative,
                        seed=seed,
                        original_module=original_module,
                        safeguarded_module=safeguarded_module,
                        fixedk_module=fixedk_module,
                        args=args,
                    )
                except Exception as exc:
                    result = {
                        "setting": setting,
                        "variant": variant,
                        "seed": int(seed),
                        "n": int(args.n),
                        "d": int(args.d),
                        "d_info": int(args.d_info),
                        "k_true": int(args.k),
                        "status": "error",
                        "error": (
                            f"{type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc()}"
                        ),
                        "ARI": np.nan,
                        "NMI": np.nan,
                        "runtime_sec": np.nan,
                        "final_K": np.nan,
                        "correct_K": np.nan,
                        "over_split": np.nan,
                        "under_split": np.nan,
                        "split_proposals": np.nan,
                        "accepted_splits": np.nan,
                        "rejected_splits": np.nan,
                        "active_dims": np.nan,
                        "feature_f1_thresh": np.nan,
                        "feature_f1_topd": np.nan,
                    }
                    print(result["error"])

                new_rows.append(result)

                current = pd.concat(
                    [existing, pd.DataFrame(new_rows)],
                    ignore_index=True,
                )
                current.to_csv(raw_path, index=False)

    raw = pd.concat(
        [existing, pd.DataFrame(new_rows)],
        ignore_index=True,
    )

    if raw.empty:
        raise RuntimeError("No component-ablation results were produced.")

    raw = (
        raw.drop_duplicates(
            subset=["setting", "variant", "seed"],
            keep="last",
        )
        .sort_values(["setting", "seed", "variant"])
        .reset_index(drop=True)
    )
    raw.to_csv(raw_path, index=False)

    failures = raw.loc[raw["status"] != "ok"].copy()
    failures.to_csv(failures_path, index=False)

    summary = summarize(raw)
    summary.to_csv(summary_path, index=False)

    print("\n=== Component-ablation summary ===")
    display_cols = [
        "setting",
        "variant",
        "successful_runs",
        "ARI_mean",
        "ARI_std",
        "NMI_mean",
        "NMI_std",
        "final_K_mean",
        "final_K_std",
        "correct_K_mean",
        "over_split_mean",
        "under_split_mean",
        "active_dims_mean",
        "feature_f1_thresh_mean",
        "feature_f1_topd_mean",
        "runtime_sec_mean",
    ]
    print(summary[display_cols].to_string(index=False))

    print("\nOutput files:")
    print(raw_path)
    print(summary_path)
    print(config_path)
    print(failures_path)

    if len(failures):
        print(f"\nWARNING: {len(failures)} runs failed.")
    else:
        print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
