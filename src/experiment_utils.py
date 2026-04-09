from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


COMMON_COLUMNS = [
    "experiment_id",
    "script_name",
    "timestamp",
    "git_commit",
    "hostname",
    "device_type",
    "device_name",
    "dataset",
    "dataset_variant",
    "split_name",
    "N",
    "D",
    "K_true",
    "informative_ratio",
    "noise_ratio",
    "noise_sigma",
    "data_seed",
    "run_seed",
    "method",
    "prior_mode",
    "baseline_name",
    "beta",
    "beta_mult",
    "Tsplit",
    "tau_mode",
    "tau",
    "tau_mult",
    "lr",
    "temp_start",
    "temp_end",
    "max_epochs",
    "epochs_completed",
    "final_K",
    "split_count",
    "first_split_epoch",
    "last_split_epoch",
    "split_epochs_json",
    "ari",
    "nmi",
    "f1_feature",
    "acc",
    "selected_dims_count",
    "selected_dims_ratio",
    "mean_phi",
    "median_phi",
    "objective_final",
    "objective_best",
    "objective_gap",
    "nll_final",
    "kl_final",
    "wallclock_total_sec",
    "wallclock_stepA_sec",
    "wallclock_train_sec",
    "wallclock_split_diag_sec",
    "wallclock_post_sec",
    "time_per_epoch_sec",
    "peak_memory_mb",
    "status",
    "timeout_flag",
    "error_message",
    "artifact_dir",
]


def utc_timestamp() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_git_commit(cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_info() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", platform.processor() or "cpu"


def get_peak_memory_mb() -> float:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 ** 2))
    except Exception:
        return float("nan")


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)

    cost = np.zeros((len(classes_true), len(classes_pred)), dtype=int)
    for i, ct in enumerate(classes_true):
        for j, cp in enumerate(classes_pred):
            cost[i, j] = np.sum((y_true == ct) & (y_pred == cp))

    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    matched = cost[row_ind, col_ind].sum()
    return float(matched / len(y_true))


def calculate_feature_f1_from_phi(
    phi_probs: np.ndarray,
    n_signal: int = 10,
    threshold: float = 0.5,
) -> float:
    D = len(phi_probs)
    true_mask = np.zeros(D, dtype=int)
    true_mask[:n_signal] = 1
    pred_mask = (phi_probs > threshold).astype(int)
    return float(f1_score(true_mask, pred_mask))


def calculate_feature_f1_from_selected_mask(
    selected_mask: np.ndarray,
    n_signal: int = 10,
) -> float:
    D = len(selected_mask)
    true_mask = np.zeros(D, dtype=int)
    true_mask[:n_signal] = 1
    pred_mask = np.asarray(selected_mask, dtype=int)
    return float(f1_score(true_mask, pred_mask))


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "acc": float(clustering_accuracy(y_true, y_pred)),
    }


def make_experiment_id(prefix: str, dataset: str, split_name: str, run_seed: int, method: str) -> str:
    return f"{prefix}__{dataset}__{split_name}__seed{run_seed}__{method}"


def safe_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_run_record(csv_path: str | Path, record: Dict[str, Any], columns: Optional[List[str]] = None) -> None:
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    if columns is None:
        columns = list(dict.fromkeys(COMMON_COLUMNS + list(record.keys())))

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        row = {col: record.get(col, None) for col in columns}
        writer.writerow(row)


def summarize_results(
    runs_csv: str | Path,
    group_cols: List[str],
    out_csv: str | Path,
) -> pd.DataFrame:
    df = pd.read_csv(runs_csv)
    numeric_cols = [
        c for c in [
            "ari", "nmi", "f1_feature", "acc",
            "final_K", "selected_dims_count", "selected_dims_ratio",
            "wallclock_total_sec", "peak_memory_mb", "split_count"
        ] if c in df.columns
    ]
    agg_spec = {}
    for c in numeric_cols:
        agg_spec[c] = ["mean", "std", "median"]
    summary = df.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    summary.columns = [
        "__".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns.values
    ]
    ensure_dir(Path(out_csv).parent)
    summary.to_csv(out_csv, index=False)
    return summary


def load_yaml_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
