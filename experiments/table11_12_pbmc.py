
"""
PBMC-SCT experiment for DIVI (MLWA manuscript)

Key protocol decisions:
1. DIVI hyperparameters are fixed before evaluation:
   beta_mult = 1.0, tau_mult = 1.0, use_prior = 1.
2. Reference cell-type labels are used only for external ARI/NMI.
3. DIVI uses max_components = K_ref and rough_k = K_ref.
4. Final predictions are obtained through DIVIClustering.predict(),
   which must implement deterministic component-fit assignment.
5. Five seeds are used for DIVI, K-means, and diagonal GMM.

Required files in the working directory:
- DIVI_HAR.py
- X_pbmc_sct.csv
- cell_ids_pbmc_sct.csv
- y_pbmc_fixed.csv
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler

from DIVI_HAR import DIVIClustering


# ============================================================
# 0. Configuration
# ============================================================
DATA_DIR = Path(".")
OUTPUT_DIR = Path("pbmc_sct_results")

X_PATH = DATA_DIR / "X_pbmc_sct.csv"
CELL_IDS_PATH = DATA_DIR / "cell_ids_pbmc_sct.csv"
LABELS_PATH = DATA_DIR / "y_pbmc_fixed.csv"

SEEDS = [1, 2, 3, 4, 5]

PBMC_BETA_MULT = 1.0
PBMC_TAU_MULT = 1.0
PBMC_USE_PRIOR = 1
PBMC_SPLIT_INTERVAL = 40
PBMC_MAX_EPOCHS = 320
PBMC_LR = 0.01

TOP_GENE_SEED = 1
TOP_N_GENES = 50


# ============================================================
# 1. Utilities
# ============================================================
def set_seed(seed: int) -> None:
    """Set all relevant random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility settings. These may modestly reduce speed.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path.resolve()}"
        )


def mean_sd_text(mean_value: float, sd_value: float, digits: int = 3) -> str:
    return f"{mean_value:.{digits}f} ({sd_value:.{digits}f})"


# ============================================================
# 2. Load and align PBMC-SCT data
# ============================================================
def load_pbmc_sct_data() -> dict:
    for path in [X_PATH, CELL_IDS_PATH, LABELS_PATH]:
        require_file(path)

    x_df = pd.read_csv(X_PATH)

    if x_df.empty:
        raise ValueError("X_pbmc_sct.csv is empty.")

    if str(x_df.columns[0]).lower().startswith("unnamed"):
        x_df = x_df.drop(columns=x_df.columns[0])

    cell_ids_df = pd.read_csv(CELL_IDS_PATH)
    if "barcode" not in cell_ids_df.columns:
        raise ValueError(
            "cell_ids_pbmc_sct.csv must contain a 'barcode' column."
        )

    cell_ids = cell_ids_df["barcode"].astype(str).to_numpy()

    if len(cell_ids) != len(x_df):
        raise ValueError(
            "The number of barcodes does not match the number of rows "
            "in X_pbmc_sct.csv."
        )

    if pd.Index(cell_ids).duplicated().any():
        raise ValueError("Duplicate barcodes found in cell_ids_pbmc_sct.csv.")

    x_df.index = cell_ids

    y_df = pd.read_csv(LABELS_PATH)
    required_label_columns = {"barcode", "label"}
    missing = required_label_columns.difference(y_df.columns)
    if missing:
        raise ValueError(
            f"y_pbmc_fixed.csv is missing columns: {sorted(missing)}"
        )

    y_df = y_df.copy()
    y_df["barcode"] = y_df["barcode"].astype(str)

    if y_df["barcode"].duplicated().any():
        raise ValueError("Duplicate barcodes found in y_pbmc_fixed.csv.")

    y_df = y_df.set_index("barcode")

    # Preserve the row order of X rather than sorting the intersection.
    common_cells = x_df.index[x_df.index.isin(y_df.index)]

    if len(common_cells) == 0:
        raise ValueError("No common barcodes were found between X and labels.")

    x_df = x_df.loc[common_cells]
    labels_ref = y_df.loc[common_cells, "label"].astype(str).to_numpy()

    if x_df.isna().any().any():
        raise ValueError("Missing values found in the PBMC-SCT matrix.")

    gene_names = x_df.columns.astype(str).to_numpy()
    cell_ids_aligned = x_df.index.astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y_ref = label_encoder.fit_transform(labels_ref)
    k_ref = int(np.unique(y_ref).size)

    x_raw = x_df.to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw).astype(np.float32)

    d_pbmc = int(x_scaled.shape[1])
    base_tau_pbmc = float(
        0.5 * d_pbmc * (1.0 + np.log(2.0 * np.pi))
    )

    print("=" * 72)
    print("PBMC-SCT data summary")
    print("=" * 72)
    print("Aligned X shape:", x_scaled.shape)
    print("Number of genes:", len(gene_names))
    print("Number of aligned labels:", len(y_ref))
    print("Reference K:", k_ref)
    print("Base tau:", base_tau_pbmc)
    print("\nReference-label counts:")
    print(pd.Series(labels_ref).value_counts())
    print("=" * 72)

    return {
        "X": x_scaled,
        "X_raw": x_raw,
        "y_ref": y_ref,
        "labels_ref": labels_ref,
        "gene_names": gene_names,
        "cell_ids": cell_ids_aligned,
        "K_ref": k_ref,
        "base_tau": base_tau_pbmc,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }


# ============================================================
# 3. DIVI experiment with fixed default hyperparameters
# ============================================================
def build_divi(k_ref: int, base_tau: float, verbose: bool = False):
    return DIVIClustering(
        split_threshold=PBMC_TAU_MULT * base_tau,
        split_interval=PBMC_SPLIT_INTERVAL,
        max_epochs=PBMC_MAX_EPOCHS,
        lr=PBMC_LR,
        max_components=k_ref,
        beta_mult=PBMC_BETA_MULT,
        rough_k=k_ref,
        verbose=verbose,
    )


def evaluate_divi_seed(
    seed: int,
    X: np.ndarray,
    y_ref: np.ndarray,
    k_ref: int,
    base_tau: float,
    return_model: bool = False,
) -> tuple[dict, object | None]:
    set_seed(seed)

    divi = build_divi(
        k_ref=k_ref,
        base_tau=base_tau,
        verbose=False,
    )

    start = time.perf_counter()
    divi.fit(X, use_prior=PBMC_USE_PRIOR)
    elapsed = time.perf_counter() - start

    # DIVI_HAR.predict() must use deterministic component-fit scores.
    z_pred_1 = np.asarray(divi.predict(X))
    z_pred_2 = np.asarray(divi.predict(X))

    if not np.array_equal(z_pred_1, z_pred_2):
        raise RuntimeError(
            "Repeated DIVI predictions differ. "
            "Check that DIVI_HAR.predict() uses deterministic gates."
        )

    q_phi = np.asarray(divi.get_feature_relevance())

    result = {
        "dataset": "PBMC-SCT",
        "method": "DIVI-Info",
        "seed": seed,
        "beta_mult": PBMC_BETA_MULT,
        "tau_mult": PBMC_TAU_MULT,
        "use_prior": PBMC_USE_PRIOR,
        "max_components": k_ref,
        "rough_k": k_ref,
        "Final K": int(divi.model.K),
        "Label K": int(np.unique(z_pred_1).size),
        "ARI": float(adjusted_rand_score(y_ref, z_pred_1)),
        "NMI": float(normalized_mutual_info_score(y_ref, z_pred_1)),
        "Active genes": int((q_phi > 0.5).sum()),
        "Mean q_phi": float(q_phi.mean()),
        "Max q_phi": float(q_phi.max()),
        "Min q_phi": float(q_phi.min()),
        "Time (s)": float(elapsed),
    }

    return result, divi if return_model else None


def run_divi_multiseed(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    print("\nRunning DIVI with fixed default hyperparameters:")
    print(f"  beta_mult = {PBMC_BETA_MULT}")
    print(f"  tau_mult = {PBMC_TAU_MULT}")
    print(f"  use_prior = {PBMC_USE_PRIOR}")
    print(f"  K_max = K_ref = {data['K_ref']}")
    print(f"  rough_k = K_ref = {data['K_ref']}")
    print(f"  seeds = {SEEDS}\n")

    for seed in SEEDS:
        result, _ = evaluate_divi_seed(
            seed=seed,
            X=data["X"],
            y_ref=data["y_ref"],
            k_ref=data["K_ref"],
            base_tau=data["base_tau"],
            return_model=False,
        )
        rows.append(result)

        print(
            f"[DIVI seed {seed}] "
            f"ARI={result['ARI']:.6f}, "
            f"NMI={result['NMI']:.6f}, "
            f"K={result['Final K']}, "
            f"active={result['Active genes']}, "
            f"time={result['Time (s)']:.2f}s"
        )

    df = pd.DataFrame(rows)

    summary_columns = [
        "ARI",
        "NMI",
        "Final K",
        "Label K",
        "Active genes",
        "Mean q_phi",
        "Time (s)",
    ]
    summary = df[summary_columns].agg(["mean", "std"])

    return df, summary


# ============================================================
# 4. Classical baselines
# ============================================================
def run_baselines_multiseed(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    X = data["X"]
    y_ref = data["y_ref"]
    k_ref = data["K_ref"]

    print("\nRunning K-means and diagonal GMM baselines...\n")

    for seed in SEEDS:
        set_seed(seed)

        start = time.perf_counter()
        kmeans = KMeans(
            n_clusters=k_ref,
            random_state=seed,
            n_init=20,
        )
        z_kmeans = kmeans.fit_predict(X)
        kmeans_time = time.perf_counter() - start

        rows.append({
            "dataset": "PBMC-SCT",
            "method": "K-means",
            "seed": seed,
            "ARI": float(adjusted_rand_score(y_ref, z_kmeans)),
            "NMI": float(normalized_mutual_info_score(y_ref, z_kmeans)),
            "Final K": k_ref,
            "Label K": int(np.unique(z_kmeans).size),
            "Active genes": np.nan,
            "Time (s)": float(kmeans_time),
        })

        start = time.perf_counter()
        gmm = GaussianMixture(
            n_components=k_ref,
            covariance_type="diag",
            random_state=seed,
            max_iter=500,
            n_init=5,
            reg_covar=1e-3,
        )
        z_gmm = gmm.fit_predict(X)
        gmm_time = time.perf_counter() - start

        rows.append({
            "dataset": "PBMC-SCT",
            "method": "diag-GMM",
            "seed": seed,
            "ARI": float(adjusted_rand_score(y_ref, z_gmm)),
            "NMI": float(normalized_mutual_info_score(y_ref, z_gmm)),
            "Final K": k_ref,
            "Label K": int(np.unique(z_gmm).size),
            "Active genes": np.nan,
            "Time (s)": float(gmm_time),
        })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("method")[
            ["ARI", "NMI", "Final K", "Label K", "Time (s)"]
        ]
        .agg(["mean", "std"])
    )

    return df, summary


# ============================================================
# 5. Top-gene relevance analysis at a fixed seed
# ============================================================
def run_top_gene_analysis(data: dict) -> tuple[pd.DataFrame, dict]:
    result, model = evaluate_divi_seed(
        seed=TOP_GENE_SEED,
        X=data["X"],
        y_ref=data["y_ref"],
        k_ref=data["K_ref"],
        base_tau=data["base_tau"],
        return_model=True,
    )

    if model is None:
        raise RuntimeError("DIVI model was not returned.")

    q_phi = np.asarray(model.get_feature_relevance())
    top_n = min(TOP_N_GENES, len(q_phi))
    top_idx = np.argsort(-q_phi)[:top_n]

    top_genes = pd.DataFrame({
        "rank": np.arange(1, top_n + 1),
        "gene": data["gene_names"][top_idx],
        "q_phi": q_phi[top_idx],
    })

    return top_genes, result


# ============================================================
# 6. Manuscript-ready summary table
# ============================================================
def build_manuscript_table(
    divi_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    divi_metrics = divi_df[
        ["ARI", "NMI", "Final K", "Active genes", "Time (s)"]
    ]
    rows.append({
        "Dataset": "PBMC-SCT",
        "Method": "DIVI-Info",
        "ARI": mean_sd_text(
            divi_metrics["ARI"].mean(),
            divi_metrics["ARI"].std(ddof=1),
        ),
        "NMI": mean_sd_text(
            divi_metrics["NMI"].mean(),
            divi_metrics["NMI"].std(ddof=1),
        ),
        "Final K": mean_sd_text(
            divi_metrics["Final K"].mean(),
            divi_metrics["Final K"].std(ddof=1),
            digits=1,
        ),
        "Active genes": mean_sd_text(
            divi_metrics["Active genes"].mean(),
            divi_metrics["Active genes"].std(ddof=1),
            digits=1,
        ),
        "Time (s)": mean_sd_text(
            divi_metrics["Time (s)"].mean(),
            divi_metrics["Time (s)"].std(ddof=1),
            digits=2,
        ),
    })

    for method in ["K-means", "diag-GMM"]:
        sub = baseline_df.loc[baseline_df["method"] == method]
        rows.append({
            "Dataset": "PBMC-SCT",
            "Method": method,
            "ARI": mean_sd_text(
                sub["ARI"].mean(),
                sub["ARI"].std(ddof=1),
            ),
            "NMI": mean_sd_text(
                sub["NMI"].mean(),
                sub["NMI"].std(ddof=1),
            ),
            "Final K": mean_sd_text(
                sub["Final K"].mean(),
                sub["Final K"].std(ddof=1),
                digits=1,
            ),
            "Active genes": "--",
            "Time (s)": mean_sd_text(
                sub["Time (s)"].mean(),
                sub["Time (s)"].std(ddof=1),
                digits=2,
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. Main
# ============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_pbmc_sct_data()

    # Fixed-default DIVI, five seeds
    divi_df, divi_summary = run_divi_multiseed(data)

    # Classical baselines, five seeds
    baseline_df, baseline_summary = run_baselines_multiseed(data)

    # Top genes from fixed seed 1, fixed default hyperparameters
    top_genes_df, top_gene_run = run_top_gene_analysis(data)

    # Manuscript-ready table
    manuscript_table = build_manuscript_table(
        divi_df=divi_df,
        baseline_df=baseline_df,
    )

    # Save all outputs
    divi_df.to_csv(
        OUTPUT_DIR / "pbmc_sct_divi_multiseed_default.csv",
        index=False,
    )
    divi_summary.to_csv(
        OUTPUT_DIR / "pbmc_sct_divi_multiseed_default_summary.csv"
    )
    baseline_df.to_csv(
        OUTPUT_DIR / "pbmc_sct_baselines_multiseed.csv",
        index=False,
    )
    baseline_summary.to_csv(
        OUTPUT_DIR / "pbmc_sct_baselines_multiseed_summary.csv"
    )
    top_genes_df.to_csv(
        OUTPUT_DIR / "pbmc_sct_top_genes_divi_default.csv",
        index=False,
    )
    pd.DataFrame([top_gene_run]).to_csv(
        OUTPUT_DIR / "pbmc_sct_top_gene_run_metadata.csv",
        index=False,
    )
    manuscript_table.to_csv(
        OUTPUT_DIR / "pbmc_sct_manuscript_table.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("DIVI multi-seed summary")
    print("=" * 72)
    print(divi_summary)

    print("\n" + "=" * 72)
    print("Baseline multi-seed summary")
    print("=" * 72)
    print(baseline_summary)

    print("\n" + "=" * 72)
    print("Manuscript-ready table")
    print("=" * 72)
    print(manuscript_table.to_string(index=False))

    print("\n" + "=" * 72)
    print("Top genes")
    print("=" * 72)
    print(top_genes_df.head(20).to_string(index=False))

    print("\nSaved outputs to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
