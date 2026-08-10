
"""
Reproducible UCI Wine experiment for DIVI.

Protocol:
- StandardScaler preprocessing.
- Step A uses K_A=3 and combines Kruskal-Wallis and Gaussian LLR scores.
- Background distribution: mean 0, log-variance 2.197 (variance about 9).
- Stochastic relaxed gates during optimization.
- Deterministic gate values sigmoid(eta) for split diagnostics and final labels.
- Component-fit hard assignment, excluding mixture weights.
- Original schedule-controlled direct growth.
- No explicit reference-K cap; growth is limited only by the finite epoch/split schedule.
- Five seeds are reported; seed 1 is used for the interpretability figures.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. Fixed experiment configuration
# ============================================================
SEEDS = [1, 2, 3, 4, 5]
K_A = 3
MAX_EPOCHS = 300
SPLIT_INTERVAL = 50
LEARNING_RATE = 0.03
BETA_MULT = 1.0
TAU_MULT = 1.0
TEMPERATURE_START = 1.0
TEMPERATURE_END = 0.1
TEMPERATURE_DECAY = 0.98
PRIOR_LOGVAR0 = 2.197
SPLIT_PERTURB_SCALE = 0.2
ACTIVE_THRESHOLD = 0.5
FIGURE_SEED = 1
OUTPUT_DIR = Path("wine_divi_results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_threshold(d: int) -> float:
    return TAU_MULT * 0.5 * d * (1.0 + np.log(2.0 * np.pi))


# ============================================================
# 1. Model
# ============================================================
class DiagnosableGMM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_components: int,
        prior_phi_probs: torch.Tensor,
        init_means: torch.Tensor | None = None,
        init_logvar: torch.Tensor | None = None,
        init_pi: torch.Tensor | None = None,
        temperature: float = TEMPERATURE_START,
    ) -> None:
        super().__init__()
        self.D = int(input_dim)
        self.K = int(num_components)
        self.temperature = float(temperature)

        prior_phi_probs = torch.clamp(
            prior_phi_probs.float(), 1e-4, 1.0 - 1e-4
        )
        self.register_buffer("prior_phi_probs", prior_phi_probs)
        self.register_buffer("prior_mu_0", torch.zeros(self.D))
        self.register_buffer(
            "prior_logvar_0",
            torch.full((self.D,), float(PRIOR_LOGVAR0)),
        )

        prior_logits = torch.log(
            prior_phi_probs / (1.0 - prior_phi_probs)
        )
        self.phi_logits = nn.Parameter(prior_logits.clone())

        if init_means is None:
            init_means = torch.randn(self.K, self.D)
        self.q_mu = nn.Parameter(init_means.float())

        if init_logvar is None:
            init_logvar = torch.ones(self.K, self.D) * -1.0
        self.q_logvar = nn.Parameter(init_logvar.float())

        if init_pi is None:
            init_pi = torch.full((self.K,), 1.0 / self.K)
        init_pi = torch.clamp(init_pi.float(), 1e-6, 1.0)
        init_pi = init_pi / init_pi.sum()
        self.pi_logits = nn.Parameter(torch.log(init_pi))

    def sample_relaxed_gate(self) -> torch.Tensor:
        u = torch.rand_like(self.phi_logits)
        g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
        return torch.sigmoid(
            (self.phi_logits + g) / max(self.temperature, 1e-6)
        )

    def component_log_density(
        self,
        X: torch.Tensor,
        stochastic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi = (
            self.sample_relaxed_gate()
            if stochastic
            else torch.sigmoid(self.phi_logits)
        )

        x_exp = X.unsqueeze(1)
        mu_exp = self.q_mu.unsqueeze(0)
        logvar_exp = self.q_logvar.unsqueeze(0)

        log_prob_cluster = -0.5 * (
            math.log(2.0 * math.pi)
            + logvar_exp
            + (x_exp - mu_exp) ** 2 / torch.exp(logvar_exp)
        )

        log_prob_background = -0.5 * (
            math.log(2.0 * math.pi)
            + self.prior_logvar_0
            + (x_exp - self.prior_mu_0) ** 2
            / torch.exp(self.prior_logvar_0)
        )

        weighted = (
            phi.view(1, 1, self.D) * log_prob_cluster
            + (1.0 - phi.view(1, 1, self.D))
            * log_prob_background
        )
        component_scores = weighted.sum(dim=2)

        if component_scores.shape != (X.shape[0], self.K):
            raise RuntimeError(
                "Unexpected component-score shape: "
                f"{tuple(component_scores.shape)}; "
                f"expected {(X.shape[0], self.K)}."
            )

        return component_scores, phi

    def objective(
        self,
        X: torch.Tensor,
        beta: float,
        stochastic: bool,
    ) -> dict[str, torch.Tensor]:
        component_scores, phi = self.component_log_density(
            X, stochastic=stochastic
        )

        pi = torch.softmax(self.pi_logits, dim=0)
        log_joint = component_scores + torch.log(pi + 1e-9).unsqueeze(0)
        log_likelihood = torch.logsumexp(log_joint, dim=1).sum()

        q = torch.clamp(
            torch.sigmoid(self.phi_logits), 1e-6, 1.0 - 1e-6
        )
        rho = torch.clamp(
            self.prior_phi_probs, 1e-6, 1.0 - 1e-6
        )
        kl = (
            q * (torch.log(q) - torch.log(rho))
            + (1.0 - q)
            * (torch.log(1.0 - q) - torch.log(1.0 - rho))
        ).sum()

        loss = -log_likelihood + beta * kl
        return {
            "loss": loss,
            "component_scores": component_scores,
            "q_phi": q,
            "log_likelihood": log_likelihood,
            "kl_phi": kl,
        }

    @torch.no_grad()
    def deterministic_labels(self, X: torch.Tensor) -> np.ndarray:
        scores, _ = self.component_log_density(X, stochastic=False)
        return torch.argmax(scores, dim=1).cpu().numpy()

    @torch.no_grad()
    def diagnostics(self, X: torch.Tensor) -> np.ndarray:
        scores, _ = self.component_log_density(X, stochastic=False)
        labels = torch.argmax(scores, dim=1)
        diagnostics = []

        for k in range(self.K):
            mask = labels == k
            if int(mask.sum()) == 0:
                diagnostics.append(0.0)
            else:
                diagnostics.append(
                    float(-scores[mask, k].mean().item())
                )

        return np.asarray(diagnostics)


# ============================================================
# 2. DIVI wrapper
# ============================================================
class DIVIClustering:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.model: DiagnosableGMM | None = None
        self.history: list[dict] = []
        self.split_history: list[dict] = []

    def step_a_reference(self, X: np.ndarray) -> torch.Tensor:
        n, d = X.shape
        rough = KMeans(
            n_clusters=K_A,
            random_state=self.seed,
            n_init=20,
        ).fit_predict(X)

        combined = np.zeros(d, dtype=float)

        for j in range(d):
            feature = X[:, j]
            groups = [
                feature[rough == k]
                for k in range(K_A)
                if np.sum(rough == k) > 0
            ]

            kw_score = 0.0
            if len(groups) > 1:
                try:
                    kw_stat, _ = stats.kruskal(*groups)
                    kw_score = np.log1p(max(float(kw_stat), 0.0))
                except Exception:
                    kw_score = 0.0

            pooled_var = np.var(feature) + 1e-6
            pooled_ll = (
                -0.5
                * np.sum((feature - np.mean(feature)) ** 2)
                / pooled_var
                - 0.5 * n * np.log(pooled_var)
            )

            clustered_ll = 0.0
            for group in groups:
                if len(group) > 1:
                    group_var = np.var(group) + 1e-6
                    clustered_ll += (
                        -0.5
                        * np.sum((group - np.mean(group)) ** 2)
                        / group_var
                        - 0.5 * len(group) * np.log(group_var)
                    )

            llr_score = np.log1p(
                max(float(clustered_ll - pooled_ll), 0.0)
            )
            combined[j] = kw_score + llr_score

        if np.allclose(combined.max(), combined.min()):
            rho = np.full(d, 0.5, dtype=np.float32)
        else:
            normalized = (
                (combined - combined.min())
                / (combined.max() - combined.min() + 1e-9)
            )
            rho = 1.0 / (
                1.0 + np.exp(-6.0 * (normalized - 0.5))
            )
            rho = np.clip(rho, 0.01, 0.99).astype(np.float32)

        return torch.tensor(rho)

    def expand_model(self, split_idx: int) -> None:
        assert self.model is not None
        old = self.model
        d, k = old.D, old.K

        new_model = DiagnosableGMM(
            input_dim=d,
            num_components=k + 1,
            prior_phi_probs=old.prior_phi_probs.detach().clone(),
            temperature=old.temperature,
        )

        with torch.no_grad():
            new_model.phi_logits.copy_(old.phi_logits)

            old_mu = old.q_mu.detach()
            old_logvar = old.q_logvar.detach()

            parent_mu = old_mu[split_idx]
            child_a = (
                parent_mu
                + torch.randn(d) * SPLIT_PERTURB_SCALE
            )
            child_b = (
                parent_mu
                - torch.randn(d) * SPLIT_PERTURB_SCALE
            )

            keep_idx = [idx for idx in range(k) if idx != split_idx]
            kept_mu = old_mu[keep_idx] if keep_idx else old_mu[:0]
            kept_logvar = (
                old_logvar[keep_idx] if keep_idx else old_logvar[:0]
            )

            new_mu = torch.cat(
                [kept_mu, child_a.unsqueeze(0), child_b.unsqueeze(0)],
                dim=0,
            )
            parent_logvar = old_logvar[split_idx]
            new_logvar = torch.cat(
                [
                    kept_logvar,
                    parent_logvar.unsqueeze(0),
                    parent_logvar.unsqueeze(0),
                ],
                dim=0,
            )

            new_model.q_mu.copy_(new_mu)
            new_model.q_logvar.copy_(new_logvar)

            # Original direct-growth protocol:
            # expanded mixture logits are reinitialized uniformly.
            new_model.pi_logits.zero_()

        self.model = new_model

    def fit(self, X: np.ndarray) -> "DIVIClustering":
        set_seed(self.seed)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        n, d = X.shape

        rho = self.step_a_reference(X)
        init_mean = X_tensor.mean(dim=0, keepdim=True)

        self.model = DiagnosableGMM(
            input_dim=d,
            num_components=1,
            prior_phi_probs=rho,
            init_means=init_mean,
            temperature=TEMPERATURE_START,
        )

        optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE
        )
        beta = BETA_MULT * n
        threshold = auto_threshold(d)

        for epoch in range(1, MAX_EPOCHS + 1):
            self.model.train()
            optimizer.zero_grad()

            output = self.model.objective(
                X_tensor, beta=beta, stochastic=True
            )
            output["loss"].backward()
            optimizer.step()

            self.model.temperature = max(
                TEMPERATURE_END,
                self.model.temperature * TEMPERATURE_DECAY,
            )

            if epoch % SPLIT_INTERVAL == 0:
                diagnostics = self.model.diagnostics(X_tensor)
                worst_idx = int(np.argmax(diagnostics))
                worst_score = float(diagnostics[worst_idx])

                self.history.append({
                    "epoch": epoch,
                    "K": self.model.K,
                    "worst_score": worst_score,
                    "threshold": threshold,
                })

                if worst_score > threshold:
                    old_k = self.model.K
                    self.expand_model(worst_idx)
                    optimizer = optim.Adam(
                        self.model.parameters(),
                        lr=LEARNING_RATE,
                    )
                    self.split_history.append({
                        "epoch": epoch,
                        "split_component": worst_idx,
                        "old_K": old_k,
                        "new_K": self.model.K,
                        "worst_score": worst_score,
                        "threshold": threshold,
                    })

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        labels_1 = self.model.deterministic_labels(X_tensor)
        labels_2 = self.model.deterministic_labels(X_tensor)

        if not np.array_equal(labels_1, labels_2):
            raise RuntimeError(
                "Deterministic prediction repeatability check failed."
            )
        return labels_1

    def feature_relevance(self) -> np.ndarray:
        assert self.model is not None
        return (
            torch.sigmoid(self.model.phi_logits)
            .detach()
            .cpu()
            .numpy()
        )


# ============================================================
# 3. Experiment
# ============================================================
def fit_one_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict, DIVIClustering]:
    set_seed(seed)
    model = DIVIClustering(seed=seed)

    start = time.perf_counter()
    model.fit(X)
    elapsed = time.perf_counter() - start

    labels = model.predict(X)
    q_phi = model.feature_relevance()

    result = {
        "seed": seed,
        "ari": adjusted_rand_score(y, labels),
        "nmi": normalized_mutual_info_score(y, labels),
        "time_sec": elapsed,
        "final_k": model.model.K,
        "label_k": len(np.unique(labels)),
        "active_dims": int(np.sum(q_phi > ACTIVE_THRESHOLD)),
        "split_count": len(model.split_history),
    }
    return result, model


def plot_feature_relevance(
    q_phi: np.ndarray,
    feature_names: np.ndarray,
    path: Path,
) -> None:
    order = np.argsort(-q_phi)
    names = feature_names[order]
    values = q_phi[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(values)), values)
    ax.axhline(ACTIVE_THRESHOLD, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel(r"Gate relevance $q_{\phi_j}$")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top2_reference(
    X,
    y_ref,
    q_phi,
    feature_names,
    path,
):
    top2 = np.argsort(-q_phi)[:2]

    fig, ax = plt.subplots(figsize=(7, 6))

    for class_id in np.unique(y_ref):
        mask = y_ref == class_id
        ax.scatter(
            X[mask, top2[0]],
            X[mask, top2[1]],
            s=45,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            label=f"{class_id + 1}",
        )

    ax.set_xlabel("Flavanoids (Top 1)")
    ax.set_ylabel("Proline (Top 2)")

    ax.legend(
        title="Reference category",
        frameon=True,
    )

    ax.grid(
        linestyle="--",
        alpha=0.45,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wine = load_wine()
    X_raw = wine.data
    y = wine.target
    feature_names = np.asarray(wine.feature_names)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    rows = []
    figure_model = None

    for seed in SEEDS:
        result, model = fit_one_seed(seed, X, y)
        rows.append(result)

        print(
            f"[seed {seed}] "
            f"ARI={result['ari']:.6f}, "
            f"NMI={result['nmi']:.6f}, "
            f"K={result['final_k']}, "
            f"active={result['active_dims']}, "
            f"time={result['time_sec']:.2f}s"
        )

        if seed == FIGURE_SEED:
            figure_model = model

    results = pd.DataFrame(rows)
    summary = results[
        [
            "ari",
            "nmi",
            "time_sec",
            "final_k",
            "label_k",
            "active_dims",
            "split_count",
        ]
    ].agg(["mean", "std"])

    results.to_csv(
        OUTPUT_DIR / "wine_divi_deterministic_multiseed.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "wine_divi_deterministic_summary.csv"
    )

    # Oracle-K baselines, reported separately.
    baseline_rows = []
    for seed in SEEDS:
        km = KMeans(
            n_clusters=3,
            random_state=seed,
            n_init=20,
        )
        km_labels = km.fit_predict(X)

        gmm = GaussianMixture(
            n_components=3,
            covariance_type="diag",
            random_state=seed,
            n_init=10,
            max_iter=500,
            reg_covar=1e-4,
        )
        gmm_labels = gmm.fit_predict(X)

        baseline_rows.extend([
            {
                "seed": seed,
                "method": "K-means",
                "ari": adjusted_rand_score(y, km_labels),
                "nmi": normalized_mutual_info_score(y, km_labels),
            },
            {
                "seed": seed,
                "method": "diag-GMM",
                "ari": adjusted_rand_score(y, gmm_labels),
                "nmi": normalized_mutual_info_score(y, gmm_labels),
            },
        ])

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(
        OUTPUT_DIR / "wine_baselines_multiseed.csv",
        index=False,
    )
    baseline_df.groupby("method")[["ari", "nmi"]].agg(
        ["mean", "std"]
    ).to_csv(
        OUTPUT_DIR / "wine_baselines_summary.csv"
    )

    if figure_model is None:
        raise RuntimeError("Figure model was not retained.")

    predicted = figure_model.predict(X)
    q_phi = figure_model.feature_relevance()

    feature_df = pd.DataFrame({
        "feature": feature_names,
        "q_phi": q_phi,
        "active": q_phi > ACTIVE_THRESHOLD,
    }).sort_values("q_phi", ascending=False)
    feature_df.to_csv(
        OUTPUT_DIR / "wine_feature_relevance_seed1.csv",
        index=False,
    )

    plot_feature_relevance(
        q_phi,
        feature_names,
        OUTPUT_DIR / "wine_feature_relevance.png",
    )

    # Save both versions so the manuscript caption can be explicit.
    plot_top2_reference(
        X,
        y,
        q_phi,
        feature_names,
        OUTPUT_DIR / "wine_scatter_reference_labels_final.png",
    )

    print("\nDIVI summary")
    print(summary)
    print("\nTop features for fixed figure seed 1")
    print(feature_df.head(10).to_string(index=False))
    print("\nOutputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
