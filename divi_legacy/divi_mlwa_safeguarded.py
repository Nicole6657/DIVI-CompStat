# -*- coding: utf-8 -*-
"""Safeguarded DIVI implementation for the MLWA/Computational Statistics revision.

This version is derived from ``divi_mlwa.py`` and adds safeguards against
premature, irreversible component proliferation:

1. burn-in before the first split check;
2. persistence across consecutive split checks;
3. minimum candidate-cluster size;
4. matched-optimization-budget acceptance testing;
5. an optional maximum number of components;
6. detailed split-event logging.

The split proposal is accepted only when the deterministically evaluated,
per-observation objective improves by more than ``split_accept_tol_per_sample``.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt  # retained for backward compatibility
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.cluster import KMeans


# ==============================================================================
# 1. Diagnosable variational GMM
# ==============================================================================
class DiagnosableGMM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_components: int,
        prior_phi_probs: torch.Tensor,
        init_means: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        beta_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.D = int(input_dim)
        self.K = int(num_components)
        self.temperature = float(temperature)
        self.beta_mult = float(beta_mult)

        prior_phi_probs = prior_phi_probs.detach().clone().float()
        self.register_buffer("prior_phi_probs", prior_phi_probs)
        self.register_buffer("prior_mu_0", torch.zeros(self.D))
        self.register_buffer(
            "prior_logvar_0",
            torch.full((self.D,), 2.197, dtype=torch.float32),
        )

        prior_logits = torch.log(
            prior_phi_probs.clamp(1e-6, 1.0 - 1e-6)
            / (1.0 - prior_phi_probs.clamp(1e-6, 1.0 - 1e-6))
        )
        self.phi_logits = nn.Parameter(prior_logits.clone())

        if init_means is not None:
            init_means = init_means.detach().clone().float()
            if init_means.shape[1] != self.D:
                raise ValueError(
                    f"init_means has D={init_means.shape[1]}, expected {self.D}."
                )
            if init_means.shape[0] < self.K:
                pad = torch.randn(self.K - init_means.shape[0], self.D)
                init_means = torch.cat([init_means, pad], dim=0)
            elif init_means.shape[0] > self.K:
                init_means = init_means[: self.K]
            self.q_mu = nn.Parameter(init_means)
        else:
            self.q_mu = nn.Parameter(torch.randn(self.K, self.D))

        self.q_logvar = nn.Parameter(torch.full((self.K, self.D), -1.0))
        self.pi_logits = nn.Parameter(torch.ones(self.K))

    def gumbel_sigmoid_sample(self, logits: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
        gumbel = -torch.log(-torch.log(uniform))
        return torch.sigmoid((logits + gumbel) / self.temperature)

    def forward(
        self,
        X: torch.Tensor,
        sample_phi: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return objective, posterior gate probabilities, and component log scores.

        ``sample_phi=False`` uses the posterior mean gate and is therefore used for
        diagnostics and split acceptance. This prevents a split decision from being
        driven by a single Gumbel draw.
        """
        N, _ = X.shape
        q_phi = torch.sigmoid(self.phi_logits)
        phi = self.gumbel_sigmoid_sample(self.phi_logits) if sample_phi else q_phi
        phi = phi.unsqueeze(0)

        x_exp = X.unsqueeze(1)
        mu_exp = self.q_mu.unsqueeze(0)
        logvar_exp = self.q_logvar.unsqueeze(0)

        log_prob_cluster = -0.5 * (
            np.log(2.0 * np.pi)
            + logvar_exp
            + (x_exp - mu_exp) ** 2 / torch.exp(logvar_exp)
        )
        log_prob_bg = -0.5 * (
            np.log(2.0 * np.pi)
            + self.prior_logvar_0
            + (x_exp - self.prior_mu_0) ** 2
            / torch.exp(self.prior_logvar_0)
        )

        weighted_log_prob = (
            phi.unsqueeze(1) * log_prob_cluster
            + (1.0 - phi.unsqueeze(1)) * log_prob_bg
        )
        log_p_x_given_z = weighted_log_prob.sum(dim=2)

        pi = torch.softmax(self.pi_logits, dim=0)
        log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
        log_likelihood = torch.logsumexp(log_joint, dim=1).sum()

        q_phi_safe = q_phi.clamp(1e-6, 1.0 - 1e-6)
        p_phi_safe = self.prior_phi_probs.clamp(1e-6, 1.0 - 1e-6)
        kl_raw = (
            q_phi_safe * (torch.log(q_phi_safe) - torch.log(p_phi_safe))
            + (1.0 - q_phi_safe)
            * (
                torch.log(1.0 - q_phi_safe)
                - torch.log(1.0 - p_phi_safe)
            )
        ).sum()
        kl_phi = self.beta_mult * N * kl_raw

        loss = -log_likelihood + kl_phi
        return loss, q_phi, log_p_x_given_z

    def get_hard_assignments(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, _, log_p_x_given_z = self.forward(X, sample_phi=False)
            return torch.argmax(log_p_x_given_z, dim=1)

    def get_cluster_sizes(self, X: torch.Tensor) -> np.ndarray:
        z_hard = self.get_hard_assignments(X)
        sizes = torch.bincount(z_hard, minlength=self.K)
        return sizes.detach().cpu().numpy()

    def get_cluster_diagnostics(self, X: torch.Tensor) -> np.ndarray:
        """Mean negative component log-density within each hard cluster."""
        with torch.no_grad():
            _, _, log_p_x_given_z = self.forward(X, sample_phi=False)
            z_hard = torch.argmax(log_p_x_given_z, dim=1)
            cluster_scores: List[float] = []

            for k in range(self.K):
                mask = z_hard == k
                if int(mask.sum().item()) == 0:
                    cluster_scores.append(0.0)
                else:
                    log_probs = log_p_x_given_z[mask, k]
                    cluster_scores.append(float(-log_probs.mean().item()))

        return np.asarray(cluster_scores, dtype=float)


# ==============================================================================
# 2. Safeguarded DIVI wrapper
# ==============================================================================
class DIVIClustering:
    def __init__(
        self,
        split_threshold: Optional[float] = 22.0,
        split_interval: int = 60,
        max_epochs: int = 300,
        lr: float = 0.05,
        max_components: Optional[int] = None,
        beta_mult: float = 1.0,
        rough_k: int = 3,
        verbose: bool = True,
        # Split safeguards
        split_burn_in: int = 120,
        split_persistence: int = 2,
        split_warmup_epochs: int = 20,
        split_accept_tol_per_sample: float = 1e-3,
        min_cluster_size_for_split: int = 25,
    ) -> None:
        self.split_threshold = split_threshold
        self.split_interval = int(split_interval)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.max_components = max_components
        self.beta_mult = float(beta_mult)
        self.rough_k = int(rough_k)
        self.verbose = bool(verbose)

        self.split_burn_in = int(split_burn_in)
        self.split_persistence = int(split_persistence)
        self.split_warmup_epochs = int(split_warmup_epochs)
        self.split_accept_tol_per_sample = float(
            split_accept_tol_per_sample
        )
        self.min_cluster_size_for_split = int(min_cluster_size_for_split)

        if self.split_interval < 1:
            raise ValueError("split_interval must be at least 1.")
        if self.split_burn_in < 0:
            raise ValueError("split_burn_in must be nonnegative.")
        if self.split_persistence < 1:
            raise ValueError("split_persistence must be at least 1.")
        if self.split_warmup_epochs < 1:
            raise ValueError("split_warmup_epochs must be at least 1.")
        if self.split_accept_tol_per_sample < 0:
            raise ValueError(
                "split_accept_tol_per_sample must be nonnegative."
            )
        if self.min_cluster_size_for_split < 2:
            raise ValueError(
                "min_cluster_size_for_split must be at least 2."
            )
        if self.max_components is not None and self.max_components < 1:
            raise ValueError("max_components must be positive or None.")

        self.model: Optional[DiagnosableGMM] = None
        self.history: Dict[str, list] = {}
        self.split_history: List[dict] = []
        self._split_exceedance_counts: Dict[int, int] = {}
        self._reset_history()

    def _reset_history(self) -> None:
        self.history = {
            "loss": [],
            "k": [],
            "phi": [],
            "split_events": [],
        }
        self.split_history = []
        self._split_exceedance_counts = {}

    def _step_a_calculate_prior(
        self,
        X: np.ndarray,
        mode: int = 1,
        w_kw: float = 1.0,
        w_llr: float = 1.0,
    ) -> torch.Tensor:
        """Construct Step-A prior inclusion probabilities."""
        N, D = X.shape

        if mode == 1:
            if self.verbose:
                print("   -> Mode 1: Computing Combined KW & LLR scores...")

            k_rough = min(self.rough_k, N)
            if k_rough < 1:
                raise ValueError("rough_k must result in at least one cluster.")
            kmeans = KMeans(n_clusters=k_rough, random_state=42).fit(X)
            labels = kmeans.labels_

            final_scores = []
            for j in range(D):
                feat = X[:, j]
                groups = [
                    feat[labels == k]
                    for k in range(k_rough)
                    if np.sum(labels == k) > 0
                ]

                kw_stat = 0.0
                if len(groups) > 1:
                    try:
                        stat, _ = stats.kruskal(*groups)
                        kw_stat = float(np.log1p(stat))
                    except (ValueError, FloatingPointError):
                        kw_stat = 0.0

                var_0 = np.var(feat) + 1e-6
                ll_0 = (
                    -0.5 * np.sum((feat - np.mean(feat)) ** 2) / var_0
                    - N * 0.5 * np.log(var_0)
                )

                ll_1 = 0.0
                for group in groups:
                    if len(group) > 1:
                        v_k = np.var(group) + 1e-6
                        ll_1 += (
                            -0.5
                            * np.sum((group - np.mean(group)) ** 2)
                            / v_k
                            - len(group) * 0.5 * np.log(v_k)
                        )

                llr_stat = float(np.log1p(max(0.0, ll_1 - ll_0)))
                final_scores.append(w_kw * kw_stat + w_llr * llr_stat)

            final_scores = np.asarray(final_scores, dtype=float)
            norm = (
                (final_scores - final_scores.min())
                / (final_scores.max() - final_scores.min() + 1e-9)
            )
            logits = (norm - 0.5) * 6.0
            rho = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
            return rho.clamp(0.01, 0.99)

        if mode == 2:
            return torch.full((D,), 0.5, dtype=torch.float32)
        if mode == 3:
            return torch.rand(D, dtype=torch.float32)

        raise ValueError("use_prior/mode must be one of {1, 2, 3}.")

    def _expand_model(
        self,
        old_model: DiagnosableGMM,
        split_idx: int,
    ) -> DiagnosableGMM:
        """Create a K+1 proposal by replacing one component with two children."""
        D, K = old_model.D, old_model.K
        if not 0 <= split_idx < K:
            raise IndexError(f"split_idx={split_idx} outside [0, {K - 1}].")

        device = old_model.q_mu.device
        dtype = old_model.q_mu.dtype

        new_model = DiagnosableGMM(
            D,
            K + 1,
            old_model.prior_phi_probs,
            temperature=old_model.temperature,
            beta_mult=old_model.beta_mult,
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            new_model.phi_logits.copy_(old_model.phi_logits)

            old_mu = old_model.q_mu.detach()
            target_mu = old_mu[split_idx]
            perturbation = torch.randn(D, device=device, dtype=dtype) * 0.2
            mu_a = target_mu + perturbation
            mu_b = target_mu - perturbation

            keep_idx = [i for i in range(K) if i != split_idx]
            if keep_idx:
                new_mus = torch.cat(
                    [
                        old_mu[keep_idx],
                        mu_a.unsqueeze(0),
                        mu_b.unsqueeze(0),
                    ],
                    dim=0,
                )
            else:
                new_mus = torch.stack([mu_a, mu_b], dim=0)
            new_model.q_mu.copy_(new_mus)

            old_logvar = old_model.q_logvar.detach()
            target_logvar = old_logvar[split_idx]
            if keep_idx:
                new_logvars = torch.cat(
                    [
                        old_logvar[keep_idx],
                        target_logvar.unsqueeze(0),
                        target_logvar.unsqueeze(0),
                    ],
                    dim=0,
                )
            else:
                new_logvars = torch.stack(
                    [target_logvar, target_logvar], dim=0
                )
            new_model.q_logvar.copy_(new_logvars)

            # Preserve mixture probabilities and divide the parent mass equally.
            old_pi = torch.softmax(old_model.pi_logits.detach(), dim=0)
            kept_pi = old_pi[keep_idx] if keep_idx else old_pi.new_empty((0,))
            child_mass = old_pi[split_idx] / 2.0
            new_pi = torch.cat(
                [kept_pi, child_mass.view(1), child_mass.view(1)], dim=0
            )
            new_model.pi_logits.copy_(torch.log(new_pi.clamp_min(1e-9)))

        return new_model

    @staticmethod
    def _evaluate_loss(
        model: DiagnosableGMM,
        X_tensor: torch.Tensor,
    ) -> float:
        model.eval()
        with torch.no_grad():
            loss, _, _ = model(X_tensor, sample_phi=False)
        return float(loss.item())

    def _train_candidate(
        self,
        model: DiagnosableGMM,
        X_tensor: torch.Tensor,
        epochs: int,
    ) -> Tuple[DiagnosableGMM, optim.Optimizer, float]:
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        model.train()

        for _ in range(epochs):
            optimizer.zero_grad()
            loss, _, _ = model(X_tensor, sample_phi=True)
            if not torch.isfinite(loss):
                return model, optimizer, float("inf")
            loss.backward()
            optimizer.step()
            model.temperature = max(0.1, model.temperature * 0.98)

        final_loss = self._evaluate_loss(model, X_tensor)
        return model, optimizer, final_loss

    def _propose_and_evaluate_split(
        self,
        X_tensor: torch.Tensor,
        split_idx: int,
    ) -> Tuple[optim.Optimizer, dict]:
        """Compare K and K+1 candidates under matched optimization budgets."""
        if self.model is None:
            raise RuntimeError("Model has not been initialized.")

        N = int(X_tensor.shape[0])
        null_model = copy.deepcopy(self.model)
        split_model = self._expand_model(copy.deepcopy(self.model), split_idx)

        null_model, null_optimizer, null_loss = self._train_candidate(
            null_model, X_tensor, self.split_warmup_epochs
        )
        split_model, split_optimizer, split_loss = self._train_candidate(
            split_model, X_tensor, self.split_warmup_epochs
        )

        improvement = null_loss - split_loss
        improvement_per_sample = improvement / max(N, 1)
        accepted = bool(
            np.isfinite(improvement_per_sample)
            and improvement_per_sample
            > self.split_accept_tol_per_sample
        )

        event = {
            "split_idx": int(split_idx),
            "old_K": int(null_model.K),
            "proposed_K": int(split_model.K),
            "null_loss": float(null_loss),
            "split_loss": float(split_loss),
            "improvement": float(improvement),
            "improvement_per_sample": float(improvement_per_sample),
            "accept_tol_per_sample": float(
                self.split_accept_tol_per_sample
            ),
            "accepted": accepted,
        }

        if accepted:
            self.model = split_model
            return split_optimizer, event

        # Continue from the equally trained K-component candidate.
        self.model = null_model
        return null_optimizer, event

    def fit(self, X_np: np.ndarray, use_prior: int = 1) -> "DIVIClustering":
        X_np = np.asarray(X_np, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError("X_np must be a two-dimensional array.")
        if not np.isfinite(X_np).all():
            raise ValueError("X_np contains NaN or infinite values.")

        self._reset_history()
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        N, D = X_tensor.shape

        if self.split_threshold is None:
            self.split_threshold = 0.5 * D * (
                1.0 + np.log(2.0 * np.pi) + np.log(1.0)
            )
            if self.verbose:
                print(
                    "Auto-configured Split Threshold: "
                    f"{self.split_threshold:.2f} (D={D})"
                )

        rho = self._step_a_calculate_prior(X_np, mode=use_prior)
        global_mean = torch.mean(X_tensor, dim=0, keepdim=True)
        self.model = DiagnosableGMM(
            D,
            1,
            rho,
            init_means=global_mean,
            beta_mult=self.beta_mult,
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.verbose:
            print("Starting Training (Initial K=1)...")
            print(
                "Split safeguards: "
                f"burn-in={self.split_burn_in}, "
                f"persistence={self.split_persistence}, "
                f"warm-up={self.split_warmup_epochs}, "
                "accept_tol/N="
                f"{self.split_accept_tol_per_sample:g}, "
                f"min_size={self.min_cluster_size_for_split}"
            )

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            loss, q_phi, _ = self.model(X_tensor, sample_phi=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss encountered at epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            self.model.temperature = max(
                0.1, self.model.temperature * 0.98
            )

            self.history["loss"].append(float(loss.item()))
            self.history["k"].append(int(self.model.K))
            self.history["phi"].append(
                q_phi.detach().cpu().numpy().copy()
            )

            should_check_split = (
                epoch >= self.split_burn_in
                and epoch % self.split_interval == 0
            )
            if not should_check_split:
                continue

            scores = self.model.get_cluster_diagnostics(X_tensor)
            cluster_sizes = self.model.get_cluster_sizes(X_tensor)
            worst_idx = int(np.argmax(scores))
            worst_score = float(scores[worst_idx])
            worst_size = int(cluster_sizes[worst_idx])

            # Persistence is tracked for every current component.
            updated_counts: Dict[int, int] = {}
            for k in range(self.model.K):
                if scores[k] > float(self.split_threshold):
                    updated_counts[k] = (
                        self._split_exceedance_counts.get(k, 0) + 1
                    )
                else:
                    updated_counts[k] = 0
            self._split_exceedance_counts = updated_counts

            persistence_count = self._split_exceedance_counts.get(
                worst_idx, 0
            )
            can_split = (
                self.max_components is None
                or self.model.K < self.max_components
            )
            exceeds_threshold = worst_score > float(self.split_threshold)
            persistent = persistence_count >= self.split_persistence
            large_enough = (
                worst_size >= self.min_cluster_size_for_split
            )

            if self.verbose:
                print(
                    f"Epoch {epoch}: K={self.model.K}, "
                    f"candidate={worst_idx}, "
                    f"Max NLL={worst_score:.4f}, "
                    f"size={worst_size}, "
                    f"persistence={persistence_count}/"
                    f"{self.split_persistence}"
                )

            eligible = (
                exceeds_threshold
                and persistent
                and large_enough
                and can_split
            )

            if eligible:
                optimizer, event = self._propose_and_evaluate_split(
                    X_tensor=X_tensor,
                    split_idx=worst_idx,
                )
                event.update(
                    {
                        "epoch": int(epoch),
                        "diagnostic_score": worst_score,
                        "cluster_size": worst_size,
                        "persistence_count": int(persistence_count),
                    }
                )
                self.split_history.append(event)
                self.history["split_events"].append(event.copy())

                if self.verbose:
                    decision = "accepted" if event["accepted"] else "rejected"
                    print(
                        f"   >>> Split {decision}: cluster={worst_idx}, "
                        f"gain/N={event['improvement_per_sample']:.6g}"
                    )

                # Accepted proposals change component identities; rejected proposals
                # should also demonstrate persistence again before re-proposal.
                self._split_exceedance_counts = {}
            elif self.verbose:
                reasons = []
                if not exceeds_threshold:
                    reasons.append("below threshold")
                if not persistent:
                    reasons.append("insufficient persistence")
                if not large_enough:
                    reasons.append("cluster too small")
                if not can_split:
                    reasons.append("max_components reached")
                print("   >>> Split not proposed: " + ", ".join(reasons))

        if self.verbose:
            accepted = sum(e["accepted"] for e in self.split_history)
            rejected = len(self.split_history) - accepted
            print(
                "Training Completed. "
                f"Final K={self.model.K}; "
                f"accepted splits={accepted}; rejected proposals={rejected}."
            )

        return self

    def predict(self, X_np: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit before predict.")
        X_tensor = torch.tensor(np.asarray(X_np), dtype=torch.float32)
        with torch.no_grad():
            _, _, log_p_x_given_z = self.model(
                X_tensor, sample_phi=False
            )
            labels = torch.argmax(log_p_x_given_z, dim=1)
        return labels.detach().cpu().numpy()

    def get_feature_probabilities(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit before requesting feature probabilities.")
        with torch.no_grad():
            phi = torch.sigmoid(self.model.phi_logits)
        return phi.detach().cpu().numpy()
