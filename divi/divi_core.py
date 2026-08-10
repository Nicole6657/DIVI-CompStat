from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.cluster import KMeans


@dataclass
class FitTimings:
    step_a_sec: float = 0.0
    train_sec: float = 0.0
    split_diag_sec: float = 0.0
    post_sec: float = 0.0
    total_sec: float = 0.0


class DiagnosableGMM(nn.Module):
    """
    Variational Gaussian mixture with global feature gates phi_j.

    This is a cleaned-up version of the model logic in the original DIVI.py:
    - informative prior from Step A is still used through prior_phi_probs
    - background distribution remains wide (default log-variance = 2.197)
    - q(phi) is parameterized by logits and sampled via Gumbel-Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        num_components: int,
        prior_phi_probs: torch.Tensor,
        init_means: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        prior_logvar0: float = 2.197,
    ) -> None:
        super().__init__()
        self.D = int(input_dim)
        self.K = int(num_components)
        self.temperature = float(temperature)

        prior_phi_probs = torch.clamp(prior_phi_probs.float(), 1e-4, 1 - 1e-4)
        self.register_buffer("prior_phi_probs", prior_phi_probs)
        self.register_buffer("prior_mu_0", torch.zeros(self.D))
        self.register_buffer("prior_logvar_0", torch.full((self.D,), float(prior_logvar0)))

        prior_logits = torch.log(prior_phi_probs / (1 - prior_phi_probs))
        self.phi_logits = nn.Parameter(prior_logits.clone())

        if init_means is not None:
            if init_means.shape[0] < num_components:
                pad = torch.randn(num_components - init_means.shape[0], input_dim)
                init_means = torch.cat([init_means, pad], dim=0)
            self.q_mu = nn.Parameter(init_means.float())
        else:
            self.q_mu = nn.Parameter(torch.randn(num_components, input_dim))

        self.q_logvar = nn.Parameter(torch.ones(num_components, input_dim) * -1.0)
        self.pi_logits = nn.Parameter(torch.ones(num_components))

    def gumbel_sigmoid_sample(self, logits: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(uniform + 1e-9) + 1e-9)
        return torch.sigmoid((logits + gumbel) / max(self.temperature, 1e-6))

    def loss_components(
        self,
        X: torch.Tensor,
        beta: float,
        sample_phi: bool = True,
    ) -> Dict[str, torch.Tensor]:
        N, _ = X.shape
        phi = (
            self.gumbel_sigmoid_sample(self.phi_logits)
            if sample_phi
            else torch.sigmoid(self.phi_logits)
        ).unsqueeze(0)

        x_exp = X.unsqueeze(1)
        mu_exp = self.q_mu.unsqueeze(0)
        logvar_exp = self.q_logvar.unsqueeze(0)

        log_prob_cluster = -0.5 * (
            math.log(2 * math.pi) + logvar_exp + (x_exp - mu_exp) ** 2 / torch.exp(logvar_exp)
        )
        log_prob_bg = -0.5 * (
            math.log(2 * math.pi)
            + self.prior_logvar_0
            + (x_exp - self.prior_mu_0) ** 2 / torch.exp(self.prior_logvar_0)
        )

        weighted_log_prob = phi.unsqueeze(1) * log_prob_cluster + (1 - phi.unsqueeze(1)) * log_prob_bg
        log_p_x_given_z = weighted_log_prob.sum(dim=2)

        pi = torch.softmax(self.pi_logits, dim=0)
        log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
        log_likelihood = torch.logsumexp(log_joint, dim=1).sum()

        q_phi = torch.clamp(torch.sigmoid(self.phi_logits), 1e-6, 1 - 1e-6)
        p_phi = torch.clamp(self.prior_phi_probs, 1e-6, 1 - 1e-6)

        kl_phi = (
            q_phi * (torch.log(q_phi) - torch.log(p_phi))
            + (1 - q_phi) * (torch.log(1 - q_phi) - torch.log(1 - p_phi))
        ).sum()

        loss = -log_likelihood + beta * kl_phi
        return {
            "loss": loss,
            "q_phi": q_phi,
            "log_p_x_given_z": log_p_x_given_z,
            "log_likelihood": log_likelihood,
            "kl_phi": kl_phi,
        }

    def forward(self, X: torch.Tensor, beta: float, sample_phi: bool = True):
        out = self.loss_components(X=X, beta=beta, sample_phi=sample_phi)
        return out["loss"], out["q_phi"], out["log_p_x_given_z"]

    def get_cluster_diagnostics(self, X: torch.Tensor, beta: float) -> np.ndarray:
        with torch.no_grad():
            out = self.loss_components(X=X, beta=beta, sample_phi=False)
            log_p_x_given_z = out["log_p_x_given_z"]
            z_hard = torch.argmax(log_p_x_given_z, dim=1)
            cluster_scores: List[float] = []

            for k in range(self.K):
                mask = z_hard == k
                if mask.sum() == 0:
                    cluster_scores.append(0.0)
                else:
                    log_probs = log_p_x_given_z[mask, k]
                    score = -log_probs.mean().item()
                    cluster_scores.append(score)
            return np.asarray(cluster_scores, dtype=float)


class DIVIClustering:
    """
    Clean wrapper built from the original DIVI.py, with:
    - explicit beta scaling (default beta = N)
    - logging of split history and timings
    - predict / predict_proba / get_phi methods for experiment scripts
    """

    PRIOR_MODE_NAME = {
        1: "informative_kw_llr",
        2: "noninformative_uniform",
        3: "random_prior",
    }

    def __init__(
        self,
        split_threshold: Optional[float] = None,
        split_interval: int = 80,
        max_epochs: int = 300,
        lr: float = 0.01,
        beta_mult: float = 1.0,
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        temperature_decay: Optional[float] = None,
        split_perturb_scale: float = 0.2,
        prior_logvar0: float = 2.197,
        verbose: bool = True,
    ) -> None:
        self.split_threshold = split_threshold
        self.split_interval = int(split_interval)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.beta_mult = float(beta_mult)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.temperature_decay = temperature_decay
        self.split_perturb_scale = float(split_perturb_scale)
        self.prior_logvar0 = float(prior_logvar0)
        self.verbose = bool(verbose)

        self.model: Optional[DiagnosableGMM] = None
        self.history: Dict[str, List[Any]] = {"loss": [], "k": [], "phi": [], "temperature": []}
        self.split_history: List[Dict[str, Any]] = []
        self.fit_timings_ = FitTimings()
        self.fit_summary_: Dict[str, Any] = {}
        self.use_prior_: Optional[int] = None

    @staticmethod
    def auto_split_threshold(D: int, sigma2: float = 1.0) -> float:
        return 0.5 * D * (1 + np.log(2 * np.pi) + np.log(sigma2))

    def _resolve_temperature_decay(self) -> float:
        if self.temperature_decay is not None:
            return float(self.temperature_decay)
        if self.temperature_start <= self.temperature_end:
            return 1.0
        return float((self.temperature_end / self.temperature_start) ** (1.0 / max(self.max_epochs, 1)))

    def _step_a_calculate_prior(
        self,
        X: np.ndarray,
        mode: int = 1,
        w_kw: float = 1.0,
        w_llr: float = 1.0,
        rough_k: int = 3,
    ) -> torch.Tensor:
        """
        Mode 1: informative prior using KW + rough-clustering Gaussian LLR
        Mode 2: uniform 0.5 prior
        Mode 3: random prior
        """
        N, D = X.shape

        if mode == 1:
            if self.verbose:
                print("   -> Mode 1: computing KW + LLR prior.")
            k_rough = max(2, min(int(rough_k), N))
            labels = KMeans(n_clusters=k_rough, random_state=42, n_init=10).fit(X).labels_

            final_scores = []
            for j in range(D):
                feat = X[:, j]
                groups = [feat[labels == k] for k in range(k_rough) if np.sum(labels == k) > 0]

                kw_stat = 0.0
                if len(groups) > 1:
                    try:
                        stat, _ = stats.kruskal(*groups)
                        kw_stat = np.log1p(stat)
                    except Exception:
                        kw_stat = 0.0

                var_0 = np.var(feat) + 1e-6
                ll_0 = -0.5 * np.sum((feat - np.mean(feat)) ** 2) / var_0 - N * 0.5 * np.log(var_0)

                ll_1 = 0.0
                for group in groups:
                    if len(group) > 1:
                        v_k = np.var(group) + 1e-6
                        ll_1 += -0.5 * np.sum((group - np.mean(group)) ** 2) / v_k - len(group) * 0.5 * np.log(v_k)

                llr_stat = np.log1p(max(0.0, ll_1 - ll_0))
                final_scores.append(w_kw * kw_stat + w_llr * llr_stat)

            final_scores = np.asarray(final_scores, dtype=float)
            if np.allclose(final_scores.max(), final_scores.min()):
                rho = torch.full((D,), 0.5, dtype=torch.float32)
            else:
                norm = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min() + 1e-9)
                logits = (norm - 0.5) * 6.0
                rho = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
                rho = torch.clamp(rho, 0.01, 0.99)
            return rho

        if mode == 2:
            return torch.full((D,), 0.5, dtype=torch.float32)
        if mode == 3:
            return torch.rand(D, dtype=torch.float32)
        raise ValueError(f"Unsupported prior mode: {mode}")

    def _expand_model(self, old_model: DiagnosableGMM, split_idx: int) -> DiagnosableGMM:
        D, K = old_model.D, old_model.K
        new_model = DiagnosableGMM(
            input_dim=D,
            num_components=K + 1,
            prior_phi_probs=old_model.prior_phi_probs,
            temperature=old_model.temperature,
            prior_logvar0=self.prior_logvar0,
        )

        with torch.no_grad():
            new_model.phi_logits.copy_(old_model.phi_logits)

            old_mu = old_model.q_mu.data
            target_mu = old_mu[split_idx]
            noise_a = torch.randn(D, device=target_mu.device) * self.split_perturb_scale
            noise_b = torch.randn(D, device=target_mu.device) * self.split_perturb_scale
            mu_a = target_mu + noise_a
            mu_b = target_mu - noise_b

            keep_idx = [i for i in range(K) if i != split_idx]
            if keep_idx:
                new_mus = torch.cat([old_mu[keep_idx], mu_a.unsqueeze(0), mu_b.unsqueeze(0)], dim=0)
            else:
                new_mus = torch.cat([mu_a.unsqueeze(0), mu_b.unsqueeze(0)], dim=0)
            new_model.q_mu.copy_(new_mus)

            old_logvar = old_model.q_logvar.data
            target_logvar = old_logvar[split_idx]
            if keep_idx:
                new_logvars = torch.cat(
                    [old_logvar[keep_idx], target_logvar.unsqueeze(0), target_logvar.unsqueeze(0)],
                    dim=0,
                )
            else:
                new_logvars = torch.cat([target_logvar.unsqueeze(0), target_logvar.unsqueeze(0)], dim=0)
            new_model.q_logvar.copy_(new_logvars)

            old_pi = torch.softmax(old_model.pi_logits.data, dim=0)
            if keep_idx:
                kept_pi = old_pi[keep_idx]
            else:
                kept_pi = torch.empty(0, device=old_pi.device)
            split_mass = old_pi[split_idx] / 2.0
            new_pi = torch.cat([kept_pi, split_mass.unsqueeze(0), split_mass.unsqueeze(0)], dim=0)
            new_model.pi_logits.copy_(torch.log(new_pi + 1e-9))

        return new_model

    def fit(self, X_np: np.ndarray, use_prior: int = 1) -> "DIVIClustering":
        start_total = time.perf_counter()
        self.use_prior_ = int(use_prior)
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        N, D = X_tensor.shape
        beta = self.beta_mult * N

        if self.split_threshold is None:
            self.split_threshold = self.auto_split_threshold(D=D, sigma2=1.0)
            if self.verbose:
                print(f"Auto-configured split threshold = {self.split_threshold:.2f}")

        t0 = time.perf_counter()
        rho = self._step_a_calculate_prior(X_np, mode=use_prior)
        self.fit_timings_.step_a_sec = time.perf_counter() - t0

        global_mean = torch.mean(X_tensor, dim=0, keepdim=True)
        self.model = DiagnosableGMM(
            input_dim=D,
            num_components=1,
            prior_phi_probs=rho,
            init_means=global_mean,
            temperature=self.temperature_start,
            prior_logvar0=self.prior_logvar0,
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        temp_decay = self._resolve_temperature_decay()

        best_loss = float("inf")
        train_start = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            assert self.model is not None
            self.model.train()
            optimizer.zero_grad()
            out = self.model.loss_components(X=X_tensor, beta=beta, sample_phi=True)
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            self.model.temperature = max(self.temperature_end, self.model.temperature * temp_decay)

            q_phi_np = out["q_phi"].detach().cpu().numpy()
            loss_value = float(loss.item())
            best_loss = min(best_loss, loss_value)

            self.history["loss"].append(loss_value)
            self.history["k"].append(self.model.K)
            self.history["phi"].append(q_phi_np)
            self.history["temperature"].append(float(self.model.temperature))

            if epoch % self.split_interval == 0:
                td0 = time.perf_counter()
                scores = self.model.get_cluster_diagnostics(X_tensor, beta=beta)
                self.fit_timings_.split_diag_sec += time.perf_counter() - td0

                worst_idx = int(np.argmax(scores))
                worst_score = float(scores[worst_idx])

                if self.verbose:
                    print(
                        f"Epoch {epoch:4d} | K={self.model.K:2d} | "
                        f"loss={loss_value:.3f} | max_diag={worst_score:.3f}"
                    )

                if worst_score > float(self.split_threshold):
                    if self.verbose:
                        print(f"   >>> split cluster {worst_idx} at epoch {epoch}")
                    self.split_history.append(
                        {
                            "epoch": epoch,
                            "split_idx": worst_idx,
                            "score": worst_score,
                            "K_before": self.model.K,
                            "K_after": self.model.K + 1,
                        }
                    )
                    self.model = self._expand_model(self.model, worst_idx)
                    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.fit_timings_.train_sec = time.perf_counter() - train_start

        post_start = time.perf_counter()
        pred = self.predict(X_np)
        phi = self.get_phi()
        final_components = self.model.loss_components(X=X_tensor, beta=beta, sample_phi=False)
        self.fit_timings_.post_sec = time.perf_counter() - post_start
        self.fit_timings_.total_sec = time.perf_counter() - start_total

        self.fit_summary_ = {
            "beta": float(beta),
            "beta_mult": float(self.beta_mult),
            "epochs_completed": self.max_epochs,
            "final_K": int(self.model.K),
            "split_count": len(self.split_history),
            "first_split_epoch": self.split_history[0]["epoch"] if self.split_history else None,
            "last_split_epoch": self.split_history[-1]["epoch"] if self.split_history else None,
            "split_epochs": [s["epoch"] for s in self.split_history],
            "prior_mode": int(use_prior),
            "prior_mode_name": self.PRIOR_MODE_NAME.get(int(use_prior), f"mode_{use_prior}"),
            "selected_dims_count": int(np.sum(phi >= 0.5)),
            "selected_dims_ratio": float(np.mean(phi >= 0.5)),
            "mean_phi": float(np.mean(phi)),
            "median_phi": float(np.median(phi)),
            "objective_final": float(final_components["loss"].item()),
            "objective_best": float(best_loss),
            "objective_gap": float(final_components["loss"].item() - best_loss),
            "nll_final": float((-final_components["log_likelihood"]).item()),
            "kl_final": float(final_components["kl_phi"].item()),
            "time_per_epoch_sec": float(self.fit_timings_.train_sec / max(self.max_epochs, 1)),
            "wallclock_stepA_sec": float(self.fit_timings_.step_a_sec),
            "wallclock_train_sec": float(self.fit_timings_.train_sec),
            "wallclock_split_diag_sec": float(self.fit_timings_.split_diag_sec),
            "wallclock_post_sec": float(self.fit_timings_.post_sec),
            "wallclock_total_sec": float(self.fit_timings_.total_sec),
            "predictions": pred,
        }
        return self

    def predict_proba(self, X_np: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        beta = self.beta_mult * X_tensor.shape[0]
        with torch.no_grad():
            out = self.model.loss_components(X=X_tensor, beta=beta, sample_phi=False)
            log_p = out["log_p_x_given_z"] + torch.log(torch.softmax(self.model.pi_logits, dim=0) + 1e-9).unsqueeze(0)
            probs = torch.softmax(log_p, dim=1)
        return probs.detach().cpu().numpy()

    def predict(self, X_np: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_np)
        return np.argmax(probs, axis=1)

    def get_phi(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        with torch.no_grad():
            return torch.sigmoid(self.model.phi_logits).detach().cpu().numpy()

    def get_prior_phi(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.prior_phi_probs.detach().cpu().numpy()

    def get_split_epochs(self) -> List[int]:
        return [x["epoch"] for x in self.split_history]
