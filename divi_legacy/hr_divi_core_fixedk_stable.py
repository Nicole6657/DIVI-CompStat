from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    structure_sec: float = 0.0
    post_sec: float = 0.0
    total_sec: float = 0.0


class HierarchicalDiagnosableGMM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_components: int,
        prior_g_probs: torch.Tensor,
        prior_r_probs: Optional[torch.Tensor] = None,
        init_mu0: Optional[torch.Tensor] = None,
        init_delta: Optional[torch.Tensor] = None,
        init_logvar: Optional[torch.Tensor] = None,
        init_pi: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        prior_logvar0: float = 2.197,
        min_logvar: float = -8.0,
        max_logvar: float = 8.0,
    ) -> None:
        super().__init__()
        self.D = int(input_dim)
        self.K = int(num_components)
        self.temperature = float(temperature)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        prior_g_probs = torch.clamp(prior_g_probs.float(), 1e-4, 1.0 - 1e-4)
        self.register_buffer("prior_g_probs", prior_g_probs)

        if prior_r_probs is None:
            prior_r_probs = torch.full((self.K, self.D), 0.5, dtype=torch.float32)
        prior_r_probs = torch.clamp(prior_r_probs.float(), 1e-4, 1.0 - 1e-4)
        self.register_buffer("prior_r_probs", prior_r_probs)
        self.register_buffer("prior_logvar_0", torch.full((self.D,), float(prior_logvar0)))

        self.g_logits = nn.Parameter(torch.log(prior_g_probs / (1.0 - prior_g_probs)).clone())
        self.r_logits = nn.Parameter(torch.log(prior_r_probs / (1.0 - prior_r_probs)).clone())

        if init_mu0 is None:
            init_mu0 = torch.zeros(self.D)
        self.mu0 = nn.Parameter(init_mu0.float().reshape(self.D))

        if init_delta is None:
            init_delta = torch.zeros(self.K, self.D)
        self.q_delta = nn.Parameter(torch.nan_to_num(init_delta.float(), nan=0.0, posinf=0.0, neginf=0.0))

        if init_logvar is None:
            init_logvar = torch.ones(self.K, self.D) * -1.0
        init_logvar = torch.nan_to_num(init_logvar.float(), nan=-1.0, posinf=self.max_logvar, neginf=self.min_logvar)
        self.q_logvar = nn.Parameter(torch.clamp(init_logvar, self.min_logvar, self.max_logvar))

        if init_pi is None:
            init_pi = torch.full((self.K,), 1.0 / self.K)
        init_pi = torch.clamp(init_pi.float(), 1e-6, 1.0)
        init_pi = init_pi / init_pi.sum()
        self.pi_logits = nn.Parameter(torch.log(init_pi))

    def gumbel_sigmoid_sample(self, logits: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(logits)
        uniform = torch.clamp(uniform, 1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        return torch.sigmoid((logits + gumbel) / max(self.temperature, 1e-4))

    def get_relevance_tensors(self, sample_relevance: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_relevance:
            g = self.gumbel_sigmoid_sample(self.g_logits)
            r = self.gumbel_sigmoid_sample(self.r_logits)
        else:
            g = torch.sigmoid(self.g_logits)
            r = torch.sigmoid(self.r_logits)
        w = g.unsqueeze(0) * r
        return g, r, w

    def get_effective_means(self, sample_relevance: bool = True):
        g, r, w = self.get_relevance_tensors(sample_relevance=sample_relevance)
        mu_eff = self.mu0.unsqueeze(0) + w * self.q_delta
        return g, r, w, mu_eff

    def loss_components(self, X: torch.Tensor, beta_g: float, beta_r: Optional[float] = None, sample_relevance: bool = True) -> Dict[str, torch.Tensor]:
        if beta_r is None:
            beta_r = beta_g
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        g, r, w, mu_eff = self.get_effective_means(sample_relevance=sample_relevance)
        x_exp = X.unsqueeze(1)
        mu_exp = mu_eff.unsqueeze(0)
        logvar = torch.clamp(self.q_logvar, self.min_logvar, self.max_logvar)
        logvar_exp = logvar.unsqueeze(0)
        var_exp = torch.exp(logvar_exp).clamp_min(1e-6)

        sq = (x_exp - mu_exp) ** 2
        log_prob = -0.5 * (math.log(2.0 * math.pi) + logvar_exp + sq / var_exp)
        log_p_x_given_z = torch.nan_to_num(log_prob.sum(dim=2), nan=-1e12, posinf=1e12, neginf=-1e12)

        pi = torch.softmax(self.pi_logits, dim=0)
        log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
        log_likelihood = torch.logsumexp(log_joint, dim=1).sum()

        q_g = torch.clamp(torch.sigmoid(self.g_logits), 1e-6, 1.0 - 1e-6)
        p_g = torch.clamp(self.prior_g_probs, 1e-6, 1.0 - 1e-6)
        kl_g = (q_g * (torch.log(q_g) - torch.log(p_g)) + (1.0 - q_g) * (torch.log(1.0 - q_g) - torch.log(1.0 - p_g))).sum()

        q_r = torch.clamp(torch.sigmoid(self.r_logits), 1e-6, 1.0 - 1e-6)
        p_r = torch.clamp(self.prior_r_probs, 1e-6, 1.0 - 1e-6)
        kl_r = (q_r * (torch.log(q_r) - torch.log(p_r)) + (1.0 - q_r) * (torch.log(1.0 - q_r) - torch.log(1.0 - p_r))).sum()

        loss = -log_likelihood + beta_g * kl_g + beta_r * kl_r
        loss = torch.nan_to_num(loss, nan=1e12, posinf=1e12, neginf=1e12)
        return {"loss": loss, "q_g": q_g, "q_r": q_r, "w": w, "mu_eff": mu_eff, "log_p_x_given_z": log_p_x_given_z, "log_likelihood": log_likelihood, "kl_g": kl_g, "kl_r": kl_r}


class HRDIVIClustering:
    PRIOR_MODE_NAME = {1: "informative_kw_llr", 2: "noninformative_uniform", 3: "random_prior"}

    def __init__(self, split_threshold: Optional[float] = None, split_interval: int = 80, max_epochs: int = 300, lr: float = 0.01,
                 beta_g_mult: float = 1.0, beta_r_mult: float = 1.0, temperature_start: float = 1.0, temperature_end: float = 0.1,
                 temperature_decay: Optional[float] = None, split_perturb_scale: float = 0.2, prior_logvar0: float = 2.197,
                 verbose: bool = True, targeted_birth_enabled: bool = False, birth_interval: Optional[int] = None,
                 birth_refine_steps: int = 20, birth_accept_tol: float = 10.0, max_birth_epoch: Optional[int] = None,
                 min_cluster_size_for_birth: int = 25, target_subset_max_points: int = 500, target_subset_resp_threshold: float = 0.4,
                 birth_top_m_features: int = 4, birth_score_weight: float = 0.01, init_num_components: int = 1,
                 init_method: str = "kmeans", random_state: int = 42, grad_clip_norm: float = 5.0) -> None:
        self.split_threshold = split_threshold
        self.split_interval = int(split_interval)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.beta_g_mult = float(beta_g_mult)
        self.beta_r_mult = float(beta_r_mult)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.temperature_decay = temperature_decay
        self.split_perturb_scale = float(split_perturb_scale)
        self.prior_logvar0 = float(prior_logvar0)
        self.verbose = bool(verbose)
        self.targeted_birth_enabled = bool(targeted_birth_enabled)
        self.birth_interval = int(birth_interval) if birth_interval is not None else int(split_interval)
        self.birth_refine_steps = int(birth_refine_steps)
        self.birth_accept_tol = float(birth_accept_tol)
        self.max_birth_epoch = int(max_birth_epoch) if max_birth_epoch is not None else max(1, int(0.5 * max_epochs))
        self.min_cluster_size_for_birth = int(min_cluster_size_for_birth)
        self.target_subset_max_points = int(target_subset_max_points)
        self.target_subset_resp_threshold = float(target_subset_resp_threshold)
        self.birth_top_m_features = int(birth_top_m_features)
        self.birth_score_weight = float(birth_score_weight)
        self.init_num_components = int(init_num_components)
        self.init_method = str(init_method)
        self.random_state = int(random_state)
        self.grad_clip_norm = float(grad_clip_norm)

        self.model: Optional[HierarchicalDiagnosableGMM] = None
        self.history: Dict[str, List[Any]] = {"loss": [], "k": [], "g": [], "r": [], "w": [], "temperature": []}
        self.split_history: List[Dict[str, Any]] = []
        self.fit_timings_ = FitTimings()
        self.fit_summary_: Dict[str, Any] = {}
        self.use_prior_: Optional[int] = None

    @staticmethod
    def auto_split_threshold(D: int, sigma2: float = 1.0) -> float:
        return 0.5 * D * (1.0 + np.log(2.0 * np.pi) + np.log(sigma2))

    def _resolve_temperature_decay(self) -> float:
        if self.temperature_decay is not None:
            return float(self.temperature_decay)
        if self.temperature_start <= self.temperature_end:
            return 1.0
        return float((self.temperature_end / self.temperature_start) ** (1.0 / max(self.max_epochs, 1)))

    @staticmethod
    def _neutral_prior_r(K: int, D: int, fill: float = 0.5) -> torch.Tensor:
        return torch.full((K, D), float(fill), dtype=torch.float32)

    def _step_a_calculate_prior(self, X: np.ndarray, mode: int = 1, w_kw: float = 1.0, w_llr: float = 1.0, rough_k: int = 3) -> torch.Tensor:
        N, D = X.shape
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if mode == 1:
            k_rough = max(2, min(int(rough_k), N))
            labels = KMeans(n_clusters=k_rough, random_state=self.random_state, n_init=10).fit(X).labels_
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
            final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
            if np.allclose(final_scores.max(), final_scores.min()):
                return torch.full((D,), 0.5, dtype=torch.float32)
            norm = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min() + 1e-9)
            logits = (norm - 0.5) * 6.0
            rho = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
            return torch.clamp(rho, 0.01, 0.99)
        if mode == 2:
            return torch.full((D,), 0.5, dtype=torch.float32)
        if mode == 3:
            return torch.rand(D, dtype=torch.float32)
        raise ValueError(f"Unsupported prior mode: {mode}")

    def _initialize_components(self, X_np: np.ndarray, X_tensor: torch.Tensor, K0: int):
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        N, D = X_np.shape
        mu0 = torch.mean(X_tensor, dim=0)
        if K0 <= 1:
            return mu0, torch.zeros(1, D), torch.ones(1, D) * -1.0, torch.ones(1)
        km = KMeans(n_clusters=K0, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(X_np)
        centers = km.cluster_centers_
        delta = centers - mu0.detach().cpu().numpy()
        logvars = np.zeros((K0, D), dtype=np.float32)
        pis = np.bincount(labels, minlength=K0).astype(np.float32)
        pis = np.maximum(pis, 1.0)
        pis = pis / pis.sum()
        global_var = X_np.var(axis=0) + 1e-3
        for k in range(K0):
            Xk = X_np[labels == k]
            var_k = global_var if Xk.shape[0] <= 1 else (Xk.var(axis=0) + 1e-3)
            logvars[k] = np.log(np.clip(var_k, 1e-6, 1e6))
        return mu0, torch.tensor(delta, dtype=torch.float32), torch.tensor(logvars, dtype=torch.float32), torch.tensor(pis, dtype=torch.float32)

    def fit(self, X_np: np.ndarray, use_prior: int = 1) -> "HRDIVIClustering":
        start_total = time.perf_counter()
        self.use_prior_ = int(use_prior)
        X_np = np.asarray(X_np, dtype=np.float32)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        N, D = X_tensor.shape
        beta_g = self.beta_g_mult * N
        beta_r = self.beta_r_mult * N

        if self.split_threshold is None:
            self.split_threshold = self.auto_split_threshold(D=D, sigma2=1.0)

        t0 = time.perf_counter()
        prior_g = self._step_a_calculate_prior(X_np, mode=use_prior)
        self.fit_timings_.step_a_sec = time.perf_counter() - t0

        K0 = max(1, self.init_num_components)
        init_mu0, init_delta, init_logvar, init_pi = self._initialize_components(X_np, X_tensor, K0)
        prior_r = self._neutral_prior_r(K0, D, fill=0.5)
        self.model = HierarchicalDiagnosableGMM(input_dim=D, num_components=K0, prior_g_probs=prior_g, prior_r_probs=prior_r,
                                               init_mu0=init_mu0, init_delta=init_delta, init_logvar=init_logvar, init_pi=init_pi,
                                               temperature=self.temperature_start, prior_logvar0=self.prior_logvar0)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        temp_decay = self._resolve_temperature_decay()
        best_loss = float("inf")
        train_start = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            out = self.model.loss_components(X=X_tensor, beta_g=beta_g, beta_r=beta_r, sample_relevance=True)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered before backward at epoch {epoch}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            optimizer.step()
            with torch.no_grad():
                self.model.q_logvar.clamp_(-8.0, 8.0)
                self.model.temperature = max(self.temperature_end, self.model.temperature * temp_decay)
            loss_value = float(loss.item())
            best_loss = min(best_loss, loss_value)
            self.history["loss"].append(loss_value)
            self.history["k"].append(self.model.K)
            self.history["g"].append(out["q_g"].detach().cpu().numpy())
            self.history["r"].append(out["q_r"].detach().cpu().numpy())
            self.history["w"].append(out["w"].detach().cpu().numpy())
            self.history["temperature"].append(float(self.model.temperature))

        self.fit_timings_.train_sec = time.perf_counter() - train_start
        post_start = time.perf_counter()
        pred = self.predict(X_np)
        g = self.get_global_relevance()
        r = self.get_cluster_relevance()
        w = self.get_effective_relevance()
        final_components = self.model.loss_components(X=X_tensor, beta_g=beta_g, beta_r=beta_r, sample_relevance=False)
        self.fit_timings_.post_sec = time.perf_counter() - post_start
        self.fit_timings_.total_sec = time.perf_counter() - start_total
        self.fit_summary_ = {"beta_g": float(beta_g), "beta_r": float(beta_r), "beta_g_mult": float(self.beta_g_mult), "beta_r_mult": float(self.beta_r_mult),
                             "epochs_completed": self.max_epochs, "final_K": int(self.model.K), "split_count": len(self.split_history),
                             "split_epochs": [s.get("epoch") for s in self.split_history], "prior_mode": int(use_prior),
                             "prior_mode_name": self.PRIOR_MODE_NAME.get(int(use_prior), f"mode_{use_prior}"),
                             "selected_global_dims_count": int(np.sum(g >= 0.5)), "selected_global_dims_ratio": float(np.mean(g >= 0.5)),
                             "mean_global_relevance": float(np.mean(g)), "median_global_relevance": float(np.median(g)),
                             "mean_cluster_relevance": float(np.mean(r)), "mean_effective_relevance": float(np.mean(w)),
                             "objective_final": float(final_components["loss"].item()), "objective_best": float(best_loss),
                             "objective_gap": float(final_components["loss"].item() - best_loss),
                             "nll_final": float((-final_components["log_likelihood"]).item()), "kl_g_final": float(final_components["kl_g"].item()),
                             "kl_r_final": float(final_components["kl_r"].item()), "time_per_epoch_sec": float(self.fit_timings_.train_sec / max(self.max_epochs, 1)),
                             "wallclock_stepA_sec": float(self.fit_timings_.step_a_sec), "wallclock_train_sec": float(self.fit_timings_.train_sec),
                             "wallclock_structure_sec": float(self.fit_timings_.structure_sec), "wallclock_post_sec": float(self.fit_timings_.post_sec),
                             "wallclock_total_sec": float(self.fit_timings_.total_sec), "predictions": pred}
        return self

    def predict_proba(self, X_np: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X_np = np.nan_to_num(np.asarray(X_np, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        beta_g = self.beta_g_mult * X_tensor.shape[0]
        beta_r = self.beta_r_mult * X_tensor.shape[0]
        with torch.no_grad():
            out = self.model.loss_components(X=X_tensor, beta_g=beta_g, beta_r=beta_r, sample_relevance=False)
            log_p = out["log_p_x_given_z"] + torch.log(torch.softmax(self.model.pi_logits, dim=0) + 1e-9).unsqueeze(0)
            probs = torch.softmax(log_p, dim=1)
        return probs.detach().cpu().numpy()

    def predict(self, X_np: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X_np), axis=1)

    def get_global_relevance(self) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self.model.g_logits).detach().cpu().numpy()

    def get_cluster_relevance(self) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self.model.r_logits).detach().cpu().numpy()

    def get_effective_relevance(self) -> np.ndarray:
        with torch.no_grad():
            g = torch.sigmoid(self.model.g_logits)
            r = torch.sigmoid(self.model.r_logits)
            return (g.unsqueeze(0) * r).detach().cpu().numpy()

    def get_effective_means(self) -> np.ndarray:
        with torch.no_grad():
            g = torch.sigmoid(self.model.g_logits)
            r = torch.sigmoid(self.model.r_logits)
            w = g.unsqueeze(0) * r
            mu_eff = self.model.mu0.unsqueeze(0) + w * self.model.q_delta
            return mu_eff.detach().cpu().numpy()
