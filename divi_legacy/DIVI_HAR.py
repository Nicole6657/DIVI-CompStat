# -*- coding: utf-8 -*-
"""DIVI-MLWA: Data-informed feature-gated clustering with adaptive split growth.

This version includes the MLWA-oriented fixes for HAR / known-K benchmarks:
  1. max_components: optional cap for split growth, e.g. K_max=6 for HAR.
  2. beta_mult: controls the KL strength beta = beta_mult * N.
  3. rough_k: controls the rough K-means size in Step A initialization.
  4. deterministic prediction: final labels use q_phi = sigmoid(phi_logits),
     not stochastic Gumbel samples.
  5. split expansion inherits mixture weights instead of resetting them.
  6. split_threshold=None is handled per fit without permanently overwriting it.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.cluster import KMeans


class DiagnosableGMM(nn.Module):
    """Feature-gated diagonal Gaussian mixture used by DIVI.

    Parameters
    ----------
    input_dim : int
        Number of features.
    num_components : int
        Current number of mixture components.
    prior_phi_probs : torch.Tensor, shape (D,)
        Data-informed reference probabilities for feature gates.
    init_means : torch.Tensor, optional, shape (K, D)
        Optional initialization of component means.
    temperature : float
        Temperature for the Gumbel-Sigmoid relaxation.
    beta_mult : float
        KL scaling multiplier, beta = beta_mult * N.
    """

    def __init__(self, input_dim, num_components, prior_phi_probs,
                 init_means=None, temperature=1.0, beta_mult=1.0):
        super().__init__()
        self.D = input_dim
        self.K = num_components
        self.temperature = temperature
        self.beta_mult = beta_mult

        prior_phi_probs = prior_phi_probs.float()
        self.register_buffer("prior_phi_probs", prior_phi_probs)

        # Shared background distribution. log(9) = 2.197, i.e., variance about 9.
        self.register_buffer("prior_mu_0", torch.zeros(input_dim))
        self.register_buffer("prior_logvar_0", torch.full((input_dim,), 2.197))

        # Initialize feature-gate logits from the data-informed reference probabilities.
        prior_phi_probs_clamped = torch.clamp(prior_phi_probs, 1e-6, 1 - 1e-6)
        prior_logits = torch.log(prior_phi_probs_clamped / (1 - prior_phi_probs_clamped))
        self.phi_logits = nn.Parameter(prior_logits.clone())

        if init_means is not None:
            init_means = init_means.float()
            if init_means.shape[0] < num_components:
                pad = torch.randn(num_components - init_means.shape[0], input_dim)
                init_means = torch.cat([init_means, pad], dim=0)
            self.q_mu = nn.Parameter(init_means[:num_components].clone())
        else:
            self.q_mu = nn.Parameter(torch.randn(num_components, input_dim))

        # Diagonal log-variances and mixture logits.
        self.q_logvar = nn.Parameter(torch.ones(num_components, input_dim) * -1.0)
        self.pi_logits = nn.Parameter(torch.ones(num_components))

    def gumbel_sigmoid_sample(self, logits):
        uniform = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(uniform + 1e-9) + 1e-9)
        return torch.sigmoid((logits + gumbel) / self.temperature)

    def component_log_density(self, X, phi=None, stochastic=False):
        """Return component-wise gated log densities, shape (N, K).

        If stochastic=True, a Gumbel-Sigmoid gate sample is used.
        Otherwise, deterministic gates sigmoid(phi_logits) are used.
        """
        if stochastic:
            gate = self.gumbel_sigmoid_sample(self.phi_logits)
        elif phi is not None:
            gate = phi
        else:
            gate = torch.sigmoid(self.phi_logits)

        x_exp = X.unsqueeze(1)              # (N, 1, D)
        mu_exp = self.q_mu.unsqueeze(0)     # (1, K, D)
        logvar_exp = self.q_logvar.unsqueeze(0)

        log_prob_cluster = -0.5 * (
            np.log(2 * np.pi)
            + logvar_exp
            + (x_exp - mu_exp) ** 2 / torch.exp(logvar_exp)
        )

        log_prob_bg = -0.5 * (
            np.log(2 * np.pi)
            + self.prior_logvar_0
            + (x_exp - self.prior_mu_0) ** 2 / torch.exp(self.prior_logvar_0)
        )

        gate = gate.view(1, 1, -1)
        weighted_log_prob = gate * log_prob_cluster + (1 - gate) * log_prob_bg
        return weighted_log_prob.sum(dim=2)

    def forward(self, X):
        N, _ = X.shape

        # Stochastic gate sample for training objective.
        log_p_x_given_z = self.component_log_density(X, stochastic=True)

        pi = torch.softmax(self.pi_logits, dim=0)
        log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
        log_likelihood = torch.logsumexp(log_joint, dim=1).sum()

        q_phi = torch.sigmoid(self.phi_logits)
        q_phi = torch.clamp(q_phi, 1e-6, 1 - 1e-6)
        p_phi = torch.clamp(self.prior_phi_probs, 1e-6, 1 - 1e-6)

        kl_raw = (
            q_phi * (torch.log(q_phi) - torch.log(p_phi))
            + (1 - q_phi) * (torch.log(1 - q_phi) - torch.log(1 - p_phi))
        ).sum()
        kl_phi = self.beta_mult * N * kl_raw

        loss = -log_likelihood + kl_phi
        return loss, q_phi, log_p_x_given_z

    def get_cluster_diagnostics(self, X, deterministic=True):
        """Average negative gated log-density for each current component."""
        with torch.no_grad():
            if deterministic:
                log_p_x_given_z = self.component_log_density(X, stochastic=False)
            else:
                _, _, log_p_x_given_z = self.forward(X)

            z_hard = torch.argmax(log_p_x_given_z, dim=1)
            cluster_scores = []
            for k in range(self.K):
                mask = (z_hard == k)
                if mask.sum() == 0:
                    cluster_scores.append(0.0)
                else:
                    cluster_scores.append((-log_p_x_given_z[mask, k].mean()).item())
            return np.array(cluster_scores)


class DIVIClustering:
    """DIVI wrapper for fitting and deterministic prediction."""

    def __init__(self,
                 split_threshold=22.0,
                 split_interval=60,
                 max_epochs=300,
                 lr=0.05,
                 max_components=None,
                 beta_mult=1.0,
                 rough_k=3,
                 verbose=True):
        self.split_threshold = split_threshold
        self.split_interval = split_interval
        self.max_epochs = max_epochs
        self.lr = lr
        self.max_components = max_components
        self.beta_mult = beta_mult
        self.rough_k = rough_k
        self.verbose = verbose

        self.model = None
        self.history = {"loss": [], "k": [], "phi": [], "max_nll": []}

    def _step_a_calculate_prior(self, X, mode=1, w_kw=1.0, w_llr=1.0):
        """Data-informed feature initialization.

        mode=1: informative initialization using KW + LLR scores.
        mode=2: non-informative 0.5 probabilities.
        mode=3: random probabilities.
        """
        N, D = X.shape

        if mode == 1:
            if self.verbose:
                print("   -> Mode 1: Computing combined KW and LLR scores...")

            k_rough = int(min(max(1, self.rough_k), N))
            kmeans = KMeans(n_clusters=k_rough, random_state=42, n_init=10).fit(X)
            labels = kmeans.labels_

            final_scores = []
            for j in range(D):
                feat = X[:, j]
                groups = [feat[labels == k] for k in range(k_rough) if np.sum(labels == k) > 0]

                # Kruskal-Wallis statistic.
                kw_stat = 0.0
                if len(groups) > 1:
                    try:
                        stat, _ = stats.kruskal(*groups)
                        kw_stat = np.log1p(stat)
                    except Exception:
                        kw_stat = 0.0

                # Log-likelihood-ratio proxy: clustered Gaussian vs pooled Gaussian.
                var_0 = np.var(feat) + 1e-6
                ll_0 = -0.5 * np.sum((feat - np.mean(feat)) ** 2) / var_0 - N * 0.5 * np.log(var_0)

                ll_1 = 0.0
                for group in groups:
                    if len(group) > 1:
                        v_k = np.var(group) + 1e-6
                        ll_1 += -0.5 * np.sum((group - np.mean(group)) ** 2) / v_k - len(group) * 0.5 * np.log(v_k)

                llr_stat = np.log1p(max(0.0, ll_1 - ll_0))
                final_scores.append(w_kw * kw_stat + w_llr * llr_stat)

            final_scores = np.asarray(final_scores)
            score_range = final_scores.max() - final_scores.min()
            norm = (final_scores - final_scores.min()) / (score_range + 1e-9)

            logits = (norm - 0.5) * 6.0
            rho = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
            rho = torch.clamp(rho, 0.01, 0.99)

        elif mode == 2:
            rho = torch.full((D,), 0.5, dtype=torch.float32)
        elif mode == 3:
            rho = torch.rand(D, dtype=torch.float32)
            rho = torch.clamp(rho, 0.01, 0.99)
        else:
            raise ValueError("use_prior/mode must be 1, 2, or 3.")

        return rho

    def _expand_model(self, old_model, split_idx):
        """Split one component while inheriting gates, variances, and mixture mass."""
        D, K = old_model.D, old_model.K

        new_model = DiagnosableGMM(
            D,
            K + 1,
            old_model.prior_phi_probs,
            temperature=old_model.temperature,
            beta_mult=old_model.beta_mult,
        )

        with torch.no_grad():
            # Inherit global feature gates.
            new_model.phi_logits.copy_(old_model.phi_logits)

            old_mu = old_model.q_mu.data
            old_logvar = old_model.q_logvar.data
            target_mu = old_mu[split_idx]
            target_logvar = old_logvar[split_idx]

            # Opposite perturbations around the selected component mean.
            delta = torch.randn(D) * 0.2
            mu_a = target_mu + delta
            mu_b = target_mu - delta

            keep_idx = [i for i in range(K) if i != split_idx]
            if keep_idx:
                new_mus = torch.cat([old_mu[keep_idx], mu_a.unsqueeze(0), mu_b.unsqueeze(0)], dim=0)
                new_logvars = torch.cat(
                    [old_logvar[keep_idx], target_logvar.unsqueeze(0), target_logvar.unsqueeze(0)],
                    dim=0,
                )
            else:
                new_mus = torch.cat([mu_a.unsqueeze(0), mu_b.unsqueeze(0)], dim=0)
                new_logvars = torch.cat([target_logvar.unsqueeze(0), target_logvar.unsqueeze(0)], dim=0)

            new_model.q_mu.copy_(new_mus)
            new_model.q_logvar.copy_(new_logvars)

            # Inherit mixture weights: split the selected component mass in half.
            old_pi = torch.softmax(old_model.pi_logits.data, dim=0)
            if keep_idx:
                kept_pi = old_pi[keep_idx]
                split_pi = old_pi[split_idx] / 2.0
                new_pi = torch.cat([kept_pi, split_pi.view(1), split_pi.view(1)], dim=0)
            else:
                new_pi = torch.tensor([0.5, 0.5], dtype=old_pi.dtype, device=old_pi.device)
            new_pi = new_pi / new_pi.sum()
            new_model.pi_logits.copy_(torch.log(new_pi + 1e-8))

        return new_model

    def fit(self, X_np, use_prior=1):
        """Fit DIVI to a numpy array X_np of shape (N, D)."""
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        N, D = X_tensor.shape

        if self.split_threshold is None:
            current_split_threshold = 0.5 * D * (1 + np.log(2 * np.pi) + np.log(1.0))
            if self.verbose:
                print(f"Auto-configured Split Threshold: {current_split_threshold:.2f} based on D={D}")
        else:
            current_split_threshold = float(self.split_threshold)

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

        # Reset history for each fit.
        self.history = {"loss": [], "k": [], "phi": [], "max_nll": []}

        if self.verbose:
            print(
                f"Starting Training (Initial K=1, beta_mult={self.beta_mult}, "
                f"rough_k={self.rough_k}, max_components={self.max_components})..."
            )

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            loss, q_phi, _ = self.model(X_tensor)
            loss.backward()
            optimizer.step()

            self.model.temperature = max(0.1, self.model.temperature * 0.98)

            self.history["loss"].append(float(loss.item()))
            self.history["k"].append(int(self.model.K))
            self.history["phi"].append(q_phi.detach().cpu().numpy())

            if epoch % self.split_interval == 0:
                scores = self.model.get_cluster_diagnostics(X_tensor, deterministic=True)
                worst_idx = int(np.argmax(scores))
                worst_score = float(scores[worst_idx])
                self.history["max_nll"].append(worst_score)

                if self.verbose:
                    print(
                        f"Epoch {epoch}: K={self.model.K}, "
                        f"Max NLL={worst_score:.2f}, threshold={current_split_threshold:.2f}"
                    )

                can_split = (self.max_components is None) or (self.model.K < self.max_components)

                if (worst_score > current_split_threshold) and can_split:
                    if self.verbose:
                        print(f"   >>> Splitting Cluster {worst_idx}...")
                    self.model = self._expand_model(self.model, worst_idx)
                    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
                elif (worst_score > current_split_threshold) and (not can_split):
                    if self.verbose:
                        print(f"   >>> Split blocked: K reached max_components={self.max_components}")

        if self.verbose:
            print("Training Completed.")
        return self

    def predict(self, X_np):
        """Deterministic cluster labels using q_phi = sigmoid(phi_logits)."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")

        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            log_p_x_given_z = self.model.component_log_density(X_tensor, stochastic=False)
            labels = torch.argmax(log_p_x_given_z, dim=1).cpu().numpy()
        return labels

    def predict_proba(self, X_np):
        """Deterministic posterior-like responsibilities over current components."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")

        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            log_p_x_given_z = self.model.component_log_density(X_tensor, stochastic=False)
            pi = torch.softmax(self.model.pi_logits, dim=0)
            log_joint = log_p_x_given_z + torch.log(pi + 1e-9).unsqueeze(0)
            resp = torch.softmax(log_joint, dim=1).cpu().numpy()
        return resp

    def get_feature_relevance(self):
        """Return deterministic feature-relevance scores sigma(phi_logits)."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")

        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model.phi_logits).cpu().numpy()

    def fit_predict(self, X_np, use_prior=1):
        self.fit(X_np, use_prior=use_prior)
        return self.predict(X_np)
