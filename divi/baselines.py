from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


@dataclass
class BaselineResult:
    labels: np.ndarray
    selected_mask: np.ndarray | None = None
    method: str = ""


class SparseKMeans(BaseEstimator, ClusterMixin):
    """
    Reorganized from n200andSPKM.py.
    """

    def __init__(self, n_clusters: int = 3, l1_bound: float = 5.0, max_iter: int = 20, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.l1_bound = l1_bound
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        self.weights_ = np.ones(n_features) / np.sqrt(n_features)
        tss = np.sum(X ** 2, axis=0)

        for _ in range(self.max_iter):
            w_old = self.weights_.copy()

            X_weighted = X * np.sqrt(self.weights_)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            kmeans.fit(X_weighted)
            self.labels_ = kmeans.labels_

            centers = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if mask.sum() > 0:
                    centers[k] = np.mean(X[mask], axis=0)

            wcss = np.sum((X - centers[self.labels_]) ** 2, axis=0)
            bcss = np.maximum(tss - wcss, 0.0)
            self.weights_ = self._find_weights_binary_search(bcss, self.l1_bound)

            if np.linalg.norm(self.weights_ - w_old) < self.tol:
                break
        return self

    def _find_weights_binary_search(self, bcss: np.ndarray, s: float) -> np.ndarray:
        w_l2 = bcss / (np.linalg.norm(bcss) + 1e-9)
        if np.sum(np.abs(w_l2)) <= s:
            return w_l2

        lower, upper = 0.0, float(np.max(bcss))
        for _ in range(20):
            delta = (lower + upper) / 2.0
            w_soft = np.maximum(bcss - delta, 0.0)
            if np.linalg.norm(w_soft) == 0:
                upper = delta
                continue
            ratio = np.sum(w_soft) / np.linalg.norm(w_soft)
            if ratio > s:
                lower = delta
            else:
                upper = delta

        delta = (lower + upper) / 2.0
        w_soft = np.maximum(bcss - delta, 0.0)
        return w_soft / (np.linalg.norm(w_soft) + 1e-9)


def run_kmeans(X: np.ndarray, K: int, seed: int) -> BaselineResult:
    model = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    return BaselineResult(labels=labels, selected_mask=None, method="kmeans_oracle")


def run_gmm(X: np.ndarray, K: int, seed: int) -> BaselineResult:
    model = GaussianMixture(n_components=K, random_state=seed)
    labels = model.fit(X).predict(X)
    return BaselineResult(labels=labels, selected_mask=None, method="gmm_oracle")


def run_spkm(X: np.ndarray, K: int, l1_bound: float = 4.0) -> BaselineResult:
    model = SparseKMeans(n_clusters=K, l1_bound=l1_bound, max_iter=20)
    model.fit(X)
    mask = model.weights_ > 1e-4
    return BaselineResult(labels=model.labels_, selected_mask=mask, method="spkm")
