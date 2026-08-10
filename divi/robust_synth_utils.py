from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler



def standardize_features(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)



def _balanced_cluster_sizes(N: int, K: int) -> np.ndarray:
    base = [N // K] * K
    for i in range(N - sum(base)):
        base[i] += 1
    return np.asarray(base, dtype=int)



def _default_signal_means(K: int = 3) -> np.ndarray:
    if K == 3:
        return np.array([-2.0, 0.0, 2.0], dtype=float)
    return np.linspace(-2.0, 2.0, K, dtype=float)



def _make_block_corr_cov(dim: int, rho: float, block_size: int) -> np.ndarray:
    cov = np.eye(dim, dtype=float)
    if dim <= 1 or rho == 0.0:
        return cov

    for start in range(0, dim, block_size):
        end = min(start + block_size, dim)
        block_dim = end - start
        block = np.full((block_dim, block_dim), rho, dtype=float)
        np.fill_diagonal(block, 1.0)
        cov[start:end, start:end] = block
    return cov



def generate_heavy_tail_signal_data(
    N: int = 200,
    D: int = 100,
    n_signal: int = 10,
    K: int = 3,
    signal_means: Optional[np.ndarray] = None,
    signal_df: float = 5.0,
    signal_scale: float = 1.0,
    noise_scale: float = 3.0,
    standardize: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Misspecified scenario 1:
    informative coordinates are heavy-tailed Student-t rather than Gaussian.

    The first n_signal dimensions remain the true informative support so feature
    F1 remains well-defined.
    """
    if not (signal_df > 2.0):
        raise ValueError("signal_df must be > 2 so the variance is finite.")
    if not (0 < n_signal <= D):
        raise ValueError("n_signal must satisfy 0 < n_signal <= D.")

    rng = np.random.default_rng(random_state)
    if signal_means is None:
        signal_means = _default_signal_means(K)
    signal_means = np.asarray(signal_means, dtype=float)
    if len(signal_means) != K:
        raise ValueError("len(signal_means) must equal K.")

    cluster_sizes = _balanced_cluster_sizes(N=N, K=K)
    truth_mask = np.zeros(D, dtype=int)
    truth_mask[:n_signal] = 1

    # Standardize Student-t to unit variance before applying signal_scale.
    t_std = np.sqrt(signal_df / (signal_df - 2.0))

    X_list, y_list = [], []
    for k, nk in enumerate(cluster_sizes):
        Xk = np.zeros((nk, D), dtype=float)

        signal = rng.standard_t(df=signal_df, size=(nk, n_signal)) / t_std
        signal = signal_scale * signal + signal_means[k]
        Xk[:, :n_signal] = signal

        noise = rng.normal(loc=0.0, scale=noise_scale, size=(nk, D - n_signal))
        Xk[:, n_signal:] = noise

        X_list.append(Xk)
        y_list.append(np.full(nk, k, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    perm = rng.permutation(N)
    X = X[perm]
    y = y[perm]

    if standardize:
        X = standardize_features(X)

    return {
        "X": X.astype(float),
        "y": y.astype(int),
        "truth_mask": truth_mask,
        "dataset": "synthetic_heavy_tail_signal",
        "dataset_variant": f"t_df{signal_df:g}_noise{noise_scale:g}",
        "K_true": int(K),
        "n_signal": int(n_signal),
        "signal_df": float(signal_df),
        "noise_scale": float(noise_scale),
        "signal_scale": float(signal_scale),
    }



def generate_correlated_noise_data(
    N: int = 200,
    D: int = 100,
    n_signal: int = 10,
    K: int = 3,
    signal_means: Optional[np.ndarray] = None,
    signal_scale: float = 1.0,
    noise_scale: float = 3.0,
    noise_rho: float = 0.6,
    noise_block_size: int = 10,
    standardize: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Misspecified scenario 2:
    noise coordinates are block-correlated rather than conditionally independent.

    The first n_signal dimensions remain the true informative support so feature
    F1 remains well-defined.
    """
    if not (0 < n_signal <= D):
        raise ValueError("n_signal must satisfy 0 < n_signal <= D.")
    if not (-1.0 < noise_rho < 1.0):
        raise ValueError("noise_rho must lie in (-1, 1).")

    rng = np.random.default_rng(random_state)
    if signal_means is None:
        signal_means = _default_signal_means(K)
    signal_means = np.asarray(signal_means, dtype=float)
    if len(signal_means) != K:
        raise ValueError("len(signal_means) must equal K.")

    cluster_sizes = _balanced_cluster_sizes(N=N, K=K)
    truth_mask = np.zeros(D, dtype=int)
    truth_mask[:n_signal] = 1

    noise_dim = D - n_signal
    noise_cov = _make_block_corr_cov(dim=noise_dim, rho=noise_rho, block_size=noise_block_size)
    noise_cov = (noise_scale ** 2) * noise_cov

    X_list, y_list = [], []
    for k, nk in enumerate(cluster_sizes):
        Xk = np.zeros((nk, D), dtype=float)
        Xk[:, :n_signal] = rng.normal(
            loc=signal_means[k],
            scale=signal_scale,
            size=(nk, n_signal),
        )

        if noise_dim > 0:
            Xk[:, n_signal:] = rng.multivariate_normal(
                mean=np.zeros(noise_dim, dtype=float),
                cov=noise_cov,
                size=nk,
            )

        X_list.append(Xk)
        y_list.append(np.full(nk, k, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    perm = rng.permutation(N)
    X = X[perm]
    y = y[perm]

    if standardize:
        X = standardize_features(X)

    return {
        "X": X.astype(float),
        "y": y.astype(int),
        "truth_mask": truth_mask,
        "dataset": "synthetic_correlated_noise",
        "dataset_variant": f"rho{noise_rho:g}_block{noise_block_size}_noise{noise_scale:g}",
        "K_true": int(K),
        "n_signal": int(n_signal),
        "noise_rho": float(noise_rho),
        "noise_block_size": int(noise_block_size),
        "noise_scale": float(noise_scale),
        "signal_scale": float(signal_scale),
    }



def generate_rotated_signal_data(
    N: int = 200,
    D: int = 100,
    n_signal: int = 10,
    latent_signal_dim: int = 3,
    K: int = 3,
    signal_means: Optional[np.ndarray] = None,
    signal_scale: float = 1.0,
    noise_scale: float = 3.0,
    rotate_within_signal_block: bool = True,
    standardize: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Optional misspecified scenario 3:
    cluster structure is generated in a low-dimensional latent subspace and then
    rotated before observation.

    When rotate_within_signal_block=True, the informative support remains the
    first n_signal dimensions, so feature F1 is still interpretable.
    """
    if not (1 <= latent_signal_dim <= n_signal <= D):
        raise ValueError("Require 1 <= latent_signal_dim <= n_signal <= D.")

    rng = np.random.default_rng(random_state)
    if signal_means is None:
        signal_means = _default_signal_means(K)
    signal_means = np.asarray(signal_means, dtype=float)
    if len(signal_means) != K:
        raise ValueError("len(signal_means) must equal K.")

    cluster_sizes = _balanced_cluster_sizes(N=N, K=K)
    truth_mask = np.zeros(D, dtype=int)
    truth_mask[:n_signal] = 1

    # Latent cluster means live in a lower-dimensional subspace.
    latent_means = np.zeros((K, latent_signal_dim), dtype=float)
    latent_means[:, 0] = signal_means
    if latent_signal_dim >= 2 and K >= 3:
        latent_means[:, 1] = np.array([1.5, 0.0, -1.5])[:K]

    Q, _ = np.linalg.qr(rng.normal(size=(n_signal, n_signal)))
    if rotate_within_signal_block:
        rot = Q
    else:
        # Full rotation over all D observed coordinates; feature support then
        # becomes diffuse, so truth_mask is not a faithful sparse target.
        Q_full, _ = np.linalg.qr(rng.normal(size=(D, D)))
        rot = Q_full[:n_signal, :n_signal]

    X_list, y_list = [], []
    for k, nk in enumerate(cluster_sizes):
        latent = rng.normal(size=(nk, latent_signal_dim)) * signal_scale
        latent += latent_means[k]

        block = np.zeros((nk, n_signal), dtype=float)
        block[:, :latent_signal_dim] = latent
        block = block @ rot

        Xk = np.zeros((nk, D), dtype=float)
        Xk[:, :n_signal] = block
        if D > n_signal:
            Xk[:, n_signal:] = rng.normal(scale=noise_scale, size=(nk, D - n_signal))

        X_list.append(Xk)
        y_list.append(np.full(nk, k, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    perm = rng.permutation(N)
    X = X[perm]
    y = y[perm]

    if standardize:
        X = standardize_features(X)

    return {
        "X": X.astype(float),
        "y": y.astype(int),
        "truth_mask": truth_mask,
        "dataset": "synthetic_rotated_signal",
        "dataset_variant": f"latent{latent_signal_dim}_noise{noise_scale:g}",
        "K_true": int(K),
        "n_signal": int(n_signal),
        "latent_signal_dim": int(latent_signal_dim),
        "noise_scale": float(noise_scale),
        "signal_scale": float(signal_scale),
    }


SCENARIO_GENERATORS = {
    "heavy_tail_signal": generate_heavy_tail_signal_data,
    "correlated_noise": generate_correlated_noise_data,
    "rotated_signal": generate_rotated_signal_data,
}
