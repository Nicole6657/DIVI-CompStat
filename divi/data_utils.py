from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.datasets import fetch_20newsgroups, fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize


def standardize_features(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def generate_synthetic_data_sweet_spot(
    N: int = 100,
    D: int = 100,
    n_signal: int = 10,
    signal_means: Optional[np.ndarray] = None,
    noise_scale: float = 3.0,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorganized from StabilityRandomSeed.py and n200andSPKM.py:
    - 3 clusters
    - first n_signal dimensions are informative
    - remaining dimensions are high-variance noise
    """
    rng = np.random.default_rng(random_state)

    if signal_means is None:
        signal_means = np.array([-2.0, 0.0, 2.0])

    means = np.array(
        [[m] * n_signal + [0.0] * (D - n_signal) for m in signal_means],
        dtype=float,
    )
    n_samples = [N // 3, N // 3, N - 2 * (N // 3)]

    X_list, y_list = [], []
    for k, n in enumerate(n_samples):
        data = rng.normal(size=(n, D))
        data += means[k]
        data[:, n_signal:] *= noise_scale
        X_list.append(data)
        y_list.append(np.full(n, k))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(N)
    return X[idx], y[idx]


def load_isolet_subset(
    target_classes: int = 5,
    standardize: bool = True,
) -> Dict[str, np.ndarray]:
    isolet = fetch_openml(data_id=300, as_frame=False, parser="auto")
    X_raw = isolet.data
    y_raw = isolet.target

    le = LabelEncoder()
    y_all = le.fit_transform(y_raw)

    mask = y_all < target_classes
    X = X_raw[mask]
    y = y_all[mask]

    if standardize:
        X = StandardScaler().fit_transform(X)

    return {
        "X": X.astype(float),
        "y": y.astype(int),
        "dataset": f"isolet_first_{target_classes}",
        "target_names": [str(x) for x in le.classes_[:target_classes]],
    }


def load_20ng_subset_embeddings(
    categories: Optional[List[str]] = None,
    subset: str = "train",
    max_docs: int = 2000,
    model_name: str = "all-MiniLM-L6-v2",
    data_home: Optional[str] = None,
    download_if_missing: bool = True,
    l2_normalize: bool = True,
    standardize: bool = True,
) -> Dict[str, np.ndarray]:
    if categories is None:
        categories = [
            "sci.space",
            "rec.autos",
            "talk.politics.mideast",
            "comp.graphics",
        ]

    dataset = fetch_20newsgroups(
        data_home=data_home,
        subset=subset,
        categories=categories,
        download_if_missing=download_if_missing,
        shuffle=True,
        random_state=42,
    )
    texts = dataset.data[:max_docs]
    y = dataset.target[:max_docs]

    from sentence_transformers import SentenceTransformer  # local import

    encoder = SentenceTransformer(model_name)
    X = encoder.encode(texts, show_progress_bar=False)

    if l2_normalize:
        X = normalize(X, norm="l2")
    if standardize:
        X = StandardScaler().fit_transform(X)

    return {
        "X": X.astype(float),
        "y": np.asarray(y, dtype=int),
        "dataset": f"20ng_{subset}_{len(categories)}cats_{max_docs}docs",
        "target_names": list(dataset.target_names),
        "texts": texts,
    }
