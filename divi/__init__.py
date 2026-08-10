"""DIVI: data-informed feature-gated clustering for noisy high-dimensional data.

This package accompanies

    W. P. Chen, "Data-Informed Feature-Gated Clustering for Noisy
    High-Dimensional Data", Computational Statistics (COST-D-26-00254).

The modules in this package were originally written as flat scripts and import
one another by bare module name (``import divi_core``). To keep those imports
working unchanged -- and to keep the module paths recorded in the archived
``*_config.json`` files interpretable -- this file puts the package directory on
``sys.path`` before importing the submodules. Both of the following therefore
work:

    from divi import DIVIClustering          # package-style
    import divi_core                         # flat, as used inside the package

Typical use::

    from divi import DIVIClustering, generate_synthetic_data_sweet_spot

    X, y, informative = generate_synthetic_data_sweet_spot(N=1000, D=100)
    model = DIVIClustering(max_epochs=300, split_interval=120, lr=0.01)
    model.fit(X)
    labels = model.labels_
    relevance = model.feature_relevance_
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make the flat intra-package imports (``import divi_core`` etc.) resolvable
# regardless of the working directory the caller runs from.
_PKG_DIR = str(_Path(__file__).resolve().parent)
if _PKG_DIR not in _sys.path:
    _sys.path.insert(0, _PKG_DIR)

from divi_core import DIVIClustering, DiagnosableGMM, FitTimings  # noqa: E402
from data_utils import (  # noqa: E402
    generate_synthetic_data_sweet_spot,
    load_20ng_subset_embeddings,
    load_isolet_subset,
    standardize_features,
)
from robust_synth_utils import (  # noqa: E402
    SCENARIO_GENERATORS,
    generate_correlated_noise_data,
    generate_heavy_tail_signal_data,
    generate_rotated_signal_data,
)
from baselines import run_gmm, run_kmeans, run_spkm  # noqa: E402
from experiment_utils import (  # noqa: E402
    calculate_feature_f1_from_phi,
    calculate_feature_f1_from_selected_mask,
    clustering_accuracy,
    clustering_metrics,
    set_global_seed,
)

__version__ = "1.0.0"

CONFIG_DIR = _Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "defaults_divi.yaml"

__all__ = [
    # core model
    "DIVIClustering",
    "DiagnosableGMM",
    "FitTimings",
    # data
    "generate_synthetic_data_sweet_spot",
    "load_isolet_subset",
    "load_20ng_subset_embeddings",
    "standardize_features",
    # misspecified-scenario generators (Table 7)
    "SCENARIO_GENERATORS",
    "generate_heavy_tail_signal_data",
    "generate_correlated_noise_data",
    "generate_rotated_signal_data",
    # baselines
    "run_kmeans",
    "run_gmm",
    "run_spkm",
    # metrics and utilities
    "clustering_metrics",
    "clustering_accuracy",
    "calculate_feature_f1_from_phi",
    "calculate_feature_f1_from_selected_mask",
    "set_global_seed",
    # paths
    "CONFIG_DIR",
    "DEFAULT_CONFIG",
    "__version__",
]
