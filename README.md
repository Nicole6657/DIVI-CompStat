# DIVI — Data-Informed Feature-Gated Clustering for Noisy High-Dimensional Data

Code and processed result summaries for

> W. P. Chen. *Data-Informed Feature-Gated Clustering for Noisy High-Dimensional Data.*
> Submitted to **Computational Statistics** (manuscript COST-D-26-00254).

DIVI jointly learns cluster assignments, per-feature relevance in the **original**
coordinate space, and the number of mixture components, within a single
differentiable objective. A feature-gated diagonal Gaussian mixture attenuates
nuisance dimensions; a data-informed reference distribution (Step A) initializes
and regularizes the gates; and a diagnostic-triggered split mechanism grows the
component structure, optionally under safeguards that reduce irreversible
over-expansion.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Reproducing the paper](#reproducing-the-paper)
- [Implementation modules](#implementation-modules)
- [Results and configuration files](#results-and-configuration-files)
- [Datasets](#datasets)
- [Environment](#environment)
- [Known gaps](#known-gaps)
- [Citation](#citation)
- [License](#license)

---

## Installation

```bash
git clone <repository-url>
cd DIVI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Sparse K-means (SPKM) is run through R. Install the `sparcl` package once:

```bash
Rscript R/install_sparcl.R
```

Several scripts in `experiments/` import the core modules by bare name
(`from divi_core import DIVIClustering`), so add the package directory to the
module search path before running them:

```bash
export PYTHONPATH="$PWD/divi"
```

All commands in [Reproducing the paper](#reproducing-the-paper) assume this has
been set and that they are run from the repository root.

---

## Quick start

```python
from divi import DIVIClustering, generate_synthetic_data_sweet_spot

X, y, informative = generate_synthetic_data_sweet_spot(N=1000, D=100, random_state=0)

model = DIVIClustering(max_epochs=300, split_interval=120, lr=0.01, beta_mult=1.0)
model.fit(X)

labels    = model.labels_              # cluster assignments
relevance = model.feature_relevance_   # per-feature gate probability in [0, 1]
K_final   = model.n_components_        # number of components after split growth
```

The learned relevance scores are most reliable as a **ranking**. The fixed
threshold `relevance >= 0.5` is not calibrated when `D` is large relative to
`N`; see Sections 4.2–4.5 of the paper.

---

## Repository layout

```
DIVI/
├── divi/                  Core package (runtime, sensitivity, real-data experiments)
│   ├── divi_core.py         DIVIClustering, DiagnosableGMM
│   ├── data_utils.py        synthetic generators, ISOLET, 20NG loaders
│   ├── robust_synth_utils.py  heavy-tailed and correlated-noise generators
│   ├── baselines.py         K-means, diagonal GMM, SPKM wrapper
│   ├── experiment_utils.py  seeding, metrics, CSV logging, summaries
│   └── configs/defaults_divi.yaml
├── divi_legacy/           Implementations used by the simulation experiments
├── experiments/           One script per paper table or figure
├── R/                     R scripts: sparcl bridge, PBMC preprocessing
├── results/               Raw runs, summaries, configs, aggregated outputs
└── requirements.txt
```

---

## Reproducing the paper

Every command below writes to `results/`. The committed contents of `results/`
are the outputs that produced the reported tables and figures; re-running
overwrites them.

> **Legacy module paths.** Scripts in `experiments/` that take a `--divi_path`
> (or `--hrdivi_path`) argument load an implementation from `divi_legacy/` by
> file path. The filenames in `divi_legacy/` are kept as they appear in the
> archived `*_config.json` files, so that recorded paths remain interpretable.

| Paper item | Command | Output directory |
|---|---|---|
| Table 2 — matched benchmark | `python experiments/table02_matched.py` | `results/table1_current120_sparcl/` |
| Table 2 — two-stage rows | `python experiments/table02_two_stage.py` | `results/two_stage_baselines/` |
| Tables 3–4, S1–S2 — scaling and weak signal | `python experiments/table03_04_scaling_weak.py --divi_path divi_legacy/DIVI_HAR.py` | `results/divi_mlwa_supplement_results/` |
| Table 4 — Laplacian Score rows | `python experiments/table02_two_stage.py --weak-signal` | `results/weak_signal_laplacian_gmm/` |
| Table 5 — Step A diagnostics | `python experiments/table05_stepA.py --divi_path divi_legacy/divi_core_fixedk.py` | `results/stepA_prior_diagnostics_delta10/` |
| Table 6 — cluster-specific relevance | `python experiments/table06_cluster_specific.py --hrdivi_path divi_legacy/hr_divi_core_fixedk_stable.py` | `results/cluster_specific_delta10/` |
| Table 7 — misspecification (DIVI, K-means, GMM) | `python experiments/table07_misspecification.py --scenarios heavy_tail_signal correlated_noise --N-list 200 1000 --D 100 --n-signal 10 --n-runs 20 --prior-modes 1 2 3 --include-baselines --tau-mult 1.00 --split-interval 120 --config divi/configs/defaults_divi.yaml --output-dir results/misspec_robustness_tuned` | `results/misspec_robustness_tuned/` |
| Table 7 — SPKM rows | `python experiments/table07_spkm.py` | `results/spkm_misspecification_results_v3/` |
| Table 8, S11 — component ablation | `python experiments/table08_component_ablation.py --original_divi divi_legacy/divi_mlwa.py --safeguarded_divi divi_legacy/divi_mlwa_safeguarded.py --fixedk_divi divi_legacy/divi_core_fixedk.py` | `results/component_ablation/` |
| Table 9, S10 — split-safeguard ablation | `python experiments/table09_split_safeguard.py --original_divi divi_legacy/divi_mlwa.py --safeguarded_divi divi_legacy/divi_mlwa_safeguarded.py` | `results/split_safeguard_ablation/` |
| Table 10 — paired comparisons | `python experiments/table10_paired.py` | `results/paired_tests/results/` |
| Table S11 — ablation paired tests | `python experiments/s11_ablation_paired.py` | `results/component_ablation/paired_tests/` |
| Tables 11–12 — PBMC-SCT | `python experiments/table11_12_pbmc.py` | `results/pbmc_sct_results/` |
| Table 13 — ISOLET / 20NG runtime | `python experiments/table13_runtime_real.py --dataset isolet`<br>`python experiments/table13_runtime_real.py --dataset 20ng` | `results/runtime_real/` |
| Figure 1 — Wine relevance | `python experiments/fig01_wine.py` | `results/wine_divi_results/` |
| Figure 2 — runtime scaling | `python experiments/fig02_runtime_synth.py --config divi/configs/defaults_divi.yaml` | `results/runtime_synth/` |
| Figure 3, Tables S3–S7 — sensitivity | `python experiments/fig03_sensitivity.py --factor Tsplit --values 10,20,40,80 --outdir results/sensitivity_synth --n-list 200,1000 --D 100 --n-signal 10 --noise-scale 3.0 --n-runs 10 --prior-mode 1 --beta-mult 1.0 --split-interval 80 --max-epochs 300 --lr 0.01 --temp-start 1.0 --temp-end 0.1`<br>Repeat with `--factor beta_mult --values 0.25,0.5,1.0,2.0,4.0`, `--factor tau_mult --values 0.9,1.0,1.1,1.2`, `--factor lr --values 0.005,0.01,0.02,0.05`, `--factor temp_end --values 0.01,0.05,0.10,0.20`. | `results/sensitivity_synth/<factor>/` |

### Aggregation

After the runtime and sensitivity runs have completed, build the paper tables,
LaTeX fragments and figures in one step:

```bash
python experiments/make_paper_outputs.py \
    --root results --outdir results/aggregated --make-plots
```

This produces `results/aggregated/`:

- `master_runs.csv` — every run, one row each
- `paper_tables/*.csv` — formatted tables (mean and standard deviation)
- `latex/*.tex` — booktabs table fragments
- `figure_csv/*.csv` — long-format data behind each figure
- `figures/*.pdf`, `*.png` — Figures 2 and 3

### Supporting analyses

The conservative schedule `(tau_mult, T_split) = (1.00, 120)` used in the
misspecification experiments was selected from a grid over the split-threshold
multiplier and the split interval. This supports the statement in Section 3.6 of
the paper but is not reported as a table:

```bash
python experiments/run_divi_tau_interval_sensitivity.py \
    --scenarios heavy_tail_signal correlated_noise \
    --N-list 200 1000 --D 100 --n-signal 10 --n-runs 20 \
    --prior-modes 1 \
    --tau-mults 1.00 1.05 1.10 1.20 \
    --split-intervals 80 120 160 \
    --config divi/configs/defaults_divi.yaml \
    --output-dir results/tau_interval_sensitivity
```

Output: `results/tau_interval_sensitivity/` (960 runs — 4 multipliers x 3
intervals x 2 scenarios x 2 sample sizes x 20 replicates).

---

## Implementation modules

The manuscript describes one method with two structural-growth protocols
(direct and safeguarded). Several implementation files exist because the
experiments were developed incrementally. **The model and objective are the same
in all of them**; they differ only in which structural-growth path is enabled and
whether the component count is fixed externally.

| Module | Growth protocol | Component count | Used by |
|---|---|---|---|
| `divi/divi_core.py` | direct | schedule-limited | Table 7, Table 13, Figures 2–3, Tables S3–S7 |
| `divi_legacy/divi_mlwa.py` | direct, unsafeguarded | schedule-limited or capped | Table 6; `Unsafe-Split` and `Original split` variants of Tables 8–9 |
| `divi_legacy/divi_mlwa_safeguarded.py` | safeguarded (burn-in, persistence, minimum size, matched-budget acceptance) | adaptive, `K_max = 8` | Tables 8–9 |
| `divi_legacy/DIVI_HAR.py` | direct, capped at the true `K` | `K_max = K_true = 3` | Tables 3–4 |
| `divi_legacy/divi_core_fixedk.py` | none (no splitting) | fixed, oracle `K` | Table 5; `Fixed-K` variants of Table 8 |
| `divi_legacy/hr_divi_core_fixedk_stable.py` | direct | schedule-limited | Table 6 |

Learning rates and structural parameters differ by experiment family and are
recorded per run; see the next section.

---

## Results and configuration files

Each experiment directory under `results/` contains

- `*_raw.csv` or `runs.csv` — one row per replicate, with every hyperparameter
  used for that run (`lr`, `beta_mult`, `Tsplit`, `tau_mult`, `max_epochs`,
  `temp_start`, `temp_end`, seeds, timings, peak memory);
- `*_summary.csv` — means and standard deviations over replicates, matching the
  values printed in the manuscript;
- `*_config.json` — the complete invocation configuration;
- `*_table.tex` — a LaTeX fragment of the corresponding table, where produced.

Two points worth noting:

- `results/component_ablation/component_ablation_failures.csv` contains only a
  header row: **no run failed** in the component ablation.
- The safeguard parameters reported in the paper are recorded in
  `results/split_safeguard_ablation/split_safeguard_config.json` and
  `results/component_ablation/component_ablation_config.json`:
  burn-in `B = 120`, persistence `P = 2`, minimum component size
  `n_min = 25`, matched budget `W = 20`, acceptance margin
  `delta_split = 0.001`, with `T_split = 60`, `E = 300`, `K_max = 8`.

---

## Datasets

| Dataset | Size | Source |
|---|---|---|
| Synthetic | generated at run time | `divi/data_utils.py`; seeds recorded in each `runs.csv` |
| Wine | N = 178, D = 13 | UCI Machine Learning Repository |
| ISOLET (letters A–E) | N = 1560, D = 617 | OpenML, `data_id=300` (downloaded automatically) |
| 20 Newsgroups | N = 2000, D = 384 | scikit-learn `fetch_20newsgroups`, embedded with `all-MiniLM-L6-v2` |
| PBMC-SCT | N = 2638, D = 2000 | 10x Genomics `pbmc3k`; see below |

### PBMC preprocessing

Raw and preprocessed single-cell matrices are **not committed** (the expression
matrix alone is ~95 MB). `R/pbmc/` regenerates them from the public 10x
`pbmc3k` download:

```bash
Rscript R/pbmc/preprocess_pbmc_sctransform_for_divi.R
Rscript R/pbmc/make_pbmc_labels_with_seurat_mapping.R
Rscript R/pbmc/run_pbmc_classic_baselines_from_csv.R   # K-means and GMM rows of Table 11
```

Preprocessing settings, as recorded in the `meta_pbmc_sct.json` written by the
first script:
QC retains cells with 200–2500 detected features and mitochondrial content
below 5%; SCTransform regresses out `percent.mt` with `clip.range` 10 and
`seed = 1`; the top 2000 highly variable genes are retained. Reference
cell-type labels are obtained by Seurat label transfer from the PBMCsca
reference and are used **only** for external evaluation, never during fitting.

---

## Environment

All experiments were executed on **Google Colab** (CPU). Two Python versions
appear across the runs, which were carried out at different dates:

| Experiments | Python |
|---|---|
| Table 7, Table 13, Figures 2–3, Tables S3–S7 | 3.13 |
| Tables 2–6, 8–12, Figure 1 | 3.12 |

Sparse K-means is fitted in R. These versions were recorded by `sessionInfo()`
during the Table 2 run (`DIVI_CSDA.ipynb`, cell 29):

```
R version 4.5.3 (2026-03-11)
sparcl 1.0.4
jsonlite 2.0.0
```

**Python package versions were not pinned at the time of the original runs.**
Colab resolves dependencies at session start, so the same notebook executed on a
different date installs a different set of wheels, and no lockfile exists for the
sessions that produced the committed results. Rather than reconstruct versions
after the fact, the repository records the environment of a *verification* run:

```bash
python tools/capture_environment.py
```

This writes `ENVIRONMENT.md` and `requirements-frozen.txt`. `requirements.txt`
lists direct dependencies only and is not a lockfile.

### Verification

The split-interval sensitivity grid — the experiment behind Figure 3 and
Table S3 — can be re-executed and compared against the committed results
automatically:

```bash
python tools/verify_reproduction.py
```

The grid takes a few minutes on CPU and exercises the whole `divi/` package:
data generation, the Step A gate reference, the gated mixture, split growth and
the metric pipeline. Structural quantities (`final_K`, `split_count`) are
compared exactly, since they have zero replicate variance in the committed run;
ARI, NMI and feature F1 are compared with a tolerance of 0.01.

A verification run was performed on a fresh Colab session, months after the
original execution and therefore with independently resolved package versions.
All 48 compared quantities agreed **to four decimal places**, including the
terminal component counts (31, 16, 8, 4) and the mean active-set sizes. The
environment of that run is recorded in `ENVIRONMENT.md`.

This is a same-platform check across sessions and package versions, not a
cross-platform one; see [Known gaps](#known-gaps).

All runs used fixed seeds, recorded in the `data_seed` and `run_seed` columns of
each `runs.csv`. Results are reported as means with standard deviations over
20 independently generated datasets, except for the high-dimensional scaling and
weak-signal studies (10 datasets) and the real-data experiments (5 seeds).

Exact package versions for the release environment are pinned in
`requirements-frozen.txt`; `requirements.txt` lists direct dependencies only.

---

## Known gaps

Reported here rather than left for the reader to discover.

1. **Reproducibility has been verified within Colab, not across platforms.**
   The check described under [Verification](#verification) reproduced the
   committed results exactly on a fresh Colab session with independently
   resolved package versions. It does not establish invariance to different
   hardware, BLAS backends, or PyTorch builds. Because structural decisions are
   discrete and irreversible, a numerical difference large enough to move a
   split diagnostic across its threshold would change the terminal component
   count rather than perturb the metrics slightly.
2. **`divi/robust_synth_utils.py` also provides a `rotated_signal` scenario**
   that is not reported in the paper. It is retained because it is part of the
   `SCENARIO_GENERATORS` registry used by
   `experiments/table07_misspecification.py`; only `heavy_tail_signal` and
   `correlated_noise` correspond to Table 7.

---

## Citation

```bibtex
@article{chen2026divi,
  author  = {Chen, Wan Ping},
  title   = {Data-Informed Feature-Gated Clustering for Noisy High-Dimensional Data},
  journal = {Computational Statistics},
  year    = {2026},
  note    = {Manuscript COST-D-26-00254}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
