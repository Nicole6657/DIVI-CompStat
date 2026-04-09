# DIVI: A Data-Informed Variational Clustering Framework for Noisy High-Dimensional Data

This repository contains code and result-generation scripts for the paper:

**Wan Ping Chen**  
**A Data-Informed Variational Clustering Framework for Noisy High-Dimensional Data**  
Submitted to *Computational Statistics & Data Analysis (CSDA)*.

## Overview

DIVI is a practical variational clustering framework for noisy high-dimensional data.  
It combines:

- data-informed prior initialization,
- differentiable global feature gating, and
- split-based adaptive structure growth.

The repository supports the main experiments reported in the manuscript, including:

- matched synthetic experiments,
- misspecified robustness experiments,
- real-data experiments,
- runtime scaling analyses, and
- sensitivity analyses.

## Repository Contents

### Core modules
- `divi_core.py` — main DIVI model and training logic
- `baselines.py` — baseline clustering methods
- `experiment_utils.py` — experiment logging and summary utilities
- `robust_synth_utils.py` — misspecified synthetic-data generators
- `data_utils.py` — dataset loading and preprocessing utilities

### Experiment scripts
- `run_divi_misspec_robustness.py`
- `run_divi_misspec_robustness_tuned.py`
- `run_divi_sensitivity_synth.py`
- `run_divi_sensitivity_real.py`
- `run_divi_tau_interval_sensitivity.py`
- `run_divi_runtime_synth.py`
- `run_divi_runtime_synth_direct.py`
- `run_divi_runtime_real.py`

### Output / aggregation
- `aggregate_divi_revision_results.py`
- `make_paper_outputs.py`

### Configuration
- `defaults_divi.yaml`

## Environment

This code was developed in Python 3.10+.

Install dependencies with:

```bash
pip install -r requirements.txt
```


## Data

This repository uses both synthetic and public real datasets.

### Synthetic data

Synthetic datasets are generated directly by the experiment scripts.

### Public real datasets

The real-data experiments use public datasets such as:

- UCI Wine
- ISOLET
- 20 Newsgroups (20NG)

Some scripts may require local preprocessing or precomputed embeddings.
Please check the relevant paths in the scripts and update them as needed for your environment.

## Running Main Experiments

### Misspecified robustness
```bash
python run_divi_misspec_robustness.py
```
### Tuned misspecified robustness
```bash
python run_divi_misspec_robustness_tuned.py
```
### Synthetic sensitivity analysis
```bash
python run_divi_sensitivity_synth.py
```
### Real-data sensitivity analysis
```bash
python run_divi_sensitivity_real.py
```
### Tau / split-interval sensitivity
```bash
python run_divi_tau_interval_sensitivity.py
```
### Runtime on synthetic data
```bash
python run_divi_runtime_synth.py
```
### Runtime on synthetic data (direct version)
```bash
python run_divi_runtime_synth_direct.py
```
### Runtime on real data
```bash
python run_divi_runtime_real.py
```

## Reproducing Paper Outputs

To aggregate experiment summaries and generate paper-ready outputs:

```bash
python aggregate_divi_revision_results.py
python make_paper_outputs.py
```

## Notes on Reproducibility

- Small numerical differences may occur across runs due to randomness and hardware differences.
- Please fix random seeds where applicable for closer reproduction.
- Runtime results depend on CPU/GPU, Python version, and package versions.
- The repository is intended to reproduce the main qualitative and quantitative results reported in the manuscript.

## Citation

If you use this repository, please cite the corresponding paper:

```bibtex
@article{chen2026divi,
  author  = {Wan Ping Chen},
  title   = {A Data-Informed Variational Clustering Framework for Noisy High-Dimensional Data},
  journal = {Computational Statistics \& Data Analysis},
  year    = {2026},
  note    = {submitted}
}
```

## License

This repository is released under the MIT License. See the LICENSE file for details.