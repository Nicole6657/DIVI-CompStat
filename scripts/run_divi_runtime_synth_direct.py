#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------
# Robust local imports for Colab
# -----------------------------
THIS_FILE = Path(__file__).resolve()
SEARCH_DIRS = [THIS_FILE.parent, Path.cwd(), Path('/content')]
for p in SEARCH_DIRS:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from baselines import run_gmm, run_kmeans, run_spkm
    from data_utils import generate_synthetic_data_sweet_spot, standardize_features
    from divi_core import DIVIClustering
    from experiment_utils import (
        COMMON_COLUMNS,
        append_run_record,
        calculate_feature_f1_from_phi,
        calculate_feature_f1_from_selected_mask,
        clustering_metrics,
        ensure_dir,
        get_device_info,
        get_git_commit,
        get_peak_memory_mb,
        load_yaml_config,
        make_experiment_id,
        safe_json,
        set_global_seed,
        summarize_results,
        utc_timestamp,
    )
except Exception as e:
    raise ImportError(
        "Cannot import local project modules. Make sure baselines.py, data_utils.py, divi_core.py, "
        "and experiment_utils.py are in the same folder as this script or in the current working directory."
    ) from e


def parse_int_list(x: str) -> list[int]:
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def resolve_config_path(config_arg: str | None) -> str | None:
    if config_arg:
        p = Path(config_arg)
        if p.exists():
            return str(p)
        candidates = [
            Path.cwd() / config_arg,
            THIS_FILE.parent / config_arg,
            Path('/content') / config_arg,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
        raise FileNotFoundError(f"Config file not found: {config_arg}")

    default_candidates = [
        Path.cwd() / 'configs' / 'defaults_divi.yaml',
        THIS_FILE.parent / 'configs' / 'defaults_divi.yaml',
        Path('/content/configs/defaults_divi.yaml'),
        Path('/content/defaults_divi.yaml'),
        Path('/content/.config/defaults_divi.yaml'),
    ]
    for cand in default_candidates:
        if cand.exists():
            return str(cand)
    return None


def apply_config(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    for k, v in cfg.items():
        key = k.replace('-', '_')
        if hasattr(args, key):
            setattr(args, key, v)
    return args


def run_divi_once(X: np.ndarray, args: argparse.Namespace, run_seed: int) -> DIVIClustering:
    set_global_seed(run_seed)
    model = DIVIClustering(
        split_threshold=None,
        split_interval=args.split_interval,
        max_epochs=args.max_epochs,
        lr=args.lr,
        beta_mult=args.beta_mult,
        temperature_start=args.temp_start,
        temperature_end=args.temp_end,
        temperature_decay=getattr(args, 'temperature_decay', None),
        split_perturb_scale=getattr(args, 'split_perturb_scale', 0.2),
        prior_logvar0=getattr(args, 'prior_logvar0', 2.197),
        verbose=False,
    )
    model.fit(X, use_prior=args.prior_mode)
    return model


def build_record_common(
    *,
    experiment_id: str,
    script_name: str,
    git_commit: str,
    device_type: str,
    device_name: str,
    dataset_variant: str,
    split_name: str,
    N: int,
    D: int,
    K_true: int,
    n_signal: int,
    noise_scale: float,
    data_seed: int,
    run_seed: int,
    method: str,
    baseline_name: str,
) -> dict[str, Any]:
    return {
        'experiment_id': experiment_id,
        'script_name': script_name,
        'timestamp': utc_timestamp(),
        'git_commit': git_commit,
        'hostname': device_name,
        'device_type': device_type,
        'device_name': device_name,
        'dataset': 'synthetic',
        'dataset_variant': dataset_variant,
        'split_name': split_name,
        'N': N,
        'D': D,
        'K_true': K_true,
        'informative_ratio': n_signal / D,
        'noise_ratio': 1.0 - n_signal / D,
        'noise_sigma': noise_scale,
        'data_seed': data_seed,
        'run_seed': run_seed,
        'method': method,
        'baseline_name': baseline_name,
        'peak_memory_mb': get_peak_memory_mb(),
        'status': 'ok',
        'timeout_flag': False,
        'error_message': '',
        'artifact_dir': '',
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Run synthetic runtime/scalability experiments for DIVI.')
    ap.add_argument('--config', type=str, default=None)
    ap.add_argument('--outdir', type=str, default='outputs_revision/runtime_synth')
    ap.add_argument('--n-list', type=str, default='200,500,1000,2000,5000')
    ap.add_argument('--d-list', type=str, default='100,500,1000,2000,5000')
    ap.add_argument('--n-runs', type=int, default=5)
    ap.add_argument('--data-seed-d', type=int, default=20260325)
    ap.add_argument('--data-seed-n', type=int, default=20260326)
    ap.add_argument('--fixed-n', type=int, default=1000)
    ap.add_argument('--fixed-d', type=int, default=1000)
    ap.add_argument('--K-true', type=int, default=3)
    ap.add_argument('--D-signal', type=int, default=10)
    ap.add_argument('--noise-scale', type=float, default=3.0)
    ap.add_argument('--prior-mode', type=int, default=1)
    ap.add_argument('--beta-mult', type=float, default=1.0)
    ap.add_argument('--split-interval', type=int, default=80)
    ap.add_argument('--max-epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--temp-start', type=float, default=1.0)
    ap.add_argument('--temp-end', type=float, default=0.1)
    ap.add_argument('--temperature-decay', type=float, default=None)
    ap.add_argument('--split-perturb-scale', type=float, default=0.2)
    ap.add_argument('--prior-logvar0', type=float, default=2.197)
    ap.add_argument('--spkm-l1-bound', type=float, default=4.0)
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config)
    if cfg_path is not None:
        cfg = load_yaml_config(cfg_path)
        args = apply_config(args, cfg)
        print(f'[config] using: {cfg_path}')
    else:
        print('[config] no config file found; using CLI/default arguments only')

    outdir = ensure_dir(args.outdir)
    runs_csv = outdir / 'runs.csv'
    summary_csv = outdir / 'summary.csv'

    device_type, device_name = get_device_info()
    git_commit = get_git_commit(cwd=str(Path.cwd()))
    script_name = Path(__file__).name

    n_list = parse_int_list(args.n_list)
    d_list = parse_int_list(args.d_list)

    columns = COMMON_COLUMNS + [
        'runtime_axis', 'runtime_value', 'baseline_timeout_sec', 'baseline_converged_flag'
    ]

    for axis, values in [('D', d_list), ('N', n_list)]:
        for value in values:
            if axis == 'D':
                N, D = args.fixed_n, value
                data_seed = args.data_seed_d + value
            else:
                N, D = value, args.fixed_d
                data_seed = args.data_seed_n + value

            X_raw, y_true = generate_synthetic_data_sweet_spot(
                N=N,
                D=D,
                n_signal=args.D_signal,
                noise_scale=args.noise_scale,
                random_state=data_seed,
            )
            X = standardize_features(X_raw)
            dataset_variant = f'synthetic_n{N}_d{D}_noise{args.noise_scale}'
            split_name = f'runtime_{axis.lower()}scale'

            print(f'\n[{axis}-scale] value={value} | N={N}, D={D}, data_seed={data_seed}')

            for run_seed in range(args.n_runs):
                print(f'  seed={run_seed} ...')

                # DIVI
                divi = run_divi_once(X, args, run_seed)
                y_pred = divi.predict(X)
                phi = divi.get_phi()
                m = clustering_metrics(y_true, y_pred)
                f1_feat = calculate_feature_f1_from_phi(phi, n_signal=args.D_signal, threshold=0.5)
                rec = build_record_common(
                    experiment_id=make_experiment_id('runtime_synth', dataset_variant, split_name, run_seed, 'divi'),
                    script_name=script_name,
                    git_commit=git_commit,
                    device_type=device_type,
                    device_name=device_name,
                    dataset_variant=dataset_variant,
                    split_name=split_name,
                    N=N,
                    D=D,
                    K_true=args.K_true,
                    n_signal=args.D_signal,
                    noise_scale=args.noise_scale,
                    data_seed=data_seed,
                    run_seed=run_seed,
                    method='divi',
                    baseline_name='divi',
                )
                rec.update({
                    'prior_mode': args.prior_mode,
                    'beta': divi.fit_summary_['beta'],
                    'beta_mult': args.beta_mult,
                    'Tsplit': args.split_interval,
                    'tau_mode': 'auto_entropy',
                    'tau': divi.split_threshold,
                    'tau_mult': 1.0,
                    'lr': args.lr,
                    'temp_start': args.temp_start,
                    'temp_end': args.temp_end,
                    'max_epochs': args.max_epochs,
                    'epochs_completed': divi.fit_summary_['epochs_completed'],
                    'final_K': divi.fit_summary_['final_K'],
                    'split_count': divi.fit_summary_['split_count'],
                    'first_split_epoch': divi.fit_summary_['first_split_epoch'],
                    'last_split_epoch': divi.fit_summary_['last_split_epoch'],
                    'split_epochs_json': safe_json(divi.fit_summary_['split_epochs']),
                    'ari': m['ari'],
                    'nmi': m['nmi'],
                    'f1_feature': f1_feat,
                    'acc': m['acc'],
                    'selected_dims_count': divi.fit_summary_['selected_dims_count'],
                    'selected_dims_ratio': divi.fit_summary_['selected_dims_ratio'],
                    'mean_phi': divi.fit_summary_['mean_phi'],
                    'median_phi': divi.fit_summary_['median_phi'],
                    'objective_final': divi.fit_summary_['objective_final'],
                    'objective_best': divi.fit_summary_['objective_best'],
                    'objective_gap': divi.fit_summary_['objective_gap'],
                    'nll_final': divi.fit_summary_['nll_final'],
                    'kl_final': divi.fit_summary_['kl_final'],
                    'wallclock_total_sec': divi.fit_summary_['wallclock_total_sec'],
                    'wallclock_stepA_sec': divi.fit_summary_['wallclock_stepA_sec'],
                    'wallclock_train_sec': divi.fit_summary_['wallclock_train_sec'],
                    'wallclock_split_diag_sec': divi.fit_summary_['wallclock_split_diag_sec'],
                    'wallclock_post_sec': divi.fit_summary_['wallclock_post_sec'],
                    'time_per_epoch_sec': divi.fit_summary_['time_per_epoch_sec'],
                    'runtime_axis': axis,
                    'runtime_value': value,
                    'baseline_timeout_sec': np.nan,
                    'baseline_converged_flag': True,
                })
                append_run_record(runs_csv, rec, columns=columns)

                # Baselines
                runners = {
                    'kmeans_oracle': lambda: run_kmeans(X, args.K_true, run_seed),
                    'gmm_oracle': lambda: run_gmm(X, args.K_true, run_seed),
                    'spkm': lambda: run_spkm(X, args.K_true, args.spkm_l1_bound),
                }
                for baseline_name, runner in runners.items():
                    t0 = pd.Timestamp.utcnow()
                    result = runner()
                    elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
                    mb = clustering_metrics(y_true, result.labels)
                    if result.selected_mask is not None:
                        selected_count = int(np.asarray(result.selected_mask).sum())
                        selected_ratio = float(np.asarray(result.selected_mask).mean())
                        f1_b = calculate_feature_f1_from_selected_mask(np.asarray(result.selected_mask), n_signal=args.D_signal)
                    else:
                        selected_count = np.nan
                        selected_ratio = np.nan
                        f1_b = np.nan

                    rec_b = build_record_common(
                        experiment_id=make_experiment_id('runtime_synth', dataset_variant, split_name, run_seed, baseline_name),
                        script_name=script_name,
                        git_commit=git_commit,
                        device_type=device_type,
                        device_name=device_name,
                        dataset_variant=dataset_variant,
                        split_name=split_name,
                        N=N,
                        D=D,
                        K_true=args.K_true,
                        n_signal=args.D_signal,
                        noise_scale=args.noise_scale,
                        data_seed=data_seed,
                        run_seed=run_seed,
                        method=baseline_name,
                        baseline_name=baseline_name,
                    )
                    rec_b.update({
                        'prior_mode': np.nan,
                        'beta': np.nan,
                        'beta_mult': np.nan,
                        'Tsplit': np.nan,
                        'tau_mode': '',
                        'tau': np.nan,
                        'tau_mult': np.nan,
                        'lr': np.nan,
                        'temp_start': np.nan,
                        'temp_end': np.nan,
                        'max_epochs': np.nan,
                        'epochs_completed': np.nan,
                        'final_K': args.K_true,
                        'split_count': 0,
                        'first_split_epoch': np.nan,
                        'last_split_epoch': np.nan,
                        'split_epochs_json': '[]',
                        'ari': mb['ari'],
                        'nmi': mb['nmi'],
                        'f1_feature': f1_b,
                        'acc': mb['acc'],
                        'selected_dims_count': selected_count,
                        'selected_dims_ratio': selected_ratio,
                        'mean_phi': np.nan,
                        'median_phi': np.nan,
                        'objective_final': np.nan,
                        'objective_best': np.nan,
                        'objective_gap': np.nan,
                        'nll_final': np.nan,
                        'kl_final': np.nan,
                        'wallclock_total_sec': elapsed,
                        'wallclock_stepA_sec': np.nan,
                        'wallclock_train_sec': elapsed,
                        'wallclock_split_diag_sec': np.nan,
                        'wallclock_post_sec': np.nan,
                        'time_per_epoch_sec': np.nan,
                        'runtime_axis': axis,
                        'runtime_value': value,
                        'baseline_timeout_sec': np.nan,
                        'baseline_converged_flag': True,
                    })
                    append_run_record(runs_csv, rec_b, columns=columns)

    summarize_results(
        runs_csv=runs_csv,
        group_cols=['dataset', 'method', 'runtime_axis', 'runtime_value'],
        out_csv=summary_csv,
    )
    print(f'\n[done] runs saved to: {runs_csv}')
    print(f'[done] summary saved to: {summary_csv}')


if __name__ == '__main__':
    main()
