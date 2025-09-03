#!/usr/bin/env python3
"""
Train and evaluate multiple regression models (Decision Tree, Random Forest,
XGBoost, SVR) to predict median COF for each of the four materials.

- Builds a per-run dataset from raw .dwf files under group folders: `0/`, `2.5/`, `5/`, `7.5/`.
- Features are parsed from file names (normal load N, sliding distance m/mm, speed rpm, duration min).
- Target is the run-level median COF computed from the time series (FF divided by N).
- Trains one model per material and produces:
  - Metrics CSVs per algorithm/material and a combined summary.
  - Scatter plots of y_true vs. y_pred per algorithm/material.
  - Variation (descriptive) stats and correlation analyses saved as CSV and heatmaps.

Usage:
  python3 ml_cof_models.py --base . --out results/ml --measure median

Notes:
- If xgboost is not installed, the XGBoost model is skipped automatically.
- With very small datasets per material, the script uses cross-validation
  predictions for evaluation and plotting.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer

# Optional: XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBRegressor = None  # type: ignore

# Local utilities
from analyze_tribology import read_dwf, label_map  # reuse robust reader and labels


def parse_metadata_from_name(fp: Path) -> Dict[str, Optional[float]]:
    """Parse experimental parameters from filename text.

    Tries to extract:
      - N (normal load, N)
      - distance_m (sliding distance, meters)
      - rpm (rotational speed)
      - duration_min (test duration, minutes)

    Returns values as floats (meters for distance). Missing entries become None.
    """
    s = fp.name
    meta: Dict[str, Optional[float]] = {
        'N': None,
        'distance_m': None,
        'rpm': None,
        'duration_min': None,
    }

    # Load N
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*[Nn]\b", s)
    if m:
        try:
            meta['N'] = float(m.group(1).replace(',', '.'))
        except Exception:
            pass

    # Distance (m or mm). Prioritize explicit units; avoid capturing 'min'.
    # common patterns: '140m', '120mm', '140 m', '120 mm'
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*mm\b", s)
    if m:
        try:
            meta['distance_m'] = float(m.group(1).replace(',', '.')) / 1000.0
        except Exception:
            pass
    else:
        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*m\b", s)
        if m:
            try:
                meta['distance_m'] = float(m.group(1).replace(',', '.'))
            except Exception:
                pass

    # RPM
    m = re.search(r"([0-9]+)\s*rpm\b", s, flags=re.IGNORECASE)
    if m:
        try:
            meta['rpm'] = float(m.group(1))
        except Exception:
            pass
    else:
        # fallback: last hyphen-separated token before extension that is purely digits
        toks = re.split(r"[-_]+", s)
        if len(toks) >= 2:
            candidate = toks[-1].split('.')[0]
            if candidate.isdigit():
                try:
                    meta['rpm'] = float(candidate)
                except Exception:
                    pass

    # Duration minutes
    m = re.search(r"([0-9]+)\s*(?:min|mins|minutes)\b", s, flags=re.IGNORECASE)
    if m:
        try:
            meta['duration_min'] = float(m.group(1))
        except Exception:
            pass

    return meta


def read_run_cof(fp: Path) -> pd.DataFrame:
    """Read a .dwf run and return DataFrame with columns ['TIME', 'COF'].

    Uses `read_dwf` to parse FF, then divides by N parsed from the filename.
    """
    meta = parse_metadata_from_name(fp)
    N = meta.get('N')
    if not N or N == 0:
        raise ValueError(f"Normal load could not be parsed from filename: {fp}")

    df = read_dwf(fp)
    if df is None or df.empty:
        raise ValueError(f"Empty or unreadable DWF: {fp}")

    out = pd.DataFrame()
    out['TIME'] = df['TIME'].astype(float)
    out['COF'] = (df['FF'].astype(float) / float(N))
    return out


def gather_groups(base_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for sub in ['0', '2.5', '5', '7.5']:
        d = base_dir / sub
        if d.is_dir():
            files = sorted(d.glob('*.dwf'))
            if files:
                groups[sub] = files
    if not groups:
        raise FileNotFoundError(f"No groups with .dwf files found under {base_dir}")
    return groups


def build_run_level_dataset(base_dir: Path) -> pd.DataFrame:
    """Build one row per run with features and target median COF.

    Features:
      - N, distance_m, rpm, duration_min
      - group (material group label as string)
    Target:
      - y: median COF across the entire run (robust to noise)
    """
    rows = []
    groups = gather_groups(base_dir)
    for g, files in groups.items():
        for fp in files:
            try:
                meta = parse_metadata_from_name(fp)
                ts = read_run_cof(fp)
                if ts is None or ts.empty:
                    continue
                y = float(np.nanmedian(ts['COF'].values))
                row = {
                    'group': g,
                    'file': str(fp.name),
                    'y': y,
                    'N': meta.get('N'),
                    'distance_m': meta.get('distance_m'),
                    'rpm': meta.get('rpm'),
                    'duration_min': meta.get('duration_min'),
                }
                rows.append(row)
            except Exception as e:
                print(f"WARN: Skipping {fp} due to error: {e}")
                continue
    if not rows:
        raise RuntimeError("No dataset rows built; check file patterns and metadata parsing.")
    df = pd.DataFrame(rows)
    return df


def compute_variation_stats(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return pd.Series({'count': 0, 'mean': np.nan, 'std': np.nan, 'var': np.nan, 'min': np.nan, 'q1': np.nan, 'median': np.nan, 'q3': np.nan, 'max': np.nan, 'cv_percent': np.nan})
    desc = s.describe()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    mean = desc['mean']
    std = s.std(ddof=1)
    cv = (std / mean * 100.0) if mean and mean != 0 else np.nan
    return pd.Series({
        'count': desc['count'], 'mean': mean, 'std': std, 'var': s.var(ddof=1),
        'min': desc['min'], 'q1': q1, 'median': s.median(), 'q3': q3, 'max': desc['max'],
        'cv_percent': cv,
    })


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'MSE': float(mean_squared_error(y_true, y_pred)),
        'RMSE': rmse,
        'MAPE_percent': mape(y_true, y_pred),
        'PearsonCorr': float(pd.Series(y_true).corr(pd.Series(y_pred), method='pearson')),
        'SpearmanCorr': float(pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')),
    }


def get_models(random_state: int = 42) -> Dict[str, Pipeline]:
    numeric_features = ['N', 'distance_m', 'rpm', 'duration_min']
    # Preprocessors
    prep_tree = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
    ])
    prep_svr = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric_features)
    ], remainder='drop')

    models: Dict[str, Pipeline] = {}

    # Decision Tree
    models['DecisionTree'] = Pipeline([
        ('prep', prep_tree),
        ('model', DecisionTreeRegressor(random_state=random_state, max_depth=None))
    ])

    # Random Forest
    models['RandomForest'] = Pipeline([
        ('prep', prep_tree),
        ('model', RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    # SVR with scaling
    models['SVR'] = Pipeline([
        ('prep', prep_svr),
        ('model', SVR(kernel='rbf', C=10.0, epsilon=0.05, gamma='scale'))
    ])

    # XGBoost
    if HAS_XGB and XGBRegressor is not None:
        models['XGBoost'] = Pipeline([
            ('prep', prep_tree),
            ('model', XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1,
                objective='reg:squarederror',
            ))
        ])
    else:
        print("INFO: xgboost not available. Skipping XGBoost model.")

    return models


def plot_ytrue_ypred(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(5.2, 5))
    plt.scatter(y_true, y_pred, alpha=0.8, edgecolor='k')
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
    plt.xlabel('Actual COF (median)')
    plt.ylabel('Predicted COF')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def correlation_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    corr = df.corr(numeric_only=True, method='pearson')
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=False, cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='ML models to predict run-level median COF per material.')
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent,
                        help='Base directory containing group subfolders (0, 2.5, 5, 7.5).')
    parser.add_argument('--out', type=Path, default=None, help='Output directory for ML results (default: <base>/results/ml)')
    parser.add_argument('--measure', type=str, default='median', choices=['median'],
                        help='Target measure across time for each run (currently only median).')
    args = parser.parse_args()

    base_dir = args.base
    out_dir = args.out if args.out else (base_dir / 'results' / 'ml')
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    df = build_run_level_dataset(base_dir)

    # Save raw dataset used
    df.to_csv(out_dir / 'run_level_dataset.csv', index=False)

    # Variation stats overall and per material
    stats_overall = compute_variation_stats(df['y'])
    stats_overall.to_csv(out_dir / 'variation_stats_overall.csv')

    # Correlation analysis overall
    corr_df = df[['y', 'N', 'distance_m', 'rpm', 'duration_min']].copy()
    corr_df.to_csv(out_dir / 'features_and_target.csv', index=False)
    correlation_heatmap(corr_df, out_dir / 'correlation_heatmap_overall.png', 'Overall Pearson Correlation')

    # Prepare models
    models = get_models(random_state=42)

    # Evaluate per material group
    lbl = label_map()
    summary_rows = []

    for g in ['0', '2.5', '5', '7.5']:
        df_g = df[df['group'] == g].copy()
        if df_g.empty:
            continue
        material_name = lbl.get(g, g)
        X = df_g[['N', 'distance_m', 'rpm', 'duration_min']].copy()
        y = df_g['y'].values

        # Handle missing values by median imputation per column
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        # Choose evaluation strategy
        n = len(df_g)
        if n >= 8:
            # Train/validation/test split: 60/20/20
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 of 0.8 = 0.2
            eval_mode = 'holdout'
        else:
            # Cross-validation predictions (KFold)
            k = min(3, max(2, n))
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            eval_mode = f'{k}-fold CV'
            X_train = X
            y_train = y
            X_val = None
            y_val = None
            X_test = X
            y_test = y

        # Train and evaluate each model
        for model_name, pipe in models.items():
            if 'XGBoost' in model_name and not HAS_XGB:
                continue

            if eval_mode == 'holdout':
                # Fit on train, early evaluate on val (optional), final eval on test
                pipe.fit(X_train, y_train)
                y_pred_test = pipe.predict(X_test)
                metrics = evaluate_predictions(y_test, y_pred_test)
                # Also capture validation metrics
                y_pred_val = pipe.predict(X_val)
                metrics_val = evaluate_predictions(y_val, y_pred_val)
            else:
                # CV predictions on entire data for fair evaluation
                k = int(eval_mode.split('-')[0])
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                try:
                    y_pred_test = cross_val_predict(pipe, X, y, cv=kf, n_jobs=-1)
                except Exception:
                    # Some estimators with pipelines might not support n_jobs; retry without n_jobs
                    y_pred_test = cross_val_predict(pipe, X, y, cv=kf)
                metrics = evaluate_predictions(y_test, y_pred_test)
                metrics_val = {k: np.nan for k in metrics.keys()}

            # Save metrics row
            row = {
                'group': g,
                'material': material_name,
                'model': model_name,
                'eval_mode': eval_mode,
            }
            row.update({f'test_{k}': v for k, v in metrics.items()})
            row.update({f'val_{k}': v for k, v in metrics_val.items()})
            summary_rows.append(row)

            # Plot predictions scatter
            plot_title = f"{material_name} — {model_name} ({eval_mode})"
            out_plot = plots_dir / f"scatter_{model_name.replace(' ', '')}_group_{g}.png"
            plot_ytrue_ypred(y_test, y_pred_test, plot_title, out_plot)

        # Per-material variation stats and correlation
        stats_g = compute_variation_stats(df_g['y'])
        stats_g.to_csv(out_dir / f'variation_stats_group_{g}.csv')
        corr_g = df_g[['y', 'N', 'distance_m', 'rpm', 'duration_min']].copy()
        correlation_heatmap(corr_g, out_dir / f'correlation_heatmap_group_{g}.png', f'Correlation — group {g}')

    # Save summary metrics
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / 'model_performance_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved model performance summary to: {summary_csv}")


if __name__ == '__main__':
    main()
