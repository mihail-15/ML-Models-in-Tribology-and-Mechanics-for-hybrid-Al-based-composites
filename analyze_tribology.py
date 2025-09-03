#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import re


def read_dwf(file_path: Path) -> pd.DataFrame:
    """Read a .dwf file with European decimal commas and semicolon delimiter.

    Returns a DataFrame with columns ['TIME', 'FF'] in ascending TIME, dropping NaNs.
    """
    # Read raw text and normalize line endings (some files use CR-only separators)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=['TIME', 'FF'])

    # Skip header line if it contains TIME/FF
    start_idx = 0
    if 'TIME' in lines[0].upper() and 'FF' in lines[0].upper():
        start_idx = 1

    time_vals: List[float] = []
    ff_vals: List[float] = []
    for ln in lines[start_idx:]:
        parts = [p for p in ln.split(';')]
        if len(parts) < 2:
            continue
        t_raw = parts[0].strip()
        y_raw = parts[1].strip()
        if not t_raw or not y_raw:
            continue
        # Replace decimal comma with dot and strip spaces
        t_str = t_raw.replace(',', '.').replace('\u00A0', '').strip()
        y_str = y_raw.replace(',', '.').replace('\u00A0', '').strip()
        try:
            t = float(t_str)
            y = float(y_str)
        except ValueError:
            continue
        time_vals.append(t)
        ff_vals.append(y)

    if not time_vals:
        return pd.DataFrame(columns=['TIME', 'FF'])

    df = pd.DataFrame({'TIME': time_vals, 'FF': ff_vals})
    df = df.dropna(subset=['TIME', 'FF']).sort_values('TIME').reset_index(drop=True)
    return df


def build_common_grid(dfs: List[pd.DataFrame]) -> np.ndarray:
    """Create a common time grid based on overlap and highest sampling among series.

    - Overlap domain: [max(min(TIME)), min(max(TIME))]
    - Grid step: min of median sampling intervals across series
    """
    # Drop any empty dataframes defensively
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        raise ValueError("No dataframes provided to build grid")

    t_min = max(float(df['TIME'].min()) for df in dfs)
    t_max = min(float(df['TIME'].max()) for df in dfs)
    if t_max <= t_min:
        raise ValueError(f"Non-overlapping time ranges across runs (t_min={t_min}, t_max={t_max})")

    # Determine representative dt
    dts = []
    for df in dfs:
        d = df['TIME'].sort_values().diff().median()
        if pd.notnull(d) and d > 0:
            dts.append(float(d))
    grid_dt = max(min(dts) if dts else 1.0, 1e-3)  # avoid zero/negative, clamp to >=1e-3

    n = int(np.floor((t_max - t_min) / grid_dt)) + 1
    grid = t_min + np.arange(n + 1) * grid_dt  # include end if fits
    # Ensure upper bound does not exceed t_max due to rounding
    grid = grid[grid <= t_max]
    return grid


def interpolate_to_grid(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Interpolate FF to the provided time grid using linear interpolation."""
    t = df['TIME'].values.astype(float)
    y = df['FF'].values.astype(float)
    # numpy.interp requires ascending t and will clip; we want NaN outside range
    y_grid = np.interp(grid, t, y, left=np.nan, right=np.nan)
    return y_grid


def compute_group_medium(
    file_paths: List[Path],
    measure: str = 'median',
    units: str = 'COF',
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the group medium curve across multiple runs.

    Returns (grid_time, medium_ff)
    measure: 'median' (default) or 'mean'
    """
    if not file_paths:
        raise ValueError("No files provided for group")

    dfs = [read_dwf(fp) for fp in file_paths]
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        raise ValueError("All runs are empty after parsing; check file formats.")

    # Convert to desired units
    if units.upper() == 'COF':
        dfs_conv: List[pd.DataFrame] = []
        for fp, df in zip(file_paths, dfs):
            N = None
            m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*[Nn]\b", str(fp))
            if m:
                N = float(m.group(1).replace(',', '.'))
            if not N or N == 0:
                raise ValueError(f"Normal load could not be parsed from filename: {fp}")
            df2 = df.copy()
            df2['FF'] = df2['FF'] / N  # FF -> COF
            dfs_conv.append(df2)
        dfs = dfs_conv
    grid = build_common_grid(dfs)

    # Interpolate all runs
    runs = [interpolate_to_grid(df, grid) for df in dfs]
    Y = np.vstack(runs)  # shape: (n_runs, n_time)

    # Compute along runs axis
    if measure.lower() == 'mean':
        medium = np.nanmean(Y, axis=0)
    else:
        # default to median for robustness
        medium = np.nanmedian(Y, axis=0)

    # Drop grid points where medium is NaN (e.g., all NaN at that time)
    mask = ~np.isnan(medium)
    return grid[mask], medium[mask]


def gather_groups(base_dir: Path) -> Dict[str, List[Path]]:
    """Map group labels to their .dwf files from expected subfolders."""
    groups: Dict[str, List[Path]] = {}
    for sub in ['0', '2.5', '5', '7.5']:
        folder = base_dir / sub
        if folder.is_dir():
            files = sorted(folder.glob('*.dwf'))
            if files:
                groups[sub] = files
    if not groups:
        raise FileNotFoundError(f"No groups with .dwf files found under {base_dir}")
    return groups


def label_map() -> Dict[str, str]:
    return {
        '0': 'Al10Cu (unreinforced)',
        '2.5': 'Al10Cu + 2 wt.% MoS2 + 2.5 wt.% Al2O3',
        '5': 'Al10Cu + 2 wt.% MoS2 + 5 wt.% Al2O3',
        '7.5': 'Al10Cu + 2 wt.% MoS2 + 7.5 wt.% Al2O3',
    }


def main():
    parser = argparse.ArgumentParser(description='Compute medium curves and descriptive stats for tribology data (.dwf).')
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent,
                        help='Base directory containing group subfolders (0, 2.5, 5, 7.5).')
    parser.add_argument('--out', type=Path, default=None, help='Output directory for CSV results (default: <base>/results)')
    parser.add_argument('--measure', type=str, default='median', choices=['median', 'mean'],
                        help='Medium measure across runs (default: median)')
    parser.add_argument('--units', type=str, default='COF', choices=['FF', 'COF'],
                        help='Output units: FF (N) or COF (-). Default: COF')
    args = parser.parse_args()

    base_dir = args.base
    out_dir = args.out if args.out else (base_dir / 'results')
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = gather_groups(base_dir)

    # Compute group mediums
    mediums: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for g, files in groups.items():
        print(f"Processing group {g} with {len(files)} files...")
        t, y = compute_group_medium(files, measure=args.measure, units=args.units)
        mediums[g] = (t, y)

    # Build a combined time grid across groups by intersecting overlaps
    # Find common overlapping time span across group mediums
    t_start = max(t[0] for (t, _) in mediums.values())
    t_end = min(t[-1] for (t, _) in mediums.values())
    if t_end <= t_start:
        raise ValueError("No common time overlap across group mediums.")

    # Choose a grid step based on smallest dt among groups
    dt_candidates = []
    for (t, _) in mediums.values():
        if len(t) > 1:
            dt_candidates.append(np.median(np.diff(t)))
    grid_dt = max(min(dt_candidates) if dt_candidates else 1.0, 1e-3)
    n = int(np.floor((t_end - t_start) / grid_dt)) + 1
    common_grid = t_start + np.arange(n + 1) * grid_dt
    common_grid = common_grid[common_grid <= t_end]

    # Interpolate each group medium onto the common grid
    data = {'TIME': common_grid}
    for g, (t, y) in mediums.items():
        y_interp = np.interp(common_grid, t, y, left=np.nan, right=np.nan)
        data[g] = y_interp

    medium_df = pd.DataFrame(data)
    # Drop any rows with all-NaN (should not happen within overlap) and forward-fill small edge NaNs if any
    medium_df = medium_df.dropna(how='all')

    # Save medium curves (one CSV with columns per group)
    medium_csv = out_dir / f"medium_curves_{args.measure}_{args.units.lower()}.csv"
    medium_df.to_csv(medium_csv, index=False)
    print(f"Saved medium curves to: {medium_csv}")

    # Descriptive statistics across time for each material's medium curve
    stats_df = medium_df.drop(columns=['TIME']).describe().rename(index={
        '50%': 'median', '25%': 'q1', '75%': 'q3'
    })
    stats_csv = out_dir / f"descriptive_stats_medium_{args.measure}_{args.units.lower()}.csv"
    stats_df.to_csv(stats_csv)
    print(f"Saved descriptive statistics to: {stats_csv}")

    # Also save per-group medium as separate CSVs if desired
    for g, (t, y) in mediums.items():
        g_df = pd.DataFrame({'TIME': t, f'{g}': y})
        g_csv = out_dir / f"group_{g}_medium_{args.measure}_{args.units.lower()}.csv"
        g_df.to_csv(g_csv, index=False)

    # Save a README-like summary
    readme_path = out_dir / 'README.txt'
    lbl = label_map()
    with open(readme_path, 'w', encoding='utf-8') as f:
        unit_desc = 'Coefficient of Friction (COF, -)' if args.units.upper() == 'COF' else 'Friction Force (N)'
        f.write("Medium curves computed per group ({} across runs) with linear interpolation to a common grid. Units: {}.\n".format(args.measure, unit_desc))
        f.write("Groups mapping:\n")
        for k, v in lbl.items():
            if k in groups:
                f.write(f"  {k}: {v}\n")
        f.write("\nOutputs:\n")
        f.write(f"  - {medium_csv.name}: TIME and one column per group ({', '.join(groups.keys())}), values in {unit_desc}\n")
        f.write(f"  - {stats_csv.name}: descriptive stats across TIME for each group's medium curve (values in {unit_desc})\n")
        for g in groups:
            f.write(f"  - group_{g}_medium_{args.measure}_{args.units.lower()}.csv: TIME and medium values for group {g} ({unit_desc})\n")

    print(f"Wrote summary README to: {readme_path}")

    # Create separate plots per group
    for g, (t, y) in mediums.items():
        plt.figure(figsize=(7, 4))
        plt.plot(t, y, label=f"{lbl.get(g, g)} ({args.measure})", color="#1f77b4", linewidth=1.8)
        plt.xlabel("Time [s]")
        plt.ylabel("COF [-]" if args.units.upper() == 'COF' else "Friction Force [N]")
        plt.title(f"{lbl.get(g, g)} — medium ({args.measure})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = out_dir / f"plot_group_{g}_medium_{args.measure}_{args.units.lower()}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved plot for group {g} to: {out_png}")

    # Create combined family plot across all materials on one figure
    plt.figure(figsize=(8.5, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ordered_groups = ['0', '2.5', '5', '7.5']
    for i, g in enumerate(ordered_groups):
        if g in medium_df.columns:
            plt.plot(
                medium_df['TIME'],
                medium_df[g],
                label=lbl.get(g, g),
                linewidth=1.8,
                color=colors[i % len(colors)],
            )
    plt.xlabel("Time [s]")
    plt.ylabel("COF [-]" if args.units.upper() == 'COF' else "Friction Force [N]")
    plt.title(f"Medium curves by material — {args.measure}")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Material", ncol=1)
    out_png_family = out_dir / f"plot_family_medium_{args.measure}_{args.units.lower()}.png"
    plt.tight_layout()
    plt.savefig(out_png_family, dpi=150)
    plt.close()
    print(f"Saved combined family plot to: {out_png_family}")


if __name__ == '__main__':
    main()
