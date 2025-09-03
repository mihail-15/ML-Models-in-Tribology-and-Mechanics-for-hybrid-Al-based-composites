# ML-Models-in-Tribology-and-Mechanics-for-hybrid-Al-based-composites
Code and data to reproduce ML models for coefficient of friction (COF) and wear, plus stress-strain mechanics analysis for Al10Cu and composites with 2% MoS2 and 2.5/5.0/7.5% Al2O3. Includes Random Forest, SVM, and XGBoost models, visualizations, and metrics tables.

# Tribology-ML-Models: ML for Tribology & Mechanical Properties (Al–Cu, MoS2/Al2O3)

This repository contains the code and data to reproduce:
- Machine learning models for Coefficient of Friction (COF) and Mass Wear (Random Forest, SVM, XGBoost).
- Analysis of mechanical properties from stress-strain curves.
- Visualizations and tables for model performance metrics and comparisons.

## Contents
- `Mechanics/`: Scripts and results for mechanical analysis. Training and batch prediction scripts for each model type.
- Root-level scripts: [generate_performance_tables.py
- Data/Results: `group_*.csv`, `*_performance_metrics.txt`, `metrics_table_*.png`, `pred_*.png`, etc.

## Quick Start
- Requires Python ≥ 3.9.
- Install dependencies:
  - `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `openpyxl`
- To run an example:
  - Generate mechanics performance tables: `python Mechanics/ML/generate_mechanics_performance_tables.py`
  - For RF/SVM/XGB examples, see the respective subdirectories and the `*-batch.py` scripts.

## Reproducibility
The scripts expect the `group_*.csv` and `*_performance_metrics.txt` files to be present in the root and subdirectories as provided in this repository. Generated outputs (PNG/XLSX files) are saved to `results/` or the relevant subdirectories.

## License
- **Code**: The code in this repository is licensed under the MIT License (see `LICENSE`).
- **Data & Figures**: All data files are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Citation
If you use this work, please cite this repository and any accompanying publications (add details here).
