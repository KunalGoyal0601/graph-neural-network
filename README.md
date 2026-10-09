# Graph Neural Network for Geopolitical Impact on Indian Sectoral Markets

Research project evaluating whether financial, macroeconomic, and geopolitical information can help predict next-trading-day movement across Indian market sectors using graph neural networks (GNNs).

## Research question

Can relationships among Indian market sectors, macroeconomic variables, and geopolitical information be represented as a graph and used to predict next-trading-day sector direction?

This project treats that as an empirical question. GNNs are compared with classical machine-learning baselines; no model is assumed to outperform the others.

## Scope

- **Market:** NIFTY 50 and eight sector indices
- **Macroeconomic variables:** gold, oil, USD/INR, and India VIX
- **Geopolitical information:** GDELT-derived country and global event features
- **Models:** Random Forest, XGBoost, GCN, GraphSAGE, GAT, and an HGT-style heterogeneous attention model
- **Prediction target:** next-trading-day sector direction
- **Evaluation:** chronological train/validation/test split and held-out metrics

## Repository structure (current)

The existing repository contains raw, cleaned, and engineered CSV datasets; data-cleaning and feature-engineering notebooks; preprocessing and EDA; classical ML and GNN experiments; saved result summaries; and academic deliverables. Some historical folder and file names contain typos. These are being retained for now to avoid breaking notebook paths.

Key locations:

| Path | Purpose |
|---|---|
| `Data/raw_data/` | Collected source data |
| `Data/cleaned_data/` | Cleaned sector, macro, and GDELT data |
| `Data/featured_data/` | Engineered features and merged datasets |
| `Data/data_notebooks/` | Data collection, cleaning, feature engineering, and merge notebooks |
| `notebooks/eda/` | Exploratory analysis |
| `notebooks/Preprossing/` | Preprocessing and saved split files |
| `notebooks/models/` | Classical ML and GNN experiments |
| `dissereation_report/` | Dissertation materials (legacy folder name) |
| `intership ppt/` | Presentation materials (legacy folder name) |

## Main workflow

1. Collect market, macroeconomic, and GDELT data.
2. Clean sources and align observations by trading date.
3. Engineer sector, macroeconomic, and geopolitical features.
4. Merge the master dataset and create target labels.
5. Split data chronologically and fit preprocessing only on training data.
6. Train classical baselines.
7. Construct the domain-informed heterogeneous graph and train GNN variants.
8. Evaluate held-out predictions and inspect feature/edge saliency.
9. Record configurations and results so experiments can be reproduced.

## Suggested notebook execution order

1. `Data/data_notebooks/01_Data_cleaning/`
2. `Data/data_notebooks/02_Feature_engeniearing/`
3. `Data/data_notebooks/Market_sector_data.ipynb`
4. `Data/data_notebooks/geopolitical_data_exploring.ipynb`
5. `Data/data_notebooks/merge_all.ipynb`
6. `notebooks/eda/eda.ipynb`
7. `notebooks/Preprossing/preprocessing.ipynb`
8. `notebooks/models/Clssical_ml_final_executable.ipynb`
9. `notebooks/models/GNN_final_single_path_executable_ready_executed (1).ipynb`

## Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the packages used by the notebooks:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn networkx xgboost torch jupyter
```

Run Jupyter:

```bash
jupyter notebook
```

**Note:** The commands above provide a starting environment, not a verified lockfile. Package versions and hardware requirements may vary. A pinned `requirements.txt` should be created after a clean end-to-end environment test.

## Evaluation and interpretation

The repository contains saved GNN summaries and notebook outputs. Treat these values as experiment-specific until a final configuration is frozen and rerun. In particular, AUC values close to 0.50 indicate weak ranking discrimination in those saved runs, even when accuracy or F1 looks higher due to the class distribution or threshold.

Recommended next steps:

- Use walk-forward evaluation in addition to a single chronological split.
- Compare with majority-class and simple technical-indicator baselines.
- Verify that each GDELT feature was available by the prediction cutoff to avoid look-ahead leakage.
- Save model configurations, random seeds, package versions, and prediction files.
- Document the exact final experiment and avoid selecting results based solely on one split.

## Data and reproducibility notes

The repository currently tracks CSV datasets and saved split files. Before expanding the dataset, review file sizes, provenance, redistribution terms, and whether raw data can be regenerated from source. Keep secrets, API keys, personal data, and local environment files out of Git. If large files become difficult to version, consider Git LFS or documented external storage.

## Academic deliverables

The repository includes a dissertation PDF and project presentation under the legacy-named folders listed above. Ensure that these files are appropriate for public sharing before publishing the repository.

## Limitations

Financial market movements are noisy and difficult to predict. Results from this project are experimental, depend on the data period and methodology, and are not guarantees of future performance. Gradient/saliency plots are exploratory explanations and should not be interpreted as proof of causal relationships.

## Author

**Kunal Goyal**  
M.Sc. Big Data Analytics  
GitHub: [KunalGoyal0601](https://github.com/KunalGoyal0601)

For academic and research purposes only. Not financial advice.
