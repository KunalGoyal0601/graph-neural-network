# Graph Neural Network for Geopolitical Impact on Indian Sectoral Markets

A research project investigating whether financial, macroeconomic, and geopolitical information can improve prediction of next-trading-day movement across Indian market sectors using graph neural networks (GNNs).

**Research principle:** this project compares classical machine-learning baselines with graph models; it does not assume that GNNs will perform better.

## Research question

Can relationships among Indian market sectors, macroeconomic variables, and geopolitical information be represented as a graph and used to predict next-trading-day sector direction?

The target is defined as:

\[
y_{s,t}=\begin{cases}
1 & \text{if Close}_{s,t+1} > \text{Close}_{s,t} \\
0 & \text{otherwise}
\end{cases}
\]

Here, \(s\) is a sector and \(t\) is a trading day. Verify the target-generation notebook before re-running experiments, and ensure every feature is available by the prediction cutoff.

## Project scope

| Area | Included in this project |
|---|---|
| Market indices | NIFTY 50 benchmark and eight sector indices |
| Macroeconomic variables | Gold, oil, USD/INR, and India VIX |
| Geopolitical data | GDELT-derived country-level and global event features |
| Classical baselines | Random Forest and XGBoost |
| Graph models | GCN, GraphSAGE, GAT, and an HGT-style heterogeneous attention model |
| Analysis | EDA, feature engineering, graph construction, held-out metrics, feature/edge saliency |
| Deliverables | Dissertation report and presentation |

### Market nodes

- NIFTY 50
- NIFTY Auto
- NIFTY Bank
- NIFTY Energy
- NIFTY FMCG
- NIFTY IT
- NIFTY Metal
- NIFTY Pharma
- NIFTY Realty

### Macroeconomic variables

Gold, oil, USD/INR, and India VIX. The feature pipeline includes variable levels/returns and rolling features such as moving averages.

### Geopolitical features

GDELT features include country-level conflict indicators, event counts, and tone for the USA, India, China, Russia, Pakistan, Israel, and Iran. Global features include average tone, average Goldstein score, event count, total mentions, protest, threat, military, assault, fight, mass-violence, diplomacy, and cooperation counts.

## Graph design

The documented graph contains **21 nodes**:

- 9 market/sector nodes
- 4 macroeconomic nodes
- 7 country/geopolitical nodes
- 1 global geopolitical-information node

The documented design contains **56 unique undirected edges** across six relation groups:

| Relation group | Edges |
|---|---:|
| Benchmark–Sector | 8 |
| Sector–Sector | 5 |
| Macro–Sector | 12 |
| Country–Sector | 12 |
| Country–Country | 5 |
| Global–Other | 14 |
| **Total** | **56** |

The graph is domain-informed: relationships are specified using economic/geopolitical reasoning rather than learned solely from correlations. Treat these edges as modelling assumptions, not proof of causal influence. The actual graph used should be rechecked against the graph-construction code before a final publication.

### Model families

- **GCN:** aggregates neighbouring node representations using normalized graph connectivity.
- **GraphSAGE:** combines a node's representation with an aggregation of its neighbours.
- **GAT:** learns attention weights for neighbouring nodes.
- **HGT-style model:** represents heterogeneous node/relation types using type-aware attention.

The final executable GNN notebook documents an input tensor of shape `[samples, 21, 12]`, i.e. 21 nodes with 12 node features per observation, and eight sector prediction targets (excluding NIFTY 50 itself).

## Data and notebook locations

The repository currently contains raw, cleaned, and engineered CSVs alongside the notebooks that generate them. Existing folder names with typos are documented as-is to avoid breaking notebook paths. A staged cleanup should rename them only after references have been searched and tested.

| Current path | Purpose |
|---|---|
| `Data/raw_data/geo_raw/` | Raw GDELT files |
| `Data/raw_data/macro_raw/` | Raw gold, oil, USD/INR, and VIX data |
| `Data/raw_data/sector_raw_data/` | Raw sector-index data |
| `Data/cleaned_data/` | Cleaned geopolitical, macro, and sector data |
| `Data/featured_data/` | Engineered features and merged datasets |
| `Data/data_notebooks/01_Data_cleaning/` | Cleaning notebooks |
| `Data/data_notebooks/02_Feature_engeniearing/` | Feature-engineering and target notebooks |
| `notebooks/eda/` | Exploratory data analysis |
| `notebooks/Preprossing/` | Preprocessing notebook and stored split data |
| `notebooks/models/` | Classical ML and GNN experiments |
| `dissereation_report/` | Dissertation and related deliverables |
| `intership ppt/` | Presentation materials |

Important data artefacts currently include `master_dataset.csv`, `master_with_target.csv`, `featured_sector_data.csv`, `featured_macro_data.csv`, `featured_gdelt_data.csv`, `combined_sector_close_prices.csv`, `combined_macro_close_prices.csv`, and `gnn_fixed_results_summary.csv`.

## Recommended notebook execution order

1. Run the applicable notebooks in `Data/data_notebooks/01_Data_cleaning/`.
2. Run the relevant notebooks in `Data/data_notebooks/02_Feature_engeniearing/`.
3. Review `Data/data_notebooks/Market_sector_data.ipynb` and `Data/data_notebooks/geopolitical_data_exploring.ipynb`.
4. Run `Data/data_notebooks/merge_all.ipynb`.
5. Run `notebooks/eda/eda.ipynb`.
6. Run `notebooks/Preprossing/preprocessing.ipynb`.
7. Run `notebooks/models/Clssical_ml_final_executable.ipynb`.
8. Run `notebooks/models/GNN_final_single_path_executable_ready_executed (1).ipynb`.

Notebook names are historical and may be misspelled. Confirm each notebook's inputs and outputs before executing the pipeline end-to-end.

## Experimental setup and saved results

The README and final notebook document a chronological split, with preprocessing fitted on training data only. One reported execution used:

| Split | Samples | Reported period |
|---|---:|---|
| Train | 1,161 | 2018-02-16 to 2022-12-30 |
| Validation | 352 | 2023-01-03 to 2024-06-28 |
| Test | 470 | 2024-07-01 to 2026-06-17 |

These values are transcribed from a prior notebook execution. Recheck the actual split files/notebook state before relying on them: the saved files and the current master dataset may reflect different runs or data windows.

### GNN results saved in the repository

The saved `Data/featured_data/gnn_fixed_results_summary.csv` was documented with the following results:

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| GCN | 0.5137 | 0.6788 | 0.5045 |
| GraphSAGE | 0.5137 | 0.6788 | 0.5112 |
| GAT | 0.5119 | 0.6759 | 0.5255 |
| HGT | 0.5242 | 0.6619 | 0.5351 |

A separate execution recorded in the notebook output reported:

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| GCN | 0.517 | 0.681 | 0.486 |
| GraphSAGE | 0.517 | 0.681 | 0.485 |
| GAT | 0.517 | 0.681 | 0.494 |
| HGT | 0.517 | 0.681 | 0.502 |

**Interpretation:** these are historical, experiment-specific records, not one definitive benchmark. The variation between runs should be resolved by freezing the data split, seeds, preprocessing, model configuration, and metric calculation, then reproducing the final run. AUC values around 0.50 indicate weak ranking discrimination on those evaluations even where F1 is higher; report class balance and per-sector metrics alongside aggregate numbers.

### Classical ML comparison

The classical notebook documents two feature sets evaluated across eight sector targets:

- **E1 — Full features:** 96 features including sector technical, macro, country-level geopolitical, and global geopolitical features.
- **E2 — Without GDELT:** 63 features focused on sector technical and macroeconomic information.

Use the notebook outputs to populate an updated comparison table after a clean, reproducible execution; do not infer that adding geopolitical features improves performance without comparing like-for-like held-out results.

## Explainability

The GNN notebook contains feature-importance, gradient/saliency, edge-importance, and edge-importance heatmap analyses. These are exploratory attribution methods and **are not the PyTorch Geometric GNNExplainer implementation**. Attribution results should not be interpreted as proof of causal influence.

## Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install a starting set of libraries:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn networkx xgboost torch jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

These commands are a starting point, not a fully verified, pinned environment. Create a version-pinned `requirements.txt` after running the project end-to-end in a clean environment. PyTorch installation options can differ by operating system and hardware.

## Recommended research improvements

1. **Leakage audit:** confirm that GDELT data, rolling features, scalers, and all engineered features use only information available before the prediction cutoff.
2. **Stronger evaluation:** supplement one chronological holdout with walk-forward/rolling-origin validation.
3. **Fair baselines:** report majority-class and simple non-graph technical-indicator baselines alongside Random Forest and XGBoost.
4. **Reproducibility:** save seeds, parameters, data-window definitions, package versions, per-sector metrics, predictions, and a timestamped experiment log.
5. **Graph sensitivity:** compare the domain-informed graph with justified alternatives (e.g. rolling correlations or mutual-information edges) without using future data to construct edges.
6. **Data management:** document source/provenance and regeneration steps. Review file sizes and redistribution permissions before committing more raw data or academic deliverables. Consider Git LFS or documented external storage for genuinely large artifacts.
7. **Folder cleanup:** rename paths only in a dedicated change after checking notebook references, scripts, README links, and any external instructions.

## Academic deliverables

- Dissertation: `dissereation_report/Kunal_Goyal_2025MSBDA010.pdf`
- Supplementary dissertation archive: `dissereation_report/dissertation_single_file_with_figures.zip`
- Presentation: `intership ppt/NIFTY_Geopolitical_Dissertation_Presentation_Humanized_v5.pptx`

Confirm that any academic files and datasets in the repository are cleared for public sharing before changing the repository's visibility.

## Limitations and disclaimer

Financial market prediction is noisy and model performance can change across periods. Results depend on feature timing, graph assumptions, targets, splits, and training configuration. This project is for academic/research purposes and is not financial advice or a guarantee of future market movement.

## Author

**Kunal Goyal** — M.Sc. Big Data Analytics  
GitHub: [KunalGoyal0601](https://github.com/KunalGoyal0601)
