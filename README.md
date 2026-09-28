# Graph Neural Network for Geopolitical Impact on Indian Sectoral Markets

> A research project exploring whether financial, macroeconomic, and geopolitical information can be represented as a graph and used with Graph Neural Networks (GNNs) to predict next-trading-day movement of Indian market sectors.

## Overview

Financial sectors are interconnected. Movements in one sector can be related to other sectors, macroeconomic variables, country-level geopolitical conditions, and broader global events.

This project builds a **domain-informed heterogeneous market graph** containing:

- NIFTY market/sector nodes
- Macroeconomic nodes
- Country/geopolitical nodes
- A global geopolitical-information node

The project then compares classical machine-learning baselines with several graph neural network architectures:

- GCN
- GraphSAGE
- GAT
- HGT-style heterogeneous attention

The prediction task is **next-trading-day sector direction**:

\[
Target_t =
\begin{cases}
1 & \text{if Close}_{t+1} > Close_t \\
0 & \text{otherwise}
\end{cases}
\]

The repository contains the data-processing workflow, exploratory analysis, preprocessing, classical ML experiments, GNN experiments, results, dissertation report, and presentation.

---

## Research Question

**Can relationships between Indian market sectors, macroeconomic variables, and geopolitical factors be represented as a graph and used to predict next-trading-day sector movement?**

The project is designed to answer this experimentally rather than assuming that GNNs will outperform conventional models.

---

## Dataset

### Market / Sector Data

The project models the following NIFTY indices:

| Node | Description |
|---|---|
| NIFTY50 | NIFTY 50 benchmark |
| NIFTY_AUTO | Automobile sector |
| NIFTY_BANK | Banking sector |
| NIFTY_ENERGY | Energy sector |
| NIFTY_FMCG | FMCG sector |
| NIFTY_IT | Information Technology sector |
| NIFTY_METAL | Metal sector |
| NIFTY_PHARMA | Pharmaceutical sector |
| NIFTY_REALTY | Realty sector |

Market data is collected from Yahoo Finance using historical index data.

### Macroeconomic Variables

The graph includes:

- Gold
- Oil
- USD/INR
- India VIX

For these variables, the feature-engineering pipeline includes values/returns and rolling features such as MA30.

### Geopolitical Data

Geopolitical information is derived from **GDELT**.

Country-level features are constructed for:

- USA
- India
- China
- Russia
- Pakistan
- Israel
- Iran

Country features include:

- Conflict indicators
- Event counts
- Tone measures

Global GDELT features include:

- Average Tone
- Average Goldstein score
- Event Count
- Total Mentions
- Protest Count
- Threat Count
- Military Count
- Assault Count
- Fight Count
- Mass Violence Count
- Diplomacy Count
- Cooperation Count

---

# Feature Engineering

The final GNN notebook validates **135 engineered input features**.

### Sector Features

The sector feature schema includes:

- Price
- Return
- Log return
- MA30
- Volatility
- Momentum
- Relative strength
- RSI
- Drawdown
- Price/MA30 relationship

### Macro Features

For each macro variable:

- Level
- Return
- MA30

### Geopolitical Features

Country-specific:

- `<Country>_Conflict`
- `<Country>_Events`
- `<Country>_Tone`

Global GDELT:

- `AvgTone`
- `AvgGoldstein`
- `EventCount`
- `TotalMentions`
- `ProtestCount`
- `ThreatCount`
- `MilitaryCount`
- `AssaultCount`
- `FightCount`
- `MassViolenceCount`
- `DiplomacyCount`
- `CooperationCount`

---

# Data Pipeline

The repository currently organizes the data workflow as:

```
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Data Merging
   ↓
Master Dataset
   ↓
Target Creation
   ↓
Preprocessing
   ↓
Classical ML
   ↓
Graph Construction
   ↓
GNN Models
   ↓
Evaluation
```

The main prepared datasets include:

- `master_dataset.csv`
- `master_with_target.csv`
- `featured_sector_data.csv`
- `featured_macro_data.csv`
- `featured_gdelt_data.csv`
- Combined sector and macro datasets

---

# Actual Repository Structure

The current repository is organized as follows:

```
graph-neural-network/
│
├── Data/
│   ├── raw_data/
│   │   ├── geo_raw/
│   │   ├── macro_raw/
│   │   └── sector_raw_data/
│   │
│   ├── cleaned_data/
│   │   ├── geo_cleaned_data/
│   │   ├── macro_cleaned_data/
│   │   └── sector_cleaned_data/
│   │
│   ├── featured_data/
│   │   ├── combined_macro_close_prices.csv
│   │   ├── combined_sector_close_prices.csv
│   │   ├── featured_gdelt_data.csv
│   │   ├── featured_macro_data.csv
│   │   ├── featured_sector_data.csv
│   │   ├── gnn_fixed_results_summary.csv
│   │   ├── master_dataset.csv
│   │   └── master_with_target.csv
│   │
│   └── data_notebooks/
│       ├── 01_Data_cleaning/
│       ├── 02_Feature_engeniearing/
│       ├── Market_sector_data.ipynb
│       ├── geopolitical_data_exploring.ipynb
│       ├── merge_all.ipynb
│       └── oil_gold_usdinr_vix.ipynb
│
├── notebooks/
│   ├── Preprossing/
│   │   ├── preprocessing.ipynb
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── Y_train.csv
│   │   └── Y_test.csv
│   │
│   ├── eda/
│   │   └── eda.ipynb
│   │
│   └── models/
│       ├── Clssical_ml_final_executable.ipynb
│       └── GNN_final_single_path_executable_ready_executed (1).ipynb
│
├── dissereation_report/
│   ├── Kunal_Goyal_2025MSBDA010.pdf
│   └── dissertation_single_file_with_figures.zip
│
├── intership ppt/
│   └── NIFTY_Geopolitical_Dissertation_Presentation_Humanized_v5.pptx
│
└── classical_performance.png
```

> Folder names such as `Preprossing`, `02_Feature_engeniearing`, `dissereation_report`, and `intership ppt` reflect the current repository and have intentionally been preserved in this documentation.

---

# Graph Construction

The current GNN implementation contains **21 nodes**:

- 9 NIFTY/market nodes
- 4 macro nodes
- 7 country nodes
- 1 global node

The graph currently contains **56 unique undirected edges** grouped into six relation categories:

| Relation | Edges |
|---|---:|
| Benchmark–Sector | 8 |
| Sector–Sector | 5 |
| Macro–Sector | 12 |
| Country–Sector | 12 |
| Country–Country | 5 |
| Global–Other | 14 |
| **Total** | **56** |

The graph is domain-informed rather than generated solely from statistical correlation.

### Example relationships

```
NIFTY50 → Sector nodes

Oil → Energy
Oil → Auto
Oil → Metal

USD/INR → IT
USD/INR → Pharma
USD/INR → Auto

USA → IT
USA → Pharma

China → Metal
China → Energy

Russia → Energy

Israel → Pharma
Iran → Energy
```

Edges are treated as bidirectional during message passing.

---

# GNN Models

The repository's final executable GNN notebook implements four architectures using PyTorch tensor operations.

## 1. GCN

Graph Convolutional Network aggregates information from neighbouring nodes using normalized graph adjacency.

## 2. GraphSAGE

GraphSAGE combines each node's representation with an aggregated representation of its neighbours.

## 3. GAT

Graph Attention Network uses attention to assign different weights to neighbouring nodes.

## 4. HGT-style Model

The project uses a heterogeneous attention implementation that preserves:

- Node types
- Relation types
- Type-aware graph interactions

The relation groups used by the HGT-style model correspond to the six graph relation categories described above.

---

# GNN Input Representation

Each observation is transformed into:

```
[number of samples, number of nodes, node feature dimension]
```

The final notebook reports:

```
X_train: (1161, 21, 12)
X_val:   (352, 21, 12)
X_test:  (470, 21, 12)
```

Therefore, each daily observation is represented as a graph containing:

**21 nodes × 12 node features**

The model predicts the movement of **8 sector targets**, excluding NIFTY50 itself as a prediction target.

---

# Time-Series Evaluation

The project uses chronological splitting rather than random train/test splitting.

### Current split in the final GNN notebook

| Split | Samples | Period |
|---|---:|---|
| Train | 1,161 | 2018-02-16 → 2022-12-30 |
| Validation | 352 | 2023-01-03 → 2024-06-28 |
| Test | 470 | 2024-07-01 → 2026-06-17 |

The feature scaler is fitted **only on the training period** and then applied to validation and test data.

This is intended to reduce temporal leakage.

---

# Current Research Status

The repository is **past the initial data-collection and baseline stage and currently contains an executable end-to-end GNN experiment**.

### Completed / implemented

- [x] Historical sector-market data collection
- [x] Macroeconomic data collection
- [x] GDELT geopolitical data processing
- [x] Raw-data organization
- [x] Data cleaning
- [x] Feature engineering
- [x] Dataset merging
- [x] Master dataset creation
- [x] Next-trading-day target construction
- [x] Chronological preprocessing
- [x] Leakage-safe feature scaling
- [x] Exploratory data analysis
- [x] Classical ML experiments
- [x] Domain-informed graph construction
- [x] GCN implementation
- [x] GraphSAGE implementation
- [x] GAT implementation
- [x] HGT-style heterogeneous attention implementation
- [x] Train/validation model selection
- [x] Final train+validation refitting
- [x] Held-out test evaluation
- [x] Training curves
- [x] Per-sector AUC analysis
- [x] Feature-importance analysis
- [x] Edge-importance/saliency analysis
- [x] GNN result summary saved to CSV
- [x] Dissertation report
- [x] Project presentation

### Current research finding

The current saved GNN experiment does **not** show strong predictive discrimination on the held-out test period.

The repository's `gnn_fixed_results_summary.csv` reports:

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| GCN | 0.5137 | 0.6788 | 0.5045 |
| GraphSAGE | 0.5137 | 0.6788 | 0.5112 |
| GAT | 0.5119 | 0.6759 | 0.5255 |
| HGT | 0.5242 | 0.6619 | 0.5351 |

These are the saved summary results in the repository. They should be interpreted as **experimental results for the current graph, features, split, and training configuration**, not as evidence that the approach will generalize to future market periods.

The final executable notebook also records another execution of the same experiment with:

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| GCN | 0.517 | 0.681 | 0.486 |
| GraphSAGE | 0.517 | 0.681 | 0.485 |
| GAT | 0.517 | 0.681 | 0.494 |
| HGT | 0.517 | 0.681 | 0.502 |

The difference between these saved executions indicates that the exact reported values depend on the executed configuration/state. The project therefore should not claim a single definitive GNN score until the final experiment is frozen and rerun reproducibly.

---

# Classical ML Baselines

The repository contains a dedicated executable classical-ML notebook using:

- Random Forest
- XGBoost

The experiments compare:

### E1 — Full feature set

96 features consisting of:

- Sector technical features
- Macro features
- GDELT country features
- GDELT global features

### E2 — No GDELT

63 features consisting of:

- Sector technical features
- Macro features

The classical models are evaluated separately across the eight prediction sectors.

The repository also contains:

`classical_performance.png`

for visualizing classical-model performance.

---

# Research Methodology

The current methodology is:

```
Financial Data
      +
Macro Data
      +
GDELT Data
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Master Dataset
      ↓
Next-Day Targets
      ↓
Chronological Split
      ↓
Classical ML Baselines
      ↓
Domain-Informed Graph
      ↓
GCN / GraphSAGE / GAT / HGT
      ↓
Test Evaluation
      ↓
Feature / Edge Analysis
```

---

# Explainability

The current GNN notebook includes explanation-oriented analysis using:

- Feature importance
- Gradient-based saliency
- Edge importance
- Edge-importance heatmaps

The notebook explicitly notes that these plots are **not the PyTorch Geometric GNNExplainer implementation**. They are gradient/saliency-based analyses intended to investigate which inputs and relationships contribute to model predictions.

---

# Technology Stack

### Programming

- Python
- Jupyter Notebook

### Data Processing

- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-learn
- XGBoost

### Deep Learning

- PyTorch

### Graph Analysis

- NetworkX

### Visualization

- Matplotlib
- Seaborn

### Data Sources

- Yahoo Finance
- GDELT

---

# Running the Project

Clone the repository:

```bash
git clone https://github.com/KunalGoyal0601/graph-neural-network.git
cd graph-neural-network
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages used by the notebooks:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn networkx xgboost torch jupyter
```

Open Jupyter:

```bash
jupyter notebook
```

### Recommended execution order

For understanding the project, follow:

1. `Data/data_notebooks/01_Data_cleaning/`
2. `Data/data_notebooks/02_Feature_engeniearing/`
3. `Data/data_notebooks/Market_sector_data.ipynb`
4. `Data/data_notebooks/geopolitical_data_exploring.ipynb`
5. `Data/data_notebooks/merge_all.ipynb`
6. `notebooks/eda/eda.ipynb`
7. `notebooks/Preprossing/preprocessing.ipynb`
8. `notebooks/models/Clssical_ml_final_executable.ipynb`
9. `notebooks/models/GNN_final_single_path_executable_ready_executed (1).ipynb`

---

# Important Research Limitations

The current results should be interpreted carefully.

### 1. Financial prediction is difficult

An AUC near 0.50 indicates performance close to random discrimination for the current experiment.

### 2. The graph is domain-informed

The current graph uses manually specified economic/geopolitical relationships. It is therefore not equivalent to a graph learned entirely from the data.

### 3. Single chronological split

The current experiment uses one train/validation/test temporal split. A stronger research design would additionally use walk-forward or rolling-origin evaluation.

### 4. Geopolitical information timing

A publication-quality study should verify that every GDELT-derived feature was actually available before the prediction cutoff. This is important because publication/aggregation timing can create subtle look-ahead leakage.

### 5. Model comparison

The GNN models should be compared against simple baselines, including majority-class and non-graph time-series/technical-indicator baselines, before concluding that graph modelling adds value.

---

# Future Research Directions

The next research stage can investigate:

- Walk-forward validation
- Dynamic/time-varying graphs
- Data-driven graph construction
- Rolling correlations
- Granger-causality-based edges
- Mutual-information-based graph construction
- Temporal GNNs
- GNN + LSTM/GRU architectures
- Sector-specific models
- Multi-horizon prediction
- More rigorous geopolitical event representations
- Improved graph explainability
- Economic/portfolio evaluation with transaction costs

---

# Academic Deliverables

The repository currently contains the project's academic outputs:

### Dissertation

`dissereation_report/Kunal_Goyal_2025MSBDA010.pdf`

### Presentation

`intership ppt/NIFTY_Geopolitical_Dissertation_Presentation_Humanized_v5.pptx`

### Performance Visualization

`classical_performance.png`

### GNN Results

`Data/featured_data/gnn_fixed_results_summary.csv`

---

# Disclaimer

This project is intended for **academic and research purposes**.

The predictions produced by these models should not be interpreted as financial advice or guaranteed forecasts of future market movements.

---

# Author

**Kunal Goyal**

M.Sc. Big Data Analytics

GitHub: https://github.com/KunalGoyal0601
