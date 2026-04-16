# About this Repository
This repository contains the complete codebase, datasets, and results for my master's thesis "Development and evaluation of predictive machine learning methods for 
forecasting demand for battery electric vehicles".

The study benchmarks ensemble methods across multiple experimental trials and validates findings through a 2025 real-world case study.

 # Repository Structure
 ```
📂 Codes/         # Main experiment code — Trials 1–3
  ├── Trial1/
  ├── Trial2/
  └── Trial3/
📂 Codes_2025/    # 2025 Case Study code
📂 Data/          # Datasets for Trials 1–3
📂 Data_2025/     # Datasets for the 2025 Case Study
📄 README.md
```

# Codes for Trial 1 to 3
Each trial subfolder contains a consistent set of scripts following the full experimental pipeline:

| Script | Description |
|--------|-------------|
| Merging of Datasets | Combines and pre-processes raw datasets into analysis-ready form |
| Exploratory Data Analysis | Distributions, trends, correlations and outlier detection |
| KPI of the Results | Computes key performance indicators (RMSE, MAE, MAPE, etc.) across models |
| Visualisation of the Results | Generates plots and comparative charts of forecasting outcomes |

# Models Evaluated

1. Linear Regression
2. Support Vector Rgeressor (SVR)
3. ARIMAX
4. LSTM
5. Random Forest
6. XGBoost
7. Proposed ensemble model with ElasticNet Regression
8. Proposed ensemble model with Gradient Boosting
9. Proposed ensemble model with Weighted Averaging

#  Codes_2025 for Case Study To Forecast for 2025
Contains the source code for the 2025 real-world BEV market case study. 
The folder structure mirrors "Codes/" with the exact same pipeline stages (merging, EDA, KPIs, and visualisation) and identical result file naming convention apply.

# Datasets
1. Data/ - All datasets used by the scripts in Codes/ for Trials 1–3.
2. Data_2025/ - All datasets used by the scripts in Codes_2025/ for the case study.

## Note
The thesis document is also uploaded here for your reference. 











