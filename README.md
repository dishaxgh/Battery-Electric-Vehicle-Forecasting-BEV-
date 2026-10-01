# Development and evaluation of predictive machine learning methods for forecasting demand for battery electric vehicles.

## Master's Thesis Overview & Research Context

### 1. Problem Statement
* **Structural Market Break:** BEV adoption across EU member states accelerated sharply post-2020 due to stricter regulations, expanded infrastructure, lower battery costs, and post-COVID stimulus, rendering pre-2020 data unrepresentative.
* **Limitations of Conventional Models:** Standard forecasting assumes long historical time series remain stable, an assumption that fails during structural market regime shifts.
* **Research Gap:** Prior literature primarily focused on single markets (e.g., China), assumed structural stability, and lacked a unified multi-country framework covering all 27 EU member states.

### 2. Research Questions
* **RQ1:** To what degree can pre-2020 models forecast post-2020 adoption rates, and do EU states behave structurally differently post-COVID?
* **RQ2:** How does integrating transitional market data (2020–2022) influence predictive accuracy and bias?
* **RQ3:** Do regime-specific models trained exclusively on post-2020 data outperform models trained on longer historical windows?
* **RQ4:** Can cluster-specific models (segmenting countries into structural groups like Leaders vs. Followers) improve forecasting accuracy over a single pooled model?

### 3. Core Objective & Motivation
* **Objective:** Developed an advanced machine learning framework—featuring a novel ARIMAX-LSTM stacking ensemble across a three-trial experimental design and country clustering—to forecast BEV market share across all 27 EU member states (2011–2024).
* **Policy & Market Impact:** Provided vital near-term forecasts for infrastructure planners, energy grid operators, and policymakers addressing EU mandates (such as the 2035 internal combustion engine sales ban).
* **Methodological Insight:** Demonstrated that in fast-moving technology markets undergoing structural breaks, **data recency and structural coherence** are more valuable than historical data volume.

## About this Repository

This repository contains the complete codebase, datasets, results, and production deployment code for my master's thesis. The study benchmarks ensemble methods across multiple experimental trials and validates findings through a 2025 real-world case study; culminating in a fully containerized, cloud-deployed prediction API built specifically for the 2025 case study data.

[![Live API Status](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen)](https://battery-electric-vehicle-forecasting-bev.onrender.com/docs)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)

## Repository Structure

```text
Codes/                  # Main experiment code — Trials 1–3
├── Trial1/
├── Trial2/
└── Trial3/
Codes_2025/             # 2025 Case Study code
Data/                   # Datasets for Trials 1–3
Data_2025/              # Datasets for the 2025 Case Study
main.py                 # Production FastAPI backend & inference engine
Dockerfile              # Container configuration for cloud deployment
requirements.txt        # Pinned production dependencies
README.md
```

## Codes for Trial 1 to 3
Each trial subfolder contains a consistent set of scripts following the full experimental pipeline:

| Script | Description |
|--------|-------------|
| Merging of Datasets | Combines and pre-processes raw datasets into analysis-ready form |
| Exploratory Data Analysis | Distributions, trends, correlations and missing values imputation |
| KPI of the Results | Computes key performance indicators (MAE, MedAE) across models |
| Visualisation of the Results | Generates plots and comparative charts of forecasting outcomes |

## Models Evaluated

1. Linear Regression
2. Support Vector Rgeressor (SVR)
3. ARIMAX
4. LSTM
5. Random Forest
6. XGBoost
7. Proposed ensemble model with ElasticNet Regression as meta learner 
8. Proposed ensemble model with Gradient Boosting as meta learner
9. Proposed ensemble model with Weighted Averaging as meta learner

## Codes_2025 for Case Study To Forecast for 2025
Contains the source code for the 2025 real-world BEV market case study. 
The folder structure mirrors "Codes/" with the exact same pipeline stages (merging, EDA, KPIs, and visualisation) and identical result file naming convention apply.

## Datasets
1. Data/ - All datasets used by the scripts in Codes/ for Trials 1–3.
2. Data_2025/ - All datasets used by the scripts in Codes_2025/ for the case study.

## Production Architecture & Deployment

To bridge academic research and real-world software engineering, the core methodology has been productionized into a live cloud microservice:
1. **Backend Framework:** Built with FastAPI and Pydantic for automated payload validation and high-performance asynchronous request handling.
2. **Preprocessing Pipeline Simulation:** Incorporates production-aligned feature scaling and transformation logic mirroring the thesis data pipeline.
3. **Containerization:** Packaged using Docker (python:3.10-slim) for cross-platform reproducibility and deployed continuously via Render.

## Note
The master thesis document and presentation is also uploaded here for your reference. 











