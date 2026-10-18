# Public Transport Arrival Prediction

> **Advanced Machine Learning System** - Predicting public transport arrival delays using LightGBM, XGBoost, Random Forest, and Stacked Ensemble regression with realistic time-series features, temporal walk-forward cross-validation, SHAP interpretability, and an interactive Plotly dashboard.

---

## 1. Project Architecture & Highlights

- **Predictive Target**: Arrival delay in minutes (`delay_minutes`) across 8 metropolitan transit routes.
- **Algorithms**:
  - **LightGBM Regressor** (with Optuna hyperparameter optimization)
  - **XGBoost Regressor** (with Optuna hyperparameter optimization)
  - **Random Forest Regressor** (non-linear ensemble diversity)
  - **Ridge Regression** (regularized linear baseline & stacking meta-learner)
  - **Stacked Ensemble** (two-layer meta-regressor combining predictions)
- **Time-Series Feature Engineering**:
  - Cyclical temporal embeddings ($\sin/\cos$ for hour of day, day of week, month)
  - Autoregressive lag features (lag 1, 2, 3, 5, 10 for delay and traffic index per route)
  - Rolling window statistics (5 and 10 trips rolling mean, standard deviation, max)
  - Exponentially Weighted Moving Average (EWMA delay $\alpha=0.3$ per route without future leakage)
  - Cross-feature interactions (peak hour $\times$ weather severity, route length $\times$ weekend, passenger load $\times$ peak)
  - Route-level historical priors & weather condition categorical dummy indicators
- **Temporal Validation (Zero Leakage)**:
  - Strictly chronological train / validation / test splits by calendar date
  - 5-fold expanding-window **Walk-Forward Cross-Validation**
- **Comprehensive Error Analysis & Interpretability**:
  - Residual distributions, Q-Q normality tests, and actual vs. predicted diagnostics
  - Per-route, hourly peak-period, and weather condition MAE/RMSE segmentations
  - Delay severity confusion matrix (On-Time $\le 3\text{ min}$, Late $3-7\text{ min}$, Very Late $>7\text{ min}$)
  - Top-20 worst trip prediction anomaly detection exported to CSV
  - **SHAP (SHapley Additive exPlanations)** beeswarm summary & dependence curves
- **Interactive Plotly Dashboard**:
  - Standalone HTML application generated at [`outputs/reports/dashboard.html`](outputs/reports/dashboard.html) (no external web server required).

---

## 2. Directory Structure

```
Public Transport Arrival Prediction/
├── data/
│   ├── raw/
│   │   └── trips_raw.csv                 # Generated synthetic trips
│   └── processed/
│       └── trips_features.csv            # 54 engineered features
├── models/                               # Serialized model artifacts (.pkl)
│   ├── lightgbm.pkl
│   ├── xgboost.pkl
│   ├── randomforest.pkl
│   ├── ridge.pkl
│   └── ensemble.pkl
├── outputs/
│   ├── figures/                          # Diagnostic charts (.png)
│   │   ├── xgboost_residuals.png
│   │   ├── xgboost_error_by_route.png
│   │   ├── xgboost_error_by_time.png
│   │   ├── xgboost_error_by_weather.png
│   │   ├── xgboost_confusion.png
│   │   ├── lightgbm_shap_summary.png
│   │   └── model_comparison.png
│   └── reports/
│       ├── dashboard.html                # Interactive Plotly HTML dashboard
│       └── worst_predictions.csv         # Worst prediction breakdown
├── src/
│   ├── __init__.py                       # Package initializer & stream setup
│   ├── data_generator.py                 # Multi-route transit trip simulator
│   ├── feature_engineering.py            # Lag, rolling, cyclical, & EWMA features
│   ├── temporal_validation.py            # Walk-forward CV & evaluation metrics
│   ├── models.py                         # LightGBM, XGBoost, RF, Ridge, Stacking
│   ├── error_analysis.py                 # Residuals, slices, confusion matrix
│   ├── explainability.py                 # SHAP beeswarm & dependence plots
│   └── dashboard.py                      # Plotly HTML dashboard builder
├── main.py                               # Pipeline entry point & CLI
├── requirements.txt                      # Python dependencies
└── README.md                             # Documentation
```

---

## 3. Installation & Usage

### Setup Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Run Pipeline

```bash
# Fast run (recommended for testing: skips Optuna tuning, ~1 min)
python main.py --no-tune --no-shap --trips 15

# Full production pipeline with SHAP explainability
python main.py --no-tune --trips 30

# Full run with Optuna Bayesian hyperparameter search
python main.py --trips 40

# Launch Interactive Web Application & Simulator
python run_web.py
# Or run pipeline and automatically open web app upon completion
python main.py --no-tune --serve
```

### CLI Arguments
| Flag | Default | Description |
|---|---|---|
| `--no-tune` | `False` | Skip Optuna hyperparameter optimization for faster execution |
| `--no-shap` | `False` | Skip SHAP summary and dependence calculations |
| `--trips` | `40` | Number of daily trips generated per route |
| `--n-trials-lgbm` | `50` | Number of Optuna trials for LightGBM |
| `--n-trials-xgb` | `40` | Number of Optuna trials for XGBoost |
| `--cv-folds` | `5` | Number of folds for expanding walk-forward CV |

---

## 4. Benchmark Performance (Test Set Evaluation)

| Model | MAE (min) | RMSE (min) | $R^2$ Score | Within 3 min (%) | Within 5 min (%) |
|---|---|---|---|---|---|
| **XGBoost Regressor** | **2.264** | **3.147** | **0.5083** | **74.3%** | **89.7%** |
| **Stacked Ensemble** | 2.265 | 3.145 | 0.5087 | 74.3% | 89.9% |
| **LightGBM Regressor** | 2.267 | 3.153 | 0.5062 | 74.2% | 89.9% |
| **Random Forest** | 2.280 | 3.164 | 0.5027 | 73.8% | 89.9% |
| **Ridge Baseline** | 2.331 | 3.228 | 0.4827 | 73.3% | 89.2% |

*Note: Nearly 90% of bus arrival predictions are accurate to within 5 minutes of actual arrival time.*
