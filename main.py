"""
main.py
=======
Full end-to-end pipeline for Public Transport Arrival Prediction.

Steps
-----
1. Generate synthetic dataset
2. Feature engineering
3. Temporal train/val/test split
4. Walk-forward cross-validation (LightGBM baseline)
5. Train all models (LightGBM, XGBoost, Ridge, RF, Ensemble)
6. Evaluate on test set
7. Error analysis
8. SHAP explainability
9. Build interactive dashboard

Usage
-----
    python main.py                        # Full run with Optuna tuning
    python main.py --no-tune              # Skip Optuna (faster, ~3 min)
    python main.py --no-shap              # Skip SHAP (faster)
    python main.py --trips 50             # Trips per route per day (default 40)
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator       import generate_trips
from src.feature_engineering  import build_features, get_feature_columns
from src.temporal_validation  import TemporalSplit, WalkForwardCV, evaluate, print_metrics
from src.models               import build_all_models, LGBMModel
from src.error_analysis       import run_full_error_analysis, plot_model_comparison
from src.explainability       import run_full_explainability
from src.dashboard            import build_dashboard


# -- Simple coloured print using ANSI codes ------------------------------------
def _banner(text: str) -> None:
    print(f"\n\033[1;38;2;108;99;255m{'='*60}\033[0m")
    print(f"\033[1;38;2;108;99;255m  {text}\033[0m")
    print(f"\033[1;38;2;108;99;255m{'='*60}\033[0m")

def _rule(text: str) -> None:
    print(f"\n\033[1;34m--- {text} {'-'*max(0,50-len(text))}\033[0m")

def _ok(text: str) -> None:
    print(f"\033[32m  [OK] {text}\033[0m")

def _warn(text: str) -> None:
    print(f"\033[33m  [WARN] {text}\033[0m")

def _highlight(text: str) -> None:
    print(f"\033[1;96m  {text}\033[0m")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Public Transport Arrival Prediction")
    parser.add_argument("--no-tune",  action="store_true",  help="Skip Optuna tuning")
    parser.add_argument("--no-shap",  action="store_true",  help="Skip SHAP analysis")
    parser.add_argument("--trips",    type=int, default=40, help="Trips per route per day")
    parser.add_argument("--n-trials-lgbm", type=int, default=50)
    parser.add_argument("--n-trials-xgb",  type=int, default=40)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--serve",    action="store_true",  help="Launch interactive web app server upon completion")
    return parser.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    t0   = time.time()

    _banner("Public Transport Arrival Prediction")
    print("  LightGBM | XGBoost | Stacked Ensemble | SHAP | Optuna")

    # ── Step 1: Data Generation ───────────────────────────────────────────────
    _rule("Step 1: Data Generation")
    df_raw = generate_trips(trips_per_route_per_day=args.trips)

    # ── Step 2: Feature Engineering ──────────────────────────────────────────
    _rule("Step 2: Feature Engineering")
    df = build_features(df_raw)
    feature_cols = get_feature_columns(df)
    _ok(f"{len(feature_cols)} features built")

    # Drop rows with NaN lag features (first few trips per route)
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
    _ok(f"Clean dataset: {len(df_clean):,} rows")

    # ── Step 3: Temporal Split ────────────────────────────────────────────────
    _rule("Step 3: Temporal Train / Val / Test Split")
    splitter           = TemporalSplit(val_frac=0.15, test_frac=0.15)
    df_train, df_val, df_test = splitter.split(df_clean)

    X_train = df_train[feature_cols].values
    y_train = df_train["delay_minutes"].values
    X_val   = df_val[feature_cols].values
    y_val   = df_val["delay_minutes"].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test["delay_minutes"].values

    # ── Step 4: Walk-Forward Cross-Validation ─────────────────────────────────
    _rule("Step 4: Walk-Forward Cross-Validation (LightGBM)")
    wf = WalkForwardCV(n_folds=args.cv_folds)

    def _lgbm_factory(X, y):
        m = LGBMModel()
        m.fit(X, y, feature_names=feature_cols)
        return m

    cv_results = wf.cross_validate(
        df_clean, feature_cols, "delay_minutes", _lgbm_factory
    )

    # ── Step 5: Train All Models ──────────────────────────────────────────────
    _rule("Step 5: Train All Models")
    models = build_all_models(
        X_train, y_train, X_val, y_val,
        feature_names  = feature_cols,
        tune           = not args.no_tune,
        n_trials_lgbm  = args.n_trials_lgbm,
        n_trials_xgb   = args.n_trials_xgb,
    )

    # ── Step 6: Evaluate on Test Set ──────────────────────────────────────────
    _rule("Step 6: Test Set Evaluation")
    all_metrics = []
    for name, model in models.items():
        preds   = model.predict(X_test)
        metrics = evaluate(y_test, preds, label=name)
        all_metrics.append(metrics)
        print_metrics(metrics)

    # Print summary table
    print("\n" + "-"*90)
    header = f"  {'Model':<18} {'MAE':>7} {'RMSE':>7} {'R2':>8} {'MAPE%':>7} {'W3min%':>8} {'W5min%':>8}"
    print(f"\033[1;34m{header}\033[0m")
    print("-"*90)
    for m in all_metrics:
        print(f"  {m['label']:<18} {m['MAE']:>7.3f} {m['RMSE']:>7.3f} {m['R2']:>8.4f} "
              f"{m['MAPE']:>7.1f} {m['W3min%']:>8.1f} {m['W5min%']:>8.1f}")
    print("-"*90)

    # Identify best model by MAE
    best_name   = min(all_metrics, key=lambda x: x["MAE"])["label"]
    best_model  = models[best_name]
    y_pred_best = best_model.predict(X_test)
    _highlight(f"Best model: {best_name}  MAE={min(all_metrics, key=lambda x: x['MAE'])['MAE']:.4f}")

    # ── Step 7: Error Analysis ────────────────────────────────────────────────
    _rule("Step 7: Error Analysis")
    run_full_error_analysis(df_test, y_pred_best, best_name)
    plot_model_comparison(all_metrics)

    # ── Step 8: SHAP Explainability ───────────────────────────────────────────
    if not args.no_shap:
        _rule("Step 8: SHAP Explainability")
        run_full_explainability(models, X_test, feature_cols)
    else:
        _warn("SHAP skipped (--no-shap)")

    # ── Step 9: Dashboard ─────────────────────────────────────────────────────
    _rule("Step 9: Interactive Dashboard")
    dashboard_path = build_dashboard(
        df_train      = df_train,
        df_test       = df_test,
        y_pred_best   = y_pred_best,
        metrics_list  = all_metrics,
        cv_results    = cv_results,
        best_model    = best_model,
        feature_names = feature_cols,
        best_model_name = best_name,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    _banner(f"Pipeline Complete  ({elapsed:.0f}s)")
    _ok(f"Figures  -> outputs/figures/")
    _ok(f"Reports  -> outputs/reports/")
    _ok(f"Models   -> models/")
    _ok(f"Dashboard-> {dashboard_path}")

    # ── Step 10: Web Application ──────────────────────────────────────────────
    if args.serve:
        _rule("Step 10: Launching Web Application")
        from app import start_server
        start_server(port=5000)


if __name__ == "__main__":
    main()
