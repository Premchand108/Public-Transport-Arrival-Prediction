"""
app.py
======
Flask Web Application for Public Transport Arrival Prediction.
Provides:
  - Interactive Web Dashboard
  - Live Real-Time Delay Prediction Simulator (Ensemble, XGBoost, LightGBM, RF, Ridge)
  - Model Benchmarks & Metrics Explorer
  - Route Network Analytics
  - Residuals & Error Diagnostics
  - SHAP Interpretability Explorer
  - Anomaly & Worst Predictions Inspector
"""

import os
import sys
import json
import time
import threading
import webbrowser
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_generator import ROUTES, WEATHER_CONDITIONS
from src.feature_engineering import WEATHER_SEVERITY

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Load Pre-trained Models ───────────────────────────────────────────────────
MODELS = {}
MODEL_NAMES = ["ensemble", "xgboost", "lightgbm", "randomforest", "ridge"]

for name in MODEL_NAMES:
    model_path = BASE_DIR / "models" / f"{name}.pkl"
    if model_path.exists():
        try:
            MODELS[name] = joblib.load(model_path)
            print(f"Loaded model: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

# Load route reference statistics for live feature building
ROUTE_STATS = {}
stats_path = BASE_DIR / "models" / "route_stats.json"
if stats_path.exists():
    with open(stats_path, "r", encoding="utf-8") as f:
        ROUTE_STATS = json.load(f)

# Benchmark Test Metrics from evaluation
METRICS_DATA = [
    {"label": "Ridge", "MAE": 2.198, "RMSE": 3.034, "R2": 0.4995, "MAPE": 171.3, "W3min%": 75.5, "W5min%": 91.5},
    {"label": "RandomForest", "MAE": 2.163, "RMSE": 2.997, "R2": 0.5118, "MAPE": 176.3, "W3min%": 76.1, "W5min%": 91.5},
    {"label": "LightGBM", "MAE": 2.160, "RMSE": 2.992, "R2": 0.5132, "MAPE": 175.4, "W3min%": 76.2, "W5min%": 91.5},
    {"label": "XGBoost", "MAE": 2.156, "RMSE": 2.985, "R2": 0.5155, "MAPE": 174.8, "W3min%": 76.3, "W5min%": 91.5},
    {"label": "Ensemble", "MAE": 2.154, "RMSE": 2.981, "R2": 0.5169, "MAPE": 176.3, "W3min%": 76.3, "W5min%": 91.7},
]

# Walk-Forward Cross Validation Results
CV_DATA = [
    {"fold": 1, "MAE": 2.172, "RMSE": 3.044, "R2": 0.5010, "W3min": 75.9, "W5min": 91.1},
    {"fold": 2, "MAE": 2.357, "RMSE": 3.294, "R2": 0.5153, "W3min": 72.9, "W5min": 89.1},
    {"fold": 3, "MAE": 2.045, "RMSE": 2.835, "R2": 0.4663, "W3min": 78.2, "W5min": 92.7},
    {"fold": 4, "MAE": 2.188, "RMSE": 3.072, "R2": 0.4934, "W3min": 75.8, "W5min": 90.7},
    {"fold": 5, "MAE": 2.249, "RMSE": 3.106, "R2": 0.5137, "W3min": 74.8, "W5min": 90.6},
]

# Top SHAP Feature Importance Rankings
SHAP_FEATURES = [
    {"feature": "lag_1_delay", "importance": 0.284, "desc": "Previous trip delay on same route"},
    {"feature": "ewm_delay", "importance": 0.211, "desc": "Exponentially weighted moving avg delay"},
    {"feature": "traffic_index", "importance": 0.165, "desc": "Corridor congestion index"},
    {"feature": "traffic_x_peak", "importance": 0.089, "desc": "Congestion × Peak hour interaction"},
    {"feature": "roll_5_mean", "importance": 0.076, "desc": "Rolling mean delay over last 5 trips"},
    {"feature": "hour_sin", "importance": 0.052, "desc": "Time of day cyclical sinusoidal embedding"},
    {"feature": "hour_cos", "importance": 0.043, "desc": "Time of day cyclical cosine embedding"},
    {"feature": "route_hist_mean", "importance": 0.038, "desc": "Route baseline historical delay"},
    {"feature": "peak_x_weather", "importance": 0.035, "desc": "Rush hour × Weather severity penalty"},
    {"feature": "weather_severity", "importance": 0.031, "desc": "Adverse weather scale (0-5)"},
    {"feature": "passenger_load", "importance": 0.024, "desc": "Bus boarding / occupancy volume"},
    {"feature": "lag_2_delay", "importance": 0.019, "desc": "Delay two trips prior"},
    {"feature": "stops_x_weather", "importance": 0.015, "desc": "Number of stops × weather impact"},
    {"feature": "route_length_km", "importance": 0.012, "desc": "Total distance traveled by route"},
    {"feature": "load_x_peak", "importance": 0.010, "desc": "High passenger load during rush hour"},
]


# ── Feature Construction Helper ──────────────────────────────────────────────
def prepare_feature_vector(params: dict):
    """
    Constructs a 54-element feature vector matching models' training format.
    """
    route_id = params.get("route_id", "R01")
    hour = int(params.get("hour", 17))
    minute = int(params.get("minute", 30))
    dow = int(params.get("day_of_week", 1))   # 0=Mon, 6=Sun
    month = int(params.get("month", 5))
    weather = params.get("weather", "Clear")
    traffic_index = float(params.get("traffic_index", 52.0))
    passenger_load = float(params.get("passenger_load", 0.75))
    special_event = int(params.get("special_event", 0))

    route_info = ROUTES.get(route_id, ROUTES["R01"])
    route_length = route_info["length_km"]
    n_stops = route_info["stops"]

    is_weekend = 1 if dow in (5, 6) else 0
    is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0

    # Congestion multiplier
    if is_weekend:
        congestion_mult = 1.3 if (10 <= hour <= 14) else (1.2 if (18 <= hour <= 21) else 0.85)
    else:
        if 7 <= hour <= 9: congestion_mult = 1.9
        elif 12 <= hour <= 13: congestion_mult = 1.3
        elif 17 <= hour <= 19: congestion_mult = 2.0
        elif 0 <= hour <= 5: congestion_mult = 0.6
        else: congestion_mult = 1.0

    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    mins_since_midnight = hour * 60 + minute

    # Weather
    weather_severity = WEATHER_SEVERITY.get(weather, 0)
    wx_Cloudy = 1 if weather == "Cloudy" else 0
    wx_Fog = 1 if weather == "Fog" else 0
    wx_HeavyRain = 1 if weather == "HeavyRain" else 0
    wx_Rain = 1 if weather == "Rain" else 0
    wx_Snow = 1 if weather == "Snow" else 0

    # Route stats & lags from precomputed cache
    stats = ROUTE_STATS.get(route_id, {})
    route_hist_mean = stats.get("route_hist_mean", route_info["base_delay"])
    route_hist_std  = stats.get("route_hist_std", 3.0)
    route_hist_max  = stats.get("route_hist_max", 30.0)

    # Scale lags by current traffic index / nominal
    nominal_traffic = stats.get("lag_1_traffic", 50.0)
    traffic_ratio = max(0.5, traffic_index / (nominal_traffic if nominal_traffic > 0 else 50.0))

    lag_1_delay = stats.get("lag_1_delay", route_hist_mean) * traffic_ratio
    lag_1_traffic = traffic_index
    lag_2_delay = stats.get("lag_2_delay", route_hist_mean) * traffic_ratio
    lag_2_traffic = stats.get("lag_2_traffic", traffic_index)
    lag_3_delay = stats.get("lag_3_delay", route_hist_mean) * traffic_ratio
    lag_3_traffic = stats.get("lag_3_traffic", traffic_index)
    lag_5_delay = stats.get("lag_5_delay", route_hist_mean) * traffic_ratio
    lag_5_traffic = stats.get("lag_5_traffic", traffic_index)
    lag_10_delay = stats.get("lag_10_delay", route_hist_mean) * traffic_ratio
    lag_10_traffic = stats.get("lag_10_traffic", traffic_index)

    roll_5_mean = stats.get("roll_5_mean", route_hist_mean) * traffic_ratio
    roll_5_std  = stats.get("roll_5_std", 2.5)
    roll_5_max  = stats.get("roll_5_max", route_hist_mean * 1.8) * traffic_ratio
    roll_10_mean = stats.get("roll_10_mean", route_hist_mean) * traffic_ratio
    roll_10_std  = stats.get("roll_10_std", 2.8)
    roll_10_max  = stats.get("roll_10_max", route_hist_mean * 2.1) * traffic_ratio
    ewm_delay    = stats.get("ewm_delay", route_hist_mean) * traffic_ratio

    # Interactions
    peak_x_weather  = is_peak * weather_severity
    route_x_weekend = route_length * is_weekend
    traffic_x_peak  = traffic_index * is_peak
    load_x_peak     = passenger_load * is_peak
    stops_x_weather = n_stops * weather_severity

    route_keys = sorted(list(ROUTES.keys()))
    route_enc = route_keys.index(route_id) if route_id in route_keys else 0
    quarter = (month - 1) // 3 + 1
    is_monday = 1 if dow == 0 else 0
    is_friday = 1 if dow == 4 else 0

    feature_dict = {
        "route_length_km": route_length,
        "n_stops": n_stops,
        "hour": hour,
        "minute": minute,
        "day_of_week": dow,
        "month": month,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak,
        "traffic_index": traffic_index,
        "passenger_load": passenger_load,
        "special_event": special_event,
        "congestion_mult": congestion_mult,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "mins_since_midnight": mins_since_midnight,
        "weather_severity": weather_severity,
        "wx_Cloudy": wx_Cloudy,
        "wx_Fog": wx_Fog,
        "wx_HeavyRain": wx_HeavyRain,
        "wx_Rain": wx_Rain,
        "wx_Snow": wx_Snow,
        "route_hist_mean": route_hist_mean,
        "route_hist_std": route_hist_std,
        "route_hist_max": route_hist_max,
        "lag_1_delay": lag_1_delay,
        "lag_1_traffic": lag_1_traffic,
        "lag_2_delay": lag_2_delay,
        "lag_2_traffic": lag_2_traffic,
        "lag_3_delay": lag_3_delay,
        "lag_3_traffic": lag_3_traffic,
        "lag_5_delay": lag_5_delay,
        "lag_5_traffic": lag_5_traffic,
        "lag_10_delay": lag_10_delay,
        "lag_10_traffic": lag_10_traffic,
        "roll_5_mean": roll_5_mean,
        "roll_5_std": roll_5_std,
        "roll_5_max": roll_5_max,
        "roll_10_mean": roll_10_mean,
        "roll_10_std": roll_10_std,
        "roll_10_max": roll_10_max,
        "ewm_delay": ewm_delay,
        "peak_x_weather": peak_x_weather,
        "route_x_weekend": route_x_weekend,
        "traffic_x_peak": traffic_x_peak,
        "load_x_peak": load_x_peak,
        "stops_x_weather": stops_x_weather,
        "route_enc": route_enc,
        "quarter": quarter,
        "is_monday": is_monday,
        "is_friday": is_friday,
    }

    return pd.DataFrame([feature_dict])


# ── Web Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        routes=ROUTES,
        metrics=METRICS_DATA,
        cv_data=CV_DATA,
        shap_features=SHAP_FEATURES,
        weather_conditions=WEATHER_CONDITIONS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json or {}
        model_name = data.get("model_name", "ensemble").lower()
        if model_name not in MODELS:
            model_name = "ensemble" if "ensemble" in MODELS else list(MODELS.keys())[0]

        model = MODELS[model_name]
        df_feats = prepare_feature_vector(data)

        # Align columns
        cols = getattr(model, "feature_names_", list(df_feats.columns))
        X = df_feats[cols].values

        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)  # Truncate negative delays

        # Status categorization
        if pred <= 3.0:
            status = "On-Time"
            status_class = "ontime"
            status_desc = "Minor variance (within 3 min scheduled arrival)"
        elif pred <= 7.0:
            status = "Moderate Delay"
            status_class = "delayed"
            status_desc = "Moderate delay expected (3 to 7 min window)"
        else:
            status = "Severe Delay"
            status_class = "critical"
            status_desc = "Major transit delay expected (exceeds 7 min)"

        # Estimated arrival time calculation
        dep_hour = int(data.get("hour", 17))
        dep_min = int(data.get("minute", 30))
        route_id = data.get("route_id", "R01")
        route_info = ROUTES.get(route_id, ROUTES["R01"])
        nominal_travel_mins = int(route_info["length_km"] * 1.8 + route_info["stops"] * 0.8)
        total_trip_mins = nominal_travel_mins + pred

        arrival_hour = (dep_hour + int((dep_min + total_trip_mins) // 60)) % 24
        arrival_min = int((dep_min + total_trip_mins) % 60)
        arrival_time_str = f"{arrival_hour:02d}:{arrival_min:02d}"

        # 95% Confidence Interval band based on test RMSE
        rmse = 2.981
        ci_lower = max(0.0, round(pred - 1.96 * (rmse / np.sqrt(5)), 2))
        ci_upper = round(pred + 1.96 * (rmse / np.sqrt(5)), 2)

        # Contributing factor breakdown
        base_delay = route_info["base_delay"]
        weather_name = data.get("weather", "Clear")
        weather_factor = WEATHER_SEVERITY.get(weather_name, 0) * 0.95
        traffic_idx = float(data.get("traffic_index", 52))
        traffic_factor = max(0.0, (traffic_idx - 40) * 0.08)
        is_peak = 1 if (7 <= dep_hour <= 9 or 17 <= dep_hour <= 19) else 0
        peak_factor = 1.45 if is_peak else 0.2

        factors = [
            {"name": "Base Route Delay", "impact": round(base_delay, 2), "percent": 30},
            {"name": "Corridor Congestion", "impact": round(traffic_factor, 2), "percent": 35},
            {"name": "Weather Conditions", "impact": round(weather_factor, 2), "percent": 20},
            {"name": "Peak Period Surcharge", "impact": round(peak_factor, 2), "percent": 15},
        ]

        return jsonify({
            "success": True,
            "model_used": model_name.capitalize(),
            "predicted_delay_minutes": round(pred, 2),
            "status": status,
            "status_class": status_class,
            "status_description": status_desc,
            "scheduled_departure": f"{dep_hour:02d}:{dep_min:02d}",
            "nominal_travel_mins": nominal_travel_mins,
            "estimated_arrival": arrival_time_str,
            "confidence_interval": [ci_lower, ci_upper],
            "factors": factors,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/worst", methods=["GET"])
def api_worst():
    worst_path = BASE_DIR / "outputs" / "reports" / "worst_predictions.csv"
    if worst_path.exists():
        df = pd.read_csv(worst_path)
        return jsonify({"success": True, "records": df.to_dict(orient="records")})
    return jsonify({"success": False, "records": []})


@app.route("/api/routes", methods=["GET"])
def api_routes():
    res = []
    for rid, r in ROUTES.items():
        stats = ROUTE_STATS.get(rid, {})
        res.append({
            "route_id": rid,
            "name": r["name"],
            "length_km": r["length_km"],
            "stops": r["stops"],
            "base_delay": r["base_delay"],
            "hist_mean": round(stats.get("route_hist_mean", r["base_delay"]), 2),
            "hist_max": round(stats.get("route_hist_max", 30.0), 2),
        })
    return jsonify({"success": True, "routes": res})


@app.route("/figures/<path:filename>")
def get_figure(filename):
    fig_dir = BASE_DIR / "outputs" / "figures"
    return send_from_directory(fig_dir, filename)


@app.route("/dashboard-report")
def get_dashboard_html():
    dash_path = BASE_DIR / "outputs" / "reports" / "dashboard.html"
    if dash_path.exists():
        return send_file(dash_path)
    return "Dashboard report not found. Run pipeline first.", 404


def launch_browser(port):
    time.sleep(1.0)
    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"  PulseTransit AI Web Application Launched")
    print(f"  URL: {url}")
    print(f"=======================================================\n")
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass


def start_server(port=5000):
    threading.Thread(target=launch_browser, args=(port,), daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port to run web app")
    args = parser.parse_args()
    start_server(port=args.port)
