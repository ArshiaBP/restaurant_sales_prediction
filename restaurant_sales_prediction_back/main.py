"""
Restaurant Sales Estimation — FastAPI Backend v2
=================================================

New in this version:
  - JWT token authentication (signup / login / protected endpoints)
  - SQLite database for user management
  - Weather features via Open-Meteo (temperature + precipitation)
  - Feature selection (Mutual Information + Permutation Importance)
  - Normalization (StandardScaler)
  - Prediction window reduced to 14 days (within Open-Meteo's 16-day forecast)

File structure:
  main.py       — FastAPI app + endpoints
  pipeline.py   — ML pipeline (clean, features, train, evaluate)
  auth.py       — JWT auth + signup/login endpoints
  database.py   — SQLAlchemy User model + SQLite setup
  weather.py    — Open-Meteo API wrapper

Run:
  uvicorn main:app --reload

Endpoints (all except /auth/* and /health require Bearer token):
  POST /auth/signup         — create account
  POST /auth/login          — get JWT token
  GET  /auth/me             — current user profile

  GET  /health              — pipeline status (public)
  GET  /metrics             — model evaluation scores (auth required)
  GET  /predict?date=&model= — sales prediction for a future date (auth required)
"""

from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import create_tables
from auth import router as auth_router, get_current_user, User
from pipeline import run_pipeline, state, fetch_holidays, TRAINING_YEARS, US_WEEKEND_DAYS
from weather import fetch_forecast_weather
from models_training import predict_with_deep_model, TF_AVAILABLE

# ─────────────────────────────────────────────────────────────────────────────
# Prediction date window
# ─────────────────────────────────────────────────────────────────────────────
# MIN_DAYS_AHEAD = 1
#   Today is partially elapsed — full-day predictions would be misleading.
#
# MAX_DAYS_AHEAD = 14
#   Open-Meteo provides real forecast data for up to 16 days.
#   We use 14 to stay safely within that window so every prediction
#   uses actual forecast temperature/precipitation, not climate averages.
#   This makes the weather feature genuinely informative.
MIN_DAYS_AHEAD = 1
MAX_DAYS_AHEAD = 14

# All models the user can choose from
VALID_MODELS = {
    "linear"           : "Linear Regression",
    "ridge"            : "Ridge Regression",
    "random_forest"    : "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "rnn"              : "RNN",
    "lstm"             : "LSTM",
}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Restaurant Sales Estimation API",
    description = (
        "Predicts per-item daily sales quantities for a future date. "
        f"Accepts dates between tomorrow and {MAX_DAYS_AHEAD} days from today. "
        "All prediction endpoints require Bearer token authentication."
    ),
    version = "2.0.0",
)

# Allow requests from the React dev server on any localhost port.
# During local development, the browser origin can be any localhost port
# depending on how Vite is configured — allowing all localhost variants
# prevents 400/403 on OPTIONS preflight requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

app.include_router(auth_router)


@app.on_event("startup")
def startup_event():
    create_tables()
    run_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response models
# ─────────────────────────────────────────────────────────────────────────────

class ItemPrediction(BaseModel):
    item               : str
    category           : str
    avg_price          : float
    predicted_quantity : float
    predicted_revenue  : float


class PredictionResponse(BaseModel):
    date                     : str
    is_weekend               : bool
    is_holiday               : bool
    weather                  : dict
    predictions              : list[ItemPrediction]
    total_predicted_quantity : float
    total_predicted_revenue  : float
    selected_features        : list[str]
    model_used               : str
    prediction_note          : str


class HealthResponse(BaseModel):
    status      : str
    is_ready    : bool
    startup_log : list[str]


class MetricsResponse(BaseModel):
    metrics          : dict
    best_model       : str
    selected_features: list[str]
    note             : str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Public — pipeline readiness check and startup log."""
    return HealthResponse(
        status      = "ready" if state.is_ready else "initializing",
        is_ready    = state.is_ready,
        startup_log = state.startup_log,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def metrics(current_user: User = Depends(get_current_user)):
    """
    Evaluation scores for all 4 models.
    Metrics: MAE, RMSE, R² (test set + 5-fold CV).
    Requires authentication.
    """
    if not state.is_ready:
        raise HTTPException(503, "Pipeline not ready yet — retry in a moment")
    return MetricsResponse(
        metrics           = state.model_metrics,
        best_model        = "Gradient Boosting",
        selected_features = state.selected_features,
        note              = (
            "cv_* = 5-fold cross-validation on training data (sklearn models only). "
            "test_* = held-out last 90 days (temporal split). "
            "RNN/LSTM have no CV — test set evaluation only. "
            "All models retrained on full data for serving."
        ),
    )


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    date        : str  = Query(
        ...,
        description=(
            f"Target date in YYYY-MM-DD format. "
            f"Must be between tomorrow and {MAX_DAYS_AHEAD} days from today."
        ),
        example="2024-06-15",
    ),
    model       : str  = Query(
        default="gradient_boosting",
        description=(
            "Model to use for prediction. "
            f"Options: {', '.join(VALID_MODELS.keys())}"
        ),
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Returns predicted quantity and revenue per menu item for the given date,
    using real weather forecast data from Open-Meteo.
    Requires authentication.

    **Date constraints**
    - Minimum: tomorrow
    - Maximum: today + 14 days (within Open-Meteo real forecast window)
    """
    if not state.is_ready:
        raise HTTPException(503, "Pipeline not ready yet — please retry")

    # ── Validate date ─────────────────────────────────────────────────────────
    try:
        target = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(422, "Invalid date format — use YYYY-MM-DD")

    today    = datetime.utcnow().date()
    min_date = today + timedelta(days=MIN_DAYS_AHEAD)
    max_date = today + timedelta(days=MAX_DAYS_AHEAD)

    if target < min_date:
        raise HTTPException(
            422,
            f"Date too early. Earliest allowed: {min_date} (tomorrow)."
        )
    if target > max_date:
        raise HTTPException(
            422,
            f"Date too far ahead. Latest allowed: {max_date} "
            f"({MAX_DAYS_AHEAD} days from today). "
            "Predictions are limited to 14 days to ensure real forecast weather data."
        )

    # ── Validate model choice ─────────────────────────────────────────────────
    model = model.lower().strip()
    if model not in VALID_MODELS:
        raise HTTPException(
            422,
            f"Unknown model '{model}'. "
            f"Valid options: {', '.join(VALID_MODELS.keys())}"
        )
    is_deep = model in ("rnn", "lstm")
    if is_deep and not TF_AVAILABLE:
        raise HTTPException(
            503,
            "TensorFlow is not installed. "
            "Install it with: pip install tensorflow"
        )
    if is_deep and not state.deep_state.get("_models"):
        raise HTTPException(
            503,
            f"{model.upper()} model is not available — training may have failed. "
            "Check /health for details."
        )

    # ── Time features ─────────────────────────────────────────────────────────
    target_ts  = pd.Timestamp(target)
    dow        = target_ts.dayofweek
    month      = target_ts.month
    is_weekend = int(dow in US_WEEKEND_DAYS)
    quarter    = target_ts.quarter

    # Holiday check
    if target_ts in state.holidays:
        is_holiday = 1
    elif target_ts.year not in TRAINING_YEARS:
        try:
            future_holidays = fetch_holidays([target_ts.year])
            is_holiday = int(target_ts in future_holidays)
        except Exception:
            is_holiday = 0
    else:
        is_holiday = 0

    # ── Weather forecast ──────────────────────────────────────────────────────
    weather_data  = fetch_forecast_weather(target)
    temperature   = weather_data["temperature_mean"]
    precipitation = weather_data["precipitation"]

    # ── Build feature rows for every known menu item ──────────────────────────
    rows = []
    for item in state.le_item.classes_:
        cat   = state.item_category.get(item, state.le_cat.classes_[0])
        price = state.item_avg_price.get(item, 5.0)
        try:
            item_code = int(state.le_item.transform([item])[0])
            cat_code  = int(state.le_cat.transform([cat])[0])
        except Exception:
            continue

        rows.append({
            "ItemCode"     : item_code,
            "CatCode"      : cat_code,
            "DayOfWeek"    : dow,
            "Month"        : month,
            "IsWeekend"    : is_weekend,
            "Quarter"      : quarter,
            "IsHoliday"    : is_holiday,
            "AvgPrice"     : price,
            "Temperature"  : temperature,
            "Precipitation": precipitation,
            "_item"        : item,
            "_cat"         : cat,
        })

    if not rows:
        raise HTTPException(500, "No menu items found in model state")

    # ── Apply feature selection + normalization + model routing ──────────────
    feat_df  = pd.DataFrame(rows)
    selected = state.selected_features

    if is_deep:
        # RNN / LSTM: predict per item using sequence history.
        # feat_df does NOT have lag columns (those come from state.daily_scaled
        # inside predict_with_deep_model). Only pass base features here.
        LAG_COLS = {"lag_1", "lag_7", "lag_14", "rolling_7_mean", "rolling_7_std"}
        feature_values = {
            feat: feat_df[feat].iloc[0]
            for feat in selected
            if feat not in ("ItemCode", "CatCode", "AvgPrice")
            and feat not in LAG_COLS
        }
        preds = np.array([
            predict_with_deep_model(
                model_name      = model.upper(),
                item            = row["_item"],
                feature_values  = {
                    **feature_values,
                    "ItemCode": row["ItemCode"],
                    "CatCode" : row["CatCode"],
                    "AvgPrice": row["AvgPrice"],
                },
                daily_scaled    = state.daily_scaled,
                selected_features = selected,
                deep_state      = state.deep_state,
            )
            for row in rows
        ])
    else:
        # Tabular sklearn models — need to add lag features per item
        # since selected_features now includes lag_1, lag_7, lag_14, rolling_7_mean, rolling_7_std
        LAG_COLS = ["lag_1", "lag_7", "lag_14", "rolling_7_mean", "rolling_7_std"]

        enriched_rows = []
        for row in rows:
            item     = row["_item"]
            item_hist = (
                state.daily_scaled[state.daily_scaled["Item"] == item]
                .sort_values("Order Date")
            )
            # Get the most recent lag values from training history
            lag_vals = {}
            if len(item_hist) >= 1:
                last = item_hist.iloc[-1]
                for col in LAG_COLS:
                    lag_vals[col] = float(last[col]) if col in last.index else 0.0
            else:
                for col in LAG_COLS:
                    lag_vals[col] = 0.0

            enriched_rows.append({**row, **lag_vals})

        feat_df  = pd.DataFrame(enriched_rows)
        X_scaled  = state.scaler.transform(feat_df[selected])
        raw_preds = state.sklearn_models[model].predict(X_scaled)
        preds     = np.clip(raw_preds, 0, None)

    predictions = sorted(
        [
            ItemPrediction(
                item               = row["_item"],
                category           = row["_cat"],
                avg_price          = round(row["AvgPrice"], 2),
                predicted_quantity = round(float(pred), 2),
                predicted_revenue  = round(float(pred) * row["AvgPrice"], 2),
            )
            for row, pred in zip(rows, preds)
        ],
        key=lambda x: x.predicted_quantity,
        reverse=True,
    )

    gbm_metrics = state.model_metrics.get("Gradient Boosting", {})

    return PredictionResponse(
        date                     = str(target),
        is_weekend               = bool(is_weekend),
        is_holiday               = bool(is_holiday),
        weather                  = weather_data,
        predictions              = predictions,
        total_predicted_quantity = round(float(sum(p.predicted_quantity for p in predictions)), 2),
        total_predicted_revenue  = round(float(sum(p.predicted_revenue  for p in predictions)), 2),
        selected_features        = selected,
        model_used               = VALID_MODELS[model],
        prediction_note          = (
            f"Weather source: {weather_data['source']}. "
            f"Model test-set R2={gbm_metrics.get('test_R2','N/A')}, "
            f"Features used: {selected}. "
            "Estimates are based on 2022-2023 historical patterns."
        ),
    )



# ─────────────────────────────────────────────────────────────────────────────
# Charts endpoints — serve generated chart images to the frontend
# ─────────────────────────────────────────────────────────────────────────────
CHART_CATEGORIES = {
    "eda"        : "Exploratory Data Analysis",
    "diagnostics": "Model Diagnostics",
    "lag"        : "Lag Features",
    "timeseries" : "Daily Total Time Series",
}


@app.get("/charts/list", tags=["Charts"])
def list_charts(current_user: User = Depends(get_current_user)):
    """
    Returns all generated chart files grouped by category.
    Categories map to subfolders under ./charts/.
    """
    result = {}
    charts_root = Path("charts")
    for cat_key, cat_label in CHART_CATEGORIES.items():
        cat_dir = charts_root / cat_key
        files = []
        if cat_dir.exists():
            files = sorted(
                f.name for f in cat_dir.iterdir()
                if f.suffix.lower() == ".png"
            )
        result[cat_key] = {"label": cat_label, "files": files}
    return result


@app.get("/charts/{category}/{filename}", tags=["Charts"])
def get_chart(category: str, filename: str):
    """
    Serves a single chart PNG. Public (no auth) so <img> tags can load it
    directly without needing Authorization headers.
    Path traversal is blocked by validating category + filename.
    """
    if category not in CHART_CATEGORIES:
        raise HTTPException(404, f"Unknown category '{category}'")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    if not filename.endswith(".png"):
        raise HTTPException(400, "Only PNG files are served")

    path = Path("charts") / category / filename
    if not path.exists():
        raise HTTPException(404, "Chart not found")
    return FileResponse(path, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)