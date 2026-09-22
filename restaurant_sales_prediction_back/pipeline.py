"""
pipeline.py
===========
Full ML pipeline:
  1. Load & clean data
  2. Fetch US holidays (Calendarific)
  3. Fetch historical weather (Open-Meteo) and merge into training data
  4. Feature engineering + daily aggregation
  5. Feature selection (Mutual Information + Permutation Importance)
  6. Normalization (StandardScaler on numeric features)
  7. EDA charts
  8. Temporal train/test split (last 90 days = test)
  9. Train 4 models with 5-fold CV + test-set evaluation
     Metrics: MAE, RMSE, R²
 10. Retrain best model (GBM) on full data for serving
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from weather import fetch_historical_weather
from models_training import TF_AVAILABLE

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH       = "restaurant_sales_data.csv"
CHARTS_DIR      = Path("charts")
CALENDARIFIC_KEY= "7QxKCPpi2iUcFgvNxbdWelf2AxrnJOyg"
COUNTRY         = "US"
TRAINING_YEARS  = [2022, 2023]
US_WEEKEND_DAYS = [5, 6]

CLOSURE_TYPES   = {"National holiday", "Public holiday"}

# Feature selection threshold — features with MI score below this fraction
# of the max MI score are dropped
MI_THRESHOLD = 0.05

# All candidate features before selection
ALL_FEATURES = [
    "ItemCode", "CatCode", "DayOfWeek", "Month",
    "IsWeekend", "Quarter", "IsHoliday",
    "AvgPrice", "Temperature", "Precipitation",
]
TARGET = "TotalQty"


# ─────────────────────────────────────────────────────────────────────────────
# App State
# ─────────────────────────────────────────────────────────────────────────────
class PipelineState:
    model           = None
    scaler          : StandardScaler = None
    le_item         : LabelEncoder   = None
    le_cat          : LabelEncoder   = None
    selected_features: list          = []
    item_avg_price  : dict           = {}
    item_category   : dict           = {}
    holidays        : pd.DatetimeIndex = pd.DatetimeIndex([])
    model_metrics   : dict           = {}
    deep_state      : dict           = {}   # RNN/LSTM models + metadata
    df_clean        : pd.DataFrame   = None  # cleaned tx-level df (for TS models)
    timeseries_metrics : dict        = {}   # daily-total model results
    sklearn_models  : dict           = {}   # all 4 sklearn models keyed by API name
    daily_scaled    : pd.DataFrame   = None  # scaled daily df for sequence building
    is_ready        : bool           = False
    startup_log     : list           = []

state = PipelineState()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    state.startup_log.append(f"Loaded {len(df):,} rows from '{DATA_PATH}'")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Clean & Impute
# ─────────────────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    for cat in df["Category"].unique():
        mode_item = df[df["Category"] == cat]["Item"].mode()
        if len(mode_item):
            df.loc[(df["Category"] == cat) & df["Item"].isna(), "Item"] = mode_item[0]

    df["Price"] = df.groupby("Item")["Price"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Quantity"]       = df["Quantity"].fillna(df["Quantity"].median())
    df["Order Total"]    = df["Order Total"].fillna(df["Price"] * df["Quantity"])
    df["Payment Method"] = df["Payment Method"].fillna(df["Payment Method"].mode()[0])

    df["Item"]     = df["Item"].str.strip().str.title()
    df["Category"] = df["Category"].str.strip().str.title()

    for col in ["Price", "Quantity", "Order Total"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    state.startup_log.append(
        f"Cleaning: {before:,} -> {len(df):,} rows "
        f"(removed {before - len(df):,} outlier rows)"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Fetch US Public Holidays
# ─────────────────────────────────────────────────────────────────────────────
def fetch_holidays(years: list) -> pd.DatetimeIndex:
    date_set = set()
    for year in years:
        try:
            resp = requests.get(
                "https://calendarific.com/api/v2/holidays",
                params={"api_key": CALENDARIFIC_KEY, "country": COUNTRY, "year": year},
                timeout=10,
            )
            data = resp.json()
            if data.get("response") and data["response"].get("holidays"):
                for h in data["response"]["holidays"]:
                    if not set(h.get("type", [])).intersection(CLOSURE_TYPES):
                        continue
                    dt = h["date"]["datetime"]
                    date_set.add(
                        pd.Timestamp(year=dt["year"], month=dt["month"], day=dt["day"])
                    )
        except Exception as e:
            state.startup_log.append(f"Warning: Calendarific failed for {year}: {e}")

    if date_set:
        idx = pd.DatetimeIndex(sorted(date_set))
        state.startup_log.append(
            f"Fetched {len(idx)} US public holiday dates for {years}"
        )
    else:
        fallback = []
        for yr in years:
            fallback += [
                f"{yr}-01-01", f"{yr}-07-04", f"{yr}-11-11", f"{yr}-12-25",
                f"{yr}-11-24" if yr == 2022 else f"{yr}-11-23",
            ]
        idx = pd.DatetimeIndex(pd.to_datetime(fallback))
        state.startup_log.append("Using fallback US federal holidays")
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Feature Engineering + Weather Merge + Daily Aggregation
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(
    df      : pd.DataFrame,
    holidays: pd.DatetimeIndex,
    weather : dict,          # date_str -> {temperature_mean, precipitation}
) -> pd.DataFrame:
    df = df.copy()
    df["DayOfWeek"] = df["Order Date"].dt.dayofweek
    df["Month"]     = df["Order Date"].dt.month
    df["Year"]      = df["Order Date"].dt.year
    df["Quarter"]   = df["Order Date"].dt.quarter
    df["IsWeekend"] = df["DayOfWeek"].isin(US_WEEKEND_DAYS).astype(int)
    df["IsHoliday"] = df["Order Date"].isin(holidays).astype(int)

    daily = (
        df.groupby(["Order Date", "Item", "Category"])
        .agg(
            TotalQty     = ("Quantity",    "sum"),
            TotalRevenue = ("Order Total", "sum"),
            AvgPrice     = ("Price",       "mean"),
        )
        .reset_index()
    )

    daily["DayOfWeek"] = daily["Order Date"].dt.dayofweek
    daily["Month"]     = daily["Order Date"].dt.month
    daily["IsWeekend"] = daily["DayOfWeek"].isin(US_WEEKEND_DAYS).astype(int)
    daily["Quarter"]   = daily["Order Date"].dt.quarter
    daily["IsHoliday"] = daily["Order Date"].isin(holidays).astype(int)

    # ── Merge weather ──────────────────────────────────────────────────────
    date_strs = daily["Order Date"].dt.strftime("%Y-%m-%d")
    daily["Temperature"]   = date_strs.map(
        lambda d: weather.get(d, {}).get("temperature_mean", np.nan)
    )
    daily["Precipitation"] = date_strs.map(
        lambda d: weather.get(d, {}).get("precipitation", np.nan)
    )

    # Fill NaNs — first try monthly mean from available data,
    # then fall back to hardcoded NYC climate averages if still missing
    # (happens when the weather API is completely unavailable)
    TEMP_FALLBACK  = {1:0.6,2:1.7,3:6.3,4:12.2,5:17.8,6:22.8,
                      7:25.6,8:24.9,9:20.6,10:14.2,11:8.3,12:2.7}
    PRECIP_FALLBACK= {1:3.6,2:3.0,3:4.3,4:4.0,5:4.4,6:4.3,
                      7:4.7,8:4.0,9:4.1,10:4.0,11:3.8,12:3.7}

    daily["Temperature"]   = daily["Temperature"].fillna(
        daily.groupby("Month")["Temperature"].transform("mean")
    )
    daily["Temperature"]   = daily["Temperature"].fillna(
        daily["Month"].map(TEMP_FALLBACK)
    )
    daily["Precipitation"] = daily["Precipitation"].fillna(
        daily.groupby("Month")["Precipitation"].transform("mean")
    )
    daily["Precipitation"] = daily["Precipitation"].fillna(
        daily["Month"].map(PRECIP_FALLBACK)
    )

    state.startup_log.append(
        f"Daily aggregation: {len(daily):,} item-day rows "
        f"(weather coverage: {date_strs.isin(weather.keys()).mean()*100:.1f}%)"
    )
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Training — delegates ALL model training + charts to models_training.py
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(daily: pd.DataFrame):
    """
    Slim orchestrator. All actual training logic and chart generation lives
    in models_training.py. This function:
      1. Encodes categoricals and stores lookup dicts on state
      2. Runs feature selection
      3. Delegates to models_training for: baseline sklearn, deep models,
         lag experiment, and daily-total time series models
      4. Stores everything the API needs on `state`
    """
    import models_training as MT

    daily = daily.copy()

    # ── Item-level metadata for prediction-time feature construction ─────────
    state.item_category  = (
        daily.drop_duplicates("Item").set_index("Item")["Category"].to_dict()
    )
    state.item_avg_price = daily.groupby("Item")["AvgPrice"].median().to_dict()

    # ── Encode categoricals ───────────────────────────────────────────────────
    le_item = LabelEncoder()
    le_cat  = LabelEncoder()
    daily["ItemCode"] = le_item.fit_transform(daily["Item"])
    daily["CatCode"]  = le_cat.fit_transform(daily["Category"])
    state.le_item = le_item
    state.le_cat  = le_cat

    # ── Feature selection (on base features, before lag) ─────────────────────
    available = [f for f in ALL_FEATURES if f in daily.columns]
    cutoff    = daily["Order Date"].max() - pd.Timedelta(days=90)
    train_df  = daily[daily["Order Date"] <= cutoff]
    selected  = MT.feature_selection(
        train_df[available], train_df[TARGET], state.startup_log
    )

    # ── Train ALL per-item models (4 sklearn + lag, RNN, LSTM) + diagnostics ──
    result = MT.train_per_item_models(daily, selected, state.startup_log)

    state.sklearn_models    = result["sklearn_models"]
    state.scaler            = result["scaler"]
    state.selected_features = result["features"]   # includes lag features
    state.model_metrics     = result["metrics"]
    state.deep_state        = (
        {"_models": result["deep_models"], "_lookback": MT.LOOKBACK_ITEM}
        if result["deep_models"] else {}
    )

    # ── Scaled lagged daily df — for deep-model prediction sequences at serve ─
    daily_lagged = result["daily_lagged"].dropna(subset=result["features"]).copy()
    daily_lagged[result["features"]] = result["scaler"].transform(
        daily_lagged[result["features"]]
    )
    state.daily_scaled = daily_lagged

    # ── Daily-total time series models (charts only, not served) ─────────────
    if state.df_clean is not None:
        ts_results = MT.train_timeseries_models(state.df_clean, state.startup_log)
        state.timeseries_metrics = ts_results
        state.startup_log.append("Time-series charts saved to charts/timeseries/")

    # ── Retrain best model (GBM) on full lagged data for serving ─────────────
    best_feats = result["features"]
    full_df    = result["daily_lagged"].dropna(subset=best_feats)
    best = GradientBoostingRegressor(n_estimators=100, random_state=42)
    best.fit(result["scaler"].transform(full_df[best_feats]), full_df[TARGET])
    state.model = best
    state.startup_log.append("GBM retrained on full data — ready for serving")


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    df       = load_data()
    df       = clean(df)
    holidays = fetch_holidays(TRAINING_YEARS)
    state.holidays = holidays

    # Fetch historical weather for the full training period
    state.startup_log.append("Fetching historical weather from Open-Meteo...")
    weather = fetch_historical_weather(
        start=f"{min(TRAINING_YEARS)}-01-01",
        end  =f"{max(TRAINING_YEARS)}-12-31",
    )
    state.startup_log.append(
        f"Weather data: {len(weather)} days fetched"
    )

    # Need time features on df for EDA before aggregation
    df["DayOfWeek"] = df["Order Date"].dt.dayofweek
    df["Month"]     = df["Order Date"].dt.month
    df["Year"]      = df["Order Date"].dt.year
    df["IsWeekend"] = df["DayOfWeek"].isin(US_WEEKEND_DAYS).astype(int)
    df["IsHoliday"] = df["Order Date"].isin(holidays).astype(int)

    daily = engineer_features(df, holidays, weather)
    state.df_clean = df          # keep for daily-total time series models
    import models_training as MT
    MT.run_eda(df, daily, state.startup_log)
    train_and_evaluate(daily)
    state.is_ready = True
    state.startup_log.append("Pipeline complete — API is ready")