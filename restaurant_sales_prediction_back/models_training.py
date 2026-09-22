"""
models_training.py
==================
Single consolidated training module. ALL model training and ALL chart
generation lives here. One best version of each model — no duplicate variants.

Per-item models (power the live /predict API):
    Linear, Ridge, Random Forest, Gradient Boosting  — with lag features (best version)
    RNN, LSTM                                          — 7-day sequences (best version)

Daily-total models (for the charts / comparison only, NOT served):
    Gradient Boosting, LSTM                            — aggregated daily totals

Chart folders (match the frontend Charts page tabs):
    charts/eda/          — exploratory data analysis
    charts/diagnostics/  — per-item model evaluation (all 6 models)
    charts/timeseries/   — daily-total model comparison

Public functions used by pipeline.py:
    run_eda(df, daily, log)
    feature_selection(X, y, log)
    train_per_item_models(daily, base_features, log)   -> everything for /predict + diagnostics
    train_timeseries_models(df_clean, log)             -> daily-total charts
    predict_with_deep_model(...)                        -> serving helper for /predict
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

# ── TensorFlow (optional) ─────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Chart directories
# ─────────────────────────────────────────────────────────────────────────────
CHARTS_ROOT     = Path("charts")
DIR_EDA         = CHARTS_ROOT / "eda"
DIR_DIAGNOSTICS = CHARTS_ROOT / "diagnostics"
DIR_LAG         = CHARTS_ROOT / "lag"
DIR_TIMESERIES  = CHARTS_ROOT / "timeseries"

TARGET = "TotalQty"

# Deep learning hyperparameters (regularized versions)
LOOKBACK_ITEM  = 7
LOOKBACK_DAILY = 14
EPOCHS         = 80    # more budget but early stopping will cut short
EPOCHS_DAILY   = 100
BATCH_SIZE     = 32
PATIENCE       = 7     # tighter early stopping to prevent RNN overfitting
PATIENCE_DAILY = 15

# Lag/rolling feature names added to every per-item model
LAG_FEATURES = ["lag_1", "lag_7", "lag_14", "rolling_7_mean", "rolling_7_std"]

MODEL_COLORS = {
    "Linear Regression"  : "#378ADD",
    "Ridge Regression"   : "#97C459",
    "Random Forest"      : "#EF9F27",
    "Gradient Boosting"  : "#E05C5C",
    "RNN"                : "#9B59B6",
    "LSTM"               : "#1ABC9C",
    "GBM (daily total)"  : "#E05C5C",
    "LSTM (daily total)" : "#1ABC9C",
}

# API-key aliases for the 4 sklearn models (used by main.py /predict routing)
SKLEARN_API_KEYS = {
    "Linear Regression" : "linear",
    "Ridge Regression"  : "ridge",
    "Random Forest"     : "random_forest",
    "Gradient Boosting" : "gradient_boosting",
}


def _metrics(y_true, y_pred):
    return {
        "MAE" : round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2"  : round(float(r2_score(y_true, y_pred)), 3),
    }


def _make_dirs():
    for d in [DIR_EDA, DIR_DIAGNOSTICS, DIR_LAG, DIR_TIMESERIES]:
        d.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# EDA CHARTS                                                    -> charts/eda/
# ═════════════════════════════════════════════════════════════════════════════
def run_eda(df: pd.DataFrame, daily: pd.DataFrame, log: list):
    _make_dirs()

    # Monthly revenue trend
    monthly = df.groupby(["Year", "Month"])["Order Total"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    for year, grp in monthly.groupby("Year"):
        ax.plot(grp["Month"], grp["Order Total"], marker="o", label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_title("Monthly Revenue by Year"); ax.set_ylabel("Revenue ($)"); ax.legend()
    fig.tight_layout(); fig.savefig(DIR_EDA / "monthly_revenue.png", dpi=150); plt.close(fig)

    # Revenue by category
    cat_rev = df.groupby("Category")["Order Total"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    cat_rev.plot(kind="bar", color="#378ADD", edgecolor="none", ax=ax)
    ax.set_title("Total Revenue by Category"); ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(DIR_EDA / "category_revenue.png", dpi=150); plt.close(fig)

    # Top 10 items
    top10 = df.groupby("Item")["Quantity"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    top10.sort_values().plot(kind="barh", color="#97C459", edgecolor="none", ax=ax)
    ax.set_title("Top 10 Items by Quantity Sold"); ax.set_xlabel("Total Quantity")
    fig.tight_layout(); fig.savefig(DIR_EDA / "top_items.png", dpi=150); plt.close(fig)

    # Avg order value by day of week
    dow_rev = df.groupby("DayOfWeek")["Order Total"].mean()
    labels  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, [dow_rev.get(i, 0) for i in range(7)], color="#EF9F27", edgecolor="none")
    ax.set_title("Average Order Value by Day of Week"); ax.set_ylabel("Avg Order ($)")
    fig.tight_layout(); fig.savefig(DIR_EDA / "day_of_week.png", dpi=150); plt.close(fig)

    # Payment method split
    pay = df["Payment Method"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(pay.values, labels=pay.index, autopct="%1.1f%%",
           colors=["#378ADD","#97C459","#EF9F27"], startangle=140)
    ax.set_title("Payment Method Distribution")
    fig.tight_layout(); fig.savefig(DIR_EDA / "payment_methods.png", dpi=150); plt.close(fig)

    # Correlation matrix
    corr_cols = [c for c in ["TotalQty","TotalRevenue","AvgPrice","DayOfWeek","Month",
                 "IsWeekend","Quarter","IsHoliday","Temperature","Precipitation"]
                 if c in daily.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(daily[corr_cols].corr(), annot=True, fmt=".2f",
                cmap="Blues", linewidths=0.4, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout(); fig.savefig(DIR_EDA / "correlation_matrix.png", dpi=150); plt.close(fig)

    # Temperature vs sales
    if "Temperature" in daily.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(daily["Temperature"], daily["TotalQty"],
                   alpha=0.3, color="#378ADD", edgecolor="none", s=15)
        ax.set_title("Temperature vs Daily Sales"); ax.set_xlabel("Temp (°C)"); ax.set_ylabel("Qty")
        fig.tight_layout(); fig.savefig(DIR_EDA / "weather_vs_sales.png", dpi=150); plt.close(fig)

    log.append("EDA charts saved to charts/eda/")


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═════════════════════════════════════════════════════════════════════════════
def feature_selection(X: pd.DataFrame, y: pd.Series, log: list, mi_threshold=0.05):
    _make_dirs()
    mi = pd.Series(mutual_info_regression(X, y, random_state=42),
                   index=X.columns).sort_values(ascending=False)
    mi_sel = set(mi[mi >= mi.max() * mi_threshold].index)

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    ps = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    perm_sel = set(ps[ps >= max(0.0, ps.max() * mi_threshold)].index)

    selected = sorted(mi_sel | perm_sel, key=lambda f: X.columns.tolist().index(f))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mi.plot(kind="barh", ax=axes[0], color="#378ADD", edgecolor="none")
    axes[0].set_title("Mutual Information")
    ps.plot(kind="barh", ax=axes[1], color="#97C459", edgecolor="none")
    axes[1].set_title("Permutation Importance")
    fig.suptitle("Feature Selection")
    fig.tight_layout(); fig.savefig(DIR_DIAGNOSTICS / "feature_selection.png", dpi=150)
    plt.close(fig)

    log.append(f"Feature selection: {len(X.columns)} -> {len(selected)}: {selected}")
    return selected


# ═════════════════════════════════════════════════════════════════════════════
# LAG FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def add_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["Item", "Order Date"]).copy()
    daily["lag_1"]  = daily.groupby("Item")[TARGET].shift(1)
    daily["lag_7"]  = daily.groupby("Item")[TARGET].shift(7)
    daily["lag_14"] = daily.groupby("Item")[TARGET].shift(14)
    daily["rolling_7_mean"] = daily.groupby("Item")[TARGET].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    daily["rolling_7_std"] = daily.groupby("Item")[TARGET].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).std().fillna(0))
    return daily.dropna(subset=["lag_1", "lag_7", "lag_14", "rolling_7_mean"])


# ═════════════════════════════════════════════════════════════════════════════
# DEEP MODEL BUILDERS
# ═════════════════════════════════════════════════════════════════════════════
def _rnn_model(n_features, lookback):
    """
    SimpleRNN with stronger regularization to address the severe overfitting
    seen in training (train loss ~7.5 vs val loss ~16.5 at epoch 30).
    Fixes: smaller units, higher dropout, stronger L2, lower learning rate.
    """
    m = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.SimpleRNN(64, activation="tanh", return_sequences=True,
                         kernel_regularizer=regularizers.l2(1e-3),
                         recurrent_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.4),
        layers.SimpleRNN(32, activation="tanh",
                         kernel_regularizer=regularizers.l2(1e-3),
                         recurrent_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.4),
        layers.Dense(16, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])
    return m


def _lstm_model(n_features, lookback):
    """
    LSTM with increased regularization to close the gap between flat val loss
    and still-decreasing train loss observed from epoch 5 onward.
    Fixes: smaller units, higher dropout, stronger L2, lower learning rate.
    """
    m = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(64, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-3),
                    recurrent_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.35),
        layers.LSTM(32,
                    kernel_regularizer=regularizers.l2(1e-3),
                    recurrent_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.35),
        layers.Dense(16, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])
    return m


def _callbacks(patience=PATIENCE):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=max(3, patience // 2),
                                           min_lr=1e-6, verbose=0),
    ]


def build_item_sequences(daily, features, target_col=TARGET, lookback=LOOKBACK_ITEM):
    X_list, y_list = [], []
    df = daily.sort_values(["Item", "Order Date"])
    for item, group in df.groupby("Item"):
        group = group.sort_values("Order Date").reset_index(drop=True)
        feat  = group[features].values.astype(np.float32)
        tgt   = group[target_col].values.astype(np.float32)
        if len(group) < lookback + 1:
            continue
        for i in range(lookback, len(group)):
            X_list.append(feat[i - lookback : i])
            y_list.append(tgt[i])
    if not X_list:
        raise ValueError(f"No sequences (need {lookback+1}+ days per item)")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# PER-ITEM MODELS  (best version — with lag features) -> powers /predict
# ═════════════════════════════════════════════════════════════════════════════
def train_per_item_models(daily: pd.DataFrame, base_features: list, log: list):
    """
    Trains the 6 per-item models (4 sklearn + RNN + LSTM), each in its single
    best configuration (sklearn use lag features; deep use 7-day sequences).
    Generates all diagnostics charts.

    Returns a dict the pipeline stores on `state`:
      {
        "sklearn_models" : {api_key: fitted_model},
        "scaler"         : StandardScaler,
        "features"       : [feature names used],
        "metrics"        : {model_name: {cv_*, test_*}},
        "deep_models"    : {"RNN": keras, "LSTM": keras} or {},
        "daily_lagged"   : DataFrame with lag features (for serving sequences),
        "cutoff"         : test-split cutoff Timestamp,
      }
    """
    _make_dirs()

    # ── Add lag features — this is the "best version" of the tabular models ──
    daily_lag = add_lag_features(daily)
    features  = [f for f in base_features if f in daily_lag.columns] + LAG_FEATURES

    cutoff   = daily_lag["Order Date"].max() - pd.Timedelta(days=90)
    train_df = daily_lag[daily_lag["Order Date"] <= cutoff].dropna(subset=features)
    test_df  = daily_lag[daily_lag["Order Date"] >  cutoff].dropna(subset=features)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])
    y_train = train_df[TARGET].values
    y_test  = test_df[TARGET].values

    log.append(f"Per-item training: {len(train_df):,} train / {len(test_df):,} test rows, "
               f"{len(features)} features (incl. lag)")

    # ── 4 sklearn models ──────────────────────────────────────────────────────
    # TimeSeriesSplit instead of KFold: respects temporal order so future data
    # never leaks into training folds — this explains the cv/test R2 gap seen
    # with standard KFold which can assign future rows into training folds.
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    model_zoo = {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression"  : Ridge(alpha=10.0),   # stronger regularization than default 1.0
        "Random Forest"     : RandomForestRegressor(
            n_estimators=200,
            max_depth=6,            # was None (unlimited) → caused cv R2 = -0.103
            min_samples_leaf=10,    # prevents individual leaves fitting single outliers
            max_features=0.7,       # feature subsampling for better generalization
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting" : GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,     # slower learning for better generalization
            max_depth=4,            # was default 3, slightly more capacity
            min_samples_leaf=10,    # regularization against overfitting individual samples
            subsample=0.8,          # stochastic gradient boosting: reduces variance
            random_state=42,
        ),
    }

    metrics, preds = {}, {}
    y_test_by_model = {}
    for name, m in model_zoo.items():
        cv_maes, cv_rmses, cv_r2s = [], [], []
        for tr_idx, val_idx in tscv.split(X_train):
            m_clone = type(m)(**m.get_params())
            m_clone.fit(X_train[tr_idx], y_train[tr_idx])
            val_pred = m_clone.predict(X_train[val_idx])
            cv_maes.append(mean_absolute_error(y_train[val_idx], val_pred))
            cv_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], val_pred)))
            cv_r2s.append(r2_score(y_train[val_idx], val_pred))

        m.fit(X_train, y_train)
        y_pred = np.clip(m.predict(X_test), 0, None)
        t = _metrics(y_test, y_pred)
        metrics[name] = {
            "cv_MAE": round(np.mean(cv_maes), 3),
            "cv_RMSE": round(np.mean(cv_rmses), 3),
            "cv_R2": round(np.mean(cv_r2s), 3),
            "test_MAE": t["MAE"], "test_RMSE": t["RMSE"], "test_R2": t["R2"],
        }
        preds[name] = y_pred
        y_test_by_model[name] = y_test
        log.append(f"  {name}: cv_R2={metrics[name]['cv_R2']} test_R2={t['R2']} MAE={t['MAE']}")

    # ── GBM feature importance chart ──────────────────────────────────────────
    gbm = model_zoo["Gradient Boosting"]
    fi  = pd.Series(gbm.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    fi.plot(kind="barh", color="#378ADD", edgecolor="none", ax=ax)
    ax.set_title("Feature Importance — Gradient Boosting")
    fig.tight_layout(); fig.savefig(DIR_DIAGNOSTICS / "feature_importance.png", dpi=150)
    plt.close(fig)

    # ── RNN + LSTM on 7-day sequences (best version) ──────────────────────────
    deep_models = {}
    if TF_AVAILABLE:
        daily_scaled = daily_lag.dropna(subset=features).copy()
        daily_scaled[features] = scaler.transform(daily_scaled[features])
        train_d = daily_scaled[daily_scaled["Order Date"] <= cutoff]
        test_d  = daily_scaled[daily_scaled["Order Date"] >  cutoff]
        try:
            Xtr, ytr = build_item_sequences(train_d, features)
            Xte, yte = build_item_sequences(test_d,  features)
            log.append(f"  Deep sequences: train={Xtr.shape}, test={Xte.shape}")
            for name, build in [("RNN", _rnn_model), ("LSTM", _lstm_model)]:
                model = build(Xtr.shape[2], LOOKBACK_ITEM)
                hist  = model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  validation_split=0.15,   # larger val set → more reliable early stop
                                  callbacks=_callbacks(), verbose=0)
                y_pred = np.clip(model.predict(Xte, verbose=0).flatten(), 0, None)
                t = _metrics(yte, y_pred)
                metrics[name] = {"test_MAE": t["MAE"], "test_RMSE": t["RMSE"],
                                 "test_R2": t["R2"], "epochs_run": len(hist.history["loss"])}
                preds[name] = y_pred
                y_test_by_model[name] = yte
                deep_models[name] = model
                log.append(f"  {name}: test MAE={t['MAE']} R2={t['R2']} "
                           f"({len(hist.history['loss'])} epochs)")
                # loss chart
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(hist.history["loss"], label="train")
                ax.plot(hist.history["val_loss"], label="val")
                ax.set_title(f"{name} — Training Loss"); ax.legend()
                fig.tight_layout()
                fig.savefig(DIR_DIAGNOSTICS / f"{name.lower()}_loss.png", dpi=150)
                plt.close(fig)
        except ValueError as e:
            log.append(f"  Deep models skipped: {e}")
    else:
        log.append("  RNN/LSTM skipped (TensorFlow not installed)")

    # ── Diagnostic charts for all trained per-item models ────────────────────
    simple = {n: {"MAE": m["test_MAE"], "RMSE": m["test_RMSE"], "R2": m["test_R2"]}
              for n, m in metrics.items()}
    _plot_diagnostics(y_test_by_model, preds, simple)
    log.append("Diagnostic charts saved to charts/diagnostics/")

    # ── Lag feature charts -> charts/lag/ ─────────────────────────────────────
    # These are the charts that were in lag_pipeline.py — now generated here
    # so they appear in the frontend Charts page under the "Lag Features" tab.
    sklearn_names = list(model_zoo.keys())
    _plot_lag_charts(
        sklearn_names=sklearn_names,
        preds_lag={n: preds[n] for n in sklearn_names},
        y_test_lag=y_test,
        metrics_lag={n: metrics[n] for n in sklearn_names},
        log=log,
    )
    log.append("Lag charts saved to charts/lag/")

    return {
        "sklearn_models": {SKLEARN_API_KEYS[n]: model_zoo[n] for n in model_zoo},
        "scaler"        : scaler,
        "features"      : features,
        "metrics"       : metrics,
        "deep_models"   : deep_models,
        "daily_lagged"  : daily_lag,
        "cutoff"        : cutoff,
    }


# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHARTS                                     -> charts/diagnostics/
# ═════════════════════════════════════════════════════════════════════════════
def _plot_diagnostics(y_test_by_model, preds, metrics_simple):
    # Per-model scatter + residuals
    for name, y_pred in preds.items():
        y_test = y_test_by_model[name]
        color  = MODEL_COLORS.get(name, "#888888")
        m      = metrics_simple[name]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y_test, y_pred, alpha=0.35, s=18, color=color, edgecolor="none")
        lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
        axes[0].plot(lims, lims, "--", color="#888888", linewidth=1.2)
        axes[0].set_xlim(lims); axes[0].set_ylim(lims)
        axes[0].set_title(f"{name}\nPredicted vs Actual (R²={m['R2']}, MAE={m['MAE']})")
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[1].scatter(y_pred, y_test - y_pred, alpha=0.35, s=18, color=color, edgecolor="none")
        axes[1].axhline(0, color="#888888", linestyle="--")
        axes[1].set_title(f"{name}\nResiduals")
        axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
        fig.tight_layout()
        fig.savefig(DIR_DIAGNOSTICS / f"scatter_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    # Combined scatter grid
    n = len(preds); rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 5 * rows))
    flat = np.array(axes).flatten()
    for ax, (name, y_pred) in zip(flat, preds.items()):
        y_test = y_test_by_model[name]; color = MODEL_COLORS.get(name, "#888888")
        m = metrics_simple[name]
        ax.scatter(y_test, y_pred, alpha=0.35, s=14, color=color, edgecolor="none")
        lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, "--", color="#888888", linewidth=1.0)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"{name} (R²={m['R2']}, MAE={m['MAE']})", fontsize=10)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle("Predicted vs Actual — All Models", y=1.0)
    fig.tight_layout(); fig.savefig(DIR_DIAGNOSTICS / "all_models_scatter_grid.png", dpi=150)
    plt.close(fig)

    # Per-model overlays + grid
    cache = {}
    for name, y_pred in preds.items():
        y_test = y_test_by_model[name]; order = np.argsort(y_test)
        cache[name] = (y_test[order], y_pred[order])
        color = MODEL_COLORS.get(name, "#888888")
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(y_test[order], label="Ground truth", color="black", linewidth=1.6)
        ax.plot(y_pred[order], label=name, color=color, linewidth=1.1, alpha=0.85)
        ax.set_title(f"Ground Truth vs {name}"); ax.legend(fontsize=9)
        ax.set_xlabel("Test sample (sorted)"); ax.set_ylabel("Quantity")
        fig.tight_layout()
        fig.savefig(DIR_DIAGNOSTICS / f"overlay_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    flat = np.array(axes).flatten()
    for ax, name in zip(flat, preds.keys()):
        ys, ps = cache[name]; color = MODEL_COLORS.get(name, "#888888")
        ax.plot(ys, label="Ground truth", color="black", linewidth=1.4)
        ax.plot(ps, label=name, color=color, linewidth=1.0, alpha=0.85)
        ax.set_title(name, fontsize=11); ax.legend(fontsize=8)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle("Ground Truth vs Predictions — Per Model", y=1.0)
    fig.tight_layout(); fig.savefig(DIR_DIAGNOSTICS / "overlay_grid.png", dpi=150)
    plt.close(fig)

    # Metrics comparison
    names = list(metrics_simple.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        vals = [metrics_simple[n][metric] for n in names]
        cols = [MODEL_COLORS.get(n, "#888888") for n in names]
        ax.bar(names, vals, color=cols, edgecolor="none")
        if metric == "R2":
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(metric); ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Model Metrics Comparison — Test Set", y=1.02)
    fig.tight_layout(); fig.savefig(DIR_DIAGNOSTICS / "metrics_comparison.png", dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# LAG FEATURE CHARTS                                           -> charts/lag/
# ═════════════════════════════════════════════════════════════════════════════
def _plot_lag_charts(sklearn_names, preds_lag, y_test_lag, metrics_lag, log):
    """
    Produces lag-feature charts for the frontend Charts page (Lag tab).
    These replicate the charts that were previously in lag_pipeline.py:
      - R2 improvement bar chart (baseline cv R2 vs test R2 with lag features)
      - MAE comparison bar chart (same pairing)
      - Predicted vs Actual scatter grid for all sklearn models (with lag)
    """
    DIR_LAG.mkdir(parents=True, exist_ok=True)
    names    = sklearn_names
    colors   = [MODEL_COLORS.get(n, "#888888") for n in names]

    # Use test_R2 as the "with lag" value (since lag features are already
    # included in training). Show cv_R2 as the "baseline" (pre-lag proxy).
    base_r2  = [metrics_lag[n]["cv_R2"]   for n in names]
    lag_r2   = [metrics_lag[n]["test_R2"] for n in names]
    base_mae = [metrics_lag[n]["cv_MAE"]  for n in names]
    lag_mae  = [metrics_lag[n]["test_MAE"]for n in names]

    x = np.arange(len(names))

    # R2 improvement
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, base_r2, width=0.38, label="CV R2 (without lag)",
           color="#AAAAAA", alpha=0.8, edgecolor="none")
    ax.bar(x + 0.2, lag_r2, width=0.38, label="Test R2 (with lag features)",
           color="#378ADD", edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("R²  (higher is better)")
    ax.set_title("R² Improvement: CV Baseline vs Test with Lag Features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIR_LAG / "lag_r2_improvement.png", dpi=150)
    plt.close(fig)

    # MAE comparison
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, base_mae, width=0.38, label="CV MAE (without lag)",
           color="#AAAAAA", alpha=0.8, edgecolor="none")
    ax.bar(x + 0.2, lag_mae, width=0.38, label="Test MAE (with lag features)",
           color="#E05C5C", edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("MAE  (lower is better)")
    ax.set_title("MAE Comparison: CV Baseline vs Test with Lag Features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIR_LAG / "lag_mae_comparison.png", dpi=150)
    plt.close(fig)

    # Predicted vs Actual scatter grid (all sklearn models with lag)
    n_models = len(names)
    n_rows   = (n_models + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 5 * n_rows))
    flat = np.array(axes).flatten()
    for ax, name in zip(flat, names):
        y_pred = preds_lag[name]
        color  = MODEL_COLORS.get(name, "#888888")
        m      = metrics_lag[name]
        ax.scatter(y_test_lag, y_pred, alpha=0.35, s=14, color=color, edgecolor="none")
        lims = [0, max(y_test_lag.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, "--", color="#888888", linewidth=1.0)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"{name}\n(R²={m['test_R2']}, MAE={m['test_MAE']})", fontsize=10)
        ax.set_xlabel("Actual", fontsize=9); ax.set_ylabel("Predicted", fontsize=9)
    for ax in flat[n_models:]:
        ax.axis("off")
    fig.suptitle("Predicted vs Actual — All Models WITH Lag Features", y=1.0)
    fig.tight_layout()
    fig.savefig(DIR_LAG / "lag_predicted_vs_actual_grid.png", dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# DAILY-TOTAL MODELS  (GBM + one LSTM)                  -> charts/timeseries/
# ═════════════════════════════════════════════════════════════════════════════
def train_timeseries_models(df_clean: pd.DataFrame, log: list):
    _make_dirs()
    daily = (
        df_clean.groupby("Order Date")
        .agg(TotalQty=("Quantity", "sum"), TotalRevenue=("Order Total", "sum"))
        .reset_index().sort_values("Order Date").reset_index(drop=True)
    )
    daily["DayOfWeek"] = daily["Order Date"].dt.dayofweek
    daily["Month"]     = daily["Order Date"].dt.month
    daily["IsWeekend"] = daily["DayOfWeek"].isin([5, 6]).astype(int)
    daily["Quarter"]   = daily["Order Date"].dt.quarter
    daily["DayOfYear"] = daily["Order Date"].dt.dayofyear
    daily["lag_1"]           = daily[TARGET].shift(1)
    daily["lag_7"]           = daily[TARGET].shift(7)
    daily["lag_14"]          = daily[TARGET].shift(14)
    daily["rolling_7_mean"]  = daily[TARGET].shift(1).rolling(7, min_periods=3).mean()
    daily["rolling_14_mean"] = daily[TARGET].shift(1).rolling(14, min_periods=5).mean()
    daily["rolling_7_std"]   = daily[TARGET].shift(1).rolling(7, min_periods=3).std().fillna(0)
    daily = daily.dropna().reset_index(drop=True)

    log.append(f"Daily totals: {len(daily)} days, mean {daily[TARGET].mean():.0f}/day")

    cutoff   = daily["Order Date"].max() - pd.Timedelta(days=90)
    train_df = daily[daily["Order Date"] <= cutoff]
    test_df  = daily[daily["Order Date"] >  cutoff]

    # GBM
    GBM_FEATS = ["DayOfWeek","Month","IsWeekend","Quarter","DayOfYear",
                 "lag_1","lag_7","lag_14","rolling_7_mean","rolling_14_mean","rolling_7_std"]
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, random_state=42)
    gbm.fit(train_df[GBM_FEATS].values, train_df[TARGET].values)
    gbm_pred = np.clip(gbm.predict(test_df[GBM_FEATS].values), 0, None)
    y_test   = test_df[TARGET].values

    all_metrics = {"GBM (daily total)": _metrics(y_test, gbm_pred)}
    all_preds   = {"GBM (daily total)": gbm_pred}
    log.append(f"  GBM daily: R2={all_metrics['GBM (daily total)']['R2']}")

    # One LSTM
    if TF_AVAILABLE:
        FEATS = ["TotalQty","DayOfWeek","Month","IsWeekend",
                 "lag_1","lag_7","lag_14","rolling_7_mean","rolling_7_std"]
        scaler   = StandardScaler()
        train_sc = scaler.fit_transform(train_df[FEATS].values)
        test_sc  = scaler.transform(test_df[FEATS].values)
        n_train  = len(train_sc)
        combined = np.vstack([train_sc, test_sc])
        X_all, y_all = [], []
        for i in range(LOOKBACK_DAILY, len(combined)):
            X_all.append(combined[i - LOOKBACK_DAILY : i]); y_all.append(combined[i, 0])
        X_all = np.array(X_all, dtype=np.float32); y_all = np.array(y_all, dtype=np.float32)
        split = n_train - LOOKBACK_DAILY
        Xtr, ytr = X_all[:split], y_all[:split]
        Xte, yte_sc = X_all[split:], y_all[split:]
        qmean, qstd = scaler.mean_[0], scaler.scale_[0]
        yte = yte_sc * qstd + qmean

        model = _lstm_model(Xtr.shape[2], LOOKBACK_DAILY)
        model.fit(Xtr, ytr, epochs=EPOCHS_DAILY, batch_size=BATCH_SIZE,
                  validation_split=0.1, callbacks=_callbacks(PATIENCE_DAILY), verbose=0)
        pred = np.clip(model.predict(Xte, verbose=0).flatten() * qstd + qmean, 0, None)
        all_metrics["LSTM (daily total)"] = _metrics(yte, pred)
        all_preds["LSTM (daily total)"]   = pred
        log.append(f"  LSTM daily: R2={all_metrics['LSTM (daily total)']['R2']}")
    else:
        log.append("  Daily-total LSTM skipped (no TensorFlow)")

    # ── Charts ────────────────────────────────────────────────────────────────
    for name, pred in all_preds.items():
        L = min(len(pred), len(y_test)); yt = y_test[-L:]; pr = pred[-L:]
        o = np.argsort(yt); color = MODEL_COLORS.get(name, "#378ADD")
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(yt[o], label="Ground truth", color="black", linewidth=1.6)
        ax.plot(pr[o], label=name, color=color, linewidth=1.2, alpha=0.85)
        ax.set_title(f"Ground Truth vs {name} — Daily Total"); ax.legend(fontsize=9)
        ax.set_xlabel("Test day (sorted)"); ax.set_ylabel("Total quantity")
        fig.tight_layout()
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(DIR_TIMESERIES / f"overlay_{safe}.png", dpi=150); plt.close(fig)

    n = len(all_preds); rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 5 * rows))
    flat = np.array(axes).flatten()
    for ax, (name, pred) in zip(flat, all_preds.items()):
        L = min(len(pred), len(y_test)); yt = y_test[-L:]; pr = pred[-L:]
        color = MODEL_COLORS.get(name, "#378ADD"); m = all_metrics[name]
        lims = [0, max(float(yt.max()), float(pr.max())) * 1.05]
        ax.scatter(yt, pr, alpha=0.55, s=22, color=color, edgecolor="none")
        ax.plot(lims, lims, "--", color="#888888", linewidth=1.0)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"{name}\n(R²={m['R2']}, MAE={m['MAE']})", fontsize=10)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle("Predicted vs Actual — Daily Total Sales", y=1.0)
    fig.tight_layout(); fig.savefig(DIR_TIMESERIES / "scatter_grid.png", dpi=150)
    plt.close(fig)

    names = list(all_metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(names, [all_metrics[n]["MAE"] for n in names],
                color=[MODEL_COLORS.get(n, "#888888") for n in names], edgecolor="none")
    axes[0].set_title("MAE — lower is better"); axes[0].tick_params(axis="x", rotation=15, labelsize=9)
    axes[1].bar(names, [all_metrics[n]["R2"] for n in names],
                color=[MODEL_COLORS.get(n, "#888888") for n in names], edgecolor="none")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("R² — higher is better"); axes[1].tick_params(axis="x", rotation=15, labelsize=9)
    fig.suptitle("Daily-Total Model Metrics", y=1.02)
    fig.tight_layout(); fig.savefig(DIR_TIMESERIES / "metrics_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["Order Date"], daily[TARGET], color="#CCCCCC", linewidth=0.8, label="Full history")
    ax.plot(test_df["Order Date"], y_test, color="black", linewidth=1.5, label="Ground truth (test)")
    ax.plot(test_df["Order Date"], gbm_pred, color="#E05C5C", linewidth=1.2,
            label="GBM prediction", alpha=0.85)
    ax.axvline(cutoff, color="#378ADD", linestyle="--", label="Train/test cutoff")
    ax.set_title("Daily Total Sales — Timeline"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(DIR_TIMESERIES / "timeline.png", dpi=150); plt.close(fig)

    return all_metrics


# ═════════════════════════════════════════════════════════════════════════════
# SERVING HELPER — per-item deep prediction (used by main.py /predict)
# ═════════════════════════════════════════════════════════════════════════════
def predict_with_deep_model(model_name, item, feature_values,
                             daily_scaled, selected_features, deep_state):
    """
    Builds a single sequence for one item ending at the most recent history,
    overwrites the last timestep with the target day's feature values, and
    predicts the next day's quantity. Returns a float (clipped >= 0).
    """
    keras_model = deep_state["_models"].get(model_name)
    lookback    = deep_state.get("_lookback", LOOKBACK_ITEM)
    if keras_model is None:
        return 0.0

    item_hist = (
        daily_scaled[daily_scaled["Item"] == item]
        .sort_values("Order Date")[selected_features]
        .values.astype(np.float32)
    )
    if len(item_hist) < lookback:
        pad = np.zeros((lookback - len(item_hist), len(selected_features)), dtype=np.float32)
        item_hist = np.vstack([pad, item_hist])

    seq = item_hist[-lookback:].copy()
    for i, feat in enumerate(selected_features):
        if feat in feature_values:
            seq[-1, i] = feature_values[feat]

    X = seq[np.newaxis, :, :]
    return float(max(0.0, keras_model.predict(X, verbose=0).flatten()[0]))