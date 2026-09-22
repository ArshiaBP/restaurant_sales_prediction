# Restaurant Sales Prediction

A project I built to predict daily sales (quantity + revenue) per menu item for a restaurant, using 2 years of historical POS data plus weather and holiday info. Has a FastAPI backend that does all the ML work, and a React dashboard on top of it.

## Data Science Pipeline

- **Load data** — ~203k transaction rows (2022–2023), 26 menu items across 5 categories, loaded from `restaurant_sales_data.csv`.
- **Clean** — imputed missing items/prices/quantities, removed outliers with IQR, normalized item/category names.
- **Enrich** — pulled in real historical weather (Open-Meteo) and US public holidays (Calendarific) and merged them into the daily data.
- **EDA** — monthly revenue trends, top items, revenue by category, sales by day of week, correlation matrix, temperature vs. sales. Charts saved under `charts/eda/`.
- **Feature engineering** — day of week, month, quarter, weekend/holiday flags, plus lag features (`lag_1`, `lag_7`, `lag_14`) and 7-day rolling mean/std per item.
- **Feature selection** — Mutual Information + Permutation Importance to drop the noise.
- **Train/test split** — temporal split (last 90 days as test), so no future leakage into training.
- **Modeling** — trained and compared 6 models: Linear Regression, Ridge, Random Forest, Gradient Boosting (best one, used by default), RNN, and LSTM. Sklearn models get 5-fold `TimeSeriesSplit` CV + test-set eval; RNN/LSTM are trained on 7-day sequences with dropout/L2 to fight overfitting.
- **Serving** — Gradient Boosting gets retrained on the full dataset and serves live predictions; at request time it pulls the real weather forecast for the target date (up to 14 days out) instead of relying only on historical averages.

All of this runs automatically on backend startup and regenerates the diagnostic charts (metrics comparison, feature importance, predicted vs actual, etc.) every time.

## Stack

- **Backend:** FastAPI, pandas, scikit-learn, TensorFlow/Keras, SQLAlchemy + SQLite (for auth), JWT auth
- **Frontend:** React + Vite

## Structure

```
restaurant_sales_prediction_back/   # FastAPI app + ML pipeline
  pipeline.py           # load -> clean -> holidays -> weather -> features
  models_training.py    # EDA, feature selection, training, charts
  main.py                # API endpoints
  auth.py / database.py  # JWT auth

restaurant_sales_prediction_front/  # React dashboard (predict + metrics pages)
```

## Running project

Backend:
```bash
cd restaurant_sales_prediction_back
pip install -r requirements.txt
uvicorn main:app --reload
```
(first run trains all 6 models, takes a few minutes)

Frontend:
```bash
cd restaurant_sales_prediction_front
npm install
npm run dev
```

## API

- `POST /auth/signup`, `POST /auth/login` — auth
- `GET /predict?date=YYYY-MM-DD&model=gradient_boosting` — per-item prediction (date must be tomorrow to +14 days)
- `GET /metrics` — model evaluation scores
- `GET /charts/list` — generated chart files
