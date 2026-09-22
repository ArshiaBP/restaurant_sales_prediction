"""
weather.py
==========
Open-Meteo weather integration — no API key required.

Two modes:
  1. fetch_historical_weather(lat, lon, start, end)
       Returns daily temperature + precipitation for a date range.
       Used during pipeline startup to enrich training data.

  2. fetch_forecast_weather(lat, lon, target_date)
       Returns weather for a specific future date (up to 16 days ahead).
       Used at prediction time.

Default location: New York City (restaurant assumed to be US-based).
Change RESTAURANT_LAT / RESTAURANT_LON in config to match your restaurant.
"""

from datetime import date
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Restaurant location — change these to match your actual restaurant
# ─────────────────────────────────────────────────────────────────────────────
RESTAURANT_LAT = 40.7128   # New York City
RESTAURANT_LON = -74.0060

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"


# ─────────────────────────────────────────────────────────────────────────────
# Historical weather (for training data enrichment)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_historical_weather(
    lat   : float = RESTAURANT_LAT,
    lon   : float = RESTAURANT_LON,
    start : str   = "2022-01-01",
    end   : str   = "2023-12-31",
) -> dict[str, dict]:
    """
    Fetches daily historical weather for the given date range.

    Returns:
        dict mapping date string (YYYY-MM-DD) -> {
            "temperature_mean": float (°C),
            "precipitation":    float (mm),
        }
        Returns empty dict on failure — pipeline falls back to NYC climate averages.
    """
    try:
        url = (
            f"{ARCHIVE_URL}"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start}&end_date={end}"
            f"&daily=temperature_2m_mean&daily=precipitation_sum"
            f"&timezone=America%2FNew_York"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        dates  = data["daily"]["time"]
        temps  = data["daily"]["temperature_2m_mean"]
        precip = data["daily"]["precipitation_sum"]

        result = {}
        for d, t, p in zip(dates, temps, precip):
            result[d] = {
                "temperature_mean": t    if t    is not None else 15.0,
                "precipitation"   : p    if p    is not None else 0.0,
            }
        return result

    except Exception as e:
        print(f"[weather] Historical fetch failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Forecast weather (for prediction time — up to 16 days ahead)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_forecast_weather(
    target_date : date,
    lat         : float = RESTAURANT_LAT,
    lon         : float = RESTAURANT_LON,
) -> dict:
    """
    Fetches forecast weather for a specific future date (max 16 days ahead).

    Returns:
        {
            "temperature_mean": float (°C),
            "precipitation":    float (mm),
            "source":           "forecast" | "climate_average"
        }
    """
    today = date.today()
    days_ahead = (target_date - today).days

    if days_ahead <= 16:
        # ── Real forecast from Open-Meteo ────────────────────────────────────
        try:
            url = (
                f"{FORECAST_URL}"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_mean&daily=precipitation_sum"
                f"&timezone=America%2FNew_York"
                f"&forecast_days=16"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            dates  = data["daily"]["time"]
            temps  = data["daily"]["temperature_2m_mean"]
            precip = data["daily"]["precipitation_sum"]

            target_str = target_date.strftime("%Y-%m-%d")
            if target_str in dates:
                idx = dates.index(target_str)
                return {
                    "temperature_mean": temps[idx]  if temps[idx]  is not None else 15.0,
                    "precipitation"   : precip[idx] if precip[idx] is not None else 0.0,
                    "source"          : "forecast",
                }
        except Exception as e:
            print(f"[weather] Forecast fetch failed: {e}")

    # ── Fallback: monthly climate averages for NYC ────────────────────────────
    # Source: NOAA 30-year climate normals for New York City
    climate_avg = {
        1:  {"temperature_mean":  0.6, "precipitation": 3.6},
        2:  {"temperature_mean":  1.7, "precipitation": 3.0},
        3:  {"temperature_mean":  6.3, "precipitation": 4.3},
        4:  {"temperature_mean": 12.2, "precipitation": 4.0},
        5:  {"temperature_mean": 17.8, "precipitation": 4.4},
        6:  {"temperature_mean": 22.8, "precipitation": 4.3},
        7:  {"temperature_mean": 25.6, "precipitation": 4.7},
        8:  {"temperature_mean": 24.9, "precipitation": 4.0},
        9:  {"temperature_mean": 20.6, "precipitation": 4.1},
        10: {"temperature_mean": 14.2, "precipitation": 4.0},
        11: {"temperature_mean":  8.3, "precipitation": 3.8},
        12: {"temperature_mean":  2.7, "precipitation": 3.7},
    }
    avg = climate_avg[target_date.month]
    return {
        "temperature_mean": avg["temperature_mean"],
        "precipitation"   : avg["precipitation"],
        "source"          : "climate_average",
    }
