"""Historical API/AQI consistency validation.

The manuscript uses Air Pollution Index (API) records for 2003-2012 and Air
Quality Index (AQI) records for 2013-2015. The two systems are implemented
separately because the historical API used PM10, NO2 and SO2, whereas AQI under
HJ 633-2012 uses PM2.5, PM10, SO2, NO2, CO and O3.

Reconstructed pollutant metrics are aggregated to city level before conversion
to API/AQI, matching the final Methods description.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score


API_INDEX = np.array([0, 50, 100, 200, 300, 400, 500], dtype=float)
AQI_INDEX = np.array([0, 50, 100, 150, 200, 300, 400, 500], dtype=float)

API_BREAKPOINTS = {
    "PM10": np.array([0, 50, 150, 350, 420, 500, 600], dtype=float),
    "NO2": np.array([0, 80, 120, 280, 565, 750, 940], dtype=float),
    "SO2": np.array([0, 50, 150, 800, 1600, 2100, 2620], dtype=float),
}

AQI_24H_BREAKPOINTS = {
    "PM2.5": np.array([0, 35, 75, 115, 150, 250, 350, 500], dtype=float),
    "PM10": np.array([0, 50, 150, 250, 350, 420, 500, 600], dtype=float),
    "SO2": np.array([0, 50, 150, 475, 800, 1600, 2100, 2620], dtype=float),
    "NO2": np.array([0, 40, 80, 180, 280, 565, 750, 940], dtype=float),
    "CO": np.array([0, 2, 4, 14, 24, 36, 48, 60], dtype=float),
}

AQI_O3_1H = np.array([0, 160, 200, 300, 400, 800, 1000, 1200], dtype=float)
AQI_O3_8H = np.array([0, 100, 160, 215, 265, 800], dtype=float)
AQI_O3_8H_INDEX = np.array([0, 50, 100, 150, 200, 300], dtype=float)


def api_from_daily_concentrations(values: Mapping[str, float]) -> float:
    """Historical API from daily PM10, NO2 and SO2 concentrations."""
    scores = [
        _piecewise(values.get(p, np.nan), API_BREAKPOINTS[p], API_INDEX)
        for p in ("PM10", "NO2", "SO2")
    ]
    scores = [x for x in scores if np.isfinite(x)]
    return float(max(scores)) if scores else np.nan


def aqi_from_daily_metrics(values: Mapping[str, float]) -> float:
    """Daily AQI from HJ 633-2012 daily-report metrics."""
    scores = []

    for pollutant, breakpoints in AQI_24H_BREAKPOINTS.items():
        x = _piecewise(values.get(pollutant, np.nan), breakpoints, AQI_INDEX)
        if np.isfinite(x):
            scores.append(x)

    x = _piecewise(values.get("O3_1h_max", np.nan), AQI_O3_1H, AQI_INDEX)
    if np.isfinite(x):
        scores.append(x)

    o3_8h = values.get("O3_8h_max", np.nan)
    if np.isfinite(o3_8h) and o3_8h <= 800:
        x = _piecewise(o3_8h, AQI_O3_8H, AQI_O3_8H_INDEX)
        if np.isfinite(x):
            scores.append(x)

    return float(max(scores)) if scores else np.nan


def daily_index_metrics(hourly: xr.Dataset) -> xr.Dataset:
    """Build daily pollutant metrics used for API/AQI conversion."""

    required = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")
    missing = [p for p in required if p not in hourly]
    if missing:
        raise KeyError(f"Missing variables: {missing}")

    out = xr.Dataset()

    for pollutant in ("PM2.5", "PM10", "SO2", "NO2", "CO"):
        out[pollutant] = hourly[pollutant].resample(time="1D").mean()

    out["O3_1h_max"] = hourly["O3"].resample(time="1D").max()
    o3_8h = hourly["O3"].rolling(time=8, min_periods=8).mean()
    out["O3_8h_max"] = o3_8h.resample(time="1D").max()

    return out


def city_daily_indices(
    hourly: xr.Dataset,
    city_labels: xr.DataArray,
    *,
    spatial_dims=("latitude", "longitude"),
) -> pd.DataFrame:
    """Aggregate daily pollutant metrics to cities and calculate API/AQI."""

    daily = daily_index_metrics(hourly)
    city_labels = city_labels.transpose(*spatial_dims)

    template, city_labels = xr.align(
        daily["PM10"].isel(time=0, drop=True),
        city_labels,
        join="inner",
    )

    daily = daily.sel(
        {
            spatial_dims[0]: template[spatial_dims[0]],
            spatial_dims[1]: template[spatial_dims[1]],
        }
    )

    city_ids = [
        x for x in pd.unique(city_labels.values.ravel()) if pd.notna(x)
    ]
    dates = pd.DatetimeIndex(pd.to_datetime(daily.time.values))
    rows = []

    for city_id in city_ids:
        city = daily.where(city_labels == city_id).mean(
            dim=spatial_dims,
            skipna=True,
        )

        for i, date in enumerate(dates):
            values = {
                variable: float(city[variable].isel(time=i).values)
                for variable in city.data_vars
            }

            if date.year <= 2012:
                system = "API"
                index_value = api_from_daily_concentrations(values)
            else:
                system = "AQI"
                index_value = aqi_from_daily_metrics(values)

            rows.append(
                {
                    "city_id": city_id,
                    "date": date,
                    "year": int(date.year),
                    "index_system": system,
                    "reconstructed_index": index_value,
                }
            )

    return pd.DataFrame(rows)


def compare_historical_indices(
    reconstructed: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    city_col: str = "city_id",
    date_col: str = "date",
    index_col: str = "index_obs",
):
    """Pair reconstructed and reported city-level indices and score each year."""

    obs = observed[[city_col, date_col, index_col]].copy()
    obs.columns = ["city_id", "date", "observed_index"]
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")

    rec = reconstructed.copy()
    rec["date"] = pd.to_datetime(rec["date"], errors="coerce")

    paired = rec.merge(obs, on=["city_id", "date"], how="inner")
    paired = paired.dropna(
        subset=["reconstructed_index", "observed_index"]
    ).copy()

    rows = []
    for year, group in paired.groupby("year", sort=True):
        y = group["observed_index"].to_numpy(float)
        yhat = group["reconstructed_index"].to_numpy(float)

        if len(group) >= 2:
            r2 = float(r2_score(y, yhat))
            rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        else:
            r2 = np.nan
            rmse = np.nan

        rows.append(
            {
                "year": int(year),
                "index_system": "API" if int(year) <= 2012 else "AQI",
                "n_pairs": int(len(group)),
                "n_cities": int(group["city_id"].nunique()),
                "r2": r2,
                "rmse": rmse,
            }
        )

    return paired, pd.DataFrame(rows)


def _piecewise(
    concentration: float,
    concentration_breakpoints: np.ndarray,
    index_breakpoints: np.ndarray,
) -> float:
    if concentration is None or not np.isfinite(concentration):
        return np.nan

    c = float(concentration)
    if c < concentration_breakpoints[0]:
        return np.nan
    if c >= concentration_breakpoints[-1]:
        return float(index_breakpoints[-1])

    i = int(np.searchsorted(concentration_breakpoints, c, side="right") - 1)
    c0, c1 = concentration_breakpoints[i : i + 2]
    a0, a1 = index_breakpoints[i : i + 2]

    return float((a1 - a0) / (c1 - c0) * (c - c0) + a0)
