"""Station-based validation of reconstructed pollutant concentrations.

The same matching/scoring logic is used for the monitoring-era reconstruction
(2016-2025) and for the independent July 2014-December 2015 hindcast test.
The latter is independent because those observations were excluded from model
fitting, not because a different validation algorithm was used.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score


POLLUTANTS = ("PM2.5", "PM10", "CO", "NO2", "SO2", "O3")

UPPER_LIMITS = {
    "PM2.5": 2000.0,
    "PM10": 2500.0,
    "CO": 90.0,
    "NO2": 900.0,
    "SO2": 1500.0,
    "O3": 1200.0,
}


def match_observations(
    observations: pd.DataFrame,
    reconstruction: xr.Dataset,
    pollutant: str,
    *,
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    station_col: str | None = None,
    upper_limits: Mapping[str, float] = UPPER_LIMITS,
) -> pd.DataFrame:
    """Pair observations with the same-time value from the nearest model grid."""

    if pollutant not in observations or pollutant not in reconstruction:
        raise KeyError(f"Missing pollutant: {pollutant}")

    lat_name = "latitude" if "latitude" in reconstruction.coords else "lat"
    lon_name = "longitude" if "longitude" in reconstruction.coords else "lon"

    cols = [time_col, lat_col, lon_col, pollutant]
    if station_col is not None:
        cols.append(station_col)

    obs = observations[cols].copy()
    obs[time_col] = pd.to_datetime(obs[time_col], errors="coerce")
    obs[pollutant] = pd.to_numeric(obs[pollutant], errors="coerce")

    limit = upper_limits.get(pollutant)
    if limit is not None:
        obs.loc[obs[pollutant] > limit, pollutant] = np.nan

    obs = obs.dropna(subset=[time_col, lat_col, lon_col, pollutant]).copy()

    if station_col is None:
        obs["station_id"] = (
            obs[lat_col].astype(str) + "_" + obs[lon_col].astype(str)
        )
    else:
        obs["station_id"] = obs[station_col].astype(str)

    lat_values = np.asarray(reconstruction[lat_name].values)
    lon_values = np.asarray(reconstruction[lon_name].values)
    times = pd.DatetimeIndex(pd.to_datetime(reconstruction.time.values))
    time_index = {t: i for i, t in enumerate(times)}

    station_index = {}
    for station_id, g in obs.groupby("station_id", sort=False):
        lat0 = float(g.iloc[0][lat_col])
        lon0 = float(g.iloc[0][lon_col])
        station_index[station_id] = (
            int(np.abs(lat_values - lat0).argmin()),
            int(np.abs(lon_values - lon0).argmin()),
        )

    da = reconstruction[pollutant]
    rows = []

    for _, row in obs.iterrows():
        t = pd.Timestamp(row[time_col])
        it = time_index.get(t)
        if it is None:
            continue

        station_id = str(row["station_id"])
        iy, ix = station_index[station_id]

        value = da.isel(
            time=it,
            **{lat_name: iy, lon_name: ix},
        ).values

        try:
            reconstructed_value = float(np.asarray(value).item())
        except (ValueError, TypeError):
            continue

        observed_value = float(row[pollutant])
        if not np.isfinite(reconstructed_value):
            continue

        rows.append(
            {
                "pollutant": pollutant,
                "station_id": station_id,
                "time": t,
                "lat": float(row[lat_col]),
                "lon": float(row[lon_col]),
                "observed": observed_value,
                "reconstructed": reconstructed_value,
            }
        )

    return pd.DataFrame(rows)


def validation_metrics(
    matched: pd.DataFrame,
    scales: Sequence[str] = ("hourly", "daily", "monthly"),
) -> pd.DataFrame:
    """Calculate R2 and RMSE at hourly, daily, and monthly scales."""

    rows = []
    for pollutant, group in matched.groupby("pollutant", sort=False):
        for scale in scales:
            paired = _aggregate(group, scale)
            n = len(paired)

            if n >= 2:
                y = paired["observed"].to_numpy(float)
                yhat = paired["reconstructed"].to_numpy(float)
                r2 = float(r2_score(y, yhat))
                rmse = float(np.sqrt(mean_squared_error(y, yhat)))
            else:
                r2 = np.nan
                rmse = np.nan

            rows.append(
                {
                    "pollutant": pollutant,
                    "scale": scale,
                    "n": n,
                    "r2": r2,
                    "rmse": rmse,
                }
            )

    return pd.DataFrame(rows)


def annual_validation_metrics(
    matched: pd.DataFrame,
    scales: Sequence[str] = ("hourly", "daily", "monthly"),
) -> pd.DataFrame:
    """Calculate the same metrics separately for each calendar year."""

    if matched.empty:
        return pd.DataFrame()

    data = matched.copy()
    data["time"] = pd.to_datetime(data["time"])
    data["year"] = data["time"].dt.year

    out = []
    for year, group in data.groupby("year", sort=True):
        table = validation_metrics(group, scales)
        table.insert(0, "year", int(year))
        out.append(table)

    return pd.concat(out, ignore_index=True)


def _aggregate(group: pd.DataFrame, scale: str) -> pd.DataFrame:
    data = group.copy()
    data["time"] = pd.to_datetime(data["time"])

    if scale == "hourly":
        return data[["observed", "reconstructed"]].dropna()

    if scale == "daily":
        data["period"] = data["time"].dt.floor("D")
    elif scale == "monthly":
        data["period"] = data["time"].dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError("scale must be hourly, daily, or monthly")

    return (
        data.groupby(["station_id", "period"], as_index=False)[
            ["observed", "reconstructed"]
        ]
        .mean()
        .dropna()
    )
