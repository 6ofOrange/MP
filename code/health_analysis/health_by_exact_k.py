"""
Allocation of pollutant-specific short-term CVD burden across exact-k states.

This module implements the calculations underlying Fig. 4b-c and the focal
exposure-severity diagnostic in Supplementary Fig. S37.

Exact-k is used only as a co-exposure context. The calculations do not estimate
an independent health effect of k or a joint mixture-attributable mortality.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from health_impact import PRIMARY_CRFS, CRF, calculate_pollutant_burden


# Daily threshold definitions used to identify focal-pollutant exceedance days.
FOCAL_THRESHOLDS: Dict[str, float] = {
    "PM2.5": 35.0,
    "NO2": 25.0,
    "SO2": 40.0,
    "CO": 4.0,
    "O3": 100.0,
}

FOCAL_POLLUTANTS: Tuple[str, ...] = (
    "PM2.5",
    "NO2",
    "SO2",
    "CO",
    "O3",
)


def summarize_health_by_exact_k(
    daily: xr.Dataset,
    exact_k: xr.DataArray,
    population: xr.DataArray,
    annual_mortality_rate_per_person: float,
    days_in_year: int,
    previous_daily: Optional[xr.Dataset] = None,
    focal_pollutants: Sequence[str] = FOCAL_POLLUTANTS,
    crfs: Mapping[str, CRF] = PRIMARY_CRFS,
) -> pd.DataFrame:
    """
    Calculate focal burden shares, HI, and focal HIA exposure by exact-k state.

    Focal burden is restricted to days on which the target pollutant exceeds its
    own short-term study threshold. Exact-k is assigned on that day, while the
    CRF uses the pollutant-specific lag-adjusted HIA exposure.

    HI = focal attributable deaths / focal exceedance person-days * 1e6
    """
    if "time" not in exact_k.dims:
        raise ValueError("exact_k must contain a 'time' dimension.")

    rows = []

    for pollutant in focal_pollutants:
        if pollutant not in FOCAL_THRESHOLDS:
            raise KeyError(f"No focal threshold configured for {pollutant}.")
        if pollutant not in daily:
            raise KeyError(f"Daily Dataset is missing {pollutant}.")

        exposure, burden = calculate_pollutant_burden(
            daily=daily,
            population=population,
            annual_mortality_rate_per_person=annual_mortality_rate_per_person,
            pollutant=pollutant,
            days_in_year=days_in_year,
            previous_daily=previous_daily,
            crfs=crfs,
        )

        day_metric, k_aligned, pop_aligned, exposure, burden = xr.align(
            daily[pollutant],
            exact_k,
            population,
            exposure,
            burden,
            join="inner",
        )

        focal_day = (
            day_metric.notnull()
            & (day_metric > FOCAL_THRESHOLDS[pollutant])
        )

        total_focal_burden = 0.0
        state_records = []

        for k in range(7):
            mask = (
                focal_day
                & k_aligned.notnull()
                & (k_aligned == k)
                & exposure.notnull()
                & burden.notnull()
            )

            # Population is constant across days within a year but is broadcast
            # onto the daily mask here to form person-days.
            person_days = float(
                xr.where(mask, pop_aligned, 0.0).sum(skipna=True)
            )
            focal_burden = float(
                xr.where(mask, burden, 0.0).sum(skipna=True)
            )

            mean_exposure, q25, q75 = _weighted_exposure_summary(
                exposure,
                pop_aligned,
                mask,
            )

            state_records.append(
                {
                    "pollutant": pollutant,
                    "k": k,
                    "focal_exceedance_person_days": person_days,
                    "focal_burden": focal_burden,
                    "HI_per_million_person_days": (
                        focal_burden / person_days * 1.0e6
                        if person_days > 0 else np.nan
                    ),
                    "X_HIA_mean": mean_exposure,
                    "X_HIA_q25": q25,
                    "X_HIA_q75": q75,
                }
            )
            total_focal_burden += focal_burden

        total_person_days = sum(
            r["focal_exceedance_person_days"] for r in state_records
        )

        for record in state_records:
            record["focal_burden_share"] = (
                record["focal_burden"] / total_focal_burden
                if total_focal_burden > 0 else np.nan
            )
            record["focal_person_day_share"] = (
                record["focal_exceedance_person_days"] / total_person_days
                if total_person_days > 0 else np.nan
            )
            record["low_support_lt_0.1pct"] = (
                record["focal_person_day_share"] < 0.001
                if np.isfinite(record["focal_person_day_share"])
                else True
            )
            rows.append(record)

    return pd.DataFrame(rows)


def high_low_k_exposure_ratio(
    summary: pd.DataFrame,
    low_k_max: int = 2,
    high_k_min: int = 4,
) -> pd.DataFrame:
    """
    Calculate the Supplementary Fig. S37 high-/low-k focal exposure ratio.

    Both means are weighted by focal-pollutant exceedance person-days.
    """
    required = {
        "pollutant",
        "k",
        "focal_exceedance_person_days",
        "X_HIA_mean",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for pollutant, g in summary.groupby("pollutant"):
        low = g[g["k"] <= low_k_max]
        high = g[g["k"] >= high_k_min]

        low_mean = _table_weighted_mean(
            low["X_HIA_mean"].to_numpy(float),
            low["focal_exceedance_person_days"].to_numpy(float),
        )
        high_mean = _table_weighted_mean(
            high["X_HIA_mean"].to_numpy(float),
            high["focal_exceedance_person_days"].to_numpy(float),
        )

        rows.append(
            {
                "pollutant": pollutant,
                "low_k_max": low_k_max,
                "high_k_min": high_k_min,
                "X_HIA_low_k_mean": low_mean,
                "X_HIA_high_k_mean": high_mean,
                "R_HL": (
                    high_mean / low_mean
                    if np.isfinite(low_mean) and low_mean > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _weighted_exposure_summary(
    exposure: xr.DataArray,
    population: xr.DataArray,
    mask: xr.DataArray,
) -> Tuple[float, float, float]:
    exposure, population, mask = xr.align(
        exposure, population, mask, join="inner"
    )

    values = np.asarray(exposure.values, dtype=float)
    weights = np.asarray(
        xr.where(mask, population, 0.0).broadcast_like(exposure).values,
        dtype=float,
    )

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan, np.nan, np.nan

    x = values[valid]
    w = weights[valid]

    mean = float(np.average(x, weights=w))
    q25 = _weighted_quantile(x, w, 0.25)
    q75 = _weighted_quantile(x, w, 0.75)
    return mean, q25, q75


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    target = q * cumulative[-1]
    return float(values[np.searchsorted(cumulative, target, side="left")])


def _table_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))
