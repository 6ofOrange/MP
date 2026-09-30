"""
Exact-k and exact-combination multipollutant co-exposure calculations.

This module implements the calculations underlying the manuscript's Fig. 2:
daily threshold classification, exact-k states, exact pollutant combinations,
population-weighted annual occurrence, and composition shares.

The functions are data-agnostic and contain no project-specific paths.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# Computational order retained from the original implementation.
POLLUTANTS: Tuple[str, ...] = (
    "PM2.5",
    "PM10",
    "CO",
    "NO2",
    "SO2",
    "O3",
)

# Fixed study-specific short-term thresholds used throughout 2003-2025.
THRESHOLDS: Dict[str, float] = {
    "PM2.5": 35.0,   # ug m-3, 24-h mean
    "PM10": 50.0,    # ug m-3, 24-h mean
    "CO": 4.0,       # mg m-3, 24-h mean
    "NO2": 25.0,     # ug m-3, 24-h mean
    "SO2": 40.0,     # ug m-3, 24-h mean
    "O3": 100.0,     # ug m-3, MDA8
}


def exceedance_indicators(
    daily: xr.Dataset,
    thresholds: Mapping[str, float] = THRESHOLDS,
    pollutants: Sequence[str] = POLLUTANTS,
) -> xr.Dataset:
    """
    Convert daily concentration metrics to pollutant-specific exceedance flags.

    A value is classified as an exceedance when the daily metric is strictly
    greater than the corresponding study threshold. Missing daily values remain
    missing rather than being treated as non-exceedances.
    """
    _validate_daily_dataset(daily, pollutants)

    flags = {}
    for pollutant in pollutants:
        da = daily[pollutant]
        flags[pollutant] = xr.where(
            da.notnull(),
            da > thresholds[pollutant],
            np.nan,
        )

    return xr.Dataset(flags)


def valid_joint_days(
    indicators: xr.Dataset,
    pollutants: Sequence[str] = POLLUTANTS,
) -> xr.DataArray:
    """Return True where all six pollutant indicators are valid."""
    valid = None
    for pollutant in pollutants:
        this = indicators[pollutant].notnull()
        valid = this if valid is None else (valid & this)
    return valid


def exact_k_state(
    indicators: xr.Dataset,
    pollutants: Sequence[str] = POLLUTANTS,
) -> xr.DataArray:
    """
    Classify every valid grid-day into exact-k = 0,...,6.

    ``k`` is the exact number of the six daily pollutant metrics exceeding their
    respective thresholds on that day. The result is NaN when at least one of
    the six pollutant indicators is missing.
    """
    valid = valid_joint_days(indicators, pollutants)

    stacked = xr.concat(
        [indicators[p].fillna(False).astype(np.int8) for p in pollutants],
        dim="pollutant",
    )
    stacked = stacked.assign_coords(pollutant=list(pollutants))
    k = stacked.sum(dim="pollutant").astype(np.int8)

    k = k.where(valid)
    k.name = "exact_k"
    k.attrs["description"] = (
        "Exact number of six daily pollutant metrics exceeding their thresholds."
    )
    return k


def exact_combination_code(
    indicators: xr.Dataset,
    pollutants: Sequence[str] = POLLUTANTS,
) -> xr.DataArray:
    """
    Encode the exact pollutant combination on each valid grid-day as a bit mask.

    Codes range from 0 to 63. Code 0 denotes no exceedance; codes 1-63 represent
    exact combinations according to ``pollutants``. Invalid joint days are NaN.
    """
    valid = valid_joint_days(indicators, pollutants)
    code = None

    for bit, pollutant in enumerate(pollutants):
        term = (
            indicators[pollutant]
            .fillna(False)
            .astype(np.uint8)
            * np.uint8(1 << bit)
        )
        code = term if code is None else code + term

    code = code.astype(np.uint8).where(valid)
    code.name = "combination_code"
    code.attrs["pollutant_order"] = ",".join(pollutants)
    return code


def combination_name(
    code: int,
    pollutants: Sequence[str] = POLLUTANTS,
) -> str:
    """Convert a 0-63 bit code to an exact-combination label."""
    if not 0 <= int(code) < (1 << len(pollutants)):
        raise ValueError("Combination code is outside the valid range.")
    if int(code) == 0:
        return "None"
    return "+".join(
        pollutant
        for bit, pollutant in enumerate(pollutants)
        if (int(code) >> bit) & 1
    )


def combination_size(code: int) -> int:
    """Return the number of exceedances represented by a bit-mask code."""
    return int(code).bit_count()


def population_weighted_exact_k_days(
    exact_k: xr.DataArray,
    population: xr.DataArray,
    spatial_dims: Tuple[str, str] = ("latitude", "longitude"),
) -> pd.DataFrame:
    """
    Calculate annual population-weighted occurrence days for exact-k = 0,...,6.

    For state k, the statistic is the population-weighted mean number of days
    assigned to that state across grid cells.
    """
    exact_k, population = _align_state_and_population(exact_k, population)
    denominator = population.sum(dim=spatial_dims, skipna=True)

    rows = []
    for k in range(7):
        n_days = (exact_k == k).sum(dim="time")
        weighted_days = (
            (n_days * population).sum(dim=spatial_dims, skipna=True)
            / denominator
        )
        rows.append({"k": k, "population_weighted_days": float(weighted_days)})

    return pd.DataFrame(rows)


def population_weighted_combination_days(
    codes: xr.DataArray,
    population: xr.DataArray,
    pollutants: Sequence[str] = POLLUTANTS,
    spatial_dims: Tuple[str, str] = ("latitude", "longitude"),
    include_clean: bool = False,
) -> pd.DataFrame:
    """
    Calculate annual population-weighted occurrence days for exact combinations.
    """
    codes, population = _align_state_and_population(codes, population)
    denominator = population.sum(dim=spatial_dims, skipna=True)

    first_code = 0 if include_clean else 1
    rows = []

    for code in range(first_code, 1 << len(pollutants)):
        n_days = (codes == code).sum(dim="time")
        weighted_days = (
            (n_days * population).sum(dim=spatial_dims, skipna=True)
            / denominator
        )
        rows.append(
            {
                "code": code,
                "k": combination_size(code),
                "combination": combination_name(code, pollutants),
                "population_weighted_days": float(weighted_days),
            }
        )

    return pd.DataFrame(rows)


def combination_shares_within_k(
    combination_days: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert exact-combination occurrence to composition shares within each k.

    The resulting shares are suitable for the stacked-composition panels in
    Fig. 2. The function does not apply an 'Others' grouping; that is a plotting
    choice rather than part of the scientific calculation.
    """
    required = {"k", "combination", "population_weighted_days"}
    if not required.issubset(combination_days.columns):
        raise ValueError(f"Input must contain columns: {sorted(required)}")

    out = combination_days.copy()
    totals = out.groupby("k")["population_weighted_days"].transform("sum")
    out["share_within_k"] = np.where(
        totals > 0,
        out["population_weighted_days"] / totals,
        np.nan,
    )
    return out


def exact_pair_occurrence(
    indicators: xr.Dataset,
    population: xr.DataArray,
    pollutants: Sequence[str] = POLLUTANTS,
    spatial_dims: Tuple[str, str] = ("latitude", "longitude"),
) -> pd.DataFrame:
    """
    Calculate exact two-pollutant combinations.

    These are *exact* pairs: both named pollutants exceed and the other four do
    not. This differs from inclusive pairwise co-exceedance used in the
    independence-baseline analysis.
    """
    codes = exact_combination_code(indicators, pollutants)
    all_combos = population_weighted_combination_days(
        codes,
        population,
        pollutants=pollutants,
        spatial_dims=spatial_dims,
        include_clean=False,
    )
    return all_combos.loc[all_combos["k"] == 2].reset_index(drop=True)


def _validate_daily_dataset(
    daily: xr.Dataset,
    pollutants: Sequence[str],
) -> None:
    missing = [p for p in pollutants if p not in daily.data_vars]
    if missing:
        raise KeyError(f"Missing pollutant variables: {missing}")
    if "time" not in daily.dims:
        raise ValueError("Daily Dataset must contain a 'time' dimension.")


def _align_state_and_population(
    state: xr.DataArray,
    population: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    if "time" not in state.dims:
        raise ValueError("State array must contain a 'time' dimension.")
    state, population = xr.align(state, population, join="inner")
    population = population.where(np.isfinite(population) & (population >= 0))
    if float(population.sum(skipna=True)) <= 0:
        raise ValueError("Population weights sum to zero.")
    return state, population