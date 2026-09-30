"""
Independence-baseline analysis for daily multipollutant co-exceedance.

This publication-facing module implements the analysis underlying Fig. 3 of the
manuscript. It starts from daily pollutant-specific exceedance indicators and
calculates:

1. reconstructed exact-k occurrence;
2. an analytic independence baseline that preserves pollutant-specific
   grid-, calendar-month-, and year-level marginal exceedance probabilities;
3. reconstructed-minus-baseline residual occurrence; and
4. inclusive pairwise co-exceedance and lift relative to independence.

The module deliberately avoids the older "marginal component" / "dependence
component" terminology. Changes in residual occurrence or lift do not isolate
changes in dependence at fixed marginal distributions.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


POLLUTANTS: Tuple[str, ...] = (
    "PM2.5",
    "PM10",
    "CO",
    "NO2",
    "SO2",
    "O3",
)

EPS = 1.0e-12


def poisson_binomial_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculate exact-k probabilities for independent Bernoulli indicators.

    Parameters
    ----------
    probabilities
        Array with shape [n_pollutants, ...]. For this study n_pollutants = 6.

    Returns
    -------
    numpy.ndarray
        Array with shape [n_pollutants + 1, ...], containing probabilities for
        exactly k = 0, ..., n_pollutants exceedances.

    Notes
    -----
    Because there are only six pollutants, the independence expectation can be
    evaluated analytically without Monte Carlo permutations.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim < 1:
        raise ValueError("probabilities must have at least one dimension")
    if probabilities.shape[0] == 0:
        raise ValueError("At least one pollutant is required")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Exceedance probabilities must lie in [0, 1]")

    n_pollutants = probabilities.shape[0]
    spatial_shape = probabilities.shape[1:]

    exact = np.zeros((n_pollutants + 1,) + spatial_shape, dtype=np.float64)
    exact[0] = 1.0

    for i in range(n_pollutants):
        p = probabilities[i]
        updated = np.zeros_like(exact)
        updated += exact * (1.0 - p)[None, ...]
        updated[1:] += exact[:-1] * p[None, ...]
        exact = updated

    return exact


def analyze_independence_baseline(
    indicators: xr.Dataset,
    population: xr.DataArray,
    pollutants: Sequence[str] = POLLUTANTS,
    spatial_dims: Tuple[str, str] = ("latitude", "longitude"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyse one calendar year of daily exceedance indicators.

    Parameters
    ----------
    indicators
        Daily exceedance Dataset. Each pollutant variable must contain values
        interpretable as 0/1, with NaN for invalid daily metrics. Dimensions are
        ``time`` plus the two spatial dimensions.
    population
        Annual population count on the same spatial grid.
    pollutants
        Pollutant order.
    spatial_dims
        Names of the two spatial dimensions.

    Returns
    -------
    exact_k_table, pairwise_table
        Annual population-weighted exact-k and pairwise statistics.

    Independence baseline
    ---------------------
    For pollutant p, grid cell g, calendar month m, and year y,

        p[p,g,m,y] = N_exc[p,g,m,y] / N_valid[g,m,y]

    where N_valid is the number of days with a valid six-pollutant joint
    classification. Under same-day independence, exact-k follows a
    Poisson-binomial distribution with the six local monthly marginal
    probabilities.

    Pairwise lift is calculated for inclusive pairwise co-exceedance:
    both named pollutants exceed, irrespective of whether additional pollutants
    also exceed on that day.
    """
    data, pop, dates, year = _prepare_inputs(
        indicators,
        population,
        pollutants,
        spatial_dims,
    )

    pop_mask = np.isfinite(pop) & (pop > 0)
    pop_use = np.where(pop_mask, pop, 0.0)
    population_sum = float(pop_use.sum())
    if population_sum <= 0:
        raise ValueError("Population weights sum to zero.")

    n_pollutants = len(pollutants)

    reconstructed_exact_pd = np.zeros(n_pollutants + 1, dtype=np.float64)
    baseline_exact_pd = np.zeros(n_pollutants + 1, dtype=np.float64)

    marginal_pd = np.zeros(n_pollutants, dtype=np.float64)
    reconstructed_pair_pd = np.zeros((n_pollutants, n_pollutants), dtype=np.float64)
    baseline_pair_pd = np.zeros((n_pollutants, n_pollutants), dtype=np.float64)

    valid_population_days = 0.0

    month_numbers = dates.month.to_numpy()

    for month in range(1, 13):
        idx = np.flatnonzero(month_numbers == month)
        if len(idx) == 0:
            continue

        month_data = data[:, idx, ...]  # [pollutant, time, y, x]

        # The exact-k state is defined only where all six daily indicators are valid.
        valid_joint = np.isfinite(month_data).all(axis=0)
        valid_joint &= pop_mask[None, ...]

        n_valid = valid_joint.sum(axis=0).astype(np.float64)
        valid_population_days += float(np.sum(pop_use * n_valid))

        binary = (month_data > 0.5) & valid_joint[None, ...]

        marginal_counts = binary.sum(axis=1).astype(np.float64)
        for i in range(n_pollutants):
            marginal_pd[i] += float(np.sum(pop_use * marginal_counts[i]))

        with np.errstate(divide="ignore", invalid="ignore"):
            marginal_probabilities = np.divide(
                marginal_counts,
                n_valid[None, ...],
                out=np.zeros_like(marginal_counts, dtype=np.float64),
                where=n_valid[None, ...] > 0,
            )

        # Reconstructed exact-k occurrence.
        k_daily = binary.sum(axis=0)
        for k in range(n_pollutants + 1):
            counts = ((k_daily == k) & valid_joint).sum(axis=0).astype(np.float64)
            reconstructed_exact_pd[k] += float(np.sum(pop_use * counts))

        # Independence-baseline exact-k expectation.
        exact_probabilities = poisson_binomial_probabilities(
            marginal_probabilities
        )
        expected_counts = exact_probabilities * n_valid[None, ...]
        for k in range(n_pollutants + 1):
            baseline_exact_pd[k] += float(
                np.sum(pop_use * expected_counts[k])
            )

        # Inclusive pairwise occurrence and independence expectation.
        for i, j in combinations(range(n_pollutants), 2):
            reconstructed_counts = (
                binary[i] & binary[j]
            ).sum(axis=0).astype(np.float64)

            baseline_counts = (
                n_valid
                * marginal_probabilities[i]
                * marginal_probabilities[j]
            )

            reconstructed_pair_pd[i, j] += float(
                np.sum(pop_use * reconstructed_counts)
            )
            baseline_pair_pd[i, j] += float(
                np.sum(pop_use * baseline_counts)
            )

    calendar_days = len(dates)
    coverage_pct = (
        valid_population_days / (population_sum * calendar_days) * 100.0
    )

    exact_rows = []
    for k in range(n_pollutants + 1):
        reconstructed_pd = float(reconstructed_exact_pd[k])
        baseline_pd = float(baseline_exact_pd[k])
        residual_pd = reconstructed_pd - baseline_pd

        exact_rows.append(
            {
                "year": year,
                "k": k,
                "calendar_days": calendar_days,
                "population_sum": population_sum,
                "valid_population_days": valid_population_days,
                "coverage_pct": coverage_pct,
                "reconstructed_person_days": reconstructed_pd,
                "independence_person_days": baseline_pd,
                "residual_person_days": residual_pd,
                "reconstructed_pw_days": reconstructed_pd / population_sum,
                "independence_pw_days": baseline_pd / population_sum,
                "residual_pw_days": residual_pd / population_sum,
            }
        )

    pair_rows = []
    for i, j in combinations(range(n_pollutants), 2):
        reconstructed_pd = float(reconstructed_pair_pd[i, j])
        baseline_pd = float(baseline_pair_pd[i, j])
        residual_pd = reconstructed_pd - baseline_pd

        lift = (
            reconstructed_pd / baseline_pd
            if baseline_pd > EPS
            else np.nan
        )

        pair_rows.append(
            {
                "year": year,
                "pollutant_i": pollutants[i],
                "pollutant_j": pollutants[j],
                "pair": f"{pollutants[i]}-{pollutants[j]}",
                "calendar_days": calendar_days,
                "population_sum": population_sum,
                "valid_population_days": valid_population_days,
                "coverage_pct": coverage_pct,
                "marginal_i_pw_days": marginal_pd[i] / population_sum,
                "marginal_j_pw_days": marginal_pd[j] / population_sum,
                "reconstructed_pair_person_days": reconstructed_pd,
                "independence_pair_person_days": baseline_pd,
                "residual_pair_person_days": residual_pd,
                "reconstructed_pair_pw_days": reconstructed_pd / population_sum,
                "independence_pair_pw_days": baseline_pd / population_sum,
                "residual_pair_pw_days": residual_pd / population_sum,
                "lift_relative_to_independence": lift,
                "log2_lift": (
                    float(np.log2(lift))
                    if np.isfinite(lift) and lift > 0
                    else np.nan
                ),
            }
        )

    exact_table = pd.DataFrame(exact_rows)
    pairwise_table = pd.DataFrame(pair_rows)

    _check_exact_k_closure(
        exact_table,
        valid_population_days / population_sum,
    )

    return exact_table, pairwise_table


def _prepare_inputs(
    indicators: xr.Dataset,
    population: xr.DataArray,
    pollutants: Sequence[str],
    spatial_dims: Tuple[str, str],
):
    missing = [p for p in pollutants if p not in indicators.data_vars]
    if missing:
        raise KeyError(f"Missing pollutant indicators: {missing}")

    required_dims = {"time", *spatial_dims}
    for pollutant in pollutants:
        if not required_dims.issubset(indicators[pollutant].dims):
            raise ValueError(
                f"{pollutant} must contain dimensions {sorted(required_dims)}"
            )

    if not set(spatial_dims).issubset(population.dims):
        raise ValueError(
            f"Population must contain spatial dimensions {spatial_dims}"
        )

    arrays = [indicators[p].transpose("time", *spatial_dims) for p in pollutants]
    arrays = list(xr.align(*arrays, join="inner"))

    population = population.transpose(*spatial_dims)
    aligned = xr.align(arrays[0].isel(time=0, drop=True), population, join="inner")
    spatial_template, population = aligned

    # Re-align every pollutant to the final spatial coordinates.
    arrays = [
        da.sel(
            {
                spatial_dims[0]: spatial_template[spatial_dims[0]],
                spatial_dims[1]: spatial_template[spatial_dims[1]],
            }
        )
        for da in arrays
    ]

    dates = pd.DatetimeIndex(pd.to_datetime(arrays[0]["time"].values))
    if dates.hasnans:
        raise ValueError("Time coordinate contains invalid dates.")

    years = np.unique(dates.year)
    if len(years) != 1:
        raise ValueError(
            "analyze_independence_baseline expects one calendar year at a time."
        )
    year = int(years[0])

    data = np.stack(
        [np.asarray(da.values, dtype=np.float64) for da in arrays],
        axis=0,
    )
    pop = np.asarray(population.values, dtype=np.float64)

    return data, pop, dates, year


def _check_exact_k_closure(
    exact_table: pd.DataFrame,
    valid_pw_days: float,
    tolerance: float = 1.0e-5,
) -> None:
    reconstructed_sum = float(exact_table["reconstructed_pw_days"].sum())
    baseline_sum = float(exact_table["independence_pw_days"].sum())

    if abs(reconstructed_sum - valid_pw_days) > tolerance:
        raise RuntimeError(
            "Reconstructed exact-k states do not close to valid population-days."
        )
    if abs(baseline_sum - valid_pw_days) > tolerance:
        raise RuntimeError(
            "Independence-baseline exact-k probabilities do not close to valid "
            "population-days."
        )
