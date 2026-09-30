"""
Regional aggregation of exact-k and exact-combination co-exposure metrics.

The functions accept a two-dimensional region-label array rather than any
particular shapefile. This keeps the publication-facing code independent of
specific administrative-boundary files while reproducing the logic used for
county-level and national summaries.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from coexposure_metrics import (
    POLLUTANTS,
    population_weighted_combination_days,
    population_weighted_exact_k_days,
)


def summarize_exact_k_by_region(
    exact_k: xr.DataArray,
    population: xr.DataArray,
    region_labels: xr.DataArray,
    spatial_dims: Tuple[str, str] = ("latitude", "longitude"),
) -> pd.DataFrame:
    """
    Calculate population-weighted exact-k occurrence for each region.

    Parameters
    ----------
    exact_k
        Daily exact-k state array [time, spatial...].
    population
        Population count on the same spatial grid.
    region_labels
        Two-dimensional array on the same grid. Each finite/non-null label
        identifies a region (e.g. county).
    """
    exact_k, population, region_labels = xr.align(
        exact_k, population, region_labels, join="inner"
    )

    rows = []
    for region in _region_values(region_labels):
        mask = region_labels == region
        pop_region = population.where(mask)
        state_region = exact_k.where(mask)

        if float(pop_region.sum(skipna=True)) <= 0:
            continue

        table = population_weighted_exact_k_days(
            state_region,
            pop_region,
            spatial_dims=spatial_dims,
        )
        table.insert(0, "region", region)
        rows.append(table)

    if not rows:
        return pd.DataFrame(
            columns=["region", "k", "population_weighted_days"]
        )
    return pd.concat(rows, ignore_index=True)


def dominant_exact_k_by_region(
    regional_exact_k: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify the modal exact-k state in each region.

    The dominant state is the k = 0,...,6 category with the greatest annual
    population-weighted occurrence. It is not an annual mean k and does not
    identify the dominant pollutant combination within that category.
    """
    required = {"region", "k", "population_weighted_days"}
    if not required.issubset(regional_exact_k.columns):
        raise ValueError(f"Input must contain columns: {sorted(required)}")

    idx = regional_exact_k.groupby("region")["population_weighted_days"].idxmax()
    out = regional_exact_k.loc[idx, ["region", "k", "population_weighted_days"]]
    return out.rename(
        columns={
            "k": "dominant_k",
            "population_weighted_days": "dominant_state_days",
        }
    ).reset_index(drop=True)


def summarize_combinations_by_region(
    codes: xr.DataArray,
    population: xr.DataArray,
    region_labels: xr.DataArray,
    pollutants: Sequence[str] = POLLUTANTS,
    spatial_dims: Tuple[str, str] = ("latitude", "longitude"),
) -> pd.DataFrame:
    """Calculate exact-combination occurrence separately for each region."""
    codes, population, region_labels = xr.align(
        codes, population, region_labels, join="inner"
    )

    rows = []
    for region in _region_values(region_labels):
        mask = region_labels == region
        pop_region = population.where(mask)
        code_region = codes.where(mask)

        if float(pop_region.sum(skipna=True)) <= 0:
            continue

        table = population_weighted_combination_days(
            code_region,
            pop_region,
            pollutants=pollutants,
            spatial_dims=spatial_dims,
            include_clean=False,
        )
        table.insert(0, "region", region)
        rows.append(table)

    if not rows:
        return pd.DataFrame(
            columns=[
                "region",
                "code",
                "k",
                "combination",
                "population_weighted_days",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _region_values(region_labels: xr.DataArray):
    values = np.asarray(region_labels.values).ravel()
    values = pd.unique(values)
    return [v for v in values if not pd.isna(v)]