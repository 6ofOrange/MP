"""
Conversion of hourly six-pollutant concentration fields to daily exposure metrics.

This module contains only the calculation logic used by the manuscript. It does
not include project-specific file paths, grid definitions, or data-download code.

Expected input
--------------
xarray.Dataset with an hourly ``time`` coordinate and variables:
PM2.5, PM10, CO, NO2, SO2, O3

Daily metrics
-------------
- PM2.5, PM10, CO, NO2, SO2: 24-hour mean concentration
- O3: maximum daily 8-hour mean (MDA8)

The same daily metrics are used for the single-pollutant exposure analyses and
the exact-k multipollutant co-exposure analysis.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import xarray as xr


POLLUTANTS: Tuple[str, ...] = (
    "PM2.5",
    "PM10",
    "CO",
    "NO2",
    "SO2",
    "O3",
)


def hourly_to_daily_metrics(
    hourly: xr.Dataset,
    pollutants: Tuple[str, ...] = POLLUTANTS,
) -> xr.Dataset:
    """
    Convert hourly reconstructed concentrations to the manuscript daily metrics.

    Parameters
    ----------
    hourly
        Hourly xarray Dataset containing the six pollutant variables.
    pollutants
        Pollutant-variable names. The default order follows the computational
        implementation used in the study.

    Returns
    -------
    xarray.Dataset
        Daily pollutant metrics on the original spatial grid.

    Notes
    -----
    The O3 MDA8 calculation follows the original analysis logic: a continuous
    8-hour rolling mean is calculated on the hourly time axis and the maximum
    rolling mean is then selected within each calendar day.
    """
    missing = [p for p in pollutants if p not in hourly.data_vars]
    if missing:
        raise KeyError(f"Missing pollutant variables: {missing}")
    if "time" not in hourly.dims:
        raise ValueError("The input Dataset must contain a 'time' dimension.")

    daily = {}

    for pollutant in pollutants:
        da = hourly[pollutant]

        if pollutant == "O3":
            rolling_8h = da.rolling(time=8, min_periods=8).mean()
            daily[pollutant] = rolling_8h.resample(time="1D").max(dim="time")
        else:
            daily[pollutant] = da.resample(time="1D").mean(dim="time")

    result = xr.Dataset(daily)
    result.attrs["daily_metric_definition"] = (
        "24-h mean for PM2.5, PM10, CO, NO2 and SO2; "
        "maximum daily 8-h mean (MDA8) for O3."
    )
    return result