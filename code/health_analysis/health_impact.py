"""
Pollutant-specific short-term cardiovascular health-impact assessment.

This publication-facing module implements the calculation used for Fig. 4a and
Supplementary Fig. S39. It intentionally contains no project-specific paths,
filenames, or data-download logic.

Required inputs
---------------
1. Daily pollutant metrics on a common grid:
   PM2.5, PM10, NO2, SO2, CO, O3.
2. Annual gridded population.
3. An annual all-age CVD mortality rate for China.

The annual all-age mortality rate can be derived from age-specific GBD rates and
the corresponding national age distribution with ``aggregate_all_age_rate``.

Important interpretation
------------------------
Each pollutant-specific burden is calculated independently. The six estimates
must NOT be summed to obtain a multipollutant or mixture-attributable burden.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class CRF:
    """Concentration-response function used in the primary health assessment."""

    pollutant: str
    source_variable: str
    effect_percent: float
    increment: float
    lag_days: int
    c0: float
    unit: str

    @property
    def beta(self) -> float:
        return float(np.log1p(self.effect_percent / 100.0) / self.increment)


# Final manuscript / Table S8 parameterization.
PRIMARY_CRFS: Dict[str, CRF] = {
    "PM2.5": CRF(
        pollutant="PM2.5",
        source_variable="PM2.5",
        effect_percent=0.27,
        increment=10.0,
        lag_days=2,   # lag 0-1 moving average
        c0=0.0,
        unit="ug m-3",
    ),
    "PM2.5-10": CRF(
        pollutant="PM2.5-10",
        source_variable="PM2.5-10",
        effect_percent=0.25,
        increment=10.0,
        lag_days=2,   # lag 0-1 moving average
        c0=0.0,
        unit="ug m-3",
    ),
    "NO2": CRF(
        pollutant="NO2",
        source_variable="NO2",
        effect_percent=0.90,
        increment=10.0,
        lag_days=2,   # lag 0-1 moving average
        c0=0.0,
        unit="ug m-3",
    ),
    "SO2": CRF(
        pollutant="SO2",
        source_variable="SO2",
        effect_percent=0.70,
        increment=10.0,
        lag_days=2,   # lag 0-1 moving average
        c0=0.0,
        unit="ug m-3",
    ),
    "CO": CRF(
        pollutant="CO",
        source_variable="CO",
        effect_percent=1.12,
        increment=1.0,
        lag_days=2,   # lag 0-1 moving average
        c0=0.0,
        unit="mg m-3",
    ),
    "O3": CRF(
        pollutant="O3",
        source_variable="O3",
        effect_percent=0.27,
        increment=10.0,
        lag_days=4,   # 4-day moving-average approximation to cumulative lag 0-3
        c0=70.0,
        unit="ug m-3",
    ),
}


def aggregate_all_age_rate(
    age_specific_rates: np.ndarray,
    age_group_population: np.ndarray,
    rate_denominator: float = 100000.0,
) -> float:
    """
    Aggregate age-specific mortality rates using the national age distribution.

    Parameters
    ----------
    age_specific_rates
        Age-specific CVD mortality rates for one year.
    age_group_population
        Corresponding national population counts for the same age groups.
    rate_denominator
        Denominator of the supplied mortality rates. Use 100000 when rates are
        expressed per 100,000 population. Use 1 when rates are already
        per-person per year.

    Returns
    -------
    float
        Annual all-age CVD mortality rate per person.
    """
    rates = np.asarray(age_specific_rates, dtype=float)
    pop = np.asarray(age_group_population, dtype=float)

    if rates.shape != pop.shape:
        raise ValueError("Rates and age-group populations must have the same shape.")

    valid = np.isfinite(rates) & np.isfinite(pop) & (pop >= 0)
    if not valid.any() or pop[valid].sum() <= 0:
        raise ValueError("No valid age-specific mortality/population pairs.")

    weighted_rate = np.average(rates[valid], weights=pop[valid])
    return float(weighted_rate / rate_denominator)


def derive_coarse_pm(daily: xr.Dataset) -> xr.DataArray:
    """Derive PM2.5-10 as max(PM10 - PM2.5, 0)."""
    _require_variables(daily, ("PM2.5", "PM10"))
    coarse = xr.where(
        daily["PM10"].notnull() & daily["PM2.5"].notnull(),
        xr.where(daily["PM10"] - daily["PM2.5"] > 0,
                 daily["PM10"] - daily["PM2.5"], 0.0),
        np.nan,
    )
    coarse.name = "PM2.5-10"
    coarse.attrs["definition"] = "max(PM10 - PM2.5, 0)"
    return coarse


def moving_average(
    current: xr.DataArray,
    window_days: int,
    previous_tail: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Calculate a calendar-day moving average ending on the current day.

    ``previous_tail`` may contain the final ``window_days - 1`` daily values from
    the preceding year so that year-boundary lags are preserved.
    """
    if "time" not in current.dims:
        raise ValueError("Input must contain a 'time' dimension.")
    if window_days < 1:
        raise ValueError("window_days must be >= 1.")

    if window_days == 1:
        return current.copy()

    if previous_tail is not None:
        previous_tail, current = xr.align(
            previous_tail, current, join="inner", exclude={"time"}
        )
        needed = window_days - 1
        previous_tail = previous_tail.isel(time=slice(-needed, None))
        combined = xr.concat([previous_tail, current], dim="time")
        rolled = combined.rolling(time=window_days, min_periods=window_days).mean()
        return rolled.sel(time=current["time"])

    return current.rolling(
        time=window_days,
        min_periods=window_days,
    ).mean()


def build_hia_exposure(
    daily: xr.Dataset,
    pollutant: str,
    previous_daily: Optional[xr.Dataset] = None,
    crfs: Mapping[str, CRF] = PRIMARY_CRFS,
) -> xr.DataArray:
    """
    Build the lag-adjusted exposure entering the HIA for one pollutant.
    """
    if pollutant not in crfs:
        raise KeyError(f"No CRF configured for {pollutant}.")
    crf = crfs[pollutant]

    current_source = _source_exposure(daily, crf.source_variable)

    previous_tail = None
    if previous_daily is not None and crf.lag_days > 1:
        previous_source = _source_exposure(previous_daily, crf.source_variable)
        previous_tail = previous_source.isel(
            time=slice(-(crf.lag_days - 1), None)
        )

    exposure = moving_average(
        current_source,
        window_days=crf.lag_days,
        previous_tail=previous_tail,
    )
    exposure.name = f"X_HIA_{pollutant}"
    exposure.attrs["pollutant"] = pollutant
    exposure.attrs["lag_days"] = crf.lag_days
    exposure.attrs["beta"] = crf.beta
    exposure.attrs["c0"] = crf.c0
    return exposure


def relative_risk_and_paf(
    exposure: xr.DataArray,
    beta: float,
    c0: float,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate log-linear relative risk and population attributable fraction.

    RR = exp(beta * max(X - C0, 0))
    PAF = (RR - 1) / RR = 1 - exp(-beta * max(X - C0, 0))
    """
    excess = xr.where(exposure.notnull(), np.maximum(exposure - c0, 0.0), np.nan)
    rr = np.exp(beta * excess)
    paf = 1.0 - np.exp(-beta * excess)

    rr.name = "RR"
    paf.name = "PAF"
    return rr, paf


def daily_attributable_burden(
    exposure: xr.DataArray,
    population: xr.DataArray,
    annual_mortality_rate_per_person: float,
    days_in_year: int,
    beta: float,
    c0: float,
) -> xr.DataArray:
    """
    Calculate daily pollutant-specific attributable CVD mortality by grid cell.

    B[g,d,y] = PAF[g,d,y] * Pop[g,y] * m[y] / N[y]
    """
    if days_in_year not in (365, 366):
        raise ValueError("days_in_year should be 365 or 366.")
    if annual_mortality_rate_per_person < 0:
        raise ValueError("Mortality rate must be non-negative.")

    _, paf = relative_risk_and_paf(exposure, beta=beta, c0=c0)
    paf, population = xr.align(paf, population, join="inner")
    burden = (
        paf
        * population
        * float(annual_mortality_rate_per_person)
        / float(days_in_year)
    )
    burden.name = "attributable_cvd_deaths"
    return burden


def calculate_pollutant_burden(
    daily: xr.Dataset,
    population: xr.DataArray,
    annual_mortality_rate_per_person: float,
    pollutant: str,
    days_in_year: int,
    previous_daily: Optional[xr.Dataset] = None,
    crfs: Mapping[str, CRF] = PRIMARY_CRFS,
    c0_override: Optional[float] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Return lag-adjusted HIA exposure and daily attributable burden for one pollutant.

    ``c0_override`` can be used for the counterfactual-concentration sensitivity
    analysis without changing the primary CRF definition.
    """
    crf = crfs[pollutant]
    exposure = build_hia_exposure(
        daily,
        pollutant,
        previous_daily=previous_daily,
        crfs=crfs,
    )
    c0 = crf.c0 if c0_override is None else float(c0_override)

    burden = daily_attributable_burden(
        exposure=exposure,
        population=population,
        annual_mortality_rate_per_person=annual_mortality_rate_per_person,
        days_in_year=days_in_year,
        beta=crf.beta,
        c0=c0,
    )
    burden.attrs["pollutant"] = pollutant
    burden.attrs["beta"] = crf.beta
    burden.attrs["c0"] = c0
    return exposure, burden


def calculate_annual_burdens(
    daily: xr.Dataset,
    population: xr.DataArray,
    annual_mortality_rate_per_person: float,
    days_in_year: int,
    previous_daily: Optional[xr.Dataset] = None,
    crfs: Mapping[str, CRF] = PRIMARY_CRFS,
) -> Dict[str, float]:
    """
    Calculate annual national pollutant-specific attributable burdens.

    The returned values are intentionally kept separate by pollutant.
    """
    out: Dict[str, float] = {}

    for pollutant in crfs:
        _, burden = calculate_pollutant_burden(
            daily=daily,
            population=population,
            annual_mortality_rate_per_person=annual_mortality_rate_per_person,
            pollutant=pollutant,
            days_in_year=days_in_year,
            previous_daily=previous_daily,
            crfs=crfs,
        )
        out[pollutant] = float(burden.sum(skipna=True))

    return out


def crf_table(
    crfs: Mapping[str, CRF] = PRIMARY_CRFS,
) -> "pd.DataFrame":
    """Return the primary CRF parameter table used by the public implementation."""
    import pandas as pd

    rows = []
    for crf in crfs.values():
        rows.append(
            {
                "pollutant": crf.pollutant,
                "source_variable": crf.source_variable,
                "effect_percent": crf.effect_percent,
                "increment": crf.increment,
                "lag_days": crf.lag_days,
                "beta": crf.beta,
                "c0": crf.c0,
                "unit": crf.unit,
            }
        )
    return pd.DataFrame(rows)


def _source_exposure(daily: xr.Dataset, source_variable: str) -> xr.DataArray:
    if source_variable == "PM2.5-10":
        return derive_coarse_pm(daily)
    _require_variables(daily, (source_variable,))
    return daily[source_variable]


def _require_variables(daily: xr.Dataset, variables: Sequence[str]) -> None:
    missing = [v for v in variables if v not in daily.data_vars]
    if missing:
        raise KeyError(f"Missing variables: {missing}")
    if "time" not in daily.dims:
        raise ValueError("Daily Dataset must contain a 'time' dimension.")
