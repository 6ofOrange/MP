"""
Illustrative PM2.5-O3 health-impact scenario comparison.

This module implements Supplementary Fig. S40 using the published cardiovascular
mortality coefficients applied in the final manuscript. The coefficients are
external assumptions; they are not re-estimated from the reconstructed exposure
fields.

The output compares:
- a reference-CRF scenario; and
- a co-pollutant-stratified conditional-CRF scenario.

The difference between the two scenarios must not be interpreted as newly
estimated causal interaction-attributable mortality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from health_impact import moving_average, relative_risk_and_paf


@dataclass(frozen=True)
class PMO3ScenarioParameters:
    # lag 0-2 application
    lag_days: int = 3

    # Exposure-context cutoffs
    pm25_cutoff: float = 75.0
    o3_cutoff: float = 100.0

    # Counterfactual concentrations
    pm25_c0: float = 15.0
    o3_c0: float = 100.0

    # CVD mortality change per 10 ug m-3 from Xu et al. (2024)
    pm25_reference_percent: float = 0.19
    pm25_low_o3_percent: float = 0.17
    pm25_high_o3_percent: float = 0.79

    o3_reference_percent: float = 0.57
    o3_low_pm25_percent: float = 0.61
    o3_high_pm25_percent: float = 0.94


PARAMETERS = PMO3ScenarioParameters()


def beta_from_percent(effect_percent: float, increment: float = 10.0) -> float:
    return float(np.log1p(effect_percent / 100.0) / increment)


def calculate_pm25_o3_scenarios(
    daily: xr.Dataset,
    population: xr.DataArray,
    annual_mortality_rate_per_person: float,
    days_in_year: int,
    previous_daily: Optional[xr.Dataset] = None,
    parameters: PMO3ScenarioParameters = PARAMETERS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate annual PM2.5-O3 exposure contexts and burden-scenario contrasts.

    Returns
    -------
    context_table
        Population-person-day shares in LL, LH, HL and HH contexts.
    burden_table
        Reference and conditional-CRF burden estimates for PM2.5 and O3,
        together with absolute and relative scenario differences.
    """
    for variable in ("PM2.5", "O3"):
        if variable not in daily:
            raise KeyError(f"Daily Dataset is missing {variable}.")

    pm_prev = None
    o3_prev = None
    if previous_daily is not None:
        if "PM2.5" in previous_daily:
            pm_prev = previous_daily["PM2.5"].isel(
                time=slice(-(parameters.lag_days - 1), None)
            )
        if "O3" in previous_daily:
            o3_prev = previous_daily["O3"].isel(
                time=slice(-(parameters.lag_days - 1), None)
            )

    pm = moving_average(
        daily["PM2.5"],
        window_days=parameters.lag_days,
        previous_tail=pm_prev,
    )
    o3 = moving_average(
        daily["O3"],
        window_days=parameters.lag_days,
        previous_tail=o3_prev,
    )

    pm, o3, population = xr.align(pm, o3, population, join="inner")

    valid = pm.notnull() & o3.notnull() & (population > 0)
    high_pm = valid & (pm >= parameters.pm25_cutoff)
    high_o3 = valid & (o3 >= parameters.o3_cutoff)

    # Context labels correspond to:
    # LL = low PM2.5 / low O3
    # LH = low PM2.5 / high O3
    # HL = high PM2.5 / low O3
    # HH = high PM2.5 / high O3
    context_masks = {
        "LL": valid & ~high_pm & ~high_o3,
        "LH": valid & ~high_pm & high_o3,
        "HL": valid & high_pm & ~high_o3,
        "HH": valid & high_pm & high_o3,
    }

    total_person_days = float(
        xr.where(valid, population, 0.0).sum(skipna=True)
    )

    context_rows = []
    for label, mask in context_masks.items():
        person_days = float(
            xr.where(mask, population, 0.0).sum(skipna=True)
        )
        context_rows.append(
            {
                "context": label,
                "population_person_days": person_days,
                "share": (
                    person_days / total_person_days
                    if total_person_days > 0 else np.nan
                ),
            }
        )

    baseline_deaths = (
        population
        * float(annual_mortality_rate_per_person)
        / float(days_in_year)
    )

    # Reference CRFs.
    beta_pm_ref = beta_from_percent(parameters.pm25_reference_percent)
    beta_o3_ref = beta_from_percent(parameters.o3_reference_percent)

    _, paf_pm_ref = relative_risk_and_paf(
        pm, beta=beta_pm_ref, c0=parameters.pm25_c0
    )
    _, paf_o3_ref = relative_risk_and_paf(
        o3, beta=beta_o3_ref, c0=parameters.o3_c0
    )

    B_pm_ref = paf_pm_ref * baseline_deaths
    B_o3_ref = paf_o3_ref * baseline_deaths

    # Conditional CRFs selected from the co-pollutant context.
    beta_pm_low = beta_from_percent(parameters.pm25_low_o3_percent)
    beta_pm_high = beta_from_percent(parameters.pm25_high_o3_percent)
    beta_o3_low = beta_from_percent(parameters.o3_low_pm25_percent)
    beta_o3_high = beta_from_percent(parameters.o3_high_pm25_percent)

    beta_pm_cond = xr.where(high_o3, beta_pm_high, beta_pm_low)
    beta_o3_cond = xr.where(high_pm, beta_o3_high, beta_o3_low)

    excess_pm = xr.where(pm.notnull(), np.maximum(pm - parameters.pm25_c0, 0.0), np.nan)
    excess_o3 = xr.where(o3.notnull(), np.maximum(o3 - parameters.o3_c0, 0.0), np.nan)

    paf_pm_cond = 1.0 - np.exp(-beta_pm_cond * excess_pm)
    paf_o3_cond = 1.0 - np.exp(-beta_o3_cond * excess_o3)

    B_pm_cond = paf_pm_cond * baseline_deaths
    B_o3_cond = paf_o3_cond * baseline_deaths

    burden_rows = []
    for pollutant, reference, conditional in (
        ("PM2.5", B_pm_ref, B_pm_cond),
        ("O3", B_o3_ref, B_o3_cond),
    ):
        ref = float(xr.where(valid, reference, 0.0).sum(skipna=True))
        cond = float(xr.where(valid, conditional, 0.0).sum(skipna=True))
        delta = cond - ref

        burden_rows.append(
            {
                "pollutant": pollutant,
                "B_reference": ref,
                "B_conditional": cond,
                "delta_B": delta,
                "relative_scenario_difference_percent": (
                    delta / ref * 100.0 if ref != 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(context_rows), pd.DataFrame(burden_rows)
