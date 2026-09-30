# Cardiovascular health-impact analysis

This folder contains the publication-facing health-analysis code used for
Fig. 4 and Supplementary Figs. S37, S39 and S40.

## Files

- `health_impact.py`  
  Implements the pollutant-specific short-term CVD health-impact assessment for
  PM2.5, PM2.5-10, NO2, SO2, CO and O3. It includes the manuscript CRF
  parameters, lagged exposure construction, GBD age-weighted baseline-rate
  aggregation, relative risk / PAF calculation, attributable burden, and
  optional counterfactual-concentration overrides.

- `health_by_exact_k.py`  
  Allocates pollutant-specific burden to the exact-k state of each grid-day,
  restricts Fig. 4b-c summaries to focal-pollutant exceedance days, calculates
  health-impact intensity (HI), flags states with <0.1% of focal exceedance
  person-days, and summarizes the lag-adjusted focal exposure used for the HIA.

- `pm25_o3_scenarios.py`  
  Implements the supplementary PM2.5-O3 scenario comparison using the
  co-pollutant-stratified and reference CVD coefficients from Xu et al. (2024).

## Primary pollutant-specific HIA

The primary health assessment uses the following central CRF parameters:

| Pollutant | Effect estimate | Increment | Exposure window | Primary C0 |
|---|---:|---:|---|---:|
| PM2.5 | 0.27% | 10 ug m-3 | lag 0-1, 2-day mean | 0 |
| PM2.5-10 | 0.25% | 10 ug m-3 | lag 0-1, 2-day mean | 0 |
| NO2 | 0.90% | 10 ug m-3 | lag 0-1, 2-day mean | 0 |
| SO2 | 0.70% | 10 ug m-3 | lag 0-1, 2-day mean | 0 |
| CO | 1.12% | 1 mg m-3 | lag 0-1, 2-day mean | 0 |
| O3 | 0.27% | 10 ug m-3 | 4-day mean approximation to lag 0-3 | 70 ug m-3 |

PM2.5-10 is calculated as `max(PM10 - PM2.5, 0)`.

For a pollutant-specific lag-adjusted exposure X,

```text
beta = ln(1 + ER/100) / increment
RR   = exp(beta * max(X - C0, 0))
PAF  = 1 - exp(-beta * max(X - C0, 0))
B    = PAF * population * annual_CVD_mortality_rate / days_in_year
```

The six pollutant-specific burdens are kept separate and must not be summed to
represent a joint multipollutant mortality burden.

## Exact-k allocation

For PM2.5, NO2, SO2, CO and O3, burden is allocated to the exact-k state of
the day to which burden is assigned. Fig. 4b-c is restricted to days on which
the focal pollutant itself exceeds its study threshold. PM2.5-10 is not
allocated across exact-k because exact-k contains PM10 rather than the coarse
particle fraction.

HI is calculated as:

```text
focal attributable CVD deaths / focal-pollutant exceedance person-days * 1e6
```

HI is descriptive; it is not an independent health effect of k.

## PM2.5-O3 supplementary scenario

The supplementary scenario uses lag 0-2 exposure, PM2.5 = 75 ug m-3 and
O3 = 100 ug m-3 to define LL/LH/HL/HH exposure contexts.

The CVD coefficients applied from Xu et al. (2024) are:

- PM2.5 reference: 0.19% per 10 ug m-3
- PM2.5 under low/high O3: 0.17% / 0.79%
- O3 reference: 0.57% per 10 ug m-3
- O3 under low/high PM2.5: 0.61% / 0.94%

The scenario uses C0 = 15 ug m-3 for PM2.5 and C0 = 100 ug m-3 for O3.

`delta_B = B_conditional - B_reference` is a difference between externally
specified model scenarios. It is not a newly estimated causal interaction burden.

## Scope

The original research directory contained yearly runners, merging scripts,
validation reports, hard-coded HPC paths, region membership matrices and many
intermediate CSV products. These are intentionally excluded here. The public
implementation retains the calculations required to reproduce the manuscript's
health-analysis logic from harmonized daily pollutant, population, exact-k and
GBD mortality inputs.

Alternative C0 values for the counterfactual sensitivity analysis can be supplied
through `c0_override` in `health_impact.calculate_pollutant_burden`; a separate
project-specific runner is therefore unnecessary.
