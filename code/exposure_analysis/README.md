# Exposure and exact-k analysis

This folder contains the publication-facing implementation of the exposure
calculations underlying Fig. 2 of the manuscript.

## Files

- `daily_metrics.py`  
  Converts hourly six-pollutant concentration fields to the daily exposure
  metrics used in the study: 24-h means for PM2.5, PM10, CO, NO2 and SO2, and
  MDA8 for O3.

- `coexposure_metrics.py`  
  Applies the fixed study thresholds, constructs exact-k states (k = 0-6),
  encodes exact pollutant combinations, calculates population-weighted annual
  occurrence, and derives combination shares within each k category.

- `regional_exposure.py`  
  Aggregates exact-k and exact-combination occurrence for arbitrary regions and
  identifies the dominant exact-k state used for county-level mapping.

## Study thresholds

| Pollutant | Daily metric | Threshold |
|---|---|---:|
| PM2.5 | 24-h mean | 35 ug m-3 |
| PM10 | 24-h mean | 50 ug m-3 |
| CO | 24-h mean | 4 mg m-3 |
| NO2 | 24-h mean | 25 ug m-3 |
| SO2 | 24-h mean | 40 ug m-3 |
| O3 | MDA8 | 100 ug m-3 |

The same threshold definitions are held fixed over 2003-2025.

## Typical workflow

```python
import xarray as xr

from daily_metrics import hourly_to_daily_metrics
from coexposure_metrics import (
    exceedance_indicators,
    exact_k_state,
    exact_combination_code,
    population_weighted_exact_k_days,
    population_weighted_combination_days,
    combination_shares_within_k,
)

hourly = xr.open_dataset("hourly_pollutants.nc")
population = xr.open_dataarray("population.nc")

daily = hourly_to_daily_metrics(hourly)
flags = exceedance_indicators(daily)

k = exact_k_state(flags)
codes = exact_combination_code(flags)

exact_k_table = population_weighted_exact_k_days(k, population)
combination_table = population_weighted_combination_days(codes, population)
composition_table = combination_shares_within_k(combination_table)
```

For county-level summaries, first prepare a two-dimensional `region_labels`
DataArray on the pollutant grid and pass it to `regional_exposure.py`.

## Scope

The original research scripts contained project-specific paths, LandScan
resampling, shapefile clipping, multiprocessing, and figure drawing. Those
operations are intentionally excluded here. The public code retains the
scientific calculation itself and accepts already harmonized pollutant,
population, and region-label arrays.

The `Others` category shown in the composition figure is a visualization choice
applied after the exact-combination calculations and is therefore not hard-coded
in the analysis functions.