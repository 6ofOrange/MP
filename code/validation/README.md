# Reconstruction validation

This folder contains the publication-facing validation code for the MP-LSTM reconstruction.

## Files

- `station_validation.py`  
  Matches reconstructed concentrations with ground observations at the same timestamp and nearest 0.1° grid cell. The same code is used for the 2016-2025 monitoring-era comparison and for the independent July 2014-December 2015 hindcast test. R² and RMSE can be summarized at hourly, daily and monthly scales.

- `historical_index_validation.py`  
  Implements the historical city-level index consistency check used for 2003-2015: API for 2003-2012 and AQI for 2013-2015.

## 2014-2015 independent temporal validation

The validation algorithm is the same as for 2016-2025. The distinction is the fitting period: the production MP-LSTM was trained with 2016-2025 observations, whereas July 2014-December 2015 observations were excluded from fitting. Applying the fitted model to 2014-2015 therefore tests backward temporal transfer outside the model-development period.

Observed values are quality controlled with the upper bounds used in the study:

```text
PM2.5  2000 ug m-3
PM10   2500 ug m-3
CO       90 mg m-3
NO2     900 ug m-3
SO2    1500 ug m-3
O3     1200 ug m-3
```

Each station is mapped to the nearest reconstruction grid cell and paired with the reconstruction at the same timestamp. Daily and monthly comparisons are generated after matching by averaging within station and period.

## Historical API/AQI consistency check

The final manuscript distinguishes the two historical index systems:

```text
2003-2012: Air Pollution Index (API)
2013-2015: Air Quality Index (AQI)
```

The historical API uses daily-average PM10, NO2 and SO2. The AQI calculation follows HJ 633-2012 and uses 24-h means for PM2.5, PM10, SO2, NO2 and CO, together with daily maximum 1-h O3 and daily maximum 8-h O3.

For the public implementation, pollutant metrics are spatially aggregated to the city before conversion to API/AQI, matching the final Methods description.

## Difference from the old year-specific scripts

The old file named `aqi_2010.py` used a dictionary called `API_BREAKPOINTS`, but the table itself contained PM2.5, CO and O3 and therefore followed an AQI-like six-pollutant scheme rather than the historical three-pollutant API. The publication-facing code separates API and AQI explicitly.

The old 2014 script also first reduced all six pollutants to simple daily means, including O3. The publication-facing AQI calculation instead uses the daily-report O3 metrics specified by HJ 633-2012.

## Scope

This module covers direct station-to-grid concentration validation, the independent 2014-2015 temporal-generalization test, and the historical 2003-2015 API/AQI consistency check. It does not recreate the sample-, grid-, and year-based training holdout experiments or the Monte Carlo-dropout uncertainty analysis.
