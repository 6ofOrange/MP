# Independence-baseline analysis

This folder contains the publication-facing implementation of the analysis
underlying Fig. 3.

## Files

- `independence_baseline.py`  
  Calculates the analytic independence baseline for exact-k occurrence while
  retaining pollutant-specific marginal exceedance probabilities within each
  grid cell, calendar month and year. It also calculates inclusive pairwise
  co-exceedance and lift relative to independence.

- `fig3_summary.py`  
  Aggregates annual outputs for the Fig. 3 states (`k = 2`, `k = 3`,
  `k >= 4`), calculates the 2013-2025 accounting of change, extracts the
  reconstructed-minus-baseline residual for exact `k = 1-6`, and summarizes
  endpoint changes in pairwise occurrence and lift.

## Input

The analysis starts from the daily exceedance indicators produced by the
`exposure_analysis` module and the corresponding annual gridded population.

For each pollutant, grid cell, calendar month and year, the marginal exceedance
probability is

```text
number of exceedance days / number of valid joint-classification days
```

Under same-day independence of the six binary exceedance indicators, exact-k
probabilities are evaluated analytically with the Poisson-binomial distribution.

## Terminology

The public code follows the final manuscript terminology:

- **reconstructed occurrence** rather than "observed occurrence";
- **independence baseline** rather than "marginal-preserving null";
- **reconstructed-minus-baseline residual** rather than "dependence excess";
- **change in independence baseline** and **change in residual** rather than
  "marginal component" and "dependence component";
- **lift relative to independence** rather than "marginal-adjusted lift".

The identity

```text
change in reconstructed occurrence
= change in independence baseline
+ change in residual
```

is an accounting identity. Neither changes in the residual nor changes in lift
isolate changes in atmospheric dependence at fixed marginal exceedance
probabilities.

## Pairwise definition

Pairwise co-exceedance in this module is **inclusive**: both named pollutants
exceed on the same day, irrespective of whether any of the other four
pollutants also exceed. This differs from an exact two-pollutant combination in
the Fig. 2 analysis.

## Scope

The original research directory also contained plotting scripts, network
visualizations, path configuration, preflight checks, and intermediate-file
merging code. These are intentionally omitted from the publication-facing
version because they are not required to reproduce the scientific calculation.
