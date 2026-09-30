"""
Summary calculations for Fig. 3 from annual independence-baseline outputs.

Terminology follows the final manuscript:

- reconstructed occurrence
- independence baseline
- reconstructed-minus-baseline residual
- lift relative to independence

The accounting identity

    change in reconstructed occurrence
      = change in independence baseline
      + change in residual

is descriptive. It is not a fixed-marginal decomposition of atmospheric
dependence and should not be interpreted as such.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_METRICS: Dict[str, Sequence[int]] = {
    "k2": (2,),
    "k3": (3,),
    "k_ge4": (4, 5, 6),
}

EPS = 1.0e-12


def build_annual_metric_table(
    exact_k_annual: pd.DataFrame,
    metrics: Mapping[str, Sequence[int]] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """
    Aggregate annual exact-k outputs into the states displayed in Fig. 3a-b.
    """
    required = {
        "year",
        "k",
        "reconstructed_pw_days",
        "independence_pw_days",
        "residual_pw_days",
    }
    _require_columns(exact_k_annual, required)

    rows = []
    for year, year_table in exact_k_annual.groupby("year", sort=True):
        for metric, k_values in metrics.items():
            subset = year_table[year_table["k"].isin(k_values)]

            reconstructed = float(subset["reconstructed_pw_days"].sum())
            baseline = float(subset["independence_pw_days"].sum())
            residual = float(subset["residual_pw_days"].sum())

            rows.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "k_members": "|".join(map(str, k_values)),
                    "reconstructed_pw_days": reconstructed,
                    "independence_pw_days": baseline,
                    "residual_pw_days": residual,
                }
            )

    return pd.DataFrame(rows)


def change_accounting(
    annual_metrics: pd.DataFrame,
    start_year: int = 2013,
    end_year: int = 2025,
) -> pd.DataFrame:
    """
    Calculate the Fig. 3b accounting of change between two endpoint years.

    For each metric:

        Δreconstructed = Δindependence_baseline + Δresidual

    This identity does not isolate changes in dependence at fixed margins.
    """
    required = {
        "year",
        "metric",
        "reconstructed_pw_days",
        "independence_pw_days",
        "residual_pw_days",
    }
    _require_columns(annual_metrics, required)

    rows = []
    for metric in annual_metrics["metric"].drop_duplicates():
        start = annual_metrics[
            (annual_metrics["year"] == start_year)
            & (annual_metrics["metric"] == metric)
        ]
        end = annual_metrics[
            (annual_metrics["year"] == end_year)
            & (annual_metrics["metric"] == metric)
        ]

        if len(start) != 1 or len(end) != 1:
            raise ValueError(
                f"Could not resolve exactly one row for {metric}: "
                f"{start_year}->{end_year}"
            )

        start = start.iloc[0]
        end = end.iloc[0]

        delta_reconstructed = (
            end["reconstructed_pw_days"] - start["reconstructed_pw_days"]
        )
        delta_baseline = (
            end["independence_pw_days"] - start["independence_pw_days"]
        )
        delta_residual = end["residual_pw_days"] - start["residual_pw_days"]

        rows.append(
            {
                "start_year": start_year,
                "end_year": end_year,
                "metric": metric,
                "delta_reconstructed_pw_days": delta_reconstructed,
                "delta_independence_baseline_pw_days": delta_baseline,
                "delta_residual_pw_days": delta_residual,
                "closure_error_days": (
                    delta_reconstructed - delta_baseline - delta_residual
                ),
            }
        )

    result = pd.DataFrame(rows)
    if (result["closure_error_days"].abs() > 1.0e-8).any():
        raise RuntimeError("Change-accounting closure check failed.")
    return result


def pairwise_change(
    pairwise_annual: pd.DataFrame,
    start_year: int = 2013,
    end_year: int = 2025,
) -> pd.DataFrame:
    """
    Summarize endpoint changes in inclusive pairwise co-exceedance and lift.

    This table provides the quantities used for Fig. 3d. The figure may display
    a selected subset of pollutant pairs; pair selection is a presentation
    choice and is therefore not hard-coded here.
    """
    required = {
        "year",
        "pollutant_i",
        "pollutant_j",
        "reconstructed_pair_pw_days",
        "independence_pair_pw_days",
        "lift_relative_to_independence",
    }
    _require_columns(pairwise_annual, required)

    rows = []
    pairs = (
        pairwise_annual[["pollutant_i", "pollutant_j"]]
        .drop_duplicates()
        .sort_values(["pollutant_i", "pollutant_j"])
    )

    for _, pair in pairs.iterrows():
        pi = pair["pollutant_i"]
        pj = pair["pollutant_j"]

        subset = pairwise_annual[
            (pairwise_annual["pollutant_i"] == pi)
            & (pairwise_annual["pollutant_j"] == pj)
        ]

        start = subset[subset["year"] == start_year]
        end = subset[subset["year"] == end_year]

        if len(start) != 1 or len(end) != 1:
            raise ValueError(
                f"Could not resolve exactly one row for {pi}-{pj}: "
                f"{start_year}->{end_year}"
            )

        start = start.iloc[0]
        end = end.iloc[0]

        rows.append(
            {
                "pollutant_i": pi,
                "pollutant_j": pj,
                "pair": f"{pi}-{pj}",
                "start_year": start_year,
                "end_year": end_year,
                "reconstructed_pair_pw_days_start": start[
                    "reconstructed_pair_pw_days"
                ],
                "reconstructed_pair_pw_days_end": end[
                    "reconstructed_pair_pw_days"
                ],
                "delta_reconstructed_pair_pw_days": (
                    end["reconstructed_pair_pw_days"]
                    - start["reconstructed_pair_pw_days"]
                ),
                "independence_pair_pw_days_start": start[
                    "independence_pair_pw_days"
                ],
                "independence_pair_pw_days_end": end[
                    "independence_pair_pw_days"
                ],
                "lift_start": start["lift_relative_to_independence"],
                "lift_end": end["lift_relative_to_independence"],
                "delta_lift": (
                    end["lift_relative_to_independence"]
                    - start["lift_relative_to_independence"]
                ),
            }
        )

    return pd.DataFrame(rows)


def exact_k_residual_table(exact_k_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Return exact-k = 1,...,6 residual occurrence for Fig. 3c.
    """
    required = {"year", "k", "residual_pw_days"}
    _require_columns(exact_k_annual, required)

    return (
        exact_k_annual.loc[
            exact_k_annual["k"].between(1, 6),
            ["year", "k", "residual_pw_days"],
        ]
        .sort_values(["k", "year"])
        .reset_index(drop=True)
    )


def _require_columns(table: pd.DataFrame, required) -> None:
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
