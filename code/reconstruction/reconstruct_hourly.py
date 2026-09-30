"""
Hourly reconstruction logic for the trained six-pollutant MP-LSTM.

This module contains only model inference. It does not assume specific input
filenames, geographic dimensions, NetCDF conventions, or project directories.

The input predictor grid must already contain the same 24 preprocessed predictor
variables used for training, in the same order and units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import joblib
import numpy as np
import torch

from mp_lstm_model import ModelConfig, MultiTaskLSTMRegressor, POLLUTANTS


def load_model_and_scalers(
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    device: str | torch.device | None = None,
):
    """Load the trained model and fitted min-max scalers."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_config = ModelConfig(**checkpoint["model_config"])

    model = MultiTaskLSTMRegressor(model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    scaler_bundle = joblib.load(scaler_path)
    x_scaler = scaler_bundle["x_scaler"]
    y_scalers = scaler_bundle["y_scalers"]

    return model, model_config, x_scaler, y_scalers, device


@torch.no_grad()
def predict_sequences(
    sequences: np.ndarray,
    model: MultiTaskLSTMRegressor,
    x_scaler,
    y_scalers: Sequence,
    device: str | torch.device | None = None,
    batch_size: int = 5120,
) -> np.ndarray:
    """
    Predict pollutant concentrations for preconstructed 48-hour sequences.

    Parameters
    ----------
    sequences
        Array [n_samples, 48, 24].
    model
        Trained MP-LSTM.
    x_scaler, y_scalers
        Scalers fitted during model development.
    device
        CPU or CUDA device.
    batch_size
        Number of sequences evaluated per inference batch.

    Returns
    -------
    numpy.ndarray
        Array [n_samples, 6] in ``POLLUTANTS`` order, inverse-transformed to
        the original pollutant units.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    cfg = model.config
    if sequences.ndim != 3:
        raise ValueError("sequences must have shape [samples, time, features]")
    if sequences.shape[1:] != (cfg.sequence_length, cfg.input_size):
        raise ValueError(
            "Expected sequence shape "
            f"[N, {cfg.sequence_length}, {cfg.input_size}], "
            f"got {sequences.shape}"
        )

    flat = sequences.reshape(-1, cfg.input_size)
    scaled = x_scaler.transform(flat).reshape(sequences.shape).astype(np.float32)

    outputs = []
    for start in range(0, len(scaled), batch_size):
        stop = min(start + batch_size, len(scaled))
        batch = torch.from_numpy(scaled[start:stop]).to(device)
        pred = model(batch).cpu().numpy()
        outputs.append(pred)

    normalized = np.concatenate(outputs, axis=0)

    reconstructed = np.column_stack(
        [
            y_scalers[i].inverse_transform(
                normalized[:, i].reshape(-1, 1)
            ).ravel()
            for i in range(normalized.shape[1])
        ]
    ).astype(np.float32)

    return reconstructed


@torch.no_grad()
def reconstruct_hourly_grid(
    predictors: np.ndarray,
    model: MultiTaskLSTMRegressor,
    x_scaler,
    y_scalers: Sequence,
    device: str | torch.device | None = None,
    batch_size: int = 5120,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct hourly pollutant concentrations on a regular predictor grid.

    Parameters
    ----------
    predictors
        Array [time, y, x, 24]. The time axis must be continuous and ordered.
        Predictor preparation (including relative humidity) is performed
        upstream, following the Supplementary Methods.
    model, x_scaler, y_scalers
        Trained model artifacts.
    device
        CPU or CUDA device.
    batch_size
        Maximum number of grid-cell sequences per model call.

    Returns
    -------
    target_indices
        Integer indices on the input time axis corresponding to reconstructed
        hours.
    concentrations
        Array [reconstructed_time, y, x, 6] in ``POLLUTANTS`` order.

    Notes
    -----
    For each reconstructed hour, the function uses the 48-hour predictor
    sequence ending at that hour. Grid cells with non-finite values anywhere
    in the corresponding sequence are returned as NaN.
    """
    cfg = model.config

    if predictors.ndim != 4:
        raise ValueError("predictors must have shape [time, y, x, features]")
    if predictors.shape[-1] != cfg.input_size:
        raise ValueError(
            f"Expected {cfg.input_size} predictor variables, "
            f"got {predictors.shape[-1]}"
        )
    if predictors.shape[0] < cfg.sequence_length:
        raise ValueError(
            f"At least {cfg.sequence_length} hourly predictor steps are required"
        )

    n_time, ny, nx, _ = predictors.shape
    target_indices = np.arange(cfg.sequence_length - 1, n_time)
    out = np.full(
        (len(target_indices), ny, nx, cfg.num_tasks),
        np.nan,
        dtype=np.float32,
    )

    for out_t, target_t in enumerate(target_indices):
        window = predictors[
            target_t - cfg.sequence_length + 1 : target_t + 1
        ]  # [48, y, x, 24]

        # Move grid cells to the sample axis: [y*x, 48, 24]
        sequences = np.transpose(window, (1, 2, 0, 3)).reshape(
            ny * nx,
            cfg.sequence_length,
            cfg.input_size,
        )

        valid = np.isfinite(sequences).all(axis=(1, 2))
        if not np.any(valid):
            continue

        predicted = predict_sequences(
            sequences[valid],
            model,
            x_scaler,
            y_scalers,
            device=device,
            batch_size=batch_size,
        )

        frame = np.full(
            (ny * nx, cfg.num_tasks),
            np.nan,
            dtype=np.float32,
        )
        frame[valid] = predicted
        out[out_t] = frame.reshape(ny, nx, cfg.num_tasks)

    return target_indices, out


def pollutant_order() -> Tuple[str, ...]:
    """Return the output-variable order used by the public reconstruction code."""
    return POLLUTANTS