"""
Training logic for the six-pollutant MP-LSTM reconstruction model.

The code is data-agnostic: it expects already harmonized model-development
samples and contains no project-specific paths, filenames, or raw-data access.

Expected arrays
---------------
X : numpy.ndarray
    Shape [n_samples, 48, 24].
y : numpy.ndarray
    Shape [n_samples, 6], ordered as:
    PM2.5, PM10, CO, NO2, SO2, O3.

The manuscript model-development period was 2016-2025. Sample construction and
predictor assembly should therefore be performed upstream using the procedures
described in the Supplementary Methods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple
import os
import random

import joblib
import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from mp_lstm_model import ModelConfig, MultiTaskLSTMRegressor, POLLUTANTS


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings reported in the manuscript."""

    epochs: int = 18
    batch_size: int = 256
    learning_rate: float = 2.0e-4
    test_fraction: float = 0.20
    split_seed: int = 42
    task_weights: Tuple[float, ...] = (
        1.0,  # PM2.5
        1.0,  # PM10
        1.0,  # CO
        1.0,  # NO2
        0.8,  # SO2
        1.0,  # O3
    )


def _distributed_context() -> Tuple[bool, int, int, torch.device]:
    """
    Initialize torch.distributed when launched with torchrun.

    The same code also runs on a single CPU/GPU without torchrun.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return distributed, rank, local_rank, device


def _set_split_seed(seed: int) -> None:
    """
    Set deterministic seeds for the publication-facing reference implementation.

    The original model-development workflow used random_state=42 for sample/file
    splitting. The model manuscript does not rely on deterministic initialization
    as a scientific result, but fixing the seed here makes the public reference
    implementation easier to reproduce.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_samples(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig = TrainingConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the 80:20 development split used by the training implementation."""
    _validate_arrays(X, y)
    return train_test_split(
        X,
        y,
        test_size=config.test_fraction,
        random_state=config.split_seed,
        shuffle=True,
    )


def fit_minmax_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[MinMaxScaler, Tuple[MinMaxScaler, ...]]:
    """
    Fit one predictor scaler and one independent target scaler per pollutant.

    Predictor values are scaled feature-wise. Each pollutant target is scaled
    independently so task losses are optimized on comparable numerical ranges.
    """
    _validate_arrays(X_train, y_train)

    x_scaler = MinMaxScaler()
    x_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    y_scalers = []
    for i in range(y_train.shape[1]):
        scaler = MinMaxScaler()
        scaler.fit(y_train[:, i].reshape(-1, 1))
        y_scalers.append(scaler)

    return x_scaler, tuple(y_scalers)


def transform_samples(
    X: np.ndarray,
    y: np.ndarray,
    x_scaler: MinMaxScaler,
    y_scalers: Sequence[MinMaxScaler],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the fitted predictor and pollutant-specific target scalers."""
    _validate_arrays(X, y)

    X_scaled = x_scaler.transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(X.shape).astype(np.float32)

    y_scaled = np.column_stack(
        [
            y_scalers[i].transform(y[:, i].reshape(-1, 1)).ravel()
            for i in range(y.shape[1])
        ]
    ).astype(np.float32)

    return X_scaled, y_scaled


def weighted_multitask_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_weights: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pollutant-specific MSEs in normalized target space and their
    weighted sum.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape; "
            f"got {tuple(predictions.shape)} and {tuple(targets.shape)}"
        )

    losses = torch.stack(
        [
            nn.functional.mse_loss(predictions[:, i], targets[:, i])
            for i in range(predictions.shape[1])
        ]
    )
    weights = predictions.new_tensor(task_weights)
    if len(task_weights) != predictions.shape[1]:
        raise ValueError("task_weights length must equal the number of tasks")

    total = torch.sum(weights * losses)
    return total, losses


def train_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str | Path,
    model_config: ModelConfig = ModelConfig(),
    training_config: TrainingConfig = TrainingConfig(),
) -> Dict[str, object]:
    """
    Train the MP-LSTM from preconstructed development samples.

    Parameters
    ----------
    X
        Float array [n_samples, 48, 24].
    y
        Float array [n_samples, 6] in ``POLLUTANTS`` order.
    output_dir
        Directory in which the checkpoint and fitted scalers are saved.

    Returns
    -------
    dict
        Training summary containing epoch losses and output paths.

    Notes
    -----
    - No early stopping is applied.
    - The primary loss is the weighted sum of six pollutant-specific MSEs.
    - When launched with torchrun, DistributedDataParallel is used.
    """
    _validate_arrays(X, y, model_config)

    if len(training_config.task_weights) != model_config.num_tasks:
        raise ValueError("Number of task weights must match num_tasks")

    distributed, rank, local_rank, device = _distributed_context()
    _set_split_seed(training_config.split_seed)

    X_train, X_test, y_train, y_test = split_samples(X, y, training_config)

    # Fit scalers on the training partition, then reuse them for held-out samples
    # and historical reconstruction.
    x_scaler, y_scalers = fit_minmax_scalers(X_train, y_train)
    X_train, y_train = transform_samples(
        X_train, y_train, x_scaler, y_scalers
    )
    X_test, y_test = transform_samples(
        X_test, y_test, x_scaler, y_scalers
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            shuffle=True,
            seed=training_config.split_seed,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        drop_last=False,
    )

    model = MultiTaskLSTMRegressor(model_config).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
    )

    history = []
    for epoch in range(training_config.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        weighted_sum = 0.0
        task_sum = np.zeros(model_config.num_tasks, dtype=np.float64)
        n_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            total_loss, task_losses = weighted_multitask_mse(
                predictions,
                targets,
                training_config.task_weights,
            )
            total_loss.backward()
            optimizer.step()

            weighted_sum += float(total_loss.detach().cpu())
            task_sum += task_losses.detach().cpu().numpy()
            n_batches += 1

        if distributed:
            stats = torch.tensor(
                [weighted_sum, *task_sum.tolist(), float(n_batches)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            weighted_sum = float(stats[0].item())
            task_sum = stats[1 : 1 + model_config.num_tasks].cpu().numpy()
            n_batches = int(stats[-1].item())

        if rank == 0:
            epoch_record = {
                "epoch": epoch + 1,
                "weighted_loss": weighted_sum / max(n_batches, 1),
                "task_mse": {
                    pollutant: float(task_sum[i] / max(n_batches, 1))
                    for i, pollutant in enumerate(POLLUTANTS)
                },
            }
            history.append(epoch_record)
            print(
                f"Epoch {epoch + 1:02d}/{training_config.epochs}: "
                f"weighted loss = {epoch_record['weighted_loss']:.6f}"
            )

    if distributed:
        dist.barrier()

    output_dir = Path(output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

        base_model = model.module if isinstance(model, DDP) else model
        checkpoint_path = output_dir / "mp_lstm_checkpoint.pt"
        torch.save(
            {
                "state_dict": base_model.state_dict(),
                "model_config": asdict(model_config),
                "training_config": asdict(training_config),
                "pollutants": POLLUTANTS,
            },
            checkpoint_path,
        )

        scaler_path = output_dir / "mp_lstm_scalers.joblib"
        joblib.dump(
            {
                "x_scaler": x_scaler,
                "y_scalers": y_scalers,
                "pollutants": POLLUTANTS,
            },
            scaler_path,
        )

        summary = {
            "checkpoint": str(checkpoint_path),
            "scalers": str(scaler_path),
            "history": history,
            "n_train": len(train_ds),
            "n_test": len(test_ds),
        }
    else:
        summary = {}

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    return summary


def _validate_arrays(
    X: np.ndarray,
    y: np.ndarray,
    model_config: ModelConfig = ModelConfig(),
) -> None:
    if X.ndim != 3:
        raise ValueError("X must have shape [samples, time, features]")
    if y.ndim != 2:
        raise ValueError("y must have shape [samples, tasks]")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if X.shape[1] != model_config.sequence_length:
        raise ValueError(
            f"Expected sequence length {model_config.sequence_length}, "
            f"got {X.shape[1]}"
        )
    if X.shape[2] != model_config.input_size:
        raise ValueError(
            f"Expected {model_config.input_size} predictors, got {X.shape[2]}"
        )
    if y.shape[1] != model_config.num_tasks:
        raise ValueError(
            f"Expected {model_config.num_tasks} targets, got {y.shape[1]}"
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(
            "Training arrays contain NaN/inf. Missing-value handling belongs "
            "to the preprocessing step described in the Supplementary Methods."
        )