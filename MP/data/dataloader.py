import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    DistributedSampler,
    ConcatDataset,
)

from .read_data import read_data
from .preprocess import preprocess_data, split_dataset
from ..utils.memory import log_mem


def concatFile(files_with_year, idx, X_scaler, y_scalers, y_root):
    """
    Read and concatenate a chunk of npy files, then preprocess and split.
    """
    local_files = files_with_year[380 * idx: 380 * (idx + 1)]
    X_list, y_list = [], []

    for year, npy_file in local_files:
        try:
            X, y = read_data(npy_file, year, y_root)
        except FileNotFoundError:
            continue
        X_list.append(X)
        y_list.append(y)

    log_mem(f"before np.concatenate idx={idx}")
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    del X_list, y_list
    log_mem(f"after concatenate idx={idx}")

    X_tensor, y_tensor = preprocess_data(X, y, X_scaler, y_scalers)
    del X, y

    train_dataset, test_dataset = split_dataset(X_tensor, y_tensor)
    del X_tensor, y_tensor

    return train_dataset, test_dataset


def update_dataloader(
    files_with_year,
    local_rank,
    epoch,
    X_scaler,
    y_scalers,
    y_root,
    mode="train",
    batch_size=256,
):
    """
    Generator-style dataloader, consistent with original script.
    """
    world_size = dist.get_world_size()
    total_chunks = len(files_with_year) // 380 + int(len(files_with_year) % 380 != 0)

    if mode == "train":
        for idx in range(total_chunks):
            train_dataset, _ = concatFile(
                files_with_year,
                idx,
                X_scaler,
                y_scalers,
                y_root,
            )

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=True,
                seed=epoch,
            )
            sampler.set_epoch(epoch)

            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                drop_last=False,
            )

            del train_dataset

            for inputs, targets in loader:
                yield inputs, targets

            # sentinel: tell trainer how many batches were seen
            yield len(loader), None

            log_mem(f"after draining loader chunk={idx}")
            del loader

    elif mode == "test":
        test_datasets = []
        for idx in range(total_chunks):
            _, test_dataset = concatFile(
                files_with_year,
                idx,
                X_scaler,
                y_scalers,
                y_root,
            )
            test_datasets.append(test_dataset)

        merged = ConcatDataset(test_datasets)
        loader = DataLoader(
            merged,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        del merged, test_datasets
        yield loader

    else:
        raise ValueError("mode must be 'train' or 'test'")
