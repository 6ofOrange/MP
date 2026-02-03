import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from ..data.dataloader import update_dataloader


def evaluate_model(
    model,
    files_with_year,
    criterions,
    device,
    X_scaler,
    y_scalers,
    y_root,
    local_rank,
):
    num_tasks = len(criterions)
    model.eval()

    total_losses = [0.0] * num_tasks
    all_preds = [[] for _ in range(num_tasks)]
    all_trues = [[] for _ in range(num_tasks)]

    loader_gen = update_dataloader(
        files_with_year,
        local_rank,
        epoch=0,
        X_scaler=X_scaler,
        y_scalers=y_scalers,
        y_root=y_root,
        mode="test",
    )
    test_loader = next(loader_gen)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            for i in range(num_tasks):
                pred = outputs[f"task_{i}"].squeeze(-1)
                true = targets[:, i]
                loss = criterions[f"task_{i}"](pred, true)
                total_losses[i] += loss.item()

                pred_np = pred.cpu().numpy().reshape(-1, 1)
                true_np = true.cpu().numpy().reshape(-1, 1)
                pred_inv = y_scalers[i].inverse_transform(pred_np).flatten()
                true_inv = y_scalers[i].inverse_transform(true_np).flatten()

                all_preds[i].extend(pred_inv)
                all_trues[i].extend(true_inv)

    for i in range(num_tasks):
        r2 = r2_score(all_trues[i], all_preds[i])
        rmse = np.sqrt(mean_squared_error(all_trues[i], all_preds[i]))
        print(
            f"Task {i}: "
            f"R² = {r2:.4f}, RMSE = {rmse:.4f}"
        )
