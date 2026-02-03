import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.utils import shuffle

from models.multitask_lstm import MultiTaskLSTMRegressor
from utils.distributed import init_distributed, cleanup
from train.trainer import train_model
from train.evaluator import evaluate_model
from data.read_data import collect_npy_by_year


def main():
    local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    num_tasks = 6
    model = MultiTaskLSTMRegressor(
        input_size=23,
        hidden_size=256,
        num_tasks=num_tasks,
    )

    criterions = {f"task_{i}": nn.MSELoss() for i in range(num_tasks)}
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
    )

    x_root = "/public/home/zhaoyp01/datafolder/X_new_add2"
    y_root = "/public/home/zhaoyp01/datafolder/y_data"
    nan_txt_root = "/public/home/zhaoyp01/datafolder/nan_files"

    files = collect_npy_by_year(x_root, nan_txt_root)
    files = shuffle(files, random_state=42)
    files = [(y, f) for y, f in files if 2016 <= y <= 2022]

    X_scaler = joblib.load("scalers_23/x_scaler.joblib")
    y_scalers = [
        joblib.load(f"scalers_23/y_scaler_{i}.joblib") for i in range(num_tasks)
    ]

    train_model(
        model,
        files,
        criterions,
        optimizer,
        local_rank,
        device,
        X_scaler,
        y_scalers,
        y_root,
        num_epochs=18,
        model_path="multitask_dis23_7.pth",
        task_weights=[1, 1, 1, 1, 0.8, 1],
    )

    if torch.distributed.get_rank() == 0:
        eval_model = MultiTaskLSTMRegressor(23, 256, num_tasks)
        eval_model.load_state_dict(
            torch.load("multitask_dis23_7.pth", map_location=device)
        )
        eval_model.to(device)
        evaluate_model(
            eval_model,
            files,
            criterions,
            device,
            X_scaler,
            y_scalers,
            y_root,
            local_rank,
        )

    cleanup()


if __name__ == "__main__":
    main()
