import torch
import torch.nn as nn

class MultiTaskLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_tasks=6,
        num_layers=4,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.task_heads = nn.ModuleDict({
            f"task_{i}": nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
            for i in range(num_tasks)
        })

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device
        )
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(x, (h0, c0))
        shared = out[:, -1, :]

        return {k: head(shared) for k, head in self.task_heads.items()}
