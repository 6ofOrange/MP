import torch
import torch.distributed as dist

from ..data.dataloader import update_dataloader
from ..utils.plotting import plot_loss_curves, plot_task_losses


def train_model(
    model,
    files_with_year,
    criterions,
    optimizer,
    local_rank,
    device,
    X_scaler,
    y_scalers,
    y_root,
    num_epochs=10,
    model_path="multitask.pth",
    task_weights=None,
):
    num_tasks = len(criterions)
    if task_weights is None:
        task_weights = [1.0] * num_tasks

    model.train()
    epoch_total_loss_list = []
    epoch_task_losses_list = [[] for _ in range(num_tasks)]

    for epoch in range(num_epochs):
        running_loss = 0.0
        len_dataloader = 0
        epoch_task_losses = [0.0] * num_tasks

        loader_gen = update_dataloader(
            files_with_year,
            local_rank,
            epoch,
            X_scaler,
            y_scalers,
            y_root,
            mode="train",
        )

        for inputs, targets in loader_gen:
            if isinstance(inputs, int):
                batch_len, _ = inputs, targets
                len_dataloader += batch_len
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            total_loss = 0.0
            for i in range(num_tasks):
                pred = outputs[f"task_{i}"].squeeze(-1)
                true = targets[:, i]
                loss = criterions[f"task_{i}"](pred, true)
                total_loss += task_weights[i] * loss
                epoch_task_losses[i] += loss.item()

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        # all-reduce
        loss_tensor = torch.tensor(running_loss, device=device)
        count_tensor = torch.tensor(len_dataloader, device=device)
        dist.all_reduce(loss_tensor)
        dist.all_reduce(count_tensor)

        task_loss_tensors = [
            torch.tensor(l, device=device) for l in epoch_task_losses
        ]
        for t in task_loss_tensors:
            dist.all_reduce(t)

        if dist.get_rank() == 0:
            avg_total = loss_tensor.item() / count_tensor.item()
            epoch_total_loss_list.append(avg_total)
            print(f"[Epoch {epoch+1}/{num_epochs}] Total Loss: {avg_total:.4f}")

            for i, t in enumerate(task_loss_tensors):
                avg_task = t.item() / count_tensor.item()
                epoch_task_losses_list[i].append(avg_task)
                print(f"   Task {i} Loss: {avg_task:.4f}")

    dist.barrier()

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), model_path)
        plot_loss_curves(epoch_total_loss_list)
        plot_task_losses(epoch_task_losses_list, num_tasks)

    dist.barrier()
