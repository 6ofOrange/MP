import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def plot_loss_curves(epoch_losses, fname="total_loss.png"):
    epochs = range(1, len(epoch_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

def plot_task_losses(epoch_task_losses, num_tasks, fname="task_losses.png"):
    num_epochs = len(epoch_task_losses[0])
    epochs = np.arange(1, num_epochs + 1)

    colors = plt.cm.viridis(np.linspace(0, 1, num_tasks))
    fig, ax = plt.subplots(figsize=(9, 5))
    handles = []

    for i in range(num_tasks):
        y = np.array(epoch_task_losses[i])
        points = np.column_stack((epochs, y))
        segments = [[points[j], points[j + 1]] for j in range(len(points) - 1)]
        lc = LineCollection(segments, colors=[colors[i]], linewidths=2)
        ax.add_collection(lc)
        handles.append(Line2D([0], [0], color=colors[i], lw=2, label=f"Task {i}"))

    ax.set_xlim(epochs.min(), epochs.max())
    ax.legend(handles=handles)
    ax.grid(True)
    plt.savefig(fname)
    plt.close()
