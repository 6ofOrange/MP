import psutil
import os
import torch

_process = psutil.Process(os.getpid())

def log_mem(stage: str):
    rss = _process.memory_info().rss / 1024**2
    gpu_alloc = torch.cuda.memory_allocated() / 1024**2
    gpu_reserved = torch.cuda.memory_reserved() / 1024**2
    print(
        f"[MEM] {stage:30s} | CPU RAM: {rss:7.1f} MB | "
        f"GPU alloc: {gpu_alloc:7.1f} MB | reserved: {gpu_reserved:7.1f} MB"
    )
