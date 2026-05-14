# util.py
import os
import json
import time
import random
import logging
from typing import Any

import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seed for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_model_params(model, only_trainable: bool = True):
    """
    Count model parameters.

    Returns:
        Number of parameters.
    """
    if only_trainable:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())

    print(f"Number of {'trainable' if only_trainable else 'total'} parameters: {num_params}")
    print(f"Number of parameters in K: {num_params / 1e3:.2f}K")
    print(f"Number of parameters in M: {num_params / 1e6:.4f}M")

    return num_params


def create_experiment_dir(result_dir: str, dataset: str):
    """
    Create an experiment directory.

    Example:
        results/tox21/20260514_153000
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(result_dir, dataset, timestamp)

    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir


def save_config(args: Any, exp_dir: str):
    """
    Save argparse configuration into config.json.
    """
    os.makedirs(exp_dir, exist_ok=True)

    config = vars(args) if hasattr(args, "__dict__") else dict(args)

    save_path = os.path.join(exp_dir, "config.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    return save_path


def set_logger(exp_dir: str, filename: str = "log.txt"):
    """
    Create a logger that writes both to console and file.
    """
    os.makedirs(exp_dir, exist_ok=True)

    logger = logging.getLogger(exp_dir)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(exp_dir, filename),
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Track average values during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class Timer:
    """
    Simple timer.
    """

    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time

    def elapsed_minutes(self):
        return self.elapsed() / 60.0


def move_to_device(batch, device: str):
    """
    Move PyG batch or tensor to device.
    """
    if hasattr(batch, "to"):
        return batch.to(device)

    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]

    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)

    return batch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(model, optimizer, epoch: int, save_path: str, extra: dict = None):
    """
    Save training checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }

    if extra is not None:
        checkpoint.update(extra)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, load_path: str, device: str = "cpu"):
    """
    Load training checkpoint.
    """
    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint