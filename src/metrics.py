from __future__ import annotations

import numpy as np
import torch


def wrap_angle_deg_np(x: np.ndarray) -> np.ndarray:
    """
    Wrap angle to [-180, 180).
    """

    y = (x + 180.0) % 360.0 - 180.0
    return y.astype(np.float32)


def wrap_angle_deg_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + 180.0, 360.0) - 180.0


def angle_error_deg_np(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return wrap_angle_deg_np(est - gt)


def angle_error_deg_torch(est: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return wrap_angle_deg_torch(est - gt)


def rmse_np(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def rmse_torch(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x * x)).item())


def rmse_angle_deg_np(est: np.ndarray, gt: np.ndarray) -> float:
    return rmse_np(angle_error_deg_np(est, gt))


def rmse_angle_deg_torch(est: torch.Tensor, gt: torch.Tensor) -> float:
    return rmse_torch(angle_error_deg_torch(est, gt))

