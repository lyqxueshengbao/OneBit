from __future__ import annotations

import numpy as np
import torch

from .fda import FDAConfig, steering_vector_np, steering_vector_torch


def normalized_correlation_torch(
    a: torch.Tensor, z: torch.Tensor, *, eps: float = 1e-12
) -> torch.Tensor:
    """
    J = |a^H z|^2 / ||a||^2

    Shapes:
      a: (...,K) complex
      z: (...,K) complex
      out: (...) real
    """

    inner = torch.sum(torch.conj(a) * z, dim=-1)
    num = torch.abs(inner) ** 2
    denom = torch.sum(torch.abs(a) ** 2, dim=-1).clamp_min(eps)
    return (num / denom).real


def normalized_correlation_np(a: np.ndarray, z: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    inner = np.sum(np.conj(a) * z, axis=-1)
    num = np.abs(inner) ** 2
    denom = np.maximum(np.sum(np.abs(a) ** 2, axis=-1), eps)
    return (num / denom).astype(np.float32)


def J_torch(
    theta_deg: torch.Tensor,
    r_m: torch.Tensor,
    z: torch.Tensor,
    cfg: FDAConfig,
    *,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    a = steering_vector_torch(theta_deg, r_m, cfg, dtype=dtype)
    return normalized_correlation_torch(a, z, eps=eps)


def J_np(
    theta_deg: np.ndarray,
    r_m: np.ndarray,
    z: np.ndarray,
    cfg: FDAConfig,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    a = steering_vector_np(theta_deg, r_m, cfg)
    return normalized_correlation_np(a, z, eps=eps)

