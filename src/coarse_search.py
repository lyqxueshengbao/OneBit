from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .fda import FDAConfig, steering_vector_np, steering_vector_torch
from .objective import normalized_correlation_np


def make_grid_1d(min_v: float, max_v: float, step: float) -> np.ndarray:
    n = int(np.floor((max_v - min_v) / step + 0.5)) + 1
    return (min_v + step * np.arange(n, dtype=np.float32)).astype(np.float32)


def coarse_search_np(
    z: np.ndarray,
    cfg: FDAConfig,
    *,
    theta_range: Tuple[float, float] = (-60.0, 60.0),
    r_range: Tuple[float, float] = (0.0, 2000.0),
    theta_step: float = 1.0,
    r_step: float = 100.0,
) -> Tuple[float, float, float]:
    """
    Coarse grid search (CPU / numpy).

    Args:
      z: (K,) complex64

    Returns:
      theta0_deg, r0_m, best_score
    """

    theta_grid = make_grid_1d(theta_range[0], theta_range[1], theta_step)  # (T,)
    r_grid = make_grid_1d(r_range[0], r_range[1], r_step)  # (R,)

    # (T,R,K)
    A = steering_vector_np(theta_grid[:, None], r_grid[None, :], cfg)
    scores = normalized_correlation_np(A, z[None, None, :])  # (T,R)

    idx = int(np.argmax(scores))
    t_idx, r_idx = np.unravel_index(idx, scores.shape)
    theta0 = float(theta_grid[t_idx])
    r0 = float(r_grid[r_idx])
    return theta0, r0, float(scores[t_idx, r_idx])


@dataclass
class CoarseSearcherTorch:
    """
    Torch coarse search with a precomputed grid manifold.

    Precomputes:
      A_grid: (G,K) complex
      norm2: (G,) real

    Search:
      scores = |A^H z|^2 / ||a||^2
    """

    cfg: FDAConfig
    theta_range: Tuple[float, float] = (-60.0, 60.0)
    r_range: Tuple[float, float] = (0.0, 2000.0)
    theta_step: float = 1.0
    r_step: float = 100.0
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.complex64
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device("cpu")

        theta_grid = torch.tensor(
            make_grid_1d(self.theta_range[0], self.theta_range[1], self.theta_step),
            device=self.device,
        )
        r_grid = torch.tensor(
            make_grid_1d(self.r_range[0], self.r_range[1], self.r_step),
            device=self.device,
        )

        TT, RR = torch.meshgrid(theta_grid, r_grid, indexing="ij")  # (T,R)
        a = steering_vector_torch(TT.reshape(-1), RR.reshape(-1), self.cfg, dtype=self.dtype)
        self.theta_grid = theta_grid
        self.r_grid = r_grid
        self.A_grid = a  # (G,K)
        self.norm2 = torch.sum(torch.abs(a) ** 2, dim=-1).clamp_min(self.eps)  # (G,)

    def search(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          z: (B,K) complex (device must match)

        Returns:
          theta0: (B,) deg
          r0: (B,) m
          score0: (B,) best J value
        """

        AH = torch.conj(self.A_grid).transpose(0, 1)  # (K,G)
        inner = z @ AH  # (B,G) complex
        scores = (torch.abs(inner) ** 2) / self.norm2[None, :]

        best = torch.argmax(scores, dim=-1)  # (B,)
        score0 = scores[torch.arange(z.shape[0], device=z.device), best]

        R = self.r_grid.numel()
        t_idx = torch.div(best, R, rounding_mode="floor")
        r_idx = best - t_idx * R
        theta0 = self.theta_grid[t_idx]
        r0 = self.r_grid[r_idx]
        return theta0, r0, score0

