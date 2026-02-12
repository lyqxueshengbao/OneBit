from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .coarse_search import make_grid_1d
from .fda import FDAConfig, steering_vector_np
from .objective import J_np


def _soft_threshold(x: np.ndarray, tau: np.ndarray | float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


@dataclass
class QGAMPWarmStarter:
    """
    Lightweight quantized sparse warm-start on a fixed (theta, r) grid.

    This module performs a simple quantized logistic sparse recovery over a
    precomputed dictionary A_grid, and picks the best candidate by J-score.
    """

    cfg: FDAConfig
    theta_range: Tuple[float, float] = (-60.0, 60.0)
    r_range: Tuple[float, float] = (0.0, 2000.0)
    theta_step: float = 1.0
    r_step: float = 100.0
    beta: float = 2.0
    l1: float = 3e-3
    l2: float = 1e-4
    iters: int = 8
    topcand: int = 8
    nonneg: bool = True
    step_scale: float = 1.0

    def __post_init__(self) -> None:
        self.theta_grid = make_grid_1d(self.theta_range[0], self.theta_range[1], self.theta_step)
        self.r_grid = make_grid_1d(self.r_range[0], self.r_range[1], self.r_step)
        tt, rr = np.meshgrid(self.theta_grid, self.r_grid, indexing="ij")
        self.theta_flat = tt.reshape(-1).astype(np.float32)
        self.r_flat = rr.reshape(-1).astype(np.float32)

        # Dictionary on grid: (G,K) complex.
        self.A_grid = steering_vector_np(self.theta_flat, self.r_flat, self.cfg).astype(np.complex64)
        # Real-augmented measurement matrix: y = sign(Phi x + noise), y in {+1,-1}^{2K}.
        # Phi: (2K, G) with rows [Re, Im].
        phi_r = self.A_grid.real.T.astype(np.float32)  # (K,G)
        phi_i = self.A_grid.imag.T.astype(np.float32)  # (K,G)
        self.Phi = np.concatenate([phi_r, phi_i], axis=0).astype(np.float32)  # (2K,G)

        m = float(self.Phi.shape[0])
        # Diagonal preconditioner for stable proximal-gradient.
        self.diag_inv = 1.0 / (np.mean(self.Phi * self.Phi, axis=0) + float(self.l2) + 1e-8)
        self.inv_m = 1.0 / max(m, 1.0)

    def _solve_coeff(self, z: np.ndarray) -> np.ndarray:
        y = np.concatenate([np.sign(z.real), np.sign(z.imag)], axis=0).astype(np.float32)
        y[y == 0.0] = 1.0

        g = self.Phi.shape[1]
        x = np.zeros((g,), dtype=np.float32)
        eta = float(self.step_scale)
        beta = float(self.beta)
        l1 = float(self.l1)
        l2 = float(self.l2)
        for _ in range(max(int(self.iters), 1)):
            s = self.Phi @ x  # (2K,)
            yz = y * s
            # d/dx sum log(1+exp(-beta y s)) = -beta * Phi^T (y * sigmoid(-beta y s))
            w = y * _sigmoid(-beta * yz)
            grad = -(beta * self.inv_m) * (self.Phi.T @ w) + l2 * x
            x = x - eta * self.diag_inv * grad
            x = _soft_threshold(x, eta * l1 * self.diag_inv)
            if self.nonneg:
                x = np.maximum(x, 0.0)
            x = x.astype(np.float32, copy=False)
        return x

    def search_single(self, z: np.ndarray) -> tuple[float, float, float]:
        x = self._solve_coeff(z.astype(np.complex64, copy=False))
        topk = max(1, min(int(self.topcand), int(x.shape[0])))
        cand_idx = np.argpartition(np.abs(x), -topk)[-topk:]
        cand_th = self.theta_flat[cand_idx]
        cand_rr = self.r_flat[cand_idx]
        scores = J_np(cand_th, cand_rr, z.astype(np.complex64, copy=False), self.cfg)
        best_local = int(np.argmax(scores))
        best_idx = int(cand_idx[best_local])
        return (
            float(self.theta_flat[best_idx]),
            float(self.r_flat[best_idx]),
            float(scores[best_local]),
        )

    def search_batch(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(z.shape[0])
        theta0 = np.zeros((n,), dtype=np.float32)
        r0 = np.zeros((n,), dtype=np.float32)
        score0 = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            th, rr, sc = self.search_single(z[i])
            theta0[i] = np.float32(th)
            r0[i] = np.float32(rr)
            score0[i] = np.float32(sc)
        return theta0, r0, score0

