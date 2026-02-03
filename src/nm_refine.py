from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from .fda import FDAConfig
from .objective import J_np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


@dataclass
class NMResult:
    theta_deg: float
    r_m: float
    fun: float
    nfev: int
    nit: int
    success: bool
    message: str


def refine_nelder_mead(
    z: np.ndarray,
    theta0_deg: float,
    r0_m: float,
    cfg: FDAConfig,
    *,
    theta_range: Tuple[float, float] = (-60.0, 60.0),
    r_range: Tuple[float, float] = (0.0, 2000.0),
    maxiter: int = 80,
    xatol: float = 1e-3,
    fatol: float = 1e-4,
    eps: float = 1e-12,
) -> NMResult:
    """
    Nelder–Mead refinement in continuous domain.

    Boundary handling:
      Optimize unconstrained u, map via sigmoid into [min,max].
    Objective:
      minimize f(u) = -J(theta(u), r(u))

    Args:
      z: (K,) complex64 (1-bit measurement)
      theta0_deg, r0_m: coarse init
    """

    tmin, tmax = theta_range
    rmin, rmax = r_range

    def u_to_tr(u: np.ndarray) -> Tuple[float, float]:
        s = _sigmoid(u)
        theta = tmin + (tmax - tmin) * float(s[0])
        r = rmin + (rmax - rmin) * float(s[1])
        return theta, r

    p0 = np.array(
        [
            (theta0_deg - tmin) / (tmax - tmin + eps),
            (r0_m - rmin) / (rmax - rmin + eps),
        ],
        dtype=np.float64,
    )
    u0 = _logit(p0)

    def f(u: np.ndarray) -> float:
        theta, r = u_to_tr(u)
        val = float(J_np(np.array(theta, dtype=np.float32), np.array(r, dtype=np.float32), z, cfg))
        return -val

    res = minimize(
        f,
        u0,
        method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": float(xatol), "fatol": float(fatol)},
    )
    theta_hat, r_hat = u_to_tr(np.asarray(res.x, dtype=np.float64))
    return NMResult(
        theta_deg=float(theta_hat),
        r_m=float(r_hat),
        fun=float(res.fun),
        nfev=int(res.nfev),
        nit=int(res.nit),
        success=bool(res.success),
        message=str(res.message),
    )

