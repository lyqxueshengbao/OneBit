from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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


def refine_nelder_mead_with_history(
    z: np.ndarray,
    theta0_deg: float,
    r0_m: float,
    cfg: FDAConfig,
    *,
    theta_range: Tuple[float, float] = (-60.0, 60.0),
    r_range: Tuple[float, float] = (0.0, 2000.0),
    maxiter: int = 80,
    hist_len: int = 10,
    xatol: float = 1e-3,
    fatol: float = 1e-4,
    eps: float = 1e-12,
) -> tuple[NMResult, np.ndarray, np.ndarray]:
    """
    Same as refine_nelder_mead, but also returns a per-iteration (theta,r) history.

    History convention:
      Each element corresponds to the *current* estimate after one NM iteration (callback xk).
      If NM terminates early, the history is padded by repeating the last estimate.
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

    hist_u: List[np.ndarray] = []

    def cb(xk: np.ndarray) -> None:
        if len(hist_u) >= int(hist_len):
            return
        hist_u.append(np.asarray(xk, dtype=np.float64).copy())

    res = minimize(
        f,
        u0,
        method="Nelder-Mead",
        callback=cb,
        options={"maxiter": int(maxiter), "xatol": float(xatol), "fatol": float(fatol)},
    )

    # Ensure final point is included as the last history element when available.
    x_final = np.asarray(res.x, dtype=np.float64)
    if int(hist_len) > 0:
        if len(hist_u) == 0:
            hist_u.append(x_final.copy())
        else:
            if not np.allclose(hist_u[-1], x_final, rtol=0.0, atol=1e-12):
                hist_u.append(x_final.copy())

    theta_hat, r_hat = u_to_tr(x_final)
    out = NMResult(
        theta_deg=float(theta_hat),
        r_m=float(r_hat),
        fun=float(res.fun),
        nfev=int(res.nfev),
        nit=int(res.nit),
        success=bool(res.success),
        message=str(res.message),
    )

    # Map history to (theta,r), pad/truncate to hist_len.
    th_hist: List[float] = []
    r_hist: List[float] = []
    for u in hist_u[: int(hist_len)]:
        th, rr = u_to_tr(u)
        th_hist.append(float(th))
        r_hist.append(float(rr))

    if int(hist_len) > 0:
        if len(th_hist) == 0:
            th_hist = [float(out.theta_deg)]
            r_hist = [float(out.r_m)]
        while len(th_hist) < int(hist_len):
            th_hist.append(th_hist[-1])
            r_hist.append(r_hist[-1])
        if len(th_hist) > int(hist_len):
            th_hist = th_hist[: int(hist_len)]
            r_hist = r_hist[: int(hist_len)]

    return out, np.asarray(th_hist, dtype=np.float32), np.asarray(r_hist, dtype=np.float32)
