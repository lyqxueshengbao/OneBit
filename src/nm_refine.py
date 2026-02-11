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
    early_stop_eps_theta: float | None = None,
    early_stop_eps_r: float | None = None,
    early_stop_patience: int = 0,
    early_stop_kmin: int = 1,
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

    f_calls = 0

    def f(u: np.ndarray) -> float:
        nonlocal f_calls
        f_calls += 1
        theta, r = u_to_tr(u)
        val = float(J_np(np.array(theta, dtype=np.float32), np.array(r, dtype=np.float32), z, cfg))
        return -val

    hist_u: List[np.ndarray] = []
    early_on = (
        early_stop_eps_theta is not None
        and early_stop_eps_r is not None
        and int(early_stop_patience) > 0
        and int(early_stop_kmin) > 0
    )
    stable = 0

    class _EarlyStop(Exception):
        pass

    def _wrap_abs_deg(delta: float) -> float:
        return abs((delta + 180.0) % 360.0 - 180.0)

    def cb(xk: np.ndarray) -> None:
        if len(hist_u) >= int(hist_len):
            return
        hist_u.append(np.asarray(xk, dtype=np.float64).copy())
        if not early_on or len(hist_u) < 2:
            return
        th_prev, r_prev = u_to_tr(hist_u[-2])
        th_cur, r_cur = u_to_tr(hist_u[-1])
        dth = _wrap_abs_deg(float(th_cur - th_prev))
        dr = abs(float(r_cur - r_prev))
        if dth < float(early_stop_eps_theta) and dr < float(early_stop_eps_r):
            stable_local = 1
        else:
            stable_local = 0
        nonlocal stable
        stable = (stable + 1) if stable_local else 0
        if len(hist_u) >= int(early_stop_kmin) and stable >= int(early_stop_patience):
            raise _EarlyStop()

    early_stopped = False
    try:
        res = minimize(
            f,
            u0,
            method="Nelder-Mead",
            callback=cb,
            options={"maxiter": int(maxiter), "xatol": float(xatol), "fatol": float(fatol)},
        )
        x_final = np.asarray(res.x, dtype=np.float64)
        fun_final = float(res.fun)
        nfev = int(res.nfev)
        nit = int(res.nit)
        success = bool(res.success)
        message = str(res.message)
    except _EarlyStop:
        early_stopped = True
        x_final = hist_u[-1].copy() if len(hist_u) > 0 else u0.copy()
        fun_final = float(f(x_final))
        nfev = int(f_calls)
        nit = int(len(hist_u))
        success = True
        message = "early-stop (callback)"

    # Ensure final point is included as the last history element when available.
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
        fun=float(fun_final),
        nfev=int(nfev),
        nit=int(nit),
        success=bool(success),
        message=(f"{message}; hist={len(hist_u)}" if early_stopped else str(message)),
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
