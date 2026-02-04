from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

from .fda import FDAConfig, steering_vector_torch
from .objective import loglike_logistic, loglike_probit, normalized_correlation_torch


def _inv_sigmoid(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def _inv_tanh(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = y.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(y) - torch.log1p(-y))


def _inv_softplus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically-stable inverse-softplus for initialization (x must be positive).
    """

    x = x.clamp_min(eps)
    return torch.log(torch.expm1(x))


@dataclass(frozen=True)
class Box:
    theta_min: float = -60.0
    theta_max: float = 60.0
    r_min: float = 0.0
    r_max: float = 2000.0


def _box01_from_u(u: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "sigmoid":
        return torch.sigmoid(u)
    if mode == "tanh":
        # tanh box: s = 0.5 * (tanh(u) + 1) ∈ [0,1]
        return 0.5 * (torch.tanh(u) + 1.0)
    raise ValueError(f"Unknown r_box mode: {mode}")


def _inv_box01_to_u(s: torch.Tensor, *, mode: str, eps: float = 1e-6) -> torch.Tensor:
    s = s.clamp(eps, 1.0 - eps)
    if mode == "sigmoid":
        return _inv_sigmoid(s, eps=eps)
    if mode == "tanh":
        # tanh box inverse: u = atanh(2s-1)
        y = (2.0 * s - 1.0).clamp(-1.0 + eps, 1.0 - eps)
        return _inv_tanh(y, eps=eps)
    raise ValueError(f"Unknown r_box mode: {mode}")


def map_u_to_theta_r(u: torch.Tensor, box: Box, *, r_box: str = "tanh") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map unconstrained u -> (theta,r) within the valid box.

    Args:
      u: (B,2) where u[:,0] for theta, u[:,1] for r
    Returns:
      theta_deg: (B,) deg
      r_m: (B,) m
    """

    theta_mid = 0.5 * (box.theta_min + box.theta_max)
    theta_half = 0.5 * (box.theta_max - box.theta_min)
    theta = theta_mid + theta_half * torch.tanh(u[:, 0])

    # tanh box (default): more linear / gentler boundary saturation than sigmoid.
    s = _box01_from_u(u[:, 1], mode=str(r_box))
    r = box.r_min + (box.r_max - box.r_min) * s
    return theta, r


def map_theta_r_to_u(theta_deg: torch.Tensor, r_m: torch.Tensor, box: Box, *, r_box: str = "tanh") -> torch.Tensor:
    """
    Inverse map (theta,r) -> u via (atanh/logit).
    """

    theta_mid = 0.5 * (box.theta_min + box.theta_max)
    theta_half = 0.5 * (box.theta_max - box.theta_min)
    y_theta = (theta_deg - theta_mid) / theta_half

    s_r = (r_m - box.r_min) / (box.r_max - box.r_min)
    u_r = _inv_box01_to_u(s_r, mode=str(r_box))
    return torch.stack([_inv_tanh(y_theta), u_r], dim=-1)


class Refiner(nn.Module):
    """
    Unrolled off-grid refinement with boundary constraints.

    Objective:
      Maximize 1-bit likelihood (default logistic), or use baseline J for ablation.

    Update (simple diagonal LM-like):
      g = ∂(-obj)/∂u
      du_theta = alpha_theta[t] * g_theta / (|g_theta| + lambda_theta[t])
      du_r     = alpha_r[t]     * g_r     / (|g_r|     + lambda_r[t])
      u <- u - clamp([du_theta, du_r], [-step_clip, step_clip])

    Stability:
      Default is *first-order unroll* (second_order=False), i.e. do NOT backprop through the
      gradient operator itself (no higher-order graph). This is far more stable.
    """

    def __init__(
        self,
        cfg: FDAConfig,
        *,
        T: int = 10,
        box: Box = Box(),
        learnable: bool = True,
        r_box: str = "tanh",  # {"tanh","sigmoid"}
        objective: str = "logistic",  # {"logistic","probit","J"}
        init_alpha: float = 1e-2,
        init_lambda: float = 1e-3,
        alpha_min: float = 1e-6,
        alpha_max: float = 2e-1,
        lambda_min: float = 1e-6,
        lambda_max: float = 1.0,
        step_clip: float = 1e-1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.T = int(T)
        self.box = box
        self.r_box = str(r_box)
        self.objective = str(objective)

        self.init_alpha = float(init_alpha)
        self.init_lambda = float(init_lambda)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.step_clip = float(step_clip)

        a0 = torch.full((self.T,), float(init_alpha), dtype=torch.float32)
        l0 = torch.full((self.T,), float(init_lambda), dtype=torch.float32)
        a_raw0 = _inv_softplus(a0)
        l_raw0 = _inv_softplus(l0)

        if learnable:
            self.alpha_theta_raw = nn.Parameter(a_raw0.clone())
            self.alpha_r_raw = nn.Parameter(a_raw0.clone())
            self.lambda_theta_raw = nn.Parameter(l_raw0.clone())
            self.lambda_r_raw = nn.Parameter(l_raw0.clone())
        else:
            self.register_buffer("alpha_theta_raw", a_raw0.clone())
            self.register_buffer("alpha_r_raw", a_raw0.clone())
            self.register_buffer("lambda_theta_raw", l_raw0.clone())
            self.register_buffer("lambda_r_raw", l_raw0.clone())

    def _alpha_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.softplus(raw)
        return a.clamp(self.alpha_min, self.alpha_max)

    def _lambda_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        l = torch.nn.functional.softplus(raw)
        return l.clamp(self.lambda_min, self.lambda_max)

    def _objective_value(
        self, theta_deg: torch.Tensor, r_m: torch.Tensor, z: torch.Tensor, beta: float | torch.Tensor
    ) -> torch.Tensor:
        if self.objective == "logistic":
            return loglike_logistic(theta_deg, r_m, z, beta, self.cfg)
        if self.objective == "probit":
            return loglike_probit(theta_deg, r_m, z, beta, self.cfg)
        if self.objective == "J":
            a = steering_vector_torch(theta_deg, r_m, self.cfg, dtype=z.dtype)
            return normalized_correlation_torch(a, z)
        raise ValueError(f"Unknown objective: {self.objective}")

    def forward(
        self,
        z: torch.Tensor,
        theta0_deg: torch.Tensor,
        r0_m: torch.Tensor,
        *,
        T_run: int | None = None,
        beta: float | torch.Tensor = 2.0,
        second_order: bool = False,
        sanitize_in_forward: bool = False,
        return_trace: bool = False,
    ):
        """
        Args:
          z: (B,K) complex64
          theta0_deg: (B,) deg
          r0_m: (B,) m
          beta: likelihood sharpness
          second_order: if True, build higher-order graph (slow/unstable; default False)
          sanitize_in_forward: if True, use nan_to_num *views* of raw params for computation (no write-back)
          return_trace: if True, return per-step traces (detached)

        Returns:
          theta_T: (B,) deg
          r_T: (B,) m
          debug: dict with keys {"obj","ll_mean","alpha_mean","lambda_mean","objective","beta"}
          trace (optional): dict with keys {"J","theta","r","nonfinite_g_ratio","fallback_alpha","fallback_lambda"}
        """

        u = map_theta_r_to_u(theta0_deg, r0_m, self.box, r_box=self.r_box)

        raw_at = self.alpha_theta_raw
        raw_ar = self.alpha_r_raw
        raw_lt = self.lambda_theta_raw
        raw_lr = self.lambda_r_raw
        if sanitize_in_forward:
            raw_at = torch.nan_to_num(raw_at, nan=0.0, posinf=0.0, neginf=0.0)
            raw_ar = torch.nan_to_num(raw_ar, nan=0.0, posinf=0.0, neginf=0.0)
            raw_lt = torch.nan_to_num(raw_lt, nan=0.0, posinf=0.0, neginf=0.0)
            raw_lr = torch.nan_to_num(raw_lr, nan=0.0, posinf=0.0, neginf=0.0)

        alpha_theta = self._alpha_from_raw(raw_at).to(u.device)
        alpha_r = self._alpha_from_raw(raw_ar).to(u.device)
        lambda_theta = self._lambda_from_raw(raw_lt).to(u.device)
        lambda_r = self._lambda_from_raw(raw_lr).to(u.device)

        steps = self.T if T_run is None else min(int(T_run), self.T)

        obj_trace = []
        theta_trace = []
        r_trace = []
        nonfinite_g_ratio = []
        fallback_alpha = []
        fallback_lambda = []

        with torch.enable_grad():
            for t in range(steps):
                u = u.requires_grad_(True)
                theta, r = map_u_to_theta_r(u, self.box, r_box=self.r_box)
                obj = self._objective_value(theta, r, z, beta)  # (B,)

                g = torch.autograd.grad(
                    (-obj).sum(),
                    u,
                    create_graph=bool(second_order),
                    retain_graph=bool(second_order),
                )[0]

                # Per-step fallback statistics (raw param non-finite flags).
                fallback_alpha.append(
                    (
                        (~torch.isfinite(self.alpha_theta_raw[t]))
                        | (~torch.isfinite(self.alpha_r_raw[t]))
                    )
                    .to(torch.float32)
                    .detach()
                )
                fallback_lambda.append(
                    (
                        (~torch.isfinite(self.lambda_theta_raw[t]))
                        | (~torch.isfinite(self.lambda_r_raw[t]))
                    )
                    .to(torch.float32)
                    .detach()
                )

                g_finite = torch.isfinite(g)
                nonfinite_g_ratio.append((~g_finite).to(torch.float32).mean().detach())
                g = torch.where(g_finite, g, torch.zeros_like(g))

                g_th = g[:, 0]
                g_r = g[:, 1]

                a_th = alpha_theta[t]
                a_r = alpha_r[t]
                l_th = lambda_theta[t]
                l_r = lambda_r[t]

                # Prevent NaN propagation.
                a_th = torch.where(torch.isfinite(a_th), a_th, torch.full_like(a_th, self.init_alpha))
                a_r = torch.where(torch.isfinite(a_r), a_r, torch.full_like(a_r, self.init_alpha))
                l_th = torch.where(torch.isfinite(l_th), l_th, torch.full_like(l_th, self.init_lambda))
                l_r = torch.where(torch.isfinite(l_r), l_r, torch.full_like(l_r, self.init_lambda))

                denom_th = torch.abs(g_th).clamp_min(1e-6) + l_th
                denom_r = torch.abs(g_r).clamp_min(1e-6) + l_r
                denom_th = torch.nan_to_num(denom_th, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6)
                denom_r = torch.nan_to_num(denom_r, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6)

                step_th = a_th * g_th / denom_th
                step_r = a_r * g_r / denom_r
                step_u = torch.stack([step_th, step_r], dim=-1)
                step_u = torch.nan_to_num(step_u, nan=0.0, posinf=0.0, neginf=0.0)
                step_u = torch.clamp(step_u, -self.step_clip, self.step_clip)

                if self.r_box == "tanh":
                    # r precondition scaling:
                    # Compensate for tanh saturation so a similar step_u produces a more constant Δr.
                    u1 = u[:, 1].detach()
                    tanh_u1 = torch.tanh(u1)
                    dr_du1 = 0.5 * (self.box.r_max - self.box.r_min) * (1.0 - tanh_u1 * tanh_u1)
                    target_dr_du = 0.25 * (self.box.r_max - self.box.r_min)
                    scale_r = (float(target_dr_du) / (dr_du1 + 1e-6)).clamp(0.2, 5.0)
                    step_u[:, 1] = step_u[:, 1] * scale_r

                u = u - step_u

                if return_trace:
                    obj_trace.append(obj.detach())
                    theta_trace.append(theta.detach())
                    r_trace.append(r.detach())

        theta_T, r_T = map_u_to_theta_r(u, self.box, r_box=self.r_box)
        obj_T = self._objective_value(theta_T, r_T, z, beta)

        debug: Dict[str, Any] = {
            "obj": obj_T,
            "ll_mean": obj_T.mean(),
            "alpha_mean": torch.stack([alpha_theta[:steps], alpha_r[:steps]], dim=0).mean(),
            "lambda_mean": torch.stack([lambda_theta[:steps], lambda_r[:steps]], dim=0).mean(),
            "objective": self.objective,
            "beta": beta if isinstance(beta, float) else float(beta.detach().cpu().item()),
        }

        if not return_trace:
            return theta_T, r_T, debug
        return theta_T, r_T, debug, {
            "J": obj_trace,
            "theta": theta_trace,
            "r": r_trace,
            "nonfinite_g_ratio": nonfinite_g_ratio,
            "fallback_alpha": fallback_alpha,
            "fallback_lambda": fallback_lambda,
        }
