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
        r_precond_mul: float = 1.0,
        r_precond_pow: float = 1.0,
        r_precond_learnable: bool = False,
        use_pscale: bool = False,
        pscale_hidden: int = 32,
        pscale_detach_step: bool = True,
        pscale_logrange: float = 6.9,
        pscale_amp: float = 1.0,
        pscale_input: str = "step,u,t",
        use_t_table: bool = False,
        t_table_init: float = 0.0,
        pscale_min_theta: float = 0.7,
        pscale_max_theta: float = 1.3,
        pscale_min_r: float = 0.7,
        pscale_max_r: float = 1.3,
        objective: str = "logistic",  # {"logistic","probit","J"}
        init_alpha: float = 1e-2,
        init_lambda: float = 1e-3,
        alpha_min: float = 1e-6,
        alpha_max: float = 2e-1,
        lambda_min: float = 1e-6,
        lambda_max: float = 1.0,
        step_clip: float = 1e-1,
        delta_theta_clip_deg: float = 5.0,
        delta_r_clip_m: float = 200.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.T = int(T)
        self.box = box
        self.r_box = str(r_box)
        self.objective = str(objective)
        self.r_precond_pow = float(r_precond_pow)
        self.r_precond_learnable = bool(r_precond_learnable)
        if self.r_precond_learnable:
            self.r_precond_mul = nn.Parameter(torch.tensor(float(r_precond_mul), dtype=torch.float32))
        else:
            # Keep default behavior unchanged: when not learnable, this is a plain python float
            # and is NOT saved into the state_dict.
            self.r_precond_mul = float(r_precond_mul)

        # Tiny per-sample scale preconditioner (pscale) for update: u <- u - scale(step,t) * step_u
        self.use_pscale = bool(use_pscale)
        self.pscale_detach_step = bool(pscale_detach_step)
        self.pscale_logrange = float(pscale_logrange)
        self.pscale_amp = float(pscale_amp)
        self.use_t_table = bool(use_t_table) and self.use_pscale
        self.pscale_min_theta = float(pscale_min_theta)
        self.pscale_max_theta = float(pscale_max_theta)
        self.pscale_min_r = float(pscale_min_r)
        self.pscale_max_r = float(pscale_max_r)

        if self.use_pscale:
            # Parse pscale_input tokens (order preserved, deduped).
            allowed = {"step", "u", "t"}
            raw = [tok.strip().lower() for tok in str(pscale_input).split(",")]
            tokens: list[str] = []
            seen = set()
            for tok in raw:
                if not tok:
                    continue
                if tok not in allowed:
                    raise ValueError(
                        f"Invalid pscale_input token: {tok!r}. Allowed: {sorted(allowed)}. "
                        f"Got: {pscale_input!r}"
                    )
                if tok not in seen:
                    tokens.append(tok)
                    seen.add(tok)
            if not tokens:
                raise ValueError(f"Empty pscale_input after parsing: {pscale_input!r}")
            self.pscale_tokens = tokens
            self.pscale_feat_dim = int(len(tokens))

            hid = int(pscale_hidden)
            self.pscale_mlp = nn.Sequential(
                nn.Linear(self.pscale_feat_dim, hid),
                nn.SiLU(),
                nn.Linear(hid, 2),
            )
            # Initialize to exact identity scaling: delta=0 => log_scale≈0 => scale≈1.
            nn.init.zeros_(self.pscale_mlp[-1].weight)
            nn.init.zeros_(self.pscale_mlp[-1].bias)

        if self.use_t_table:
            self.t_log_scale_table = nn.Parameter(
                torch.full((self.T, 2), float(t_table_init), dtype=torch.float32)
            )

        self.init_alpha = float(init_alpha)
        self.init_lambda = float(init_lambda)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.step_clip = float(step_clip)
        self.delta_theta_clip_deg = float(delta_theta_clip_deg)
        self.delta_r_clip_m = float(delta_r_clip_m)

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
        trace_detach: bool = True,
        pscale_step_norm: float = 0.0,
    ):
        """
        Args:
          z: (B,K) complex64
          theta0_deg: (B,) deg
          r0_m: (B,) m
          beta: likelihood sharpness
          second_order: if True, build higher-order graph (slow/unstable; default False)
          sanitize_in_forward: if True, use nan_to_num *views* of raw params for computation (no write-back)
          return_trace: if True, return per-step traces (detached by default)
          trace_detach: if False, keep trace tensors differentiable (for training distillation)

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
        u_trace = []
        pscale_sum = None
        pscale_sumsq = None
        pscale_count = 0
        pscale_clamp_hits = None
        pscale_clamp_total = 0.0
        pscale_reg_sum = None
        nonfinite_g_ratio = []
        fallback_alpha = []
        fallback_lambda = []

        with torch.enable_grad():
            for t in range(steps):
                u = u.requires_grad_(True)
                u_prev = u
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
                    scale_base = scale_r.clamp(1e-4, 1e4)
                    # scale_final = (scale_r ** pow) * mul
                    mul = self.r_precond_mul if self.r_precond_learnable else float(self.r_precond_mul)
                    scale_final = (scale_base ** float(self.r_precond_pow)) * mul
                    # Avoid in-place update (needed when mul is learnable).
                    step_u_r = step_u[:, 1] * scale_final
                    step_u = torch.stack([step_u[:, 0], step_u_r], dim=-1)

                if self.use_pscale:
                    # Build pscale features according to tokens (each token contributes 1 scalar per sample).
                    B = int(u.shape[0])
                    feat_list = []
                    denom_train = 1.0
                    step_norm = float(pscale_step_norm)
                    step_norm = max(0.0, min(1.0, step_norm))
                    t_norm = float(t) / float(max(steps - 1, 1))

                    u_for_feat = u.detach() if self.pscale_detach_step else u
                    u_norm = torch.linalg.vector_norm(u_for_feat.to(torch.float32), ord=2, dim=-1)  # (B,)
                    u_feat = torch.log1p(u_norm).clamp(0.0, 10.0).to(u.dtype)

                    for tok in self.pscale_tokens:
                        if tok == "step":
                            feat_list.append(
                                torch.full((B, 1), step_norm, device=u.device, dtype=u.dtype)
                            )
                        elif tok == "t":
                            feat_list.append(
                                torch.full((B, 1), t_norm, device=u.device, dtype=u.dtype)
                            )
                        elif tok == "u":
                            feat_list.append(u_feat.view(B, 1))
                        else:
                            raise RuntimeError(f"Unexpected pscale token: {tok}")

                    x = torch.cat(feat_list, dim=-1)
                    delta = torch.tanh(self.pscale_mlp(x)) * float(self.pscale_amp)
                    log_scale = delta
                    if self.use_t_table:
                        log_scale = log_scale + self.t_log_scale_table[t].to(log_scale.dtype).view(1, 2)
                    log_scale = log_scale.clamp(-float(self.pscale_logrange), float(self.pscale_logrange))
                    scale_raw = torch.exp(log_scale)

                    # Clamp pscale in scale-space (per-dim), to prevent runaway/degenerate solutions.
                    scale_min = torch.tensor(
                        [self.pscale_min_theta, self.pscale_min_r],
                        device=scale_raw.device,
                        dtype=scale_raw.dtype,
                    ).view(1, 2)
                    scale_max = torch.tensor(
                        [self.pscale_max_theta, self.pscale_max_r],
                        device=scale_raw.device,
                        dtype=scale_raw.dtype,
                    ).view(1, 2)
                    scale = torch.max(torch.min(scale_raw, scale_max), scale_min)

                    step_eff = scale * step_u

                    if self.delta_theta_clip_deg > 0.0 or self.delta_r_clip_m > 0.0:
                        theta_prev, r_prev = theta, r
                        u_cand = u_prev - step_eff
                        theta_cand, r_cand = map_u_to_theta_r(u_cand, self.box, r_box=self.r_box)
                        dtheta = (theta_cand - theta_prev).detach()
                        dr = (r_cand - r_prev).detach()

                        scale_th = torch.ones_like(dtheta)
                        scale_rr = torch.ones_like(dr)
                        if self.delta_theta_clip_deg > 0.0:
                            clip_th = torch.tensor(
                                float(self.delta_theta_clip_deg),
                                device=dtheta.device,
                                dtype=dtheta.dtype,
                            )
                            scale_th = torch.minimum(scale_th, clip_th / (torch.abs(dtheta) + 1e-12))
                        if self.delta_r_clip_m > 0.0:
                            clip_rr = torch.tensor(
                                float(self.delta_r_clip_m),
                                device=dr.device,
                                dtype=dr.dtype,
                            )
                            scale_rr = torch.minimum(scale_rr, clip_rr / (torch.abs(dr) + 1e-12))

                        scale_tr = torch.stack([scale_th, scale_rr], dim=-1).clamp(0.0, 1.0)
                        step_eff = step_eff * scale_tr

                    u = u - step_eff

                    with torch.no_grad():
                        if pscale_sum is None:
                            pscale_sum = torch.zeros((2,), device=scale.device, dtype=torch.float32)
                            pscale_sumsq = torch.zeros((2,), device=scale.device, dtype=torch.float32)
                            pscale_clamp_hits = torch.zeros((2,), device=scale.device, dtype=torch.float32)
                            pscale_reg_sum = torch.zeros((), device=scale.device, dtype=torch.float32)
                        s = scale.detach().to(torch.float32)
                        pscale_sum += s.sum(dim=0)
                        pscale_sumsq += (s * s).sum(dim=0)
                        pscale_count += int(s.shape[0])
                        hit = (scale.detach() != scale_raw.detach()).to(torch.float32)  # (B,2)
                        pscale_clamp_hits += hit.sum(dim=0)
                        pscale_clamp_total += float(hit.shape[0])
                        pscale_reg_sum += torch.mean(torch.sum((s - 1.0) * (s - 1.0), dim=-1))
                else:
                    step_eff = step_u
                    if self.delta_theta_clip_deg > 0.0 or self.delta_r_clip_m > 0.0:
                        theta_prev, r_prev = theta, r
                        u_cand = u_prev - step_eff
                        theta_cand, r_cand = map_u_to_theta_r(u_cand, self.box, r_box=self.r_box)
                        dtheta = (theta_cand - theta_prev).detach()
                        dr = (r_cand - r_prev).detach()

                        scale_th = torch.ones_like(dtheta)
                        scale_rr = torch.ones_like(dr)
                        if self.delta_theta_clip_deg > 0.0:
                            clip_th = torch.tensor(
                                float(self.delta_theta_clip_deg),
                                device=dtheta.device,
                                dtype=dtheta.dtype,
                            )
                            scale_th = torch.minimum(scale_th, clip_th / (torch.abs(dtheta) + 1e-12))
                        if self.delta_r_clip_m > 0.0:
                            clip_rr = torch.tensor(
                                float(self.delta_r_clip_m),
                                device=dr.device,
                                dtype=dr.dtype,
                            )
                            scale_rr = torch.minimum(scale_rr, clip_rr / (torch.abs(dr) + 1e-12))

                        scale_tr = torch.stack([scale_th, scale_rr], dim=-1).clamp(0.0, 1.0)
                        step_eff = step_eff * scale_tr

                    u = u - step_eff

                if return_trace:
                    obj_trace.append(obj.detach())
                    if trace_detach:
                        theta_trace.append(theta.detach())
                        r_trace.append(r.detach())
                        u_trace.append(u_prev.detach())
                    else:
                        theta_trace.append(theta)
                        r_trace.append(r)
                        u_trace.append(u_prev)

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
        if self.use_pscale and pscale_sum is not None:
            mean = pscale_sum / max(float(pscale_count), 1.0)
            var = pscale_sumsq / max(float(pscale_count), 1.0) - mean * mean
            std = torch.sqrt(var.clamp_min(0.0))
            debug["pscale_scale_mean"] = mean
            debug["pscale_scale_std"] = std
            debug["pscale_clamp_hit_ratio"] = pscale_clamp_hits / max(float(pscale_clamp_total), 1.0)
            debug["pscale_reg_term"] = pscale_reg_sum / max(float(steps), 1.0)
            if self.use_t_table:
                tbl = self.t_log_scale_table.detach().to(torch.float32)
                debug["t_table_mean"] = tbl.mean(dim=0)
                debug["t_table_std"] = tbl.std(dim=0, unbiased=False)

        if not return_trace:
            return theta_T, r_T, debug
        return theta_T, r_T, debug, {
            "J": obj_trace,
            "theta": theta_trace,
            "r": r_trace,
            "u": u_trace,
            "u_T": u.detach() if trace_detach else u,
            "nonfinite_g_ratio": nonfinite_g_ratio,
            "fallback_alpha": fallback_alpha,
            "fallback_lambda": fallback_lambda,
        }
