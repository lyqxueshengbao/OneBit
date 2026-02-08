from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize

from scripts.eval_all import run_unrolled
from src.coarse_search import coarse_search_np, make_grid_1d
from src.dataset import TargetBox
from src.fda import FDAConfig, steering_vector_np
from src.metrics import angle_error_deg_np, rmse_np
from src.nm_refine import refine_nelder_mead
from src.quantize import quantize_1bit_np
from src.utils import CsvLogger, Timer, ensure_dir, seed_all, timestamp, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snr_list", type=str, default="-10,-5,0,5,10,15,20")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_targets", type=int, default=2)
    p.add_argument("--min_sep_theta_deg", type=float, default=3.0)
    p.add_argument("--min_sep_r_m", type=float, default=120.0)
    p.add_argument("--theta_step", type=float, default=1.0)
    p.add_argument("--r_step", type=float, default=100.0)
    p.add_argument("--nm_maxiter", type=int, default=60)
    p.add_argument("--T", type=int, default=1)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--sanitize_ckpt", action="store_true")
    p.add_argument("--learned_nm_tail_iters", type=int, default=0)
    p.add_argument("--learned_nm_tail_snr_min", type=float, default=0.0)
    p.add_argument("--run_topk_nm_baseline", type=int, default=1, choices=[0, 1])
    p.add_argument("--topk_init_min_sep_theta_deg", type=float, default=1.0)
    p.add_argument("--topk_init_min_sep_r_m", type=float, default=100.0)
    p.add_argument("--joint_nm2_num_samples", type=int, default=0)
    p.add_argument("--joint_nm2_maxiter", type=int, default=80)
    p.add_argument("--match_theta_scale_deg", type=float, default=1.0)
    p.add_argument("--match_r_scale_m", type=float, default=100.0)
    p.add_argument("--full", action="store_true")
    p.add_argument("--run_dir", type=str, default="")
    return p.parse_args()


def ms_per_sample(dt_s: float, n: int) -> float:
    return 1000.0 * dt_s / max(int(n), 1)


def build_grid_cache(
    cfg: FDAConfig,
    box: TargetBox,
    *,
    theta_step: float,
    r_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_grid = make_grid_1d(box.theta_min, box.theta_max, float(theta_step))
    r_grid = make_grid_1d(box.r_min, box.r_max, float(r_step))
    tt, rr = np.meshgrid(theta_grid, r_grid, indexing="ij")
    theta_flat = tt.reshape(-1).astype(np.float32)
    r_flat = rr.reshape(-1).astype(np.float32)
    a_grid = steering_vector_np(theta_flat, r_flat, cfg).astype(np.complex64)  # (G,K)
    ah_grid = np.conj(a_grid)  # (G,K)
    norm2 = np.maximum(np.sum(np.abs(a_grid) ** 2, axis=-1).real.astype(np.float32), 1e-12)
    return theta_flat, r_flat, ah_grid, norm2


def topk_grid_init_no_deflation(
    z: np.ndarray,
    theta_flat: np.ndarray,
    r_flat: np.ndarray,
    ah_grid: np.ndarray,
    norm2: np.ndarray,
    *,
    num_targets: int,
    min_sep_theta_deg: float,
    min_sep_r_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = int(z.shape[0])
    theta0 = np.zeros((n, num_targets), dtype=np.float32)
    r0 = np.zeros((n, num_targets), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n):
        inner = ah_grid @ z[i]  # (G,)
        scores = (np.abs(inner) ** 2) / norm2
        order = np.argsort(scores)[::-1]

        picks: list[int] = []
        pick_set: set[int] = set()
        for idx in order:
            th = float(theta_flat[idx])
            rr = float(r_flat[idx])
            ok = True
            for j in picks:
                dth = abs(
                    float(
                        angle_error_deg_np(
                            np.array([th], dtype=np.float32),
                            np.array([float(theta_flat[j])], dtype=np.float32),
                        )[0]
                    )
                )
                dr = abs(rr - float(r_flat[j]))
                if dth < float(min_sep_theta_deg) and dr < float(min_sep_r_m):
                    ok = False
                    break
            if ok:
                picks.append(int(idx))
                pick_set.add(int(idx))
            if len(picks) >= num_targets:
                break

        if len(picks) < num_targets:
            for idx in order:
                idx_int = int(idx)
                if idx_int in pick_set:
                    continue
                picks.append(idx_int)
                if len(picks) >= num_targets:
                    break

        picks = picks[:num_targets]
        theta0[i] = theta_flat[picks]
        r0[i] = r_flat[picks]
    init_ms = ms_per_sample(time.perf_counter() - t0, n)
    return theta0, r0, init_ms


def _sample_targets_with_separation(
    rng: np.random.Generator,
    n: int,
    num_targets: int,
    box: TargetBox,
    min_sep_theta_deg: float,
    min_sep_r_m: float,
    max_trials: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.zeros((n, num_targets), dtype=np.float32)
    r = np.zeros((n, num_targets), dtype=np.float32)
    for i in range(n):
        accepted: list[tuple[float, float]] = []
        trials = 0
        while len(accepted) < num_targets and trials < max_trials:
            th = float(rng.uniform(box.theta_min, box.theta_max))
            rr = float(rng.uniform(box.r_min, box.r_max))
            ok = True
            for th0, rr0 in accepted:
                dth = abs(float(angle_error_deg_np(np.array([th], dtype=np.float32), np.array([th0], dtype=np.float32))[0]))
                dr = abs(rr - rr0)
                if dth < float(min_sep_theta_deg) and dr < float(min_sep_r_m):
                    ok = False
                    break
            if ok:
                accepted.append((th, rr))
            trials += 1

        while len(accepted) < num_targets:
            accepted.append(
                (
                    float(rng.uniform(box.theta_min, box.theta_max)),
                    float(rng.uniform(box.r_min, box.r_max)),
                )
            )

        accepted = sorted(accepted, key=lambda x: x[0])
        theta[i] = np.asarray([x[0] for x in accepted], dtype=np.float32)
        r[i] = np.asarray([x[1] for x in accepted], dtype=np.float32)
    return theta, r


def synthesize_multi_np(
    theta_gt: np.ndarray,
    r_gt: np.ndarray,
    snr_db_vec: np.ndarray,
    cfg: FDAConfig,
    *,
    seed: int,
    sign0: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    bsz, num_targets = theta_gt.shape
    y_clean = np.zeros((bsz, cfg.K), dtype=np.complex64)
    for k in range(num_targets):
        a_k = steering_vector_np(theta_gt[:, k], r_gt[:, k], cfg).astype(np.complex64)
        y_clean += a_k
    y_clean = y_clean / np.sqrt(float(max(num_targets, 1)))

    snr_lin = np.maximum(10.0 ** (snr_db_vec.astype(np.float32) / 10.0), 1e-12).astype(np.float32)
    noise_var = 1.0 / snr_lin
    sigma = np.sqrt(noise_var).astype(np.float32)
    w = (sigma[:, None] / np.sqrt(2.0)) * (
        rng.standard_normal(y_clean.shape, dtype=np.float32)
        + 1j * rng.standard_normal(y_clean.shape, dtype=np.float32)
    )
    y = y_clean + w.astype(np.complex64)
    z = quantize_1bit_np(y, sign0=sign0)
    return y.astype(np.complex64), z.astype(np.complex64)


def deflate_residual(
    z_res: np.ndarray,
    theta_hat: np.ndarray,
    r_hat: np.ndarray,
    cfg: FDAConfig,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    a_hat = steering_vector_np(theta_hat.astype(np.float32), r_hat.astype(np.float32), cfg).astype(np.complex64)
    num = np.sum(np.conj(a_hat) * z_res, axis=-1)
    den = np.maximum(np.sum(np.abs(a_hat) ** 2, axis=-1).real.astype(np.float32), eps)
    c_hat = num / den
    return (z_res - c_hat[:, None] * a_hat).astype(np.complex64)


def greedy_grid_only(
    z: np.ndarray,
    cfg: FDAConfig,
    box: TargetBox,
    *,
    num_targets: int,
    theta_step: float,
    r_step: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = z.shape[0]
    theta_hat = np.zeros((n, num_targets), dtype=np.float32)
    r_hat = np.zeros((n, num_targets), dtype=np.float32)
    residual = z.copy()
    with Timer() as timer:
        for t in range(num_targets):
            for i in range(n):
                th0, rr0, _ = coarse_search_np(
                    residual[i],
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    theta_step=theta_step,
                    r_step=r_step,
                )
                theta_hat[i, t] = float(th0)
                r_hat[i, t] = float(rr0)
            residual = deflate_residual(residual, theta_hat[:, t], r_hat[:, t], cfg)
    return theta_hat, r_hat, ms_per_sample(timer.dt, n)


def greedy_grid_nm(
    z: np.ndarray,
    cfg: FDAConfig,
    box: TargetBox,
    *,
    num_targets: int,
    theta_step: float,
    r_step: float,
    nm_maxiter: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n = z.shape[0]
    theta_hat = np.zeros((n, num_targets), dtype=np.float32)
    r_hat = np.zeros((n, num_targets), dtype=np.float32)
    residual = z.copy()

    t0 = time.perf_counter()
    nm_dt = 0.0
    for t in range(num_targets):
        theta0 = np.zeros((n,), dtype=np.float32)
        r0 = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            th0, rr0, _ = coarse_search_np(
                residual[i],
                cfg,
                theta_range=(box.theta_min, box.theta_max),
                r_range=(box.r_min, box.r_max),
                theta_step=theta_step,
                r_step=r_step,
            )
            theta0[i] = float(th0)
            r0[i] = float(rr0)

        t_nm0 = time.perf_counter()
        for i in range(n):
            res = refine_nelder_mead(
                residual[i],
                float(theta0[i]),
                float(r0[i]),
                cfg,
                theta_range=(box.theta_min, box.theta_max),
                r_range=(box.r_min, box.r_max),
                maxiter=nm_maxiter,
            )
            theta_hat[i, t] = float(res.theta_deg)
            r_hat[i, t] = float(res.r_m)
        nm_dt += time.perf_counter() - t_nm0
        residual = deflate_residual(residual, theta_hat[:, t], r_hat[:, t], cfg)

    total_dt = time.perf_counter() - t0
    return theta_hat, r_hat, ms_per_sample(total_dt, n), ms_per_sample(nm_dt, n)


def refine_nm_from_inits(
    z: np.ndarray,
    cfg: FDAConfig,
    box: TargetBox,
    theta0: np.ndarray,
    r0: np.ndarray,
    *,
    nm_maxiter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n, num_targets = theta0.shape
    theta_hat = np.zeros((n, num_targets), dtype=np.float32)
    r_hat = np.zeros((n, num_targets), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n):
        for t in range(num_targets):
            res = refine_nelder_mead(
                z[i],
                float(theta0[i, t]),
                float(r0[i, t]),
                cfg,
                theta_range=(box.theta_min, box.theta_max),
                r_range=(box.r_min, box.r_max),
                maxiter=int(nm_maxiter),
            )
            theta_hat[i, t] = float(res.theta_deg)
            r_hat[i, t] = float(res.r_m)
    return theta_hat, r_hat, ms_per_sample(time.perf_counter() - t0, n)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


def _decode_joint_u(u: np.ndarray, box: TargetBox) -> tuple[np.ndarray, np.ndarray]:
    p = _sigmoid_np(u.astype(np.float64))
    th = np.array(
        [
            box.theta_min + (box.theta_max - box.theta_min) * p[0],
            box.theta_min + (box.theta_max - box.theta_min) * p[2],
        ],
        dtype=np.float32,
    )
    rr = np.array(
        [
            box.r_min + (box.r_max - box.r_min) * p[1],
            box.r_min + (box.r_max - box.r_min) * p[3],
        ],
        dtype=np.float32,
    )
    return th, rr


def _joint_projection_score_2target(
    z_i: np.ndarray,
    theta_pair: np.ndarray,
    r_pair: np.ndarray,
    cfg: FDAConfig,
) -> float:
    a = steering_vector_np(theta_pair.astype(np.float32), r_pair.astype(np.float32), cfg).astype(
        np.complex64
    )  # (2,K)
    a_mat = a.T  # (K,2)
    gram = a_mat.conj().T @ a_mat
    gram = gram + (1e-5 * np.eye(2, dtype=np.complex64))
    rhs = a_mat.conj().T @ z_i
    coeff = np.linalg.solve(gram, rhs)
    y_hat = a_mat @ coeff
    return float(np.real(np.vdot(y_hat, y_hat)))


def run_joint_nm2_from_inits(
    z: np.ndarray,
    cfg: FDAConfig,
    box: TargetBox,
    theta0: np.ndarray,
    r0: np.ndarray,
    *,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = int(theta0.shape[0])
    theta_hat = np.zeros((n, 2), dtype=np.float32)
    r_hat = np.zeros((n, 2), dtype=np.float32)

    t0 = time.perf_counter()
    eps = 1e-12
    for i in range(n):
        p_th = (theta0[i] - float(box.theta_min)) / (float(box.theta_max - box.theta_min) + eps)
        p_r = (r0[i] - float(box.r_min)) / (float(box.r_max - box.r_min) + eps)
        u0 = np.array(
            [
                _logit_np(np.array([p_th[0]], dtype=np.float64))[0],
                _logit_np(np.array([p_r[0]], dtype=np.float64))[0],
                _logit_np(np.array([p_th[1]], dtype=np.float64))[0],
                _logit_np(np.array([p_r[1]], dtype=np.float64))[0],
            ],
            dtype=np.float64,
        )

        def f(u: np.ndarray) -> float:
            th, rr = _decode_joint_u(np.asarray(u, dtype=np.float64), box)
            return -_joint_projection_score_2target(z[i], th, rr, cfg)

        res = minimize(
            f,
            u0,
            method="Nelder-Mead",
            options={"maxiter": int(maxiter), "xatol": 1e-3, "fatol": 1e-4},
        )
        th_est, rr_est = _decode_joint_u(np.asarray(res.x, dtype=np.float64), box)
        order = np.argsort(th_est)
        theta_hat[i] = th_est[order]
        r_hat[i] = rr_est[order]
    return theta_hat, r_hat, ms_per_sample(time.perf_counter() - t0, n)


def greedy_unrolled(
    z: np.ndarray,
    cfg: FDAConfig,
    box: TargetBox,
    *,
    device: torch.device,
    num_targets: int,
    theta_step: float,
    r_step: float,
    T_run: int,
    ckpt_path: str,
    sanitize_ckpt: bool,
    learnable: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float, int | None, int | None]:
    n = z.shape[0]
    theta_hat = np.zeros((n, num_targets), dtype=np.float32)
    r_hat = np.zeros((n, num_targets), dtype=np.float32)
    residual = z.copy()
    ms_total = 0.0
    t_model_note = None
    t_run_note = None

    for t in range(num_targets):
        if learnable:
            th, rr, ms, t_model, t_eff = run_unrolled(
                residual,
                cfg,
                box,
                device=device,
                theta_step=theta_step,
                r_step=r_step,
                T=T_run,
                ckpt_path=ckpt_path,
                sanitize_ckpt=bool(sanitize_ckpt),
                learnable=True,
                T_run=T_run,
                return_meta=True,
                batch_size=batch_size,
            )
            t_model_note = int(t_model)
            t_run_note = int(t_eff)
        else:
            th, rr, ms = run_unrolled(
                residual,
                cfg,
                box,
                device=device,
                theta_step=theta_step,
                r_step=r_step,
                T=T_run,
                ckpt_path="",
                sanitize_ckpt=False,
                learnable=False,
                T_run=T_run,
                batch_size=batch_size,
            )
        theta_hat[:, t] = th.astype(np.float32)
        r_hat[:, t] = rr.astype(np.float32)
        ms_total += float(ms)
        residual = deflate_residual(residual, theta_hat[:, t], r_hat[:, t], cfg)

    return theta_hat, r_hat, float(ms_total), t_model_note, t_run_note


def greedy_learned_nm_tail(
    z: np.ndarray,
    cfg: FDAConfig,
    box: TargetBox,
    *,
    device: torch.device,
    num_targets: int,
    theta_step: float,
    r_step: float,
    T_run: int,
    ckpt_path: str,
    sanitize_ckpt: bool,
    batch_size: int,
    tail_iters: int,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    n = z.shape[0]
    theta_hat = np.zeros((n, num_targets), dtype=np.float32)
    r_hat = np.zeros((n, num_targets), dtype=np.float32)
    residual = z.copy()

    ms_learned_total = 0.0
    ms_tail_total = 0.0
    t_model_note = -1
    t_run_note = -1

    for t in range(num_targets):
        th, rr, ms, t_model, t_eff = run_unrolled(
            residual,
            cfg,
            box,
            device=device,
            theta_step=theta_step,
            r_step=r_step,
            T=T_run,
            ckpt_path=ckpt_path,
            sanitize_ckpt=bool(sanitize_ckpt),
            learnable=True,
            T_run=T_run,
            return_meta=True,
            batch_size=batch_size,
        )
        ms_learned_total += float(ms)
        t_model_note = int(t_model)
        t_run_note = int(t_eff)

        t_nm0 = time.perf_counter()
        th_tail = np.zeros_like(th, dtype=np.float32)
        rr_tail = np.zeros_like(rr, dtype=np.float32)
        for i in range(n):
            res = refine_nelder_mead(
                residual[i],
                float(th[i]),
                float(rr[i]),
                cfg,
                theta_range=(box.theta_min, box.theta_max),
                r_range=(box.r_min, box.r_max),
                maxiter=int(tail_iters),
            )
            th_tail[i] = float(res.theta_deg)
            rr_tail[i] = float(res.r_m)
        ms_tail_total += ms_per_sample(time.perf_counter() - t_nm0, n)

        theta_hat[:, t] = th_tail
        r_hat[:, t] = rr_tail
        residual = deflate_residual(residual, theta_hat[:, t], r_hat[:, t], cfg)

    return (
        theta_hat,
        r_hat,
        float(ms_learned_total + ms_tail_total),
        float(ms_tail_total),
        int(t_model_note),
        int(t_run_note),
    )


def matched_rmse(
    theta_hat: np.ndarray,
    r_hat: np.ndarray,
    theta_gt: np.ndarray,
    r_gt: np.ndarray,
    *,
    theta_scale_deg: float,
    r_scale_m: float,
) -> tuple[float, float]:
    n, num_targets = theta_gt.shape
    err_theta_all: list[np.ndarray] = []
    err_r_all: list[np.ndarray] = []
    for i in range(n):
        c = np.zeros((num_targets, num_targets), dtype=np.float64)
        for p in range(num_targets):
            for q in range(num_targets):
                dth = float(
                    angle_error_deg_np(
                        np.array([theta_hat[i, p]], dtype=np.float32),
                        np.array([theta_gt[i, q]], dtype=np.float32),
                    )[0]
                )
                dr = float(r_hat[i, p] - r_gt[i, q])
                c[p, q] = (dth / float(theta_scale_deg)) ** 2 + (dr / float(r_scale_m)) ** 2
        ridx, cidx = linear_sum_assignment(c)
        dth_vec = angle_error_deg_np(theta_hat[i, ridx], theta_gt[i, cidx])
        dr_vec = r_hat[i, ridx] - r_gt[i, cidx]
        err_theta_all.append(dth_vec.astype(np.float32))
        err_r_all.append(dr_vec.astype(np.float32))
    err_theta = np.concatenate(err_theta_all, axis=0)
    err_r = np.concatenate(err_r_all, axis=0)
    return rmse_np(err_theta), rmse_np(err_r)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.full:
        args.num_samples = max(int(args.num_samples), 1000)
        args.nm_maxiter = max(int(args.nm_maxiter), 120)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = FDAConfig()
    box = TargetBox()
    t_run = int(args.T)
    num_targets = int(args.num_targets)
    if num_targets <= 0:
        raise ValueError("--num_targets must be >= 1")

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"eval_multi_{timestamp()}"
    ensure_dir(run_dir)
    write_json(run_dir / "config.json", {**vars(args), "device_used": str(device)})

    csv = CsvLogger(
        run_dir / "results.csv",
        fieldnames=[
            "snr_db",
            "method",
            "rmse_theta_deg",
            "rmse_r_m",
            "ms_per_sample",
            "notes",
        ],
    )

    snr_list = [float(x) for x in args.snr_list.split(",") if x.strip()]
    rng = np.random.default_rng(int(args.seed))
    need_topk_init = bool(int(args.run_topk_nm_baseline)) or (
        int(args.joint_nm2_num_samples) > 0 and num_targets == 2
    )
    topk_cache = None
    if need_topk_init:
        topk_cache = build_grid_cache(
            cfg,
            box,
            theta_step=float(args.theta_step),
            r_step=float(args.r_step),
        )

    for snr_idx, snr_db in enumerate(snr_list):
        theta_gt, r_gt = _sample_targets_with_separation(
            rng,
            int(args.num_samples),
            num_targets,
            box,
            float(args.min_sep_theta_deg),
            float(args.min_sep_r_m),
        )
        snr_vec = np.full((int(args.num_samples),), float(snr_db), dtype=np.float32)
        _, z = synthesize_multi_np(theta_gt, r_gt, snr_vec, cfg, seed=args.seed + 12345 + snr_idx)

        # 1) Greedy grid-only baseline.
        th_grid, rr_grid, ms_grid = greedy_grid_only(
            z,
            cfg,
            box,
            num_targets=num_targets,
            theta_step=float(args.theta_step),
            r_step=float(args.r_step),
        )
        rmse_th, rmse_r = matched_rmse(
            th_grid,
            rr_grid,
            theta_gt,
            r_gt,
            theta_scale_deg=float(args.match_theta_scale_deg),
            r_scale_m=float(args.match_r_scale_m),
        )
        csv.log(
            {
                "snr_db": snr_db,
                "method": "grid_only_cpu_mt",
                "rmse_theta_deg": rmse_th,
                "rmse_r_m": rmse_r,
                "ms_per_sample": ms_grid,
                "notes": f"num_targets={num_targets}",
            }
        )

        # 2) Greedy grid + NM.
        th_nm, rr_nm, ms_nm, nm_only_ms = greedy_grid_nm(
            z,
            cfg,
            box,
            num_targets=num_targets,
            theta_step=float(args.theta_step),
            r_step=float(args.r_step),
            nm_maxiter=int(args.nm_maxiter),
        )
        rmse_th, rmse_r = matched_rmse(
            th_nm,
            rr_nm,
            theta_gt,
            r_gt,
            theta_scale_deg=float(args.match_theta_scale_deg),
            r_scale_m=float(args.match_r_scale_m),
        )
        csv.log(
            {
                "snr_db": snr_db,
                "method": "grid_nm_cpu_mt",
                "rmse_theta_deg": rmse_th,
                "rmse_r_m": rmse_r,
                "ms_per_sample": ms_nm,
                "notes": f"num_targets={num_targets}; nm_only_ms={nm_only_ms:.3f}",
            }
        )

        # 2.5) Top-K init (no deflation) + per-target NM refine.
        topk_theta0 = None
        topk_r0 = None
        topk_init_ms = None
        if topk_cache is not None:
            th_flat, rr_flat, ah_grid, norm2 = topk_cache
            topk_theta0, topk_r0, topk_init_ms = topk_grid_init_no_deflation(
                z,
                th_flat,
                rr_flat,
                ah_grid,
                norm2,
                num_targets=num_targets,
                min_sep_theta_deg=float(args.topk_init_min_sep_theta_deg),
                min_sep_r_m=float(args.topk_init_min_sep_r_m),
            )

        if bool(int(args.run_topk_nm_baseline)):
            if topk_theta0 is None or topk_r0 is None or topk_init_ms is None:
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": "grid_topk_nm_cpu_mt",
                        "rmse_theta_deg": "",
                        "rmse_r_m": "",
                        "ms_per_sample": "",
                        "notes": "skip (topk init unavailable)",
                    }
                )
            else:
                th_topk_nm, rr_topk_nm, nm_refine_ms = refine_nm_from_inits(
                    z,
                    cfg,
                    box,
                    topk_theta0,
                    topk_r0,
                    nm_maxiter=int(args.nm_maxiter),
                )
                rmse_th, rmse_r = matched_rmse(
                    th_topk_nm,
                    rr_topk_nm,
                    theta_gt,
                    r_gt,
                    theta_scale_deg=float(args.match_theta_scale_deg),
                    r_scale_m=float(args.match_r_scale_m),
                )
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": "grid_topk_nm_cpu_mt",
                        "rmse_theta_deg": rmse_th,
                        "rmse_r_m": rmse_r,
                        "ms_per_sample": float(topk_init_ms) + float(nm_refine_ms),
                        "notes": (
                            f"num_targets={num_targets}; no_deflation=1; "
                            f"init_ms={float(topk_init_ms):.3f}; nm_only_ms={float(nm_refine_ms):.3f}"
                        ),
                    }
                )

        # 2.6) Joint-NM for K=2 (small-sample probe).
        if num_targets != 2:
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "joint_nm2_topk_cpu_mt",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": f"skip (num_targets={num_targets} != 2)",
                }
            )
        elif int(args.joint_nm2_num_samples) <= 0:
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "joint_nm2_topk_cpu_mt",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": "skip (joint_nm2_num_samples<=0)",
                }
            )
        elif topk_theta0 is None or topk_r0 is None:
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "joint_nm2_topk_cpu_mt",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": "skip (topk init unavailable)",
                }
            )
        else:
            n_eval = min(int(args.num_samples), int(args.joint_nm2_num_samples))
            th_joint, rr_joint, ms_joint = run_joint_nm2_from_inits(
                z[:n_eval],
                cfg,
                box,
                topk_theta0[:n_eval, :2],
                topk_r0[:n_eval, :2],
                maxiter=int(args.joint_nm2_maxiter),
            )
            rmse_th, rmse_r = matched_rmse(
                th_joint,
                rr_joint,
                theta_gt[:n_eval],
                r_gt[:n_eval],
                theta_scale_deg=float(args.match_theta_scale_deg),
                r_scale_m=float(args.match_r_scale_m),
            )
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "joint_nm2_topk_cpu_mt",
                    "rmse_theta_deg": rmse_th,
                    "rmse_r_m": rmse_r,
                    "ms_per_sample": ms_joint,
                    "notes": (
                        f"num_targets=2; n_eval={n_eval}; no_deflation=1; "
                        f"maxiter={int(args.joint_nm2_maxiter)}"
                    ),
                }
            )

        # 3) Greedy fixed unroll.
        if t_run <= 0:
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "grid_unroll_fixed_mt",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": f"skip: T_run={t_run} (no unroll steps)",
                }
            )
        else:
            th_fix, rr_fix, ms_fix, _, _ = greedy_unrolled(
                z,
                cfg,
                box,
                device=device,
                num_targets=num_targets,
                theta_step=float(args.theta_step),
                r_step=float(args.r_step),
                T_run=t_run,
                ckpt_path="",
                sanitize_ckpt=False,
                learnable=False,
                batch_size=int(args.batch_size),
            )
            rmse_th, rmse_r = matched_rmse(
                th_fix,
                rr_fix,
                theta_gt,
                r_gt,
                theta_scale_deg=float(args.match_theta_scale_deg),
                r_scale_m=float(args.match_r_scale_m),
            )
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "grid_unroll_fixed_mt",
                    "rmse_theta_deg": rmse_th,
                    "rmse_r_m": rmse_r,
                    "ms_per_sample": ms_fix,
                    "notes": f"num_targets={num_targets}; device={device.type}; T_run={t_run}",
                }
            )

        # 4) Greedy learned unroll.
        learned_ok = False
        learned_skip_note = ""
        t_model_note = None
        t_eff_note = None

        if not args.ckpt_path:
            learned_skip_note = "skip (no --ckpt_path)"
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "grid_unroll_learned_mt",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": learned_skip_note,
                }
            )
        elif t_run <= 0:
            learned_skip_note = f"skip: T_run={t_run} (no unroll steps)"
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "grid_unroll_learned_mt",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": learned_skip_note,
                }
            )
        else:
            try:
                th_ld, rr_ld, ms_ld, t_model_note, t_eff_note = greedy_unrolled(
                    z,
                    cfg,
                    box,
                    device=device,
                    num_targets=num_targets,
                    theta_step=float(args.theta_step),
                    r_step=float(args.r_step),
                    T_run=t_run,
                    ckpt_path=str(args.ckpt_path),
                    sanitize_ckpt=bool(args.sanitize_ckpt),
                    learnable=True,
                    batch_size=int(args.batch_size),
                )
                rmse_th, rmse_r = matched_rmse(
                    th_ld,
                    rr_ld,
                    theta_gt,
                    r_gt,
                    theta_scale_deg=float(args.match_theta_scale_deg),
                    r_scale_m=float(args.match_r_scale_m),
                )
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": "grid_unroll_learned_mt",
                        "rmse_theta_deg": rmse_th,
                        "rmse_r_m": rmse_r,
                        "ms_per_sample": ms_ld,
                        "notes": (
                            f"num_targets={num_targets}; ckpt={args.ckpt_path}; "
                            f"T_model={t_model_note}, T_run={t_eff_note}"
                        ),
                    }
                )
                learned_ok = True
            except RuntimeError as e:
                learned_skip_note = f"skip ({str(e)})"
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": "grid_unroll_learned_mt",
                        "rmse_theta_deg": "",
                        "rmse_r_m": "",
                        "ms_per_sample": "",
                        "notes": learned_skip_note,
                    }
                )

        # 5) Optional learned + tiny NM tail.
        tail_iters = int(args.learned_nm_tail_iters)
        if tail_iters > 0:
            method_tail = f"grid_unroll_learned_nm{tail_iters}_mt"
            if not args.ckpt_path:
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": method_tail,
                        "rmse_theta_deg": "",
                        "rmse_r_m": "",
                        "ms_per_sample": "",
                        "notes": "skip (no --ckpt_path)",
                    }
                )
            elif snr_db < float(args.learned_nm_tail_snr_min):
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": method_tail,
                        "rmse_theta_deg": "",
                        "rmse_r_m": "",
                        "ms_per_sample": "",
                        "notes": (
                            f"skip (snr_db={snr_db} < learned_nm_tail_snr_min="
                            f"{float(args.learned_nm_tail_snr_min)})"
                        ),
                    }
                )
            elif not learned_ok:
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": method_tail,
                        "rmse_theta_deg": "",
                        "rmse_r_m": "",
                        "ms_per_sample": "",
                        "notes": f"skip (learned failed: {learned_skip_note})",
                    }
                )
            else:
                th_tail, rr_tail, ms_tail, tail_only_ms, t_model_k, t_run_k = greedy_learned_nm_tail(
                    z,
                    cfg,
                    box,
                    device=device,
                    num_targets=num_targets,
                    theta_step=float(args.theta_step),
                    r_step=float(args.r_step),
                    T_run=t_run,
                    ckpt_path=str(args.ckpt_path),
                    sanitize_ckpt=bool(args.sanitize_ckpt),
                    batch_size=int(args.batch_size),
                    tail_iters=tail_iters,
                )
                rmse_th, rmse_r = matched_rmse(
                    th_tail,
                    rr_tail,
                    theta_gt,
                    r_gt,
                    theta_scale_deg=float(args.match_theta_scale_deg),
                    r_scale_m=float(args.match_r_scale_m),
                )
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": method_tail,
                        "rmse_theta_deg": rmse_th,
                        "rmse_r_m": rmse_r,
                        "ms_per_sample": ms_tail,
                        "notes": (
                            f"num_targets={num_targets}; ckpt={args.ckpt_path}; "
                            f"T_model={t_model_k}, T_run={t_run_k}; "
                            f"nm_tail_iters={tail_iters}; nm_tail_only_ms={tail_only_ms:.3f}"
                        ),
                    }
                )

        print(
            f"SNR={snr_db} done | "
            f"learned={'ok' if learned_ok else 'skip'} | "
            f"ckpt_T={t_model_note if t_model_note is not None else '-'}"
        )

    print(f"Wrote: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
