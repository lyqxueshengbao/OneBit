from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.coarse_search import coarse_search_np
from src.dataset import TargetBox, synthesize_np
from src.fda import FDAConfig
from src.metrics import angle_error_deg_np, rmse_np
from src.nm_refine import refine_nelder_mead
from src.qgamp_warmstart import QGAMPWarmStarter
from src.unroll_refine import Box, Refiner
from src.utils import CsvLogger, Timer, ensure_dir, seed_all, timestamp, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snr_list", type=str, default="-15,-10,-5,0,5,10,15,20")
    p.add_argument("--num_samples", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--theta_step", type=float, default=1.0)
    p.add_argument("--r_step", type=float, default=100.0)
    p.add_argument("--nm_maxiter", type=int, default=60)
    p.add_argument("--T", type=int, default=1)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--learned_nm_tail_iters", type=int, default=0)
    p.add_argument("--learned_nm_tail_snr_min", type=float, default=0.0)
    p.add_argument("--full", action="store_true")
    # Q-GAMP warm-start options.
    p.add_argument("--qgamp_iters", type=int, default=8)
    p.add_argument("--qgamp_beta", type=float, default=2.0)
    p.add_argument("--qgamp_l1", type=float, default=3e-3)
    p.add_argument("--qgamp_l2", type=float, default=1e-4)
    p.add_argument("--qgamp_topcand", type=int, default=8)
    p.add_argument("--qgamp_nonneg", type=int, default=1, choices=[0, 1])
    p.add_argument("--qgamp_step_scale", type=float, default=1.0)
    p.add_argument("--run_dir", type=str, default="")
    return p.parse_args()


def ms_per_sample(dt_s: float, n: int) -> float:
    return 1000.0 * dt_s / max(int(n), 1)


def _infer_t_model(sd: dict, ckpt_args: dict, t_fallback: int) -> int:
    t_arg = ckpt_args.get("T", None)
    if t_arg is not None:
        return int(t_arg)
    alpha = sd.get("alpha_theta_raw", None)
    if torch.is_tensor(alpha) and alpha.ndim >= 1:
        return int(alpha.shape[0])
    return int(t_fallback)


def _slice_state_dict_for_t(sd: dict, t_run: int, t_model: int) -> dict:
    stepwise_names = {
        "alpha_theta_raw",
        "alpha_r_raw",
        "lambda_theta_raw",
        "lambda_r_raw",
        "t_log_scale_table",
        "phys_precond_raw",
    }
    out = {}
    for name, value in sd.items():
        if not torch.is_tensor(value):
            out[name] = value
            continue
        should_slice = name in stepwise_names or (
            value.ndim >= 1
            and int(value.shape[0]) == int(t_model)
            and any(tok in name for tok in ("alpha", "lambda", "step", "t_log_scale", "precond"))
        )
        if should_slice:
            out[name] = value[: int(t_run)].clone()
        else:
            out[name] = value
    return out


def _build_learned_refiner(
    cfg: FDAConfig,
    box: TargetBox,
    device: torch.device,
    t_req: int,
    ckpt_path: str,
) -> tuple[Refiner, int, int]:
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"]
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    t_model = _infer_t_model(sd, ckpt_args, t_fallback=t_req)
    if int(t_req) > int(t_model):
        raise RuntimeError(
            f"T_run={int(t_req)} > T_model={int(t_model)} in ckpt; please eval with --T <= T_model"
        )
    t_build = max(0, int(t_req))
    if t_build <= 0:
        raise RuntimeError(f"T_run={int(t_build)} (no unroll steps)")

    refiner = Refiner(
        cfg,
        T=int(t_build),
        box=Box(box.theta_min, box.theta_max, box.r_min, box.r_max),
        learnable=True,
        r_precond_mul=float(ckpt_args.get("r_precond_mul", 1.0)),
        r_precond_pow=float(ckpt_args.get("r_precond_pow", 1.0)),
        r_precond_learnable=bool("r_precond_mul" in sd),
        use_pscale=bool(int(ckpt_args.get("use_pscale", 0))) or any(
            str(k).startswith("pscale_mlp.") for k in sd.keys()
        ),
        pscale_hidden=int(ckpt_args.get("pscale_hidden", 32)),
        pscale_detach_step=bool(int(ckpt_args.get("pscale_detach_step", 1))),
        pscale_logrange=float(ckpt_args.get("pscale_logrange", 6.9)),
        pscale_amp=float(ckpt_args.get("pscale_amp", 1.0)),
        pscale_input=str(ckpt_args.get("pscale_input", "step,u,t")),
        use_t_table=bool(int(ckpt_args.get("use_t_table", 0))) or ("t_log_scale_table" in sd),
        t_table_init=float(ckpt_args.get("t_table_init", 0.0)),
        pscale_min_theta=float(ckpt_args.get("pscale_min_theta", 0.7)),
        pscale_max_theta=float(ckpt_args.get("pscale_max_theta", 1.3)),
        pscale_min_r=float(ckpt_args.get("pscale_min_r", 0.7)),
        pscale_max_r=float(ckpt_args.get("pscale_max_r", 1.3)),
        use_step_attn=bool(int(ckpt_args.get("use_step_attn", 0))) or any(
            str(k).startswith("step_attn_mlp.") for k in sd.keys()
        ),
        step_attn_hidden=int(ckpt_args.get("step_attn_hidden", 32)),
        step_attn_detach_feat=bool(int(ckpt_args.get("step_attn_detach_feat", 1))),
        step_attn_amp=float(ckpt_args.get("step_attn_amp", 0.3)),
        step_attn_min_theta=float(ckpt_args.get("step_attn_min_theta", 0.8)),
        step_attn_max_theta=float(ckpt_args.get("step_attn_max_theta", 1.2)),
        step_attn_min_r=float(ckpt_args.get("step_attn_min_r", 0.8)),
        step_attn_max_r=float(ckpt_args.get("step_attn_max_r", 1.2)),
        use_phys_precond2=bool(int(ckpt_args.get("use_phys_precond2", 0))) or ("phys_precond_raw" in sd),
        phys_precond_diag_logmax=float(ckpt_args.get("phys_precond_diag_logmax", 0.35)),
        phys_precond_offdiag_max=float(ckpt_args.get("phys_precond_offdiag_max", 0.35)),
        accept_reject=bool(int(ckpt_args.get("accept_reject", 0))),
        ar_backtrack_max=int(ckpt_args.get("ar_backtrack_max", 0)),
        ar_backtrack_factor=float(ckpt_args.get("ar_backtrack_factor", 0.5)),
        ar_min_scale=float(ckpt_args.get("ar_min_scale", 0.1)),
        ar_accept_tol=float(ckpt_args.get("ar_accept_tol", 0.0)),
    ).to(device)
    refiner.load_state_dict(_slice_state_dict_for_t(sd, t_build, t_model), strict=True)
    refiner.eval()
    return refiner, int(t_model), int(t_build)


def _run_refiner(
    refiner: Refiner,
    z_np: np.ndarray,
    theta0: np.ndarray,
    r0: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    t_run: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    z = torch.from_numpy(z_np).to(device=device)
    if z.dtype != torch.complex64:
        z = z.to(torch.complex64)
    th0 = torch.from_numpy(theta0.astype(np.float32)).to(device=device)
    rr0 = torch.from_numpy(r0.astype(np.float32)).to(device=device)

    out_th: list[np.ndarray] = []
    out_rr: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, z.shape[0], int(batch_size)):
        zb = z[i : i + int(batch_size)]
        tb = th0[i : i + int(batch_size)]
        rb = rr0[i : i + int(batch_size)]
        with torch.enable_grad():
            th, rr, _ = refiner(zb, tb, rb, T_run=int(t_run))
        out_th.append(th.detach().cpu().numpy())
        out_rr.append(rr.detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return np.concatenate(out_th), np.concatenate(out_rr), ms_per_sample(dt, int(z.shape[0]))


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.full:
        args.num_samples = max(int(args.num_samples), 4096)
        args.nm_maxiter = max(int(args.nm_maxiter), 120)
        args.batch_size = max(int(args.batch_size), 512)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = FDAConfig()
    box = TargetBox()
    t_run = int(args.T)

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"eval_qgamp_{timestamp()}"
    ensure_dir(run_dir)
    write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "device_used": str(device),
            "cfg": cfg.__dict__,
        },
    )
    logger = CsvLogger(
        run_dir / "results.csv",
        fieldnames=["snr_db", "method", "rmse_theta_deg", "rmse_r_m", "ms_per_sample", "notes"],
    )

    warm = QGAMPWarmStarter(
        cfg,
        theta_range=(box.theta_min, box.theta_max),
        r_range=(box.r_min, box.r_max),
        theta_step=float(args.theta_step),
        r_step=float(args.r_step),
        beta=float(args.qgamp_beta),
        l1=float(args.qgamp_l1),
        l2=float(args.qgamp_l2),
        iters=int(args.qgamp_iters),
        topcand=int(args.qgamp_topcand),
        nonneg=bool(int(args.qgamp_nonneg)),
        step_scale=float(args.qgamp_step_scale),
    )

    learned_refiner = None
    t_model = None
    t_eff = None
    if args.ckpt_path and t_run > 0:
        learned_refiner, t_model, t_eff = _build_learned_refiner(
            cfg,
            box,
            device,
            t_req=t_run,
            ckpt_path=str(args.ckpt_path),
        )

    snr_list = [float(s.strip()) for s in str(args.snr_list).split(",") if s.strip()]
    for snr in snr_list:
        with Timer() as tm_syn:
            theta_gt = np.random.uniform(box.theta_min, box.theta_max, size=(int(args.num_samples),)).astype(
                np.float32
            )
            r_gt = np.random.uniform(box.r_min, box.r_max, size=(int(args.num_samples),)).astype(np.float32)
            snr_arr = np.full_like(theta_gt, float(snr), dtype=np.float32)
            _, z, _ = synthesize_np(theta_gt, r_gt, snr_arr, cfg, seed=int(args.seed + int(round(snr * 10))))
        _ = tm_syn

        # 1) Grid only.
        with Timer() as tm_grid:
            th_g = np.zeros_like(theta_gt)
            rr_g = np.zeros_like(r_gt)
            for i in range(theta_gt.shape[0]):
                th0, rr0, _ = coarse_search_np(
                    z[i],
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    theta_step=float(args.theta_step),
                    r_step=float(args.r_step),
                )
                th_g[i] = np.float32(th0)
                rr_g[i] = np.float32(rr0)
        logger.log(
            {
                "snr_db": snr,
                "method": "grid_only_cpu",
                "rmse_theta_deg": rmse_np(angle_error_deg_np(th_g, theta_gt)),
                "rmse_r_m": rmse_np(rr_g - r_gt),
                "ms_per_sample": ms_per_sample(tm_grid.dt, int(args.num_samples)),
                "notes": "",
            }
        )

        # 2) Grid + NM.
        with Timer() as tm_nm:
            th_nm = np.zeros_like(theta_gt)
            rr_nm = np.zeros_like(r_gt)
            for i in range(theta_gt.shape[0]):
                res = refine_nelder_mead(
                    z[i],
                    float(th_g[i]),
                    float(rr_g[i]),
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    maxiter=int(args.nm_maxiter),
                )
                th_nm[i] = np.float32(res.theta_deg)
                rr_nm[i] = np.float32(res.r_m)
        ms_nm_total = ms_per_sample(tm_nm.dt, int(args.num_samples))
        ms_nm_only = max(0.0, ms_nm_total - ms_per_sample(tm_grid.dt, int(args.num_samples)))
        logger.log(
            {
                "snr_db": snr,
                "method": "grid_nm_cpu",
                "rmse_theta_deg": rmse_np(angle_error_deg_np(th_nm, theta_gt)),
                "rmse_r_m": rmse_np(rr_nm - r_gt),
                "ms_per_sample": ms_nm_total,
                "notes": f"nm_only_ms={ms_nm_only:.3f}",
            }
        )

        # 3) Q-GAMP warm-start.
        with Timer() as tm_qg:
            th_qg, rr_qg, _ = warm.search_batch(z)
        logger.log(
            {
                "snr_db": snr,
                "method": "qgamp_only_cpu",
                "rmse_theta_deg": rmse_np(angle_error_deg_np(th_qg, theta_gt)),
                "rmse_r_m": rmse_np(rr_qg - r_gt),
                "ms_per_sample": ms_per_sample(tm_qg.dt, int(args.num_samples)),
                "notes": (
                    f"iters={int(args.qgamp_iters)}; beta={float(args.qgamp_beta):.3g}; "
                    f"l1={float(args.qgamp_l1):.3g}; l2={float(args.qgamp_l2):.3g}; topcand={int(args.qgamp_topcand)}"
                ),
            }
        )

        # 4) Q-GAMP + fixed unroll.
        if t_run <= 0:
            logger.log(
                {
                    "snr_db": snr,
                    "method": "qgamp_unroll_fixed",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": f"skip: T_run={int(t_run)} (no unroll steps)",
                }
            )
        else:
            fixed_refiner = Refiner(
                cfg,
                T=int(t_run),
                box=Box(box.theta_min, box.theta_max, box.r_min, box.r_max),
                learnable=False,
            ).to(device)
            fixed_refiner.eval()
            th_qf, rr_qf, ms_qf = _run_refiner(
                fixed_refiner,
                z,
                th_qg,
                rr_qg,
                device=device,
                batch_size=int(args.batch_size),
                t_run=int(t_run),
            )
            logger.log(
                {
                    "snr_db": snr,
                    "method": "qgamp_unroll_fixed",
                    "rmse_theta_deg": rmse_np(angle_error_deg_np(th_qf, theta_gt)),
                    "rmse_r_m": rmse_np(rr_qf - r_gt),
                    "ms_per_sample": ms_qf,
                    "notes": f"device={device}; T_run={int(t_run)}",
                }
            )

        # 5) Q-GAMP + learned unroll.
        learned_ok = learned_refiner is not None and t_run > 0
        if not learned_ok:
            note = "skip (no --ckpt_path)" if not args.ckpt_path else f"skip: T_run={int(t_run)} (no unroll steps)"
            logger.log(
                {
                    "snr_db": snr,
                    "method": "qgamp_unroll_learned",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": note,
                }
            )
            print(f"SNR={snr:.1f} done")
            continue

        th_ql, rr_ql, ms_ql = _run_refiner(
            learned_refiner,
            z,
            th_qg,
            rr_qg,
            device=device,
            batch_size=int(args.batch_size),
            t_run=int(t_eff),
        )
        logger.log(
            {
                "snr_db": snr,
                "method": "qgamp_unroll_learned",
                "rmse_theta_deg": rmse_np(angle_error_deg_np(th_ql, theta_gt)),
                "rmse_r_m": rmse_np(rr_ql - r_gt),
                "ms_per_sample": ms_ql,
                "notes": f"ckpt={args.ckpt_path}; T_model={int(t_model)}, T_run={int(t_eff)}",
            }
        )

        # 6) Optional small NM tail after learned output.
        tail_k = int(args.learned_nm_tail_iters)
        if tail_k <= 0:
            print(f"SNR={snr:.1f} done")
            continue
        method_tail = f"qgamp_unroll_learned_nm{tail_k}"
        if float(snr) < float(args.learned_nm_tail_snr_min):
            logger.log(
                {
                    "snr_db": snr,
                    "method": method_tail,
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": (
                        f"skip (snr_db={float(snr):.1f} < "
                        f"learned_nm_tail_snr_min={float(args.learned_nm_tail_snr_min):.1f})"
                    ),
                }
            )
            print(f"SNR={snr:.1f} done")
            continue

        with Timer() as tm_tail:
            th_tail = np.zeros_like(theta_gt)
            rr_tail = np.zeros_like(r_gt)
            for i in range(theta_gt.shape[0]):
                res = refine_nelder_mead(
                    z[i],
                    float(th_ql[i]),
                    float(rr_ql[i]),
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    maxiter=int(tail_k),
                )
                th_tail[i] = np.float32(res.theta_deg)
                rr_tail[i] = np.float32(res.r_m)
        ms_tail_only = ms_per_sample(tm_tail.dt, int(args.num_samples))
        logger.log(
            {
                "snr_db": snr,
                "method": method_tail,
                "rmse_theta_deg": rmse_np(angle_error_deg_np(th_tail, theta_gt)),
                "rmse_r_m": rmse_np(rr_tail - r_gt),
                "ms_per_sample": ms_ql + ms_tail_only,
                "notes": (
                    f"ckpt={args.ckpt_path}; T_model={int(t_model)}, T_run={int(t_eff)}; "
                    f"nm_tail_iters={int(tail_k)}; nm_tail_only_ms={ms_tail_only:.3f}"
                ),
            }
        )
        print(f"SNR={snr:.1f} done")

    print(f"Wrote: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()

