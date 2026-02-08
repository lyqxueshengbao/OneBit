from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.coarse_search import CoarseSearcherTorch, coarse_search_np
from src.dataset import TargetBox, synthesize_np
from src.fda import FDAConfig
from src.metrics import angle_error_deg_np, rmse_np
from src.nm_refine import refine_nelder_mead
from src.unroll_refine import Box, Refiner
from src.utils import CsvLogger, Timer, ensure_dir, seed_all, timestamp, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snr_list", type=str, default="-15,-10,-5,0,5,10,15,20")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--theta_step", type=float, default=1.0)
    p.add_argument("--r_step", type=float, default=100.0)
    p.add_argument("--nm_maxiter", type=int, default=60)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--sanitize_ckpt", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--run_ablations", action="store_true")
    p.add_argument("--ablation_snr_db", type=float, default=0.0)
    p.add_argument("--ablation_num_samples", type=int, default=200)
    p.add_argument("--ablate_T_list", type=str, default="1,2,5,10")
    p.add_argument("--grid_theta_steps", type=str, default="1,2,5")
    p.add_argument("--grid_r_steps", type=str, default="50,100,200")
    # Tiny per-sample scale preconditioner (pscale)
    p.add_argument("--use_pscale", type=int, default=0, choices=[0, 1])
    p.add_argument("--pscale_hidden", type=int, default=32)
    p.add_argument("--pscale_detach_step", type=int, default=1, choices=[0, 1])
    p.add_argument("--pscale_logrange", type=float, default=6.9)
    p.add_argument("--pscale_amp", type=float, default=1.0)
    p.add_argument(
        "--pscale_input",
        type=str,
        default="step,u,t",
        help='Comma-separated tokens for pscale features. Supported: "step,u,t" (order matters; deduped).',
    )
    p.add_argument("--use_t_table", type=int, default=0, choices=[0, 1])
    p.add_argument("--t_table_init", type=float, default=0.0)
    p.add_argument("--pscale_min_theta", type=float, default=0.7)
    p.add_argument("--pscale_max_theta", type=float, default=1.3)
    p.add_argument("--pscale_min_r", type=float, default=0.7)
    p.add_argument("--pscale_max_r", type=float, default=1.3)
    p.add_argument("--run_dir", type=str, default="")
    return p.parse_args()


def ms_per_sample(dt_s: float, n: int) -> float:
    return 1000.0 * dt_s / max(int(n), 1)


def _nonfinite_module_tensors(module: torch.nn.Module) -> list[str]:
    bad: list[str] = []
    for name, p in module.named_parameters(recurse=True):
        if p is not None and (not torch.isfinite(p).all().item()):
            bad.append(f"param:{name}")
    for name, b in module.named_buffers(recurse=True):
        if b is not None and (not torch.isfinite(b).all().item()):
            bad.append(f"buffer:{name}")
    return bad


def _sanitize_refiner_alpha_lambda(refiner: Refiner, *, init_alpha: float = 2e-2, init_lambda: float = 1e-3) -> None:
    """
    Best-effort sanitize for corrupted checkpoints where step parameters become NaN/Inf.
    """

    with torch.no_grad():
        device = refiner.alpha_theta_raw.device
        dtype = refiner.alpha_theta_raw.dtype

        def inv_softplus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            x = x.clamp_min(eps)
            return torch.log(torch.expm1(x))

        a0 = torch.full((refiner.T,), float(init_alpha), device=device, dtype=dtype)
        l0 = torch.full((refiner.T,), float(init_lambda), device=device, dtype=dtype)
        a_raw0 = inv_softplus(a0)
        l_raw0 = inv_softplus(l0)

        for name, raw0 in [
            ("alpha_theta_raw", a_raw0),
            ("alpha_r_raw", a_raw0),
            ("lambda_theta_raw", l_raw0),
            ("lambda_r_raw", l_raw0),
        ]:
            t = getattr(refiner, name)
            t.data = torch.where(torch.isfinite(t.data), t.data, raw0)


def _infer_refiner_T_model_from_sd(sd: dict, *, T_fallback: int, learnable: bool) -> int:
    t_model = int(T_fallback)
    if not learnable:
        return t_model
    for name in (
        "alpha_theta_raw",
        "alpha_r_raw",
        "lambda_theta_raw",
        "lambda_r_raw",
        "t_log_scale_table",
    ):
        tensor = sd.get(name, None)
        if torch.is_tensor(tensor):
            return int(tensor.numel())
    return t_model


def run_unrolled(
    z_np: np.ndarray,
    cfg: FDAConfig,
    box: TargetBox,
    *,
    device: torch.device,
    theta_step: float,
    r_step: float,
    T: int,
    ckpt_path: str = "",
    sanitize_ckpt: bool = False,
    learnable: bool = False,
    T_run: int | None = None,
    return_meta: bool = False,
    batch_size: int = 512,
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
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, int, int]:
    z = torch.from_numpy(z_np).to(device=device)
    if z.dtype != torch.complex64:
        z = z.to(torch.complex64)

    coarse = CoarseSearcherTorch(
        cfg,
        theta_range=(box.theta_min, box.theta_max),
        r_range=(box.r_min, box.r_max),
        theta_step=theta_step,
        r_step=r_step,
        device=device,
    )

    refiner = Refiner(
        cfg,
        T=T,
        box=Box(box.theta_min, box.theta_max, box.r_min, box.r_max),
        learnable=learnable,
        r_precond_mul=1.0,
        r_precond_pow=1.0,
        r_precond_learnable=False,
        use_pscale=bool(use_pscale),
        pscale_hidden=int(pscale_hidden),
        pscale_detach_step=bool(pscale_detach_step),
        pscale_logrange=float(pscale_logrange),
        pscale_amp=float(pscale_amp),
        pscale_input=str(pscale_input),
        use_t_table=bool(use_t_table),
        t_table_init=float(t_table_init),
        pscale_min_theta=float(pscale_min_theta),
        pscale_max_theta=float(pscale_max_theta),
        pscale_min_r=float(pscale_min_r),
        pscale_max_r=float(pscale_max_r),
    ).to(device)
    T_model = int(T)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt["state_dict"]
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        T_model = _infer_refiner_T_model_from_sd(sd, T_fallback=T, learnable=learnable)
        r_precond_mul = float(ckpt_args.get("r_precond_mul", 1.0))
        r_precond_pow = float(ckpt_args.get("r_precond_pow", 1.0))
        r_precond_learnable = bool("r_precond_mul" in sd)
        use_pscale_ckpt = bool(int(ckpt_args.get("use_pscale", 0))) or any(
            str(k).startswith("pscale_mlp.") for k in sd.keys()
        )
        use_t_table_ckpt = bool(int(ckpt_args.get("use_t_table", 0))) or ("t_log_scale_table" in sd)
        pscale_hidden_ckpt = int(ckpt_args.get("pscale_hidden", 32))
        pscale_detach_step_ckpt = bool(int(ckpt_args.get("pscale_detach_step", 1)))
        pscale_logrange_ckpt = float(ckpt_args.get("pscale_logrange", 6.9))
        pscale_amp_ckpt = float(ckpt_args.get("pscale_amp", 1.0))
        pscale_input_ckpt = str(ckpt_args.get("pscale_input", pscale_input))
        t_table_init_ckpt = float(ckpt_args.get("t_table_init", 0.0))
        pscale_min_theta_ckpt = float(ckpt_args.get("pscale_min_theta", pscale_min_theta))
        pscale_max_theta_ckpt = float(ckpt_args.get("pscale_max_theta", pscale_max_theta))
        pscale_min_r_ckpt = float(ckpt_args.get("pscale_min_r", pscale_min_r))
        pscale_max_r_ckpt = float(ckpt_args.get("pscale_max_r", pscale_max_r))

        refiner = Refiner(
            cfg,
            T=T_model,
            box=Box(box.theta_min, box.theta_max, box.r_min, box.r_max),
            learnable=learnable,
            r_precond_mul=r_precond_mul,
            r_precond_pow=r_precond_pow,
            r_precond_learnable=r_precond_learnable,
            use_pscale=use_pscale_ckpt,
            pscale_hidden=pscale_hidden_ckpt,
            pscale_detach_step=pscale_detach_step_ckpt,
            pscale_logrange=pscale_logrange_ckpt,
            pscale_amp=pscale_amp_ckpt,
            pscale_input=pscale_input_ckpt,
            use_t_table=use_t_table_ckpt,
            t_table_init=t_table_init_ckpt,
            pscale_min_theta=pscale_min_theta_ckpt,
            pscale_max_theta=pscale_max_theta_ckpt,
            pscale_min_r=pscale_min_r_ckpt,
            pscale_max_r=pscale_max_r_ckpt,
        ).to(device)
        if "t_log_scale_table" in sd and sd["t_log_scale_table"].shape[0] != int(T_model):
            old = sd["t_log_scale_table"]
            new = torch.zeros((int(T_model), 2), dtype=old.dtype, device=old.device)
            n = min(int(T_model), int(old.shape[0]))
            new[:n] = old[:n]
            sd = dict(sd)
            sd["t_log_scale_table"] = new
        # Reload after rebuilding the module.
        refiner.load_state_dict(sd, strict=True)
        bad = _nonfinite_module_tensors(refiner)
        if bad:
            if sanitize_ckpt:
                _sanitize_refiner_alpha_lambda(refiner)
                bad2 = _nonfinite_module_tensors(refiner)
                if bad2:
                    raise RuntimeError(f"Checkpoint has non-finite tensors after sanitize: {bad2}")
                print(f"[warn] Sanitized non-finite ckpt tensors: {bad}")
            else:
                raise RuntimeError(f"Checkpoint has non-finite tensors: {bad}")
    refiner.eval()
    T_run_eff = int(T_model if T_run is None else T_run)
    T_run_eff = max(0, min(T_run_eff, int(T_model)))

    n = z.shape[0]
    theta_hat = []
    r_hat = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        zb = z[i : i + batch_size]
        theta0, r0, _ = coarse.search(zb)
        with torch.enable_grad():
            th, rr, _ = refiner(zb, theta0, r0, T_run=T_run_eff)
        theta_hat.append(th.detach().cpu().numpy())
        r_hat.append(rr.detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if return_meta:
        return np.concatenate(theta_hat), np.concatenate(r_hat), ms_per_sample(dt, n), int(T_model), int(T_run_eff)
    return np.concatenate(theta_hat), np.concatenate(r_hat), ms_per_sample(dt, n)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.full:
        args.num_samples = max(args.num_samples, 2000)
        args.nm_maxiter = max(args.nm_maxiter, 120)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = FDAConfig()
    box = TargetBox()

    # Resolve requested runtime unroll T.
    ckpt_T = None
    if args.ckpt_path and args.T is None:
        try:
            ckpt_meta = torch.load(args.ckpt_path, map_location="cpu")
            ckpt_T = ckpt_meta.get("T", None)
            if ckpt_T is None:
                ckpt_T = ckpt_meta.get("args", {}).get("T", None)
        except Exception:
            ckpt_T = None
    T_resolved = int(args.T) if args.T is not None else int(ckpt_T) if ckpt_T is not None else 10

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"eval_{timestamp()}"
    ensure_dir(run_dir)
    write_json(run_dir / "config.json", {**vars(args), "T_resolved": T_resolved, "cfg": cfg.__dict__})

    ensure_dir("runs")
    latest = Path("runs") / "latest_eval"
    latest_ptr = Path("runs") / "latest_eval_path.txt"
    try:
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            # best-effort cleanup; not required
            pass
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
    except Exception:
        pass
    try:
        latest_ptr.write_text(str(run_dir.resolve()), encoding="utf-8")
    except Exception:
        pass

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
    rng = np.random.default_rng(args.seed)

    for snr_db in snr_list:
        theta_gt = rng.uniform(box.theta_min, box.theta_max, size=(args.num_samples,)).astype(
            np.float32
        )
        r_gt = rng.uniform(box.r_min, box.r_max, size=(args.num_samples,)).astype(np.float32)
        snr_vec = np.full((args.num_samples,), float(snr_db), dtype=np.float32)

        _, z, _ = synthesize_np(theta_gt, r_gt, snr_vec, cfg, seed=args.seed + 12345)
        z = z.astype(np.complex64)

        # 1) grid-only (CPU)
        theta_grid = np.zeros_like(theta_gt)
        r_grid = np.zeros_like(r_gt)
        with Timer() as t_grid:
            for i in range(args.num_samples):
                th0, rr0, _ = coarse_search_np(
                    z[i],
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    theta_step=args.theta_step,
                    r_step=args.r_step,
                )
                theta_grid[i] = th0
                r_grid[i] = rr0

        rmse_theta = rmse_np(angle_error_deg_np(theta_grid, theta_gt))
        rmse_r = rmse_np(r_grid - r_gt)
        csv.log(
            {
                "snr_db": snr_db,
                "method": "grid_only_cpu",
                "rmse_theta_deg": rmse_theta,
                "rmse_r_m": rmse_r,
                "ms_per_sample": ms_per_sample(t_grid.dt, args.num_samples),
                "notes": "",
            }
        )

        # 2) grid + Nelder–Mead (CPU): time NM only, report coarse+NM total
        theta_nm = np.zeros_like(theta_gt)
        r_nm = np.zeros_like(r_gt)
        with Timer() as t_nm:
            for i in range(args.num_samples):
                res = refine_nelder_mead(
                    z[i],
                    float(theta_grid[i]),
                    float(r_grid[i]),
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    maxiter=args.nm_maxiter,
                )
                theta_nm[i] = res.theta_deg
                r_nm[i] = res.r_m

        rmse_theta = rmse_np(angle_error_deg_np(theta_nm, theta_gt))
        rmse_r = rmse_np(r_nm - r_gt)
        csv.log(
            {
                "snr_db": snr_db,
                "method": "grid_nm_cpu",
                "rmse_theta_deg": rmse_theta,
                "rmse_r_m": rmse_r,
                "ms_per_sample": ms_per_sample(t_grid.dt + t_nm.dt, args.num_samples),
                "notes": f"nm_only_ms={ms_per_sample(t_nm.dt, args.num_samples):.3f}",
            }
        )

        # 3) grid + unrolled (fixed) GPU batch
        th_u, r_u, ms_u = run_unrolled(
            z,
            cfg,
            box,
            device=device,
            theta_step=args.theta_step,
            r_step=args.r_step,
            T=T_resolved,
            ckpt_path="",
            sanitize_ckpt=False,
            learnable=False,
            batch_size=args.batch_size,
            use_pscale=bool(int(args.use_pscale)),
            pscale_hidden=int(args.pscale_hidden),
            pscale_detach_step=bool(int(args.pscale_detach_step)),
            pscale_logrange=float(args.pscale_logrange),
            pscale_amp=float(args.pscale_amp),
            pscale_input=str(args.pscale_input),
            use_t_table=bool(int(args.use_t_table)),
            t_table_init=float(args.t_table_init),
            pscale_min_theta=float(args.pscale_min_theta),
            pscale_max_theta=float(args.pscale_max_theta),
            pscale_min_r=float(args.pscale_min_r),
            pscale_max_r=float(args.pscale_max_r),
        )
        rmse_theta = rmse_np(angle_error_deg_np(th_u, theta_gt))
        rmse_r = rmse_np(r_u - r_gt)
        csv.log(
            {
                "snr_db": snr_db,
                "method": "grid_unroll_fixed",
                "rmse_theta_deg": rmse_theta,
                "rmse_r_m": rmse_r,
                "ms_per_sample": ms_u,
                "notes": f"device={device.type}",
            }
        )

        # 4) grid + unrolled (learned) GPU batch
        if args.ckpt_path:
            try:
                th_ul, r_ul, ms_ul, T_model_ul, T_run_ul = run_unrolled(
                    z,
                    cfg,
                    box,
                    device=device,
                    theta_step=args.theta_step,
                    r_step=args.r_step,
                    T=T_resolved,
                    ckpt_path=args.ckpt_path,
                    sanitize_ckpt=bool(args.sanitize_ckpt),
                    learnable=True,
                    T_run=T_resolved,
                    return_meta=True,
                    batch_size=args.batch_size,
                    use_pscale=bool(int(args.use_pscale)),
                    pscale_hidden=int(args.pscale_hidden),
                    pscale_detach_step=bool(int(args.pscale_detach_step)),
                    pscale_logrange=float(args.pscale_logrange),
                    pscale_amp=float(args.pscale_amp),
                    pscale_input=str(args.pscale_input),
                    use_t_table=bool(int(args.use_t_table)),
                    t_table_init=float(args.t_table_init),
                    pscale_min_theta=float(args.pscale_min_theta),
                    pscale_max_theta=float(args.pscale_max_theta),
                    pscale_min_r=float(args.pscale_min_r),
                    pscale_max_r=float(args.pscale_max_r),
                )
                rmse_theta = rmse_np(angle_error_deg_np(th_ul, theta_gt))
                rmse_r = rmse_np(r_ul - r_gt)
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": "grid_unroll_learned",
                        "rmse_theta_deg": rmse_theta,
                        "rmse_r_m": rmse_r,
                        "ms_per_sample": ms_ul,
                        "notes": f"ckpt={args.ckpt_path}; T_model={T_model_ul}, T_run={T_run_ul}",
                    }
                )
            except RuntimeError as e:
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": "grid_unroll_learned",
                        "rmse_theta_deg": "",
                        "rmse_r_m": "",
                        "ms_per_sample": "",
                        "notes": f"skip ({str(e)})",
                    }
                )
        else:
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "grid_unroll_learned",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": "skip (no --ckpt_path)",
                }
            )

        print(f"SNR={snr_db} done")

    print(f"Wrote: {run_dir / 'results.csv'}")

    if not args.run_ablations:
        return

    # -------------------------
    # Ablations (single SNR)
    # -------------------------
    ab_snr = float(args.ablation_snr_db)
    ab_n = int(args.ablation_num_samples)

    theta_gt = rng.uniform(box.theta_min, box.theta_max, size=(ab_n,)).astype(np.float32)
    r_gt = rng.uniform(box.r_min, box.r_max, size=(ab_n,)).astype(np.float32)
    snr_vec = np.full((ab_n,), ab_snr, dtype=np.float32)
    _, z, _ = synthesize_np(theta_gt, r_gt, snr_vec, cfg, seed=args.seed + 23456)
    z = z.astype(np.complex64)

    # (A) T ablation for unrolled (fixed; learned if ckpt provided)
    ablate_T = [int(x) for x in args.ablate_T_list.split(",") if x.strip()]
    ablate_T = [t for t in ablate_T if t > 0]

    ab_csv = CsvLogger(
        run_dir / "ablate_T.csv",
        fieldnames=["snr_db", "method", "T_run", "rmse_theta_deg", "rmse_r_m", "ms_per_sample", "notes"],
    )

    def run_unrolled_with_Trun(learned: bool, ckpt_path: str, T_run: int):
        return run_unrolled(
            z[:ab_n],
            cfg,
            box,
            device=device,
            theta_step=args.theta_step,
            r_step=args.r_step,
            T=T_resolved,
            ckpt_path=ckpt_path,
            sanitize_ckpt=bool(args.sanitize_ckpt),
            learnable=bool(learned),
            T_run=int(T_run),
            return_meta=True,
            batch_size=args.batch_size,
            use_pscale=bool(int(args.use_pscale)),
            pscale_hidden=int(args.pscale_hidden),
            pscale_detach_step=bool(int(args.pscale_detach_step)),
            pscale_logrange=float(args.pscale_logrange),
            pscale_amp=float(args.pscale_amp),
            pscale_input=str(args.pscale_input),
            use_t_table=bool(int(args.use_t_table)),
            t_table_init=float(args.t_table_init),
            pscale_min_theta=float(args.pscale_min_theta),
            pscale_max_theta=float(args.pscale_max_theta),
            pscale_min_r=float(args.pscale_min_r),
            pscale_max_r=float(args.pscale_max_r),
        )

    for T_run in ablate_T:
        if T_run > T_resolved:
            continue
        th_hat, r_hat, ms_hat, _, T_run_eff = run_unrolled_with_Trun(False, "", T_run)
        ab_csv.log(
            {
                "snr_db": ab_snr,
                "method": "unroll_fixed",
                "T_run": T_run_eff,
                "rmse_theta_deg": rmse_np(angle_error_deg_np(th_hat, theta_gt)),
                "rmse_r_m": rmse_np(r_hat - r_gt),
                "ms_per_sample": ms_hat,
                "notes": f"device={device.type}",
            }
        )

        if args.ckpt_path:
            try:
                th_hat, r_hat, ms_hat, T_model_hat, T_run_hat = run_unrolled_with_Trun(
                    True, args.ckpt_path, T_run
                )
                ab_csv.log(
                    {
                        "snr_db": ab_snr,
                        "method": "unroll_learned",
                        "T_run": T_run_hat,
                        "rmse_theta_deg": rmse_np(angle_error_deg_np(th_hat, theta_gt)),
                        "rmse_r_m": rmse_np(r_hat - r_gt),
                        "ms_per_sample": ms_hat,
                        "notes": f"ckpt={args.ckpt_path}; T_model={T_model_hat}, T_run={T_run_hat}",
                    }
                )
            except RuntimeError as e:
                ab_csv.log(
                    {
                        "snr_db": ab_snr,
                        "method": "unroll_learned",
                        "T_run": T_run,
                        "rmse_theta_deg": "",
                        "rmse_r_m": "",
                        "ms_per_sample": "",
                        "notes": f"skip ({str(e)})",
                    }
                )

    # (B) coarse-grid step robustness at fixed SNR
    theta_steps = [float(x) for x in args.grid_theta_steps.split(",") if x.strip()]
    r_steps = [float(x) for x in args.grid_r_steps.split(",") if x.strip()]

    gr_csv = CsvLogger(
        run_dir / "grid_robustness.csv",
        fieldnames=[
            "snr_db",
            "theta_step",
            "r_step",
            "method",
            "rmse_theta_deg",
            "rmse_r_m",
            "ms_per_sample",
            "notes",
        ],
    )

    for th_step in theta_steps:
        for rr_step in r_steps:
            # grid-only + grid+NM on CPU (keep smaller for NM)
            n_cpu = min(ab_n, 80)
            theta_grid = np.zeros((n_cpu,), dtype=np.float32)
            r_grid = np.zeros((n_cpu,), dtype=np.float32)
            with Timer() as t_grid:
                for i in range(n_cpu):
                    th0, rr0, _ = coarse_search_np(
                        z[i],
                        cfg,
                        theta_range=(box.theta_min, box.theta_max),
                        r_range=(box.r_min, box.r_max),
                        theta_step=th_step,
                        r_step=rr_step,
                    )
                    theta_grid[i] = th0
                    r_grid[i] = rr0
            gr_csv.log(
                {
                    "snr_db": ab_snr,
                    "theta_step": th_step,
                    "r_step": rr_step,
                    "method": "grid_only_cpu",
                    "rmse_theta_deg": rmse_np(angle_error_deg_np(theta_grid, theta_gt[:n_cpu])),
                    "rmse_r_m": rmse_np(r_grid - r_gt[:n_cpu]),
                    "ms_per_sample": ms_per_sample(t_grid.dt, n_cpu),
                    "notes": "",
                }
            )

            theta_nm = np.zeros_like(theta_grid)
            r_nm = np.zeros_like(r_grid)
            with Timer() as t_nm:
                for i in range(n_cpu):
                    res = refine_nelder_mead(
                        z[i],
                        float(theta_grid[i]),
                        float(r_grid[i]),
                        cfg,
                        theta_range=(box.theta_min, box.theta_max),
                        r_range=(box.r_min, box.r_max),
                        maxiter=max(60, args.nm_maxiter // 2),
                    )
                    theta_nm[i] = res.theta_deg
                    r_nm[i] = res.r_m
            gr_csv.log(
                {
                    "snr_db": ab_snr,
                    "theta_step": th_step,
                    "r_step": rr_step,
                    "method": "grid_nm_cpu",
                    "rmse_theta_deg": rmse_np(angle_error_deg_np(theta_nm, theta_gt[:n_cpu])),
                    "rmse_r_m": rmse_np(r_nm - r_gt[:n_cpu]),
                    "ms_per_sample": ms_per_sample(t_grid.dt + t_nm.dt, n_cpu),
                    "notes": f"nm_only_ms={ms_per_sample(t_nm.dt, n_cpu):.3f}",
                }
            )

            # unrolled fixed (GPU batch) with this grid
            th_u, r_u, ms_u = run_unrolled(
                z[:ab_n],
                cfg,
                box,
                device=device,
                theta_step=th_step,
                r_step=rr_step,
                T=T_resolved,
                ckpt_path="",
                sanitize_ckpt=False,
                learnable=False,
                batch_size=args.batch_size,
                use_pscale=bool(int(args.use_pscale)),
                pscale_hidden=int(args.pscale_hidden),
                pscale_detach_step=bool(int(args.pscale_detach_step)),
                pscale_logrange=float(args.pscale_logrange),
                pscale_amp=float(args.pscale_amp),
                pscale_input=str(args.pscale_input),
                use_t_table=bool(int(args.use_t_table)),
                t_table_init=float(args.t_table_init),
                pscale_min_theta=float(args.pscale_min_theta),
                pscale_max_theta=float(args.pscale_max_theta),
                pscale_min_r=float(args.pscale_min_r),
                pscale_max_r=float(args.pscale_max_r),
            )
            gr_csv.log(
                {
                    "snr_db": ab_snr,
                    "theta_step": th_step,
                    "r_step": rr_step,
                    "method": "grid_unroll_fixed",
                    "rmse_theta_deg": rmse_np(angle_error_deg_np(th_u, theta_gt)),
                    "rmse_r_m": rmse_np(r_u - r_gt),
                    "ms_per_sample": ms_u,
                    "notes": f"device={device.type}",
                }
            )

            if args.ckpt_path:
                try:
                    th_ul, r_ul, ms_ul, T_model_ul, T_run_ul = run_unrolled(
                        z[:ab_n],
                        cfg,
                        box,
                        device=device,
                        theta_step=th_step,
                        r_step=rr_step,
                        T=T_resolved,
                        ckpt_path=args.ckpt_path,
                        sanitize_ckpt=bool(args.sanitize_ckpt),
                        learnable=True,
                        T_run=T_resolved,
                        return_meta=True,
                        batch_size=args.batch_size,
                        use_pscale=bool(int(args.use_pscale)),
                        pscale_hidden=int(args.pscale_hidden),
                        pscale_detach_step=bool(int(args.pscale_detach_step)),
                        pscale_logrange=float(args.pscale_logrange),
                        pscale_amp=float(args.pscale_amp),
                        pscale_input=str(args.pscale_input),
                        use_t_table=bool(int(args.use_t_table)),
                        t_table_init=float(args.t_table_init),
                        pscale_min_theta=float(args.pscale_min_theta),
                        pscale_max_theta=float(args.pscale_max_theta),
                        pscale_min_r=float(args.pscale_min_r),
                        pscale_max_r=float(args.pscale_max_r),
                    )
                    gr_csv.log(
                        {
                            "snr_db": ab_snr,
                            "theta_step": th_step,
                            "r_step": rr_step,
                            "method": "grid_unroll_learned",
                            "rmse_theta_deg": rmse_np(angle_error_deg_np(th_ul, theta_gt)),
                            "rmse_r_m": rmse_np(r_ul - r_gt),
                            "ms_per_sample": ms_ul,
                            "notes": f"ckpt={args.ckpt_path}; T_model={T_model_ul}, T_run={T_run_ul}",
                        }
                    )
                except RuntimeError as e:
                    gr_csv.log(
                        {
                            "snr_db": ab_snr,
                            "theta_step": th_step,
                            "r_step": rr_step,
                            "method": "grid_unroll_learned",
                            "rmse_theta_deg": "",
                            "rmse_r_m": "",
                            "ms_per_sample": "",
                            "notes": f"skip ({str(e)})",
                        }
                    )

    print(f"Wrote: {run_dir / 'ablate_T.csv'}")
    print(f"Wrote: {run_dir / 'grid_robustness.csv'}")


if __name__ == "__main__":
    main()
