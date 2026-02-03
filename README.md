# OneBit: 1-bit FDA-MIMO Radar single-target range-angle estimation

本仓库复现并对比两类 **off-grid refinement**：

1. Baseline：**粗网格搜索**初始化 + **SciPy Nelder–Mead** 连续域精修（CPU）。
2. 新方法：同一目标函数下的 **可微 / 可展开（unrolled）refinement**（PyTorch，支持 batch GPU 推理与可学习步长/阻尼）。

## 1. 问题与符号（单目标）

- 角度 `θ`：单位 **deg**
- 距离 `r`：单位 **m**
- 虚拟阵列通道数 `K = M*N`
- 1-bit 量化（复数）：

  z = sign(Re(y)) + j*sign(Im(y)),  sign(x) ∈ {+1,-1}

  提供 `sign0` 模式：当 `x==0` 时按 `+1` 处理。

## 2. Baseline 目标函数（严格对齐）

只使用如下归一化相关准则（不更换目标）：

  J(θ,r) = |a(θ,r)^H z|^2 / ||a(θ,r)||_2^2

粗网格搜索最大化 `J(θ,r)` 得到 `(θ0,r0)`；精修阶段在连续域最大化 `J`（等价最小化 `-J`）。

## 3. FDA-MIMO steering vector（“最小一致版本”，可替换）

默认参数：

- f0=1e9, c=3e8
- M=N=10
- Δf=30e3
- λ0=c/f0, d=λ0/2

ULA 假设下，对 Tx 元素 m=0..M-1、Rx 元素 n=0..N-1，定义：

- 角度相位（参考波长 λ0）：
  φ_ang(m,n;θ) = -(2π/λ0) * (m+n)*d*sin(θ)
- FDA 距离相位（Tx 频偏导致）：
  f_m = f0 + mΔf,  φ_rng(m;r) = -(4π/c) * f_m * r

最终虚拟阵列流形（展平成 K=M*N 的复向量）：

  a_{m,n}(θ,r) = exp{ j( φ_ang(m,n;θ) + φ_rng(m;r) ) }

说明：
- 这是一个可复现且与 baseline 目标一致的最小版本；你可以在 `src/fda.py` 中替换为更精确的 FDA-MIMO 相位模型。

## 4. 安装

Python >= 3.10

```bash
pip install -r requirements.txt
```

Windows 上如果 `python` 指向 Microsoft Store stub，请用 `py`（例如 `py -m scripts.eval_all`）。

## 5. 快速跑通（1–2 分钟 sanity）

```bash
python -m scripts.eval_all --device cuda
python -m scripts.plot_curves --run_dir runs/latest_eval
```

可选：跑 T 消融 + 粗网格步长鲁棒性（单个 SNR 上，额外耗时）：

```bash
python -m scripts.eval_all --device cuda --run_ablations --ablation_snr_db 0 --ablation_num_samples 200
```

## 6. 训练 unrolled refiner（示例）

```bash
python -m scripts.train_unroll --device cuda --T 10 --steps 2000
python -m scripts.eval_all --device cuda --ckpt_path runs/<your_train_run>/ckpt.pt
```

如果你的旧 ckpt 里出现了 `alpha_raw/lambda_raw = NaN`（历史训练不稳定导致），评测时可用 `--sanitize_ckpt` 先做 best-effort 修复，或直接重新训练生成干净 ckpt：

```bash
python -m scripts.eval_all --device cuda --ckpt_path runs/<your_train_run>/ckpt.pt --sanitize_ckpt
```
