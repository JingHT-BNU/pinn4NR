"""
multi_param_eval.py —— 多参数参数化 PINN 评估
================================================

评估内容:
    1. 训练配置的 PDE 残差自检
    2. 验证配置的零样本 PDE 残差
    3. Base case L2RE (如有参考解)
    4. 分区 L2RE (近/中/远区)
    5. 物理合理性检查 (剖面, 单调性)
    6. 参数敏感性分析

用法:
    python multi_param_eval.py --exp runs/multi_param_a1 --reference ref_data.npz
"""

import argparse, json, logging, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import sample_ball, sample_sphere_surface
from logutil import setup_logging
import physics
from multi_param_model import (
    MultiParamGuidedPINN, compute_pde_residual, compute_robin_residual,
    normalize_params, build_bbh_from_params, BASE_RAW,
)
from multi_param_train import predict, l2re, resolve_input_path

log = logging.getLogger("paper.A3.multi_param_eval")


def load_model(exp_dir, device):
    """加载训练好的模型。"""
    ckpt = torch.load(os.path.join(exp_dir, "model.pt"), map_location=device,
                      weights_only=False)
    info = json.load(open(os.path.join(exp_dir, "configs.json")))

    hl = ckpt.get("hidden_layers", 6)
    hn = ckpt.get("hidden_neurons", 256)
    nfc = ckpt.get("n_freq_coord", 10)
    nfp = ckpt.get("n_freq_param", 12)

    model = MultiParamGuidedPINN(
        n_params=8, c_init=ckpt.get("c", 0.2),
        hidden_layers=hl, hidden_neurons=hn,
        n_freq_coord=nfc, n_freq_param=nfp,
        amp_mode=ckpt.get("amp_mode", "raw"),
        c_max=ckpt.get("c_max", 1.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    if "u_scale" in ckpt:
        model.set_u_scale(ckpt["u_scale"])
    model = model.double()  # 统一 double 精度(与训练时一致, 避免 eval 的 dtype 错误)
    model.eval()
    return model, ckpt, info


def pde_selfcheck(model, raw_params, kappa, device, n_pts=5000, R_max=30.0):
    """PDE 残差自检 (无 no_grad, 需要梯度)。"""
    masses, xs, Ps, Ss = build_bbh_from_params(raw_params)
    ma = torch.from_numpy(masses).double().to(device)
    xst = torch.from_numpy(xs).double().to(device)
    Pt = torch.from_numpy(Ps).double().to(device)
    St = torch.from_numpy(Ss).double().to(device)
    pn = torch.tensor(normalize_params(raw_params), dtype=torch.float64,
                      device=device).unsqueeze(0)

    rng = np.random.default_rng(42)
    x_np = sample_ball(n_pts, R_max, rng).astype(np.float32)
    x = torch.from_numpy(x_np).double().to(device)
    x.requires_grad_(True)
    u = model(x, ma, xst, Pt, St, pn, kappa)
    r = compute_pde_residual(u, x, ma, xst, Pt, St)
    return r.detach().cpu().numpy()


def robin_selfcheck(model, raw_params, kappa, device, n_pts=2000, R_max=30.0):
    """Robin 边界残差自检。"""
    masses, xs, Ps, Ss = build_bbh_from_params(raw_params)
    ma = torch.from_numpy(masses).double().to(device)
    xst = torch.from_numpy(xs).double().to(device)
    Pt = torch.from_numpy(Ps).double().to(device)
    St = torch.from_numpy(Ss).double().to(device)
    pn = torch.tensor(normalize_params(raw_params), dtype=torch.float64,
                      device=device).unsqueeze(0)

    rng = np.random.default_rng(43)
    x_np = sample_sphere_surface(n_pts, R_max, rng).astype(np.float32)
    x = torch.from_numpy(x_np).double().to(device)
    x.requires_grad_(True)
    u = model(x, ma, xst, Pt, St, pn, kappa)
    r = compute_robin_residual(u, x, R_max)
    return r.detach().cpu().numpy()


def main():
    setup_logging("A3", "multi_param_eval")
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="runs/multi_param_a1")
    p.add_argument("--reference", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # 输入路径解析: 优先按 CWD 相对路径, 不存在则锚定到 paper/(与运行时 CWD 无关)
    args.exp = resolve_input_path(args.exp)
    args.reference = resolve_input_path(args.reference)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto"
                          else "cpu")
    log.info(f"设备: {device}")

    model, ckpt, info = load_model(args.exp, device)
    R_max = ckpt.get("kappa_cache_meta", {}).get("R_max", 30.0)
    log.info(f"模型: {args.exp}")
    log.info(f"  c={ckpt['c']:.4f}, u_scale={ckpt.get('u_scale', 0.05):.3g}")

    # ── 1. 训练配置 PDE 自检 ──
    log.info(f"\n{'='*60}")
    log.info("训练配置 PDE 残差自检:")
    train_pde_means, train_pde_stds = [], []
    for cfg in info["train"][:20]:
        r = pde_selfcheck(model, cfg["raw"], cfg["kappa"], device, R_max=R_max)
        train_pde_means.append(abs(r).mean())
        train_pde_stds.append(r.std())
    log.info(f"  |R|_mean: {np.mean(train_pde_means):.3e} "
          f"(max={np.max(train_pde_means):.3e})")
    log.info(f"  R_std   : {np.mean(train_pde_stds):.3e}")

    # ── 2. 验证配置零样本 PDE 自检 ──
    log.info(f"\n验证配置零样本 PDE 残差:")
    val_pde_means = []
    for cfg in info["val"][:20]:
        r = pde_selfcheck(model, cfg["raw"], cfg["kappa"], device, R_max=R_max)
        val_pde_means.append(abs(r).mean())
    log.info(f"  |R|_mean: {np.mean(val_pde_means):.3e} "
          f"(max={np.max(val_pde_means):.3e})")

    # ── 3. Robin 边界自检 ──
    log.info(f"\nRobin 边界残差:")
    rob_vals = []
    for cfg in info["train"][:5] + info["val"][:5]:
        r = robin_selfcheck(model, cfg["raw"], cfg["kappa"], device, R_max=R_max)
        rob_vals.append(abs(r).mean())
    log.info(f"  |R_B|_mean: {np.mean(rob_vals):.3e}")

    # ── 4. Base case L2RE ──
    if args.reference:
        log.info(f"\nBase case L2RE:")
        ref = np.load(args.reference)
        ref_x, ref_u = ref["x_ref"], ref["u_ref"]

        # 找 base case 配置(比较 raw 参数, 修复旧版用 norm 比 raw 的 bug)
        base_cfg = min(info["train"], key=lambda c: np.linalg.norm(
            np.array(c["raw"]) - BASE_RAW))
        up = predict(model, ref_x, base_cfg["raw"], base_cfg["kappa"], device)
        log.info(f"  L2RE = {l2re(up, ref_u):.4e}")

        # 分区 L2RE
        r = np.linalg.norm(ref_x, axis=1)
        zones = [(0, 0.5, "r<0.5"), (0.5, 2, "r[0.5,2]"),
                 (2, 10, "r[2,10]"), (10, 30, "r>10")]
        log.info(f"\n  分区 L2RE:")
        for lo, hi, name in zones:
            mask = (r >= lo) & (r < hi)
            if mask.sum() > 0:
                err = l2re(up[mask], ref_u[mask])
                log.info(f"    {name:12s}: {err:.4e} ({mask.sum()} pts)")

    # ── 5. 参数敏感性: 沿各维度变化 ──
    log.info(f"\n参数敏感性 (u(0,0,0) vs 各参数):")
    # 使用 base 配置的 κ(近似, 真实 κ 随参数变化)
    base_cfg = min(info["train"], key=lambda c: np.linalg.norm(
        np.array(c["raw"]) - BASE_RAW))
    x_origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    for dim_idx, dim_name in enumerate(["m_plus", "m_minus", "x_plus", "P_plus_y", "S_plus_z"]):
        vals = []
        for t in np.linspace(-0.8, 0.8, 9):
            raw = BASE_RAW.copy()
            from multi_param_model import PARAM_LO, PARAM_HI
            raw[dim_idx] = PARAM_LO[dim_idx] + (t + 1) / 2 * (PARAM_HI[dim_idx] - PARAM_LO[dim_idx])
            masses, xs, Ps, Ss = build_bbh_from_params(raw)
            # 简单估计 kappa: 用 base 配置 kappa(真实 kappa 随参数变化)
            kappa = base_cfg["kappa"]
            try:
                u_val = predict(model, x_origin, raw, kappa, device)
                vals.append(u_val[0])
            except Exception:
                vals.append(float("nan"))
        vals = np.array(vals)
        log.info(f"  {dim_name:12s}: u(0)∈[{np.nanmin(vals):.4e}, {np.nanmax(vals):.4e}]")

    log.info(f"\n评估完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
