"""
main.py —— 主入口:一键运行"数据 → 训练 → 推理 → 评估 → 可视化"全流程
=====================================================================

复现论文:Solving Hamiltonian Constraint Equation with Physics-Informed
Neural Networks (arXiv:2607.06002v1)

用法:
    # 本机 CPU 冒烟验证(约 1-3 分钟,小规模)
    python main.py --smoke

    # 完整训练(服务器 GPU,论文超参数,约 30 分钟/5000 步)
    python main.py --case base

    # 已有模型:跳过训练,直接评估+可视化(自动检测 runs/<exp>/model.pt)
    python main.py --case base
    python main.py --case base --reference reference_u.npz

    # 强制重新训练(覆盖已有模型)
    python main.py --case base --retrain

    # 指定算例与步数
    python main.py --case uneq_spin --steps 10000

输出:
    runs/<exp_name>/          # 每次运行一个目录
        ├── model.pt          # 训练好的模型权重
        ├── data.npz          # 采样数据与解析量(可复用)
        ├── history.json      # 损失历史
        ├── metrics.json      # 评估指标
        └── figs/             # 可视化图
            ├── loss_history.png
            ├── x_axis_profile.png
            └── equatorial_plane.png

行为约定:
    - 若 runs/<exp_name>/model.pt 已存在且未指定 --retrain:跳过训练,
      加载已有模型与 data.npz 缓存,直接评估与可视化;
    - 否则:重新采样、训练并保存。
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK 编码,无法打印 κ/± 等 Unicode 符号:
# 强制 stdout 使用 UTF-8(服务器 Linux 本身是 UTF-8,不受影响)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from logutil import setup_logging
import config as config_mod
from config import BBHConfig, PINNConfig, TrainConfig
from data import DataBundle
from model import GuidedPINN
from train import Trainer
import evaluate
import visualize

log = logging.getLogger("paper.A1.main")


def get_device(prefer: str = "auto") -> torch.device:
    """选择计算设备。

    Args:
        prefer: 'auto' | 'cuda' | 'cpu'

    Returns:
        torch.device
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(prefer)


def load_model(path: str, device) -> GuidedPINN:
    """从 model.pt 重建模型(含全部元数据:κ、c、网络结构、物理参数)。

    Args:
        path  : model.pt 路径
        device: torch 设备

    Returns:
        GuidedPINN: 训练好的模型(已 load_state_dict)
    """
    ckpt = torch.load(path, map_location=device)
    pcfg = PINNConfig(hidden_layers=ckpt["hidden_layers"],
                      hidden_neurons=ckpt["hidden_neurons"])
    # 物理参数作为 buffer 保存在 state_dict 中,直接取回
    sd = ckpt["model_state"]
    model = GuidedPINN(
        pcfg, kappa=ckpt["kappa"], c=ckpt["c"],
        masses=sd["masses"].cpu().numpy(), xs=sd["xs"].cpu().numpy(),
        Ps=sd["Ps"].cpu().numpy(), Ss=sd["Ss"].cpu().numpy(),
        u_min=ckpt["u_min"], u_max=ckpt["u_max"]).to(device)
    model.load_state_dict(sd)
    return model


def save_runs(exp_dir: str, data_bundle: DataBundle, model: GuidedPINN,
              history: dict, metrics: dict, cfg: TrainConfig, bb: BBHConfig):
    """保存模型、数据与指标到运行目录。

    Args:
        exp_dir    : 运行目录(runs/<name>)
        data_bundle: DataBundle
        model      : 训练好的模型
        history    : 损失历史
        metrics    : 评估指标
        cfg        : TrainConfig
        bb         : BBHConfig
    """
    os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)

    # 模型权重(含重建所需的全部元数据:κ、c、网络结构、引导解归一化常数)
    n_linear = sum(1 for m in model.mlp.net.modules() if isinstance(m, torch.nn.Linear))
    torch.save({"model_state": model.state_dict(),
                "kappa": model.kappa, "c": model.c,
                "u_min": model.u_min, "u_max": model.u_max,
                "hidden_layers": n_linear - 1,          # 线性层数减 1 = 隐藏层数
                "hidden_neurons": model.mlp.net[0].out_features},
               os.path.join(exp_dir, "model.pt"))

    # 数据(采样点 + 解析量 + κ),用于后续推理/复现
    np.savez(os.path.join(exp_dir, "data.npz"),
             x_int=data_bundle.x_int, x_bnd=data_bundle.x_bnd,
             ps_int=data_bundle.ps_int, kk_int=data_bundle.kk_int,
             ug_int=data_bundle.ug_int, w_int=data_bundle.w_int,
             ps_bnd=data_bundle.ps_bnd, kk_bnd=data_bundle.kk_bnd,
             ug_bnd=data_bundle.ug_bnd, w_bnd=data_bundle.w_bnd,
             u_min=data_bundle.u_min, u_max=data_bundle.u_max,
             kappa=data_bundle.kappa,
             masses=data_bundle.masses, xs=data_bundle.xs,
             Ps=data_bundle.Ps, Ss=data_bundle.Ss,
             R_max=data_bundle.R_max)

    # 历史与指标
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 配置快照(便于事后追溯)
    cfg_dict = {k: (v if not hasattr(v, "item") else v.item())
                for k, v in vars(cfg).items()}
    bb_dict = vars(bb)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({"train": cfg_dict, "bbh": bb_dict}, f, indent=2)


def load_history(exp_dir: str) -> dict:
    """加载历史损失(跳过训练时用于画损失曲线)。

    Args:
        exp_dir: 运行目录

    Returns:
        {'L2': [...], 'softLinf': [...], 'LBC': [...], 'total': [...]} 或空 dict
    """
    path = os.path.join(exp_dir, "history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    setup_logging("A1", "main")
    parser = argparse.ArgumentParser(
        description="复现 arXiv:2607.06002v1:PINN 求解双黑洞哈密顿约束方程")
    parser.add_argument("--case", default="base",
                        choices=["base", "spin_eq", "uneq_nospin", "uneq_spin"],
                        help="物理算例(论文 Table II)")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟模式:小规模快速验证")
    parser.add_argument("--steps", type=int, default=None,
                        help="训练步数(覆盖配置默认值)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率(覆盖配置默认值)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exp-name", default=None,
                        help="运行目录名(默认 <case>_<smoke|full>)")
    parser.add_argument("--reference", default=None,
                        help="TwoPunctures 参考解 npz 路径(可选,含 x_ref/u_ref)")
    parser.add_argument("--out-dir", default="runs",
                        help="输出根目录")
    parser.add_argument("--retrain", action="store_true",
                        help="强制重新训练(忽略已有 model.pt)")
    args = parser.parse_args()

    # ---- 0. 配置 ----
    bb = BBHConfig.from_case(args.case)
    cfg = config_mod.build_train_config(args.case, smoke=args.smoke)
    if args.steps is not None:
        cfg.n_steps = args.steps
    if args.lr is not None:
        cfg.lr = args.lr
    pcfg = PINNConfig()
    device = get_device(args.device)

    exp_name = args.exp_name or f"{args.case}_{'smoke' if args.smoke else 'full'}"
    exp_dir = os.path.join(args.out_dir, exp_name)
    model_path = os.path.join(exp_dir, "model.pt")
    data_path = os.path.join(exp_dir, "data.npz")

    log.info(f"设备: {device} | 算例: {args.case} | 冒烟: {args.smoke} | 输出: {exp_dir}")
    log.info(f"物理参数: m±={bb.m_plus}/{bb.m_minus}, x=±{bb.x_plus[0]}, "
          f"P+={bb.P_plus}, P-={bb.P_minus}, S+={bb.S_plus}, S-={bb.S_minus}")

    # ---- 1. 数据 + 2. 模型:优先复用已有,否则新建并训练 ----
    if os.path.exists(model_path) and os.path.exists(data_path) and not args.retrain:
        # ---- 复用已有模型与数据(跳过训练) ----
        log.info("\n[1/2] 检测到已有模型,加载缓存并跳过训练...")
        t0 = time.time()
        data = DataBundle.from_npz(data_path)
        model = load_model(model_path, device)
        # 一致性检查:缓存算例与当前 --case 是否匹配
        # (位置/质量/动量/自旋任一不同 → 缓存源项与模型都不可复用)
        cached = {"xs": np.asarray(data.xs), "masses": np.asarray(data.masses),
                  "Ps": np.asarray(data.Ps), "Ss": np.asarray(data.Ss)}
        want = {"xs": np.array([bb.x_plus, bb.x_minus]),
                "masses": np.array([bb.m_plus, bb.m_minus]),
                "Ps": np.array([bb.P_plus, bb.P_minus]),
                "Ss": np.array([bb.S_plus, bb.S_minus])}
        for key in ("xs", "masses", "Ps", "Ss"):
            if not np.allclose(cached[key], want[key], atol=1e-3):
                log.info(f"  [警告] 缓存 {key} {cached[key].ravel()} 与当前算例 "
                      f"{want[key].ravel()} 不一致!当前模型训练自其他算例,"
                      f"评估结果无意义。请用 --retrain 重新训练。")
        history = load_history(exp_dir)
        log.info(f"  已加载模型与数据 ({time.time()-t0:.1f}s),开始评估...")
    else:
        # ---- 全新训练 ----
        t0 = time.time()
        log.info("\n[1/5] 数据获取(采样配置点 + kappa 求解)...")
        data = DataBundle(cfg, bb)
        log.info(f"  N_Omega={cfg.N_Omega} 内部点, N_dOmega={cfg.N_boundary} 边界点, "
              f"kappa = {data.kappa:.4f}, u_g in [{data.u_min:.3e}, {data.u_max:.3e}]  "
              f"({time.time()-t0:.1f}s)")

        log.info("\n[2/5] 构建 PINN(3×64 SiLU,硬约束 ansatz)...")
        masses, xs, Ps, Ss = bb.as_arrays()
        model = GuidedPINN(pcfg, kappa=data.kappa, c=cfg.c_guide,
                           masses=masses, xs=xs, Ps=Ps, Ss=Ss,
                           u_min=data.u_min, u_max=data.u_max).to(device)
        log.info(f"  参数量: {sum(p.numel() for p in model.parameters())}")

        log.info(f"\n[3/5] 训练(Adam, lr={cfg.lr}, {cfg.n_steps} 步, "
              f"EMA平衡 α={cfg.ema_alpha})...")
        d = data.to_torch(device)
        trainer = Trainer(cfg, model, device)
        history = trainer.train(d, n_steps=cfg.n_steps,
                                log_every=max(1, cfg.n_steps // 10))

    # ---- 3. 评估 ----
    log.info("\n评估...")
    metrics = evaluate.evaluate_on_grid(model, data, cfg, device,
                                        reference_path=args.reference)
    metrics["kappa"] = data.kappa
    metrics["n_steps"] = cfg.n_steps if not history else len(history.get("L2", []))
    metrics["case"] = args.case
    metrics["resumed"] = (os.path.exists(model_path)
                          and os.path.exists(data_path) and not args.retrain)

    # ---- 4. 保存 + 可视化 ----
    log.info(f"\n保存结果与可视化 -> {exp_dir}")
    # 复用已有模型时只更新指标与图;新训练时全量保存
    if not metrics["resumed"]:
        save_runs(exp_dir, data, model, history, metrics, cfg, bb)
    else:
        os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    if history and history.get("L2"):
        visualize.plot_loss_history(history, os.path.join(exp_dir, "figs"))
    ref = None
    if args.reference:
        ref = np.load(args.reference)
        visualize.plot_x_axis_profile(model, data, device, os.path.join(exp_dir, "figs"),
                                      x_ref=ref["x_ref"], u_ref=ref["u_ref"])
    else:
        visualize.plot_x_axis_profile(model, data, device, os.path.join(exp_dir, "figs"))
    visualize.plot_equatorial_plane(model, data, device, os.path.join(exp_dir, "figs"),
                                    x_ref=ref["x_ref"] if ref is not None else None,
                                    u_ref=ref["u_ref"] if ref is not None else None)
    # 峰值区域局部放大图(奇点附近高分辨率一维剖面)
    visualize.plot_peak_zoom(model, data, device, os.path.join(exp_dir, "figs"),
                             x_ref=ref["x_ref"] if ref is not None else None,
                             u_ref=ref["u_ref"] if ref is not None else None)

    log.info(f"\n全部完成。结果目录: {exp_dir}")
    log.info(f"评估指标: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
