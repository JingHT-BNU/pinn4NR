# --- path bootstrap (added during repo restructure: shared code lives in core/ and tools/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while True:
    if _os.path.isdir(_os.path.join(_ROOT, "core")):
        break
    _parent = _os.path.dirname(_ROOT)
    if _parent == _ROOT:
        break
    _ROOT = _parent
for _sub in ("core", "tools"):
    _p = _os.path.join(_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---
# -*- coding: utf-8 -*-
"""a2q_train_v4.py -- v4/v5/c2 统一训练器(2026-09-01 无人值守指令 #3/#4/#5)。

与 opv3 训练的关键差异:
  - 纯监督:L_ref 唯一损失(无 PDE/Robin —— 用户指令 #5 的精神,先走纯数据路);
  - 数据:data/datasets/a2q_data_v2,refsub_<lb>.npz 含逐点权重 w
    (近峰壳 ×3、谷 ×2 —— 指令 #1"重视峰值处误差"在损失端的落实);
  - 每步每配置加权重采样 pts_per_cfg 点(默认 4096),45+ 配置可负担;
  - 参数噪声 q_eff=q·exp(σξ)(m2=0.5·q_eff 同步),增强 q∈[1,10] 泛化;
  - 显式配置清单(读 v2 目录全部 cfg_*),heldout 由 --heldout 指定。

用法:
  python a2q_train_v4.py --variant opv5 --exp-name a2q_v5 --steps 15000
  python a2q_train_v4.py --variant opv4 --exp-name a2q_v4 --steps 15000
  python a2q_train_v4.py --variant c2   --exp-name a2q_c2_50 --steps 15000
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logutil import setup_logging
import a2q_model as A2
import physics

log = logging.getLogger("paper.A2.a2q_train_v4")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(_ROOT, "data", "runs", "a2")
DATA_DIR = os.path.join(_ROOT, "data", "datasets", "a2q_data_v2")


def main():
    setup_logging("A2", "a2q_train_v4")
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["opv4", "opv5", "opv6", "c2"])
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pts-per-cfg", type=int, default=4096)
    ap.add_argument("--noise-sigma-max", type=float, default=0.0,
                    help="opv3 实际未用参数噪声(密集配置已够插值);"
                         ">0 时作为 q 增广(会抬高 L_ref 地板)")
    ap.add_argument("--smooth-w", type=float, default=0.0,
                    help="远场 Laplacian 幅值正则权重(光滑化壳,ρ>2 采样)")
    ap.add_argument("--convex-w", type=float, default=0.0,
                    help="轴向二阶导符号 hinge 权重(远场段与谷区 u''>0)")
    ap.add_argument("--smooth-n", type=int, default=1024,
                    help="每步光滑化正则采样点数")
    # ---- Hamilton 约束残差项(2026-09-11)----
    ap.add_argument("--pde-w", type=float, default=0.0,
                    help="Hamilton 约束残差 R=Δu+S 的权重(outer/far 区,"
                         "径向递增)。0 表示关闭")
    ap.add_argument("--pde-n", type=int, default=1024,
                    help="每步 PDE 残差采样点数")
    ap.add_argument("--pde-rho-min", type=float, default=2.0,
                    help="PDE 残差采样下限:ρ=min(r1,r2) 大于此值(剔除峰区)")
    ap.add_argument("--pde-rmax", type=float, default=14.0,
                    help="PDE 残差采样立方体半边长(同时为 r0 上界)")
    ap.add_argument("--pde-r-start", type=float, default=3.0,
                    help="径向权重起点:r0<=此值时权重为 0")
    ap.add_argument("--pde-r-end", type=float, default=12.0,
                    help="径向权重终点:r0>=此值时权重为 1")
    ap.add_argument("--pde-p", type=float, default=1.0,
                    help="径向权重指数:w=((r0-r_start)/(r_end-r_start))^p")
    ap.add_argument("--pde-norm", choices=("batch", "rel"), default="batch",
                    help="残差归一化方式。batch:全场单一标量 RMS_批次(S)"
                         "(旧行为,已证实远场仅占 0.2%% 权重);"
                         "rel:逐点 sqrt(S²+(eps·S_ref)²),使远场真正进入损失")
    ap.add_argument("--pde-eps", type=float, default=0.05,
                    help="[--pde-norm rel] 归一化下限,以该配置源项参考尺度 "
                         "S_ref 为单位。越小则远场放大越大(上限≈1/eps²)")
    ap.add_argument("--pde-cfgs", type=int, default=1,
                    help="每步参与 PDE 残差的配置数(总点数仍为 --pde-n)"
                         "。1=旧行为(每配置约 63 步才被监督一次)")
    ap.add_argument("--heldout", default="q15,q25,q50,q74,q86")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--init-from", default=None,
                    help="从已有 run 目录的 model.pt 载入权重后继续训练(微调)")
    ap.add_argument("--hidden-neurons", type=int, default=128)
    ap.add_argument("--n-basis", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    heldout = set(args.heldout.split(",") if args.heldout else [])
    log.info("[%s] 设备=%s 步数=%d ppc=%d", args.exp_name, device, args.steps,
             args.pts_per_cfg)
    rng = np.random.default_rng(args.seed)

    cfgs, refs = {}, {}
    data_dir = args.data_dir
    for fn in sorted(os.listdir(data_dir)):
        if not (fn.startswith("cfg_") and fn.endswith(".npz")):
            continue
        lb = fn[4:-4]
        z = np.load(os.path.join(data_dir, fn))
        cfgs[lb] = {k: z[k] for k in z.files}
        rp = os.path.join(data_dir, f"refsub_{lb}.npz")
        if not os.path.exists(rp):
            log.error("refsub_%s.npz 缺失,先运行 post_refs_v2.py", lb)
            raise SystemExit(2)
        zr = np.load(rp)
        x = zr["x"].astype(np.float64)
        u = zr["u"].astype(np.float64)
        w = zr["w"].astype(np.float64) if "w" in zr.files \
            else np.ones(len(x))
        refs[lb] = dict(x=x, u=u, w=w)
    train_labels = sorted(lb for lb in cfgs if lb not in heldout)
    all_labels = sorted(cfgs)
    log.info("数据目录 %s:共 %d 配置(训练 %d,留出 %s)",
             data_dir, len(all_labels), len(train_labels), sorted(heldout))
    if not train_labels:
        raise SystemExit("无训练配置")

    model = A2.make_model(args.variant, device,
                          hidden_neurons=args.hidden_neurons,
                          n_basis=args.n_basis).double()
    n_par = sum(p.numel() for p in model.parameters())
    log.info("参数量: %d", n_par)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                     eta_min=args.lr * 0.01)

    fp = hashlib.sha1(json.dumps(
        {"v": 4, "variant": args.variant, "steps": args.steps, "lr": args.lr,
         "seed": args.seed, "ppc": args.pts_per_cfg,
         "labels": train_labels, "heldout": sorted(heldout),
         "init": args.init_from, "hn": args.hidden_neurons,
         "nb": args.n_basis, "sw": args.smooth_w, "cw": args.convex_w,
         "sn": args.smooth_n,
         "pw": args.pde_w, "pn": args.pde_n, "prho": args.pde_rho_min,
         "prmax": args.pde_rmax, "prs": args.pde_r_start,
         "pre": args.pde_r_end, "pp": args.pde_p,
         "pnorm": args.pde_norm, "peps": args.pde_eps,
         "pcfg": args.pde_cfgs},
        sort_keys=True).encode()).hexdigest()
    ck_dir = os.path.join(RUNS, args.exp_name)
    os.makedirs(ck_dir, exist_ok=True)
    step0, hist, ema = 0, {"L_ref": [], "total": []}, {}
    ck_p = os.path.join(ck_dir, "ckpt.pt")
    if args.init_from:
        src = torch.load(os.path.join(args.init_from, "model.pt"),
                         map_location=device, weights_only=False)
        model.load_state_dict(src["model_state"])
        log.info("[微调] 权重来自 %s (variant=%s, steps=%s)", args.init_from,
                 src.get("variant"), src.get("steps_done"))
    if os.path.exists(ck_p):
        try:
            ck = torch.load(ck_p, map_location=device, weights_only=False)
            if ck.get("fingerprint") == fp:
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sch.load_state_dict(ck["sch"])
                hist, ema = ck["hist"], ck["ema"]
                step0 = ck["step"]
                rng.bit_generator.state = ck["rng"]
                log.info("[续训] 从 step %d", step0)
            else:
                log.warning("指纹不符,重头训练")
        except Exception as e:
            log.warning("ckpt 读取失败(%s)", e)

    # 每配置预取常数张量
    tens = {}
    for lb in all_labels:
        d, r = cfgs[lb], refs[lb]
        tens[lb] = dict(
            x=torch.from_numpy(r["x"]).double().to(device),
            u=torch.from_numpy(r["u"]).double().to(device),
            w=torch.from_numpy(r["w"]).double().to(device),
            q=float(d["q"]), m2=float(d["m2"]), kappa=float(d["kappa"]),
            sq=float(d["sq"]), wmin=float(d["wmin"]), wmax=float(d["wmax"]),
            norm2=float(np.sum(r["w"] * r["u"] ** 2)) or 1.0)

    def ema_bal(nm, v):
        fv = float(v)
        ema[nm] = fv if nm not in ema else 0.9 * ema[nm] + 0.1 * fv
        return v / (ema[nm] + 1e-12)

    def ref_loss():
        per = []
        for lb in train_labels:
            t = tens[lb]
            # 均匀重采样 + 权重直接进损失(加权目标的无偏估计);
            # 不用 rng.choice(p=)(replace=False+p 为慢路径,曾致 1.16s/步)
            idx = rng.integers(0, len(t["x"]), args.pts_per_cfg)
            x = t["x"][idx]
            ur = t["u"][idx]
            w = t["w"][idx]
            q_eff = t["q"]
            m2_eff = 0.5 * q_eff
            if args.noise_sigma_max > 0:
                q_eff = q_eff * np.exp(args.noise_sigma_max *
                                       rng.standard_normal() *
                                       min(1.0, 0.2 + 0.8 * s / args.steps))
                m2_eff = 0.5 * q_eff
            ma = torch.tensor([0.5, m2_eff], dtype=torch.float64,
                              device=device)
            xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]],
                              dtype=torch.float64, device=device)
            Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                              dtype=torch.float64, device=device)
            St = torch.zeros((2, 3), dtype=torch.float64, device=device)
            pv = A2.param_vec(q_eff, m2_eff, device)
            u = model(x, ma, xs, Ps, St, pv, t["kappa"], t["wmin"],
                      t["wmax"], t["sq"])
            r2 = (u - ur) ** 2
            # 分母用**该配置全量参考点的** Σw·u_ref²(启动时算一次的确定性
            # 常数),而非当前 1024 点子样本上的估计 —— 后者把抽样的随机性
            # 直接注入每一层的损失尺度,是无谓的梯度噪声来源。
            per.append((w * r2).sum() / t["norm2"])
        return torch.stack(per).mean()

    XS3 = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]],
                       dtype=torch.float64, device=device)
    PS3 = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                       dtype=torch.float64, device=device)
    ST3 = torch.zeros((2, 3), dtype=torch.float64, device=device)
    XA3 = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64, device=device)

    _PDE_EDGES_T = None                       # rel 模式下的壳层分箱边界
    # --pde-norm rel 的确定性标定量(训练前算一次,不随训练漂移)。
    #
    # 为什么必须"确定性":
    #   归一化尺度若参与梯度,它本身就变成一个可被优化的自由度,不再是尺度。
    #   故一律用**与模型无关**的 ψ_sing(而非 ψ_sing+u)计算。
    #
    # 为什么必须"逐壳层"而非"逐点":
    #   逐点 sig_j=S_j 会让单个 S_j→0 的点被放大到 1/eps²,产生 10³ 量级的
    #   批间尖峰(实测 l_pde 在 7e4~6.7e7 之间跳动,12 步就把 L_ref 从
    #   1.48e-5 打到 1.86e-5)。改用 r0 对数分箱内的 S_rms 后曲线光滑得多。
    if args.pde_w > 0 and args.pde_norm == "rel":
        t_ref = time.time()
        _K = 12                                   # 壳层数
        _edges = np.geomspace(max(args.pde_rho_min, 1e-3),
                              args.pde_rmax, _K + 1)
        _PDE_EDGES_T = torch.from_numpy(_edges).double().to(device)
        with torch.no_grad():
            for i_lb, lb in enumerate(all_labels):
                tt = tens[lb]
                _r = np.random.default_rng(args.seed + 100000 + i_lb)
                _xs = torch.from_numpy(
                    _r.uniform(-args.pde_rmax, args.pde_rmax,
                               size=(max(4 * args.pde_n, 8192), 3))
                ).double().to(device)
                _rr = torch.minimum((_xs - XA3).norm(dim=1),
                                    (_xs + XA3).norm(dim=1))
                _r0 = _xs.norm(dim=1)
                _x = _xs[(_rr > args.pde_rho_min) & (_r0 <= args.pde_rmax)]
                if _x.shape[0] < 64:
                    tt["sprof"] = None
                    tt["sref"] = float("nan")
                    continue
                _ma = torch.tensor([0.5, tt["m2"]], dtype=torch.float64,
                                   device=device)
                _S = ((1.0 / 8.0) * physics.bowen_york_KK(_x, _ma, XS3, PS3, ST3)
                      / torch.clamp(physics.psi_sing(_x, _ma, XS3),
                                    min=1e-4) ** 7).abs()
                _rn = _x.norm(dim=1).cpu().numpy()
                _Sn = _S.cpu().numpy()
                prof = np.full(_K, np.nan)
                for k in range(_K):
                    m = (_rn >= _edges[k]) & (_rn < _edges[k + 1])
                    if m.sum() >= 8:
                        prof[k] = float(np.sqrt(np.mean(_Sn[m] ** 2)))
                # 空壳层用相邻值填充(单调递减),避免出现 0 或 NaN 被放大
                ok = ~np.isnan(prof)
                if ok.sum() == 0:
                    prof = np.full(_K, float(np.sqrt(np.mean(_Sn ** 2))))
                elif not ok.all():
                    prof = np.interp(np.arange(_K), np.flatnonzero(ok),
                                     prof[ok])
                tt["sprof"] = prof
                tt["sref"] = float(np.sqrt(np.mean(_Sn ** 2)))
        _s10 = tens.get("q10", {}).get("sref", float("nan"))
        _s100 = tens.get("q100", {}).get("sref", float("nan"))
        log.info("[pde] rel 标定完成 (%.1fs): 壳层 %d 箱 [%.3g,%.3g]; "
                 "S_ref(q10)=%.3e S_ref(q100)=%.3e; --pde-eps=%.3g "
                 "→ 远场放大上限 %.1e", time.time() - t_ref, _K,
                 _edges[0], _edges[-1], _s10, _s100, args.pde_eps,
                 1.0 / max(args.pde_eps ** 2, 1e-30))

    def smooth_loss(lb0):
        """光滑化正则(2026-09-02 用户指示):远场 Laplacian 幅值 + 轴向凸性。

        - Laplacian 幅值:ρ=min(r1,r2)>2 处 |▽²u|²(真解远场光滑且 ▽²u≥0
          有界,幅值惩罚压高频抖动而不偏置形状);
        - 轴向 hinge:x∈(3.2,10)∪(−10,−3.2)(1/|x| 衰减,凹向上)与
          x∈(−1.5,1.5)(两峰间谷区,局部极小)处 relu(−u'')² —— 仅取真解
          符号鲁棒的段;近峰段不约束(奇点球附近符号复杂)。
        曲率以 sq/9(结构尺度 ~3)无量纲化,惩罚量级与配置无关。
        """
        t = tens[lb0]
        ma = torch.tensor([0.5, t["m2"]], dtype=torch.float64, device=device)
        pv = A2.param_vec(t["q"], t["m2"], device)
        xs_s = torch.from_numpy(
            rng.uniform(-10, 10, size=(max(3 * args.smooth_n // 2, 256), 3))
        ).double().to(device)
        rr = torch.minimum((xs_s - XA3).norm(dim=1),
                           (xs_s + XA3).norm(dim=1))
        x = xs_s[rr > 2.0][:args.smooth_n]
        if x.shape[0] < 64:
            return None
        x.requires_grad_(True)
        u = model(x, ma, XS3, PS3, ST3, pv, t["kappa"], t["wmin"],
                  t["wmax"], t["sq"])
        g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        lap = torch.autograd.grad(g.sum(), x, create_graph=True)[0].sum(-1)
        sc = t["sq"] / 9.0
        l_lap = ((lap / sc) ** 2).mean()
        xa = torch.cat([torch.linspace(3.2, 10, 96),
                        torch.linspace(-10, -3.2, 96),
                        torch.linspace(-1.5, 1.5, 96)]).double().to(device)
        xa_ = torch.stack([xa, torch.zeros_like(xa),
                           torch.zeros_like(xa)], dim=1)
        xa_.requires_grad_(True)
        ua = model(xa_, ma, XS3, PS3, ST3, pv, t["kappa"], t["wmin"],
                   t["wmax"], t["sq"])
        ga = torch.autograd.grad(ua.sum(), xa_, create_graph=True)[0]
        ua2 = torch.autograd.grad(ga[..., 0].sum(), xa_,
                                  create_graph=True)[0][..., 0]
        l_cx = (torch.relu(-ua2 / sc) ** 2).mean()
        return l_lap, l_cx

    def _pde_one(lb0, n_pts):
        """单配置的 Hamilton 约束残差项,返回标量 loss 或 None。

        真实残差
        --------
            R = Δu + S,   S = (1/8)·ψ^{-7}·K̄_ij K̄^ij,   ψ = ψ_sing + u

        与 smooth_w 的本质区别:后者惩罚 |Δu|²,而真解满足 Δu = −S 而非
        Δu = 0,故它把解往"调和"方向偏置,**并未使用物理信息**;此处惩罚
        的是 Hamilton 约束本身的残差。

        归一化 --pde-norm
        -----------------
        batch: sig = RMS_批次(S),全场单一标量(默认,旧行为)。
        rel  : sig_j = sqrt(S_shell(r0_j)² + (eps·S_ref)²),逐壳层。
               |S_shell| >> eps·S_ref 处即相对残差 (R/S)²;|S_shell| <<
               eps·S_ref 处退化为绝对残差 R/(eps·S_ref)。eps 即"远场放大
               上限"旋钮(上限 ≈ 1/eps²)。

        **实测:两者对远场的权重几乎相同。**
        pde_resid_profile.py 对 q10/v6a 用**真实残差** R 分解实测:
          · batch(eps=1): r0≥14 的远场已占 l_pde 的 98.97%;
          · 把 eps 从 1 调到 0.01(放大上限 1→1e4),远场份额仅由
            99.1% 变到 99.9%。
        原因是损失权重 ∝ w·R²,而 |R| 在远场**最大**(r0≥22 处 rms
        2.9e-4,r0≈10 处 5.7e-5),并非由源项 S 决定 ——
        S 虽然按 r^{-6} 衰减,但 Δu 的误差增长得更快。
        故 `batch` 并没有"忽略远场";远场精度上不去是收敛速度与批间
        方差的问题,不是权重问题。`rel` 保留为可选实验开关,默认关闭。
        """
        t = tens[lb0]
        ma = torch.tensor([0.5, t["m2"]], dtype=torch.float64, device=device)
        pv = A2.param_vec(t["q"], t["m2"], device)
        n_try = max(4 * n_pts, 512)
        xs_s = torch.from_numpy(
            rng.uniform(-args.pde_rmax, args.pde_rmax,
                        size=(n_try, 3))).double().to(device)
        rr = torch.minimum((xs_s - XA3).norm(dim=1),
                           (xs_s + XA3).norm(dim=1))
        r0 = xs_s.norm(dim=1)
        keep = (rr > args.pde_rho_min) & (r0 <= args.pde_rmax)
        x = xs_s[keep][:n_pts]
        if x.shape[0] < 64:
            return None
        x.requires_grad_(True)
        u = model(x, ma, XS3, PS3, ST3, pv, t["kappa"], t["wmin"],
                  t["wmax"], t["sq"])
        psi_s = physics.psi_sing(x, ma, XS3)
        kk = physics.bowen_york_KK(x, ma, XS3, PS3, ST3)
        R = physics.pde_residual(u, x, psi_s, kk)
        with torch.no_grad():
            psi = torch.clamp(psi_s + u, min=1e-4)
            S = (1.0 / 8.0) * kk / psi ** 7
            if args.pde_norm == "rel" and t.get("sprof") is not None:
                sp = torch.from_numpy(t["sprof"]).double().to(device)
                k = (torch.bucketize(x.norm(dim=1), _PDE_EDGES_T) - 1)
                k = k.clamp(0, sp.shape[0] - 1)
                S_sh = sp[k]
                sig = torch.sqrt(S_sh ** 2 + (args.pde_eps * t["sref"]) ** 2)
            else:
                sig = torch.sqrt(torch.mean(S ** 2))
                sig = sig.clamp(min=1e-3 * t["sq"] / 9.0, max=None)
            r0x = x.norm(dim=1)
            span = max(args.pde_r_end - args.pde_r_start, 1e-9)
            w = ((r0x - args.pde_r_start) / span).clamp(0.0, 1.0)
            if args.pde_p != 1.0:
                w = w ** args.pde_p
        # 加权平均(而非 mean):mean 会被 w=0 的点稀释,梯度传不到远场
        return (w * (R / sig) ** 2).sum() / w.sum().clamp(min=1e-12)

    def pde_loss():
        """在 --pde-cfgs 个随机配置上取残差项均值。

        单配置版本每步只更新 63 个训练配置中的一个 —— 每个配置平均要等
        63 步才拿到一次 PDE 监督。多配置(默认 8)把每配置的更新频率提高
        pde_cfgs 倍,且对配置求均值可显著压低 l_pde 的批间方差。
        """
        n_c = max(1, min(args.pde_cfgs, len(train_labels)))
        idx = rng.choice(len(train_labels), size=n_c, replace=False)
        per_n = max(args.pde_n // n_c, 64)
        vals = []
        for i in idx:
            v = _pde_one(train_labels[int(i)], per_n)
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return torch.stack(vals).mean()

    def save_ckpt(step, final=False):
        ck = {"step": step, "fingerprint": fp, "model": model.state_dict(),
              "opt": opt.state_dict(), "sch": sch.state_dict(),
              "hist": hist, "ema": ema, "rng": rng.bit_generator.state}
        tmp = ck_p + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ck_p)
        if final:
            meta = {}
            for lb in all_labels:
                d = cfgs[lb]
                meta[lb] = {k: float(d[k]) for k in
                            ("q", "m1", "m2", "kappa", "sq", "wmin", "wmax")}
            torch.save({"variant": args.variant,
                        "model_state": model.state_dict(),
                        "history": hist, "meta": meta,
                        "train_labels": train_labels,
                        "heldout_labels": sorted(heldout),
                        "steps_done": step, "args": vars(args)},
                       os.path.join(ck_dir, "model.pt"))
            json.dump(hist, open(os.path.join(ck_dir, "history.json"), "w"))

    t0 = time.time()
    log_every = max(1, args.steps // 60)
    for s in range(step0 + 1, args.steps + 1):
        model.train()
        opt.zero_grad()
        lr_ = ref_loss()
        total = ema_bal("L_ref", lr_)
        l_lap = l_cx = None
        if args.smooth_w > 0 or args.convex_w > 0:
            out_s = smooth_loss(
                train_labels[int(rng.integers(0, len(train_labels)))])
            if out_s is not None:
                l_lap, l_cx = out_s
                total = total + args.smooth_w * l_lap + args.convex_w * l_cx
        l_pde = None
        if args.pde_w > 0:
            l_pde = pde_loss()
            if l_pde is not None:
                # 与 L_ref 同样做 EMA 归一 —— 使 --pde-w 直接表示"相对 L_ref
                # 的权重占比"(否则 l_pde 的绝对量级随配置/阶段漂移,无法标定)
                total = total + args.pde_w * ema_bal("L_pde", l_pde)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sch.step()
        hist["L_ref"].append(float(lr_))
        hist["total"].append(float(total))
        if l_lap is not None:
            hist.setdefault("l_lap", []).append(float(l_lap))
            hist.setdefault("l_cx", []).append(float(l_cx))
        if l_pde is not None:
            hist.setdefault("l_pde", []).append(float(l_pde))
        if s % log_every == 0 or s == 1:
            msg = "[step %6d/%d] L_ref=%.4e" % (s, args.steps, float(lr_))
            if l_lap is not None:
                msg += " lap=%.2e cx=%.2e" % (float(l_lap), float(l_cx))
            if l_pde is not None:
                msg += " pde=%.3e" % float(l_pde)
            log.info(msg + " (%.0fs)", time.time() - t0)
        if s % 1000 == 0:
            save_ckpt(s)
    save_ckpt(args.steps, final=True)
    log.info("完成: %s 用时 %.1f min", ck_dir, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("训练失败")
        raise
