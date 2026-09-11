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
    ap.add_argument("--heldout", default="q15,q25,q50,q74,q86")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--init-from", default=None,
                    help="从已有 run 目录的 model.pt 载入权重后继续训练(微调)")
    ap.add_argument("--hidden-neurons", type=int, default=128)
    ap.add_argument("--n-basis", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "auto" else "cpu")
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
         "sn": args.smooth_n},
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
            sq=float(d["sq"]), wmin=float(d["wmin"]), wmax=float(d["wmax"]))

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
            per.append((w * r2).sum() / ((w * ur ** 2).sum() + 1e-30))
        return torch.stack(per).mean()

    XS3 = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]],
                       dtype=torch.float64, device=device)
    PS3 = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                       dtype=torch.float64, device=device)
    ST3 = torch.zeros((2, 3), dtype=torch.float64, device=device)
    XA3 = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64, device=device)

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
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sch.step()
        hist["L_ref"].append(float(lr_))
        hist["total"].append(float(total))
        if l_lap is not None:
            hist.setdefault("l_lap", []).append(float(l_lap))
            hist.setdefault("l_cx", []).append(float(l_cx))
        if s % log_every == 0 or s == 1:
            if l_lap is not None:
                log.info("[step %6d/%d] L_ref=%.4e lap=%.2e cx=%.2e (%.0fs)",
                         s, args.steps, float(lr_), float(l_lap),
                         float(l_cx), time.time() - t0)
            else:
                log.info("[step %6d/%d] L_ref=%.4e (%.0fs)", s, args.steps,
                         float(lr_), time.time() - t0)
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
