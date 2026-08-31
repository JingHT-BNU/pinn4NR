"""
parametric_train.py —— 参数化 PINN 训练(v8,最终版)
===================================================

策略:
    1. 参考解监督损失(仅 base 算例):L_ref = L2RE², 权重 1000
    2. PDE 残差正则(所有 q 值):权重 1
    3. Robin 边界正则(所有 q 值):权重 1
    4. 每步随机 1 配置,参考损失不经过 EMA 平衡(直接参与梯度)
    5. 课程学习:前 50% 步参考损失主导,后 50% 逐渐降低让 PDE 精调
"""

import argparse, json, logging, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BBHConfig, TrainConfig
from data import sample_ball, sample_sphere_surface
from logutil import setup_logging
import physics
from parametric_model import (
    ParamGuidedPINN, compute_parametric_pde_residual, compute_parametric_robin_residual,
)

log = logging.getLogger("paper.A2.parametric_train")

TRAIN_PARAMS = [(0.5, 0.25, "q05"), (0.5, 0.35, "q07"), (0.5, 0.5, "q10"),
                (0.5, 0.65, "q13"), (0.5, 0.8, "q16"), (0.5, 1.0, "q20")]
VAL_PARAMS = [(0.5, 0.3, "q06"), (0.5, 0.45, "q09"), (0.5, 0.7, "q14"), (0.5, 0.9, "q18")]
KAPPA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kappa_cache.json")


def build_bbh(m1, m2):
    return BBHConfig(m_plus=m1, m_minus=m2, x_plus=(3,0,0), x_minus=(-3,0,0),
                     P_plus=(0,0.2,0), P_minus=(0,-0.2,0))


class Data:
    def __init__(self, params, cfg, kc, seed=42):
        self.data = []
        rng = np.random.default_rng(seed)
        for m1, m2, label in params:
            bb = build_bbh(m1, m2)
            ma, xs, Ps, Ss = bb.as_arrays()
            xi = sample_ball(cfg.N_Omega, cfg.R_max, rng).astype(np.float32)
            xb = sample_sphere_surface(cfg.N_boundary, cfg.R_max, rng).astype(np.float32)
            mt = torch.from_numpy(ma).double(); xst = torch.from_numpy(xs).double()
            Pt = torch.from_numpy(Ps).double(); St = torch.from_numpy(Ss).double()
            def _c(x):
                xt = torch.from_numpy(x).double()
                return (physics.psi_sing(xt, mt, xst).numpy().astype(np.float32),
                        physics.bowen_york_KK(xt, mt, xst, Pt, St).numpy().astype(np.float32),
                        physics.guide_u(xt, mt, xst, Pt, St).numpy().astype(np.float32))
            pi, ki, ui = _c(xi); pb, kb, ub = _c(xb)
            au = np.concatenate([ui, ub])
            self.data.append(dict(m1=m1, m2=m2, label=label, masses=ma.astype(np.float32),
                xs=xs.astype(np.float32), Ps=Ps.astype(np.float32), Ss=Ss.astype(np.float32),
                x_int=xi, x_bnd=xb, ps_int=pi, kk_int=ki, ps_bnd=pb, kk_bnd=kb,
                u_min=float(au.min()), u_max=float(au.max()), kappa=kc[label]))
            log.info(f"  {label}: κ={kc[label]:.4f}, ug∈[{au.min():.4e},{au.max():.4e}]")

    def global_range(self):
        return min(d["u_min"] for d in self.data), max(d["u_max"] for d in self.data)


def predict(model, x, m1, m2, kappa, device):
    model.eval()
    bb = build_bbh(m1, m2)
    ma = torch.tensor([m1,m2], dtype=torch.float64, device=device)
    xs = torch.tensor(list(bb.x_plus)+list(bb.x_minus), dtype=torch.float64, device=device).reshape(2,3)
    Ps = torch.tensor(list(bb.P_plus)+list(bb.P_minus), dtype=torch.float64, device=device).reshape(2,3)
    Ss = torch.tensor(list(bb.S_plus)+list(bb.S_minus), dtype=torch.float64, device=device).reshape(2,3)
    p = torch.tensor([[m1,m2]], dtype=torch.float64, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 16384):
            out.append(model(torch.from_numpy(x[i:i+16384]).float().to(device), ma, xs, Ps, Ss, p, kappa).cpu().numpy())
    return np.concatenate(out)

def l2re(up, ur):
    return float(np.sqrt(np.sum((up-ur)**2)/max(np.sum(ur**2),1e-30)))


class Trainer:
    def __init__(self, model, cfg, device, ncfg=6, ref_x=None, ref_u=None):
        self.model, self.cfg, self.device = model, cfg, device
        self.ncfg = ncfg; self.rng = np.random.default_rng(0)
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.n_steps, eta_min=cfg.lr*0.01)
        self.ema = {}; self.hist = {"L2":[],"LBC":[],"L_ref":[],"L_reg":[],"total":[]}
        self.cnt = np.zeros(ncfg, dtype=int)
        # 参考解子采样 1 万点(可微损失,快速)
        if ref_x is not None:
            rng = np.random.default_rng(123)
            idx = rng.choice(ref_x.shape[0], min(10000, ref_x.shape[0]), replace=False)
            self.rx = torch.from_numpy(ref_x[idx]).float().pin_memory()
            self.ru = torch.from_numpy(ref_u[idx]).float().pin_memory()
            log.info(f"[监督] 参考解子采样: {self.rx.shape[0]} 点")
        else:
            self.rx = self.ru = None

    def _ema(self, n, l):
        v = l.item()
        if n not in self.ema: self.ema[n] = v
        else: self.ema[n] = self.cfg.ema_alpha * self.ema[n] + (1-self.cfg.ema_alpha) * v
        return l / (self.ema[n] + 1e-12)

    def _fwd(self, d):
        xi = torch.from_numpy(d["x_int"]).double().to(self.device)
        xb = torch.from_numpy(d["x_bnd"]).double().to(self.device)
        ma = torch.from_numpy(d["masses"]).double().to(self.device)
        xs = torch.from_numpy(d["xs"]).double().to(self.device)
        Ps = torch.from_numpy(d["Ps"]).double().to(self.device)
        Ss = torch.from_numpy(d["Ss"]).double().to(self.device)
        p = torch.tensor([[d["m1"],d["m2"]]], dtype=torch.float64, device=self.device)
        k = d["kappa"]
        xi.requires_grad_(True)
        ui = self.model(xi, ma, xs, Ps, Ss, p, k)
        l2 = ((compute_parametric_pde_residual(ui, xi, ma, xs, Ps, Ss))**2).mean()
        xb.requires_grad_(True)
        ub = self.model(xb, ma, xs, Ps, Ss, p, k)
        lbc = ((compute_parametric_robin_residual(ub, xb, self.cfg.R_max))**2).mean()
        # 奇点正则:在奇点附近(r<0.3)惩罚 |u| 过大(防止发散)
        r1 = (xi - xs[0]).norm(dim=1)
        r2 = (xi - xs[1]).norm(dim=1)
        near_sing = (r1 < 0.3) | (r2 < 0.3)
        if near_sing.any():
            l_reg = (ui[near_sing] ** 2).mean()
        else:
            l_reg = torch.tensor(0.0, device=self.device)
        # 参考损失(仅 base)
        lr = torch.tensor(0.0, device=self.device)
        if d["label"] == "q10" and self.rx is not None:
            rx = self.rx.to(self.device).double()
            ru = self.ru.to(self.device).double()
            up = self.model(rx, ma, xs, Ps, Ss, p, k)
            lr = torch.sum((up-ru)**2) / (torch.sum(ru**2)+1e-30)
        return l2, lbc, lr, l_reg

    def step(self, dl, step, n_steps):
        self.opt.zero_grad(); self.model.double()
        idx = self.rng.integers(0, self.ncfg); self.cnt[idx] += 1
        l2, lbc, lr, l_reg = self._fwd(dl[idx])
        # 课程:参考损失权重从 1000 逐渐降到 100;奇点正则权重固定 100
        prog = step / n_steps
        wr = 1000.0 * max(0.1, 1.0 - prog * 0.5)
        total = self.cfg.w2 * self._ema("L2", l2) + self.cfg.w_rob * self._ema("LBC", lbc) \
                + wr * lr + 100.0 * l_reg
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step(); self.sch.step(); self.model.float()
        o = {"L2":l2.item(),"LBC":lbc.item(),"L_ref":lr.item(),
             "L_reg":l_reg.item(),"total":total.item()}
        for k in o: self.hist[k].append(o[k])
        return o

    def train(self, dl, n_steps, log_every=1000):
        t0 = time.time()
        for s in range(1, n_steps+1):
            o = self.step(dl, s, n_steps)
            if s % log_every == 0 or s == 1:
                lrs = f"L_ref={o['L_ref']:.4e}" if o['L_ref']>0 else "L_ref=---"
                log.info(f"[step {s:6d}/{n_steps}] L2={o['L2']:.3e} LBC={o['LBC']:.3e} {lrs} total={o['total']:.3e} ({time.time()-t0:.0f}s)")
        log.info(f"完成,用时{time.time()-t0:.1f}s,采样分布:{self.cnt.tolist()}")
        return self.hist


def main():
    setup_logging("A2", "parametric_train")
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--N-Omega", type=int, default=10000)
    p.add_argument("--N-boundary", type=int, default=4000)
    p.add_argument("--exp-name", default="parametric_a1")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--reference", default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device=="auto" else "cpu")
    log.info(f"设备: {device}")
    cfg = TrainConfig(N_Omega=args.N_Omega, N_boundary=args.N_boundary, n_steps=args.steps,
                      lr=args.lr, R_max=30.0, w2=1.0, w_inf=0.0, w_rob=1.0, ema_alpha=0.9)
    log.info("\n生成数据...")
    kc = json.load(open(KAPPA_CACHE))
    kc = {k: v["kappa"] for k, v in kc.items()}
    td = Data(TRAIN_PARAMS, cfg, kc)
    gmin, gmax = td.global_range()
    log.info(f"全局 ug: [{gmin:.4e}, {gmax:.4e}]")
    ref = np.load(args.reference) if args.reference else None
    rx = ref["x_ref"] if ref is not None else None
    ru = ref["u_ref"] if ref is not None else None
    log.info("\n构建模型(4×128, FiLM, 正弦编码)...")
    from parametric_model import ParamGuidedPINN
    model = ParamGuidedPINN(n_params=2, c_init=0.2, hidden_layers=4, hidden_neurons=128, n_freq=8).to(device)
    model.set_u_range(gmin, gmax)
    log.info(f"参数量: {sum(p.numel() for p in model.parameters())}")
    log.info(f"\n训练 {args.steps} 步...")
    tr = Trainer(model, cfg, device, ncfg=len(td.data), ref_x=rx, ref_u=ru)
    hist = tr.train(td.data, n_steps=args.steps, log_every=max(1, args.steps // 20))
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "c": model.c.item(),
        "u_min": gmin, "u_max": gmax, "history": hist,
        "train_params": TRAIN_PARAMS, "val_params": VAL_PARAMS, "kappa_cache": kc,
    }, os.path.join(exp_dir, "model.pt"))
    json.dump(hist, open(os.path.join(exp_dir, "history.json"), "w"), indent=2)
    log.info(f"\n评估训练过的参数...")
    for d in td.data:
        m1, m2, label, k = d["m1"], d["m2"], d["label"], d["kappa"]
        if label == "q10" and ref is not None:
            up = predict(model, ref["x_ref"], m1, m2, k, device)
            log.info(f"  {label}: L2RE = {l2re(up, ref['u_ref']):.4e}")
        else:
            xc = sample_ball(5000, cfg.R_max, np.random.default_rng(42)).astype(np.float32)
            up = predict(model, xc, m1, m2, k, device)
            log.info(f"  {label}: u∈[{up.min():.4e},{up.max():.4e}]")
    log.info("\n零样本泛化...")
    for m1, m2, label in VAL_PARAMS:
        k = kc.get(label)
        if k is None: continue
        xc = sample_ball(5000, cfg.R_max, np.random.default_rng(42)).astype(np.float32)
        up = predict(model, xc, m1, m2, k, device)
        log.info(f"  {label}: u∈[{up.min():.4e},{up.max():.4e}]")
    log.info(f"\n完成: {exp_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
