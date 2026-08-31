"""a2q_train.py —— A2 单参数 q∈[1,10] 攻关:统一训练脚本(多方案变体,快路径)。

残差计算(快路径,数学上与全图自动微分严格等价):
    u = κ·u_g·(1+φ), φ = w·ψ_θ(x)  (w=(u_g−wmin)/(wmax−wmin) 逐配置归一,ψ_θ 仅经 MLP)
    Δu = κ[Δu_g(1+φ) + 2∇u_g·∇φ + u_g·Δφ]
其中 u_g/∇u_g/Δu_g 已在 prep/prep2 预计算为常数,反传只穿透 MLP —— 比穿透
LZ2008 引导解析公式快数倍。noise 变体加噪步在线重算 u_g 及其导数。

变体(--variant):
  base       A2-0 基线:q∈[1,10] 15 配置 + 逐配置窗口 + log-q 参数 + 残差尺度归一;
             仅 q10 参考监督(课程 1000→100)
  operator   A2-1 神经算子 ansatz(ψ_θ=G_θ 读入 [log1p(|u_g|/sq), w],±3 有界,零初始化)
  refweight  A2-2 全部 15 配置参考残差监督 + 自适应权重(需 refsub 全套)
  rar        A2-3 残差主动采样:每 250 步采 4096 候选,|R| 前 1024 入池;
             内点批 = (8000−n_pool) 固定集随机 + n_pool 池点
  noise      A2-4 参数分级噪声:q̃=q·exp(σ_t·N(0,1)),σ_t 0→0.06(前 50% 步线性),
             概率 0.6 加噪/0.4 锚定;u_g 导数与 ψ_sing/KK 在线重算,κ 对数插值
  champion   A2-5 组合:基线 ansatz + 修正κ + 全配置参考监督(×3,λ 下限 0.3)+ 噪声
  opv2       A2-1 v2 神经算子:邻域 patch 真泛函输入(3 半径×8 方向 u_g 采样),
             champion 配方;修正场可表示随引导场局部几何变化的形状修正
  champion2  两阶段第二阶段:从 a2q_champion 权重续训,κ→κ*(kappa_star.json),
             每步全部 15 配置 ref 全量入损失(消除"3 配置均值稀释"与多损失干扰,
             见报告 §5.7);PDE 仍按 3 配置/步

通用:每步随机 --cfgs-per-step 个配置(默认 3),内点批 8000/配置(12000 固定集
随机子采样),边界批 4000,EMA(α=0.9)逐项平衡,Adam+Cosine,裁剪 10,
奇点正则 100·mean(u²)/sq²。断点续训 runs/<exp>/ckpt.pt(指纹+原子写,每 1000 步)。
"""
import argparse, hashlib, json, logging, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import sample_ball
from logutil import setup_logging
import physics
from a2q_prep import Q_TRAIN, Q_HELDOUT, DATA_DIR, R_MAX
from a2q_model import make_model, param_vec

log = logging.getLogger("paper.A2.a2q_train")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
TRAIN_LABELS = [lb for _, lb in Q_TRAIN]
W2, W_ROB, W_SING = 1.0, 1.0, 100.0


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.rng = np.random.default_rng(args.seed)
        self.ema = {}
        self.hist = {"L2": [], "LBC": [], "L_ref": [], "L_reg": [], "total": []}
        self.step0 = 0
        self.ref_w = {lb: 1.0 for lb in TRAIN_LABELS}
        self.ema_ref = {lb: None for lb in TRAIN_LABELS}
        self.pool = {lb: None for lb in TRAIN_LABELS}

        self.cfgs = {}
        for lb in TRAIN_LABELS:
            z = np.load(os.path.join(DATA_DIR, f"cfg_{lb}.npz"))
            self.cfgs[lb] = {k: z[k] for k in z.files}
            self.cfgs[lb]["label"] = lb
        self.hmeta = {}
        for lb in [lb for _, lb in Q_HELDOUT]:
            p = os.path.join(DATA_DIR, f"cfg_{lb}.npz")
            if os.path.exists(p):
                z = np.load(p)
                self.hmeta[lb] = {k: float(z[k]) for k in
                                  ("q", "m1", "m2", "kappa", "sq", "wmin", "wmax")}
        self.labels = TRAIN_LABELS
        q_lb = sorted((float(self.cfgs[lb]["q"]), lb) for lb in TRAIN_LABELS)
        self.anchor_logq = np.log([q for q, _ in q_lb])
        self.anchor_logk = np.log([float(self.cfgs[lb]["kappa"]) for _, lb in q_lb])
        if args.variant == "champion2":
            # κ→κ* 重标定(报告 §5.7b:QMC κ 系统性高 0.8~1.5%)
            ks_path = os.path.join(DATA_DIR, "kappa_star.json")
            if not os.path.exists(ks_path):
                log.error("champion2 需要 a2q_data/kappa_star.json(先运行 a2q_kappa_fit.py)")
                raise SystemExit(2)
            ks = json.load(open(ks_path))
            for lb in TRAIN_LABELS:
                self.cfgs[lb]["kappa"] = np.float64(ks[lb]["kappa_star_all"])
            q_lb2 = sorted((float(self.cfgs[lb]["q"]), lb) for lb in TRAIN_LABELS)
            self.anchor_logk = np.log([float(self.cfgs[lb]["kappa"])
                                       for _, lb in q_lb2])
            log.info("champion2: κ 已重标定为 κ*_all(κ_qmc 系统性偏高 0.8~1.5%)")

        need_ref = {"base": ["q10"], "operator": ["q10"], "rar": ["q10"],
                    "noise": ["q10"], "refweight": TRAIN_LABELS,
                    "champion": TRAIN_LABELS, "opv2": TRAIN_LABELS,
                    "champion2": TRAIN_LABELS}[args.variant]
        self.ref = {}
        for lb in need_ref:
            p = os.path.join(DATA_DIR, f"refsub_{lb}.npz")
            if os.path.exists(p):
                z = np.load(p)
                d = self.cfgs[lb]
                if "ug" in z.files:
                    ug = z["ug"]
                else:
                    ug = self._guide_vals(z["x"], d)
                self.ref[lb] = (z["x"], z["u"], ug.astype(np.float32))
            else:
                msg = f"refsub_{lb}.npz 缺失"
                if args.variant in ("refweight", "champion", "opv2", "champion2"):
                    log.error(f"{msg};该变体需要全套参考子采样,先运行 a2q_refsub.py")
                    raise SystemExit(2)
                log.warning(f"{msg};该配置参考监督禁用")

        self.model = make_model(args.variant, device)
        if args.variant == "champion2":
            src = os.path.join(RUNS, "a2q_champion", "model.pt")
            ck = torch.load(src, map_location=device, weights_only=False)
            self.model.load_state_dict(ck["model_state"])
            log.info(f"champion2: 从 {src} 加载 champion 权重(两阶段第二阶段)")
        log.info(f"参数量: {sum(p.numel() for p in self.model.parameters())}")
        self._pf_cache = {}   # opv2: (label, kind) → 预计算 patch 特征(固定点集)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=args.steps, eta_min=args.lr * 0.01)

    def _guide_vals(self, x, d):
        with torch.no_grad():
            xt = torch.from_numpy(x).double()
            return physics.guide_u(
                xt, torch.tensor(d["masses"]).double(),
                torch.tensor(d["xs"]).double(), torch.tensor(d["Ps"]).double(),
                torch.tensor(d["Ss"]).double()).numpy().astype(np.float32)

    # ---- 指纹 / 断点续训 ----
    def fingerprint(self):
        d = {"variant": self.args.variant, "steps": self.args.steps,
             "lr": self.args.lr, "cps": self.args.cfgs_per_step,
             "nis": self.args.n_int_step, "labels": self.labels,
             "seed": self.args.seed, "sigma": self.args.noise_sigma_max}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def try_resume(self):
        p = os.path.join(RUNS, self.args.exp_name, "ckpt.pt")
        if not os.path.exists(p):
            return False
        try:
            ck = torch.load(p, map_location=self.device, weights_only=False)
        except Exception as e:
            log.warning(f"检查点读取失败({e}),重头训练")
            return False
        if ck.get("fingerprint") != self.fingerprint():
            log.warning("检查点指纹不符(配置已变),重头训练")
            return False
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sch.load_state_dict(ck["sch"])
        self.ema = ck["ema"]; self.hist = ck["hist"]
        self.ref_w = ck.get("ref_w", self.ref_w)
        self.ema_ref = ck.get("ema_ref", self.ema_ref)
        self.step0 = ck["step"]
        self.rng.bit_generator.state = ck["rng"]
        log.info(f"[续训] 从 step {self.step0} 恢复")
        return True

    def save_ckpt(self, step, final=False):
        d = os.path.join(RUNS, self.args.exp_name)
        os.makedirs(d, exist_ok=True)
        ck = {"step": step, "fingerprint": self.fingerprint(),
              "model": self.model.state_dict(), "opt": self.opt.state_dict(),
              "sch": self.sch.state_dict(), "ema": self.ema, "hist": self.hist,
              "ref_w": self.ref_w, "ema_ref": self.ema_ref,
              "rng": self.rng.bit_generator.state}
        tmp = os.path.join(d, "ckpt.tmp")
        torch.save(ck, tmp)
        os.replace(tmp, os.path.join(d, "ckpt.pt"))
        if final:
            meta = {lb: {k: float(self.cfgs[lb][k]) for k in
                         ("q", "m1", "m2", "kappa", "sq", "wmin", "wmax")}
                    for lb in TRAIN_LABELS}
            meta.update(self.hmeta)
            torch.save({"variant": self.args.variant,
                        "model_state": self.model.state_dict(),
                        "history": self.hist, "meta": meta,
                        "train_labels": TRAIN_LABELS,
                        "heldout_labels": [lb for _, lb in Q_HELDOUT],
                        "steps_done": step, "args": vars(self.args)},
                       os.path.join(d, "model.pt"))
            json.dump(self.hist, open(os.path.join(d, "history.json"), "w"))

    # ---- 数值组件 ----
    def kappa_of_q(self, qv):
        return float(np.exp(np.interp(np.log(qv), self.anchor_logq, self.anchor_logk)))

    def _guide_parts_gpu(self, xt, ma, xs, Ps, Ss, want_lap=True):
        """在线计算 u_g 及其梯度(与可选 Laplacian),全部脱离计算图返回。"""
        xt = xt.detach().requires_grad_(True)
        ug = physics.guide_u(xt, ma, xs, Ps, Ss)
        g1 = torch.autograd.grad(ug.sum(), xt, create_graph=True)[0]
        if not want_lap:
            return ug.detach(), g1.detach(), None
        lap = torch.zeros_like(ug)
        for i in range(3):
            g2 = torch.autograd.grad(g1[:, i].sum(), xt,
                                     create_graph=False, retain_graph=True)[0]
            lap = lap + g2[:, i]
        return ug.detach(), g1.detach(), lap.detach()

    def _fast_residual(self, xi, p, k, sqv, ug_t, w_t, gug_t, lug_t, ps_t, kk_t,
                       wmin=None, wmax=None, feats=None):
        """返回 (R, u, phi)。xi 必须 requires_grad。

        φ = w·ψ,但 w=(u_g−wmin)/(wmax−wmin) 亦是 x 的函数:
          ∇φ = ∇w·ψ + w·∇ψ,  Δφ = Δw·ψ + 2∇w·∇ψ + w·Δψ
          ∇w = ∇u_g/span, Δw = Δu_g/span
        ψ 的导数仅经 MLP 自动微分;u_g/∇u_g/Δu_g 为预计算常数。
        feats:opv2 的 patch 特征(no-grad 预计算,不参与反传)。"""
        span = (wmax - wmin + 1e-8) if wmax is not None else 1.0
        u, phi, psi = self.model.forward_from_parts(xi, p, k, ug_t, w_t, sq=sqv,
                                                    feats=feats)
        gpsi = torch.autograd.grad(psi.sum(), xi, create_graph=True)[0]
        lappsi = torch.zeros_like(psi)
        for i in range(3):
            g2 = torch.autograd.grad(gpsi[:, i].sum(), xi, create_graph=True)[0]
            lappsi = lappsi + g2[:, i]
        gw = gug_t / span
        lw = lug_t / span
        gphi = gw * psi.unsqueeze(1) + w_t.unsqueeze(1) * gpsi
        lapphi = lw * psi + 2.0 * (gw * gpsi).sum(1) + w_t * lappsi
        lapu = lug_t * (1.0 + phi) + 2.0 * (gug_t * gphi).sum(1) + ug_t * lapphi
        psic = torch.clamp(ps_t + u, min=1e-4)
        return k * lapu + kk_t / (8.0 * psic ** 7), u, phi

    def _components(self, d, sigma=None):
        a = self.args
        dev = self.device
        sqv = float(d["sq"]); sq = sqv ** 2
        wmin, wmax = float(d["wmin"]), float(d["wmax"])
        k = float(d["kappa"])
        q_eff, m2_eff = float(d["q"]), float(d["m2"])
        ma = torch.from_numpy(d["masses"]).double().to(dev)
        xs = torch.from_numpy(d["xs"]).double().to(dev)
        Ps = torch.from_numpy(d["Ps"]).double().to(dev)
        Ss = torch.from_numpy(d["Ss"]).double().to(dev)
        if sigma is not None:
            q_eff = q_eff * float(np.exp(sigma * self.rng.normal()))
            m2_eff = 0.5 * q_eff
            k = self.kappa_of_q(q_eff)
            ma = torch.tensor([0.5, m2_eff], dtype=torch.float64, device=dev)
        p = param_vec(q_eff, m2_eff, dev)

        # ---- 内点批 ----
        pl = self.pool[d["label"]] if a.variant == "rar" else None
        if sigma is None and pl is not None:
            n_pool = min(3000, pl["x"].shape[0])
            sel = self.rng.choice(pl["x"].shape[0], n_pool, replace=False)
            fresh = self.rng.choice(d["x_int"].shape[0], a.n_int_step - n_pool,
                                    replace=False)
            xi_np = np.concatenate([d["x_int"][fresh], pl["x"][sel]], axis=0)
            ug_np = np.concatenate([d["ug_int"][fresh], pl["ug"][sel]], axis=0)
            gug_np = np.concatenate([d["grad_ug"][fresh], pl["gug"][sel]], axis=0)
            lug_np = np.concatenate([d["lap_ug"][fresh], pl["lug"][sel]], axis=0)
            ps_np = np.concatenate([d["ps_int"][fresh], pl["ps"][sel]], axis=0)
            kk_np = np.concatenate([d["kk_int"][fresh], pl["kk"][sel]], axis=0)
        else:
            idx = self.rng.choice(d["x_int"].shape[0], a.n_int_step, replace=False)
            xi_np = d["x_int"][idx]
            ps_np, kk_np = d["ps_int"][idx], d["kk_int"][idx]
            ug_np, gug_np, lug_np = (d["ug_int"][idx], d["grad_ug"][idx],
                                     d["lap_ug"][idx])

        xi = torch.from_numpy(np.ascontiguousarray(xi_np)).double().to(dev)
        xi.requires_grad_(True)
        if sigma is not None:
            ug_t, gug_t, lug_t = self._guide_parts_gpu(xi, ma, xs, Ps, Ss)
            with torch.no_grad():
                ps_t = physics.psi_sing(xi, ma, xs)
                kk_t = physics.bowen_york_KK(xi, ma, xs, Ps, Ss)
        else:
            ug_t = torch.from_numpy(np.ascontiguousarray(ug_np)).double().to(dev)
            gug_t = torch.from_numpy(np.ascontiguousarray(gug_np)).double().to(dev)
            lug_t = torch.from_numpy(np.ascontiguousarray(lug_np)).double().to(dev)
            ps_t = torch.from_numpy(np.ascontiguousarray(ps_np)).double().to(dev)
            kk_t = torch.from_numpy(np.ascontiguousarray(kk_np)).double().to(dev)
        w_t = (ug_t - wmin) / (wmax - wmin + 1e-8)

        # opv2 patch 特征:内点集固定(12000),一次预计算按步索引;
        # 噪声步的引导场在线重算 → patch 也在线算。
        pf = None
        if a.variant == "opv2":
            if sigma is None:
                pf = self._pf_cache.get((d["label"], "int"))
                if pf is None:
                    with torch.no_grad():
                        pf = self.model.patch_feats(
                            torch.from_numpy(d["x_int"]).double().to(dev),
                            ma, xs, Ps, Ss, sqv)
                    self._pf_cache[(d["label"], "int")] = pf
                pf = pf[idx]
            else:
                with torch.no_grad():
                    pf = self.model.patch_feats(xi.detach(), ma, xs, Ps, Ss, sqv)

        R, ui, phi = self._fast_residual(xi, p, k, sqv, ug_t, w_t, gug_t, lug_t,
                                         ps_t, kk_t, wmin, wmax, feats=pf)
        l2 = (R ** 2).mean() / sq

        # ---- 边界批(Robin) ----
        xb = torch.from_numpy(d["x_bnd"]).double().to(dev)
        xb.requires_grad_(True)
        if sigma is not None:
            ug_b, gug_b, _ = self._guide_parts_gpu(xb, ma, xs, Ps, Ss, want_lap=False)
        else:
            ug_b = torch.from_numpy(d["ug_bnd"]).double().to(dev)
            gug_b = torch.from_numpy(d["grad_ug_b"]).double().to(dev)
        span = wmax - wmin + 1e-8
        w_b = (ug_b - wmin) / span
        pfb = None
        if a.variant == "opv2":
            pfb = self._pf_cache.get((d["label"], "bnd"))
            if pfb is None:
                with torch.no_grad():
                    pfb = self.model.patch_feats(xb.detach(), ma, xs, Ps, Ss, sqv)
                self._pf_cache[(d["label"], "bnd")] = pfb
        ub, phib, psib = self.model.forward_from_parts(xb, p, k, ug_b, w_b,
                                                       sq=sqv, feats=pfb)
        gpsib = torch.autograd.grad(psib.sum(), xb, create_graph=True)[0]
        gwb = gug_b / span
        gphib = gwb * psib.unsqueeze(1) + w_b.unsqueeze(1) * gpsib
        gub = k * (gug_b * (1.0 + phib).unsqueeze(1) + ug_b.unsqueeze(1) * gphib)
        robin = ((xb * gub).sum(dim=1) + ub) / R_MAX
        lbc = (robin ** 2).mean() * (R_MAX ** 2) / sq

        r1 = (xi - xs[0]).norm(dim=1)
        r2 = (xi - xs[1]).norm(dim=1)
        near = (r1 < 0.3) | (r2 < 0.3)
        lreg = (ui[near] ** 2).mean() / sq if near.any() else torch.tensor(0.0, device=dev)

        lr = torch.tensor(0.0, device=dev)
        if d["label"] in self.ref:
            lr = self._ref_loss(d["label"], p, k, sqv, wmin, wmax)
        return l2, lbc, lreg, lr

    def _ref_loss(self, lb, p=None, k=None, sqv=None, wmin=None, wmax=None):
        """单配置 refsub 全量回归损失(供 _components 与 champion2 全配置监督复用)。"""
        a = self.args
        dev = self.device
        d = self.cfgs[lb]
        if p is None:
            k = float(d["kappa"]); sqv = float(d["sq"])
            wmin, wmax = float(d["wmin"]), float(d["wmax"])
            p = param_vec(float(d["q"]), float(d["m2"]), dev)
        rx, ru, rug = self.ref[lb]
        rxt = torch.from_numpy(rx).double().to(dev)
        rut = torch.from_numpy(ru).double().to(dev)
        ugt = torch.from_numpy(rug).double().to(dev)
        wt = (ugt - wmin) / (wmax - wmin + 1e-8)
        if a.variant == "opv2":
            raise ValueError("opv2 与 champion2/ref 全量监督不组合")
        up, _, _ = self.model.forward_from_parts(rxt, p, k, ugt, wt, sq=sqv)
        return ((up - rut) ** 2).sum() / ((rut ** 2).sum() + 1e-30)

    def _ema(self, n, v):
        fv = float(v)
        if n not in self.ema:
            self.ema[n] = fv
        else:
            self.ema[n] = 0.9 * self.ema[n] + 0.1 * fv
        return v / (self.ema[n] + 1e-12)

    def _rar_refresh(self, step, labels):
        dev = self.device
        self.model.double()
        for lb in labels:
            d = self.cfgs[lb]
            cand = sample_ball(4096, R_MAX, self.rng).astype(np.float32)
            ct = torch.from_numpy(cand).double().to(dev)
            ct.requires_grad_(True)
            ma = torch.from_numpy(d["masses"]).double().to(dev)
            xs = torch.from_numpy(d["xs"]).double().to(dev)
            Ps = torch.from_numpy(d["Ps"]).double().to(dev)
            Ss = torch.from_numpy(d["Ss"]).double().to(dev)
            with torch.no_grad():
                ps = physics.psi_sing(ct, ma, xs)
                kk = physics.bowen_york_KK(ct, ma, xs, Ps, Ss)
            ug, gug, lug = self._guide_parts_gpu(ct, ma, xs, Ps, Ss)
            sqv = float(d["sq"]); wmin, wmax = float(d["wmin"]), float(d["wmax"])
            wt = (ug - wmin) / (wmax - wmin + 1e-8)
            p = param_vec(float(d["q"]), float(d["m2"]), dev)
            R, _, _ = self._fast_residual(ct, p, float(d["kappa"]), sqv,
                                          ug, wt, gug, lug, ps, kk,
                                          float(d["wmin"]), float(d["wmax"]))
            R = R.detach().abs()
            top = torch.topk(R.squeeze(-1), 1024).indices.cpu().numpy()
            sl = np.s_[top]
            new = dict(x=cand[top].astype(np.float32),
                       ug=ug.cpu().numpy()[sl].astype(np.float32),
                       gug=gug.cpu().numpy()[sl].astype(np.float32),
                       lug=lug.cpu().numpy()[sl].astype(np.float32),
                       ps=ps.cpu().numpy()[sl].astype(np.float32),
                       kk=kk.cpu().numpy()[sl].astype(np.float32))
            old = self.pool[lb]
            if old is not None:
                for kk2 in new:
                    new[kk2] = np.concatenate([old[kk2], new[kk2]], axis=0)[-4096:]
            self.pool[lb] = new
        self.model.float()

    def step(self, step):
        a = self.args
        self.model.double()
        self.opt.zero_grad()
        picks = self.rng.choice(len(self.labels), a.cfgs_per_step, replace=False)
        sigma = None
        prog = step / a.steps
        if a.variant in ("noise", "champion", "opv2") and self.rng.random() < 0.6:
            sigma = a.noise_sigma_max * min(1.0, prog / 0.5)
        comps = [self._components(self.cfgs[self.labels[i]], sigma) for i in picks]
        l2m = torch.stack([c[0] for c in comps]).mean()
        lbcm = torch.stack([c[1] for c in comps]).mean()
        regm = torch.stack([c[2] for c in comps]).mean()
        ref_terms = []
        for i, c in zip(picks, comps):
            lb = self.labels[i]
            if c[3].requires_grad or float(c[3]) > 0:
                ref_terms.append(self.ref_w[lb] * c[3])
                if a.variant in ("refweight", "champion", "opv2", "champion2"):
                    e = self.ema_ref[lb]
                    v = float(c[3])
                    self.ema_ref[lb] = v if e is None else 0.99 * e + 0.01 * v
        if a.variant == "champion2":
            # 全 15 配置 ref 全量监督(§5.7c:每步仅 3 配置会被 PDE 均值梯度淹没)
            all_terms = []
            for lb in self.labels:
                c2 = self._ref_loss(lb)
                all_terms.append(self.ref_w[lb] * c2)
                e = self.ema_ref[lb]
                v = float(c2)
                self.ema_ref[lb] = v if e is None else 0.99 * e + 0.01 * v
            refm = torch.stack(all_terms).mean()
        else:
            refm = (torch.stack(ref_terms).mean() if ref_terms
                    else torch.tensor(0.0, device=self.device))
        if a.variant in ("champion", "opv2", "champion2"):
            refm = refm * 3.0   # 对抗均值稀释(基线对单配置是全强度)
        lam = (1000.0 * max(0.3, 1.0 - 0.3 * prog)
               if a.variant in ("champion", "opv2", "champion2")
               else 1000.0 * max(0.1, 1.0 - 0.5 * prog))
        total = W2 * self._ema("L2", l2m) + W_ROB * self._ema("LBC", lbcm) \
            + lam * refm + W_SING * regm
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step()
        self.sch.step()
        self.model.float()
        if a.variant in ("refweight", "champion", "opv2") and step % 50 == 0:
            e = np.array([self.ema_ref[lb] if self.ema_ref[lb] is not None else 1.0
                          for lb in self.labels])
            w = np.sqrt(np.maximum(e, 1e-30))
            w = np.clip(w / max(w.mean(), 1e-30), 0.25, 4.0)
            for lb, wv in zip(self.labels, w):
                self.ref_w[lb] = float(wv)
        if a.variant == "rar" and step % a.rar_every == 0:
            self._rar_refresh(step, [self.labels[i] for i in picks])
        o = {"L2": float(l2m), "LBC": float(lbcm), "L_ref": float(refm),
             "L_reg": float(regm), "total": float(total)}
        for kk, v in o.items():
            self.hist[kk].append(v)
        return o


def main():
    setup_logging("A2", "a2q_train")
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base",
                    choices=["base", "operator", "refweight", "rar", "noise",
                             "champion", "opv2", "champion2"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cfgs-per-step", type=int, default=3)
    ap.add_argument("--n-int-step", type=int, default=8000)
    ap.add_argument("--noise-sigma-max", type=float, default=0.06)
    ap.add_argument("--rar-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exp-name", default=None)
    args = ap.parse_args()
    if args.exp_name is None:
        args.exp_name = f"a2q_{args.variant}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto"
                          else "cpu")
    log.info(f"[{args.exp_name}] 变体={args.variant} 设备={device} 步数={args.steps}")
    tr = Trainer(args, device)
    if not tr.try_resume():
        os.makedirs(os.path.join(RUNS, args.exp_name), exist_ok=True)
    t0 = time.time()
    log_every = max(1, args.steps // 50)
    for s in range(tr.step0 + 1, args.steps + 1):
        o = tr.step(s)
        if s % log_every == 0 or s == 1:
            log.info(f"[step {s:6d}/{args.steps}] L2={o['L2']:.3e} LBC={o['LBC']:.3e} "
                     f"L_ref={o['L_ref']:.3e} L_reg={o['L_reg']:.3e} "
                     f"total={o['total']:.3e} ({time.time()-t0:.0f}s)")
        if s % 1000 == 0:
            tr.save_ckpt(s)
    tr.save_ckpt(args.steps, final=True)
    log.info(f"完成: {os.path.join(RUNS, args.exp_name)} 用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
