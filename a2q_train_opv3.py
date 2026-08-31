"""a2q_train_opv3.py —— 算子 v3 训练(DeepONet 式 branch-trunk 自由场修正)。

设计要点(见 paper/reports/A2_opv3_设计_20260831.md §2):
  - ansatz: u = κ·u_g + Δ_θ, Δ = Σ b_k(u_g 传感器; p)·t_k(x)·χ(x),
    branch 末层零初始化(从引导解出发),加性自由场 → 远场不失杠杆;
  - κ 用 kappa_star.json 的 κ*_all(15 配置),留出用 κ*(q) 对数线性插值;
  - 损失主项 = 每步全 15 配置 refsub 全量(10240 点/配置)监督回归;
  - PDE+Robin 物理正则:每 pde_every 步一步 3 配置快路径,λ_pde 课程
    0(前 warm 步)→ 0.1;
  - 参数噪声 q̃=q·exp(σ_t ξ)(σ≤0.04)增强留出泛化。

用法:
  python a2q_train_opv3.py --steps 15000 --exp-name a2q_opv3
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

log = logging.getLogger("paper.A2.a2q_train_opv3")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "data", "runs", "a2")
# 仓库化后的数据目录(本地大文件,gitignore):优先 repo 根 data/datasets/a2q_data
DATA_DIR = os.path.join(HERE, "data", "datasets", "a2q_data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(HERE, "a2q_data")
TRAIN_LABELS = [lb for _, lb in Q_TRAIN]


def main():
    setup_logging("A2", "a2q_train_opv3")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pde-every", type=int, default=5)
    ap.add_argument("--pde-warm", type=int, default=1000)
    ap.add_argument("--lambda-pde", type=float, default=0.1)
    ap.add_argument("--n-int-step", type=int, default=6000)
    ap.add_argument("--noise-sigma-max", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exp-name", default="a2q_opv3")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "auto" else "cpu")
    log.info(f"[{args.exp_name}] opv3 设备={device} 步数={args.steps}")
    rng = np.random.default_rng(args.seed)

    # ---- 数据 ----
    ks_path = os.path.join(DATA_DIR, "kappa_star.json")
    if not os.path.exists(ks_path):
        log.error("opv3 需要 a2q_data/kappa_star.json(先运行 a2q_kappa_fit.py)")
        raise SystemExit(2)
    ks = json.load(open(ks_path))
    cfgs, refs = {}, {}
    for lb in TRAIN_LABELS:
        z = np.load(os.path.join(DATA_DIR, f"cfg_{lb}.npz"))
        d = {k: z[k] for k in z.files}
        d["kappa"] = np.float64(ks[lb]["kappa_star_all"])
        cfgs[lb] = d
        zr = np.load(os.path.join(DATA_DIR, f"refsub_{lb}.npz"))
        refs[lb] = dict(x=zr["x"], u=zr["u"])
    # ---- 扩展:tq 配置(opv3 大数据集,q∈[1,100] 补密,κ*_spec 在 cfg 内) ----
    import glob
    tq = sorted(glob.glob(os.path.join(DATA_DIR, "cfg_tq*.npz")))
    for p in tq:
        lb = os.path.basename(p)[4:-4]
        z = np.load(p)
        cfgs[lb] = {k: z[k] for k in z.files}
        rp = os.path.join(DATA_DIR, f"refsub_{lb}.npz")
        if not os.path.exists(rp):
            log.error(f"cfg_{lb} 存在但 refsub_{lb} 缺失,先运行 post_refs_opv3.py")
            raise SystemExit(2)
        zr = np.load(rp)
        refs[lb] = dict(x=zr["x"], u=zr["u"])
    if tq:
        log.info(f"扩展数据集: +{len(tq)} 个 tq 配置(共 {len(cfgs)} 配置)")
    # κ*(q) 曲线(留出插值)
    q_arr = np.array([float(cfgs[lb]["q"]) for lb in TRAIN_LABELS])
    k_arr = np.array([float(cfgs[lb]["kappa"]) for lb in TRAIN_LABELS])
    o = np.argsort(q_arr)
    q_arr, k_arr = q_arr[o], k_arr[o]

    hmeta = {}
    for lb in [lb for _, lb in Q_HELDOUT]:
        p = os.path.join(DATA_DIR, f"cfg_{lb}.npz")
        if os.path.exists(p):
            z = np.load(p)
            q = float(z["q"])
            hmeta[lb] = {k: float(z[k]) for k in
                         ("q", "m1", "m2", "kappa", "sq", "wmin", "wmax")}
            hmeta[lb]["kappa_star"] = float(np.interp(np.log(q), np.log(q_arr),
                                                      k_arr))

    model = make_model("opv3", device).double()
    log.info(f"参数量: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                     eta_min=args.lr * 0.01)

    # ---- 指纹/续训 ----
    fp = hashlib.sha1(json.dumps(
        {"v": "opv3", "steps": args.steps, "lr": args.lr, "seed": args.seed,
         "pde_every": args.pde_every, "lam": args.lambda_pde},
        sort_keys=True).encode()).hexdigest()
    ck_dir = os.path.join(RUNS, args.exp_name)
    os.makedirs(ck_dir, exist_ok=True)
    step0 = 0
    hist = {"L_ref": [], "L_pde": [], "L_bc": [], "total": []}
    ema = {}
    ck_p = os.path.join(ck_dir, "ckpt.pt")
    if os.path.exists(ck_p):
        try:
            ck = torch.load(ck_p, map_location=device, weights_only=False)
            if ck.get("fingerprint") == fp:
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sch.load_state_dict(ck["sch"])
                hist, ema = ck["hist"], ck["ema"]
                step0, rng_state = ck["step"], ck["rng"]
                rng.bit_generator.state = rng_state
                log.info(f"[续训] 从 step {step0}")
            else:
                log.warning("指纹不符,重头训练")
        except Exception as e:
            log.warning(f"ckpt 读取失败({e})")

    def ema_bal(n, v):
        fv = float(v)
        ema[n] = fv if n not in ema else 0.9 * ema[n] + 0.1 * fv
        return v / (ema[n] + 1e-12)

    def ref_loss_all():
        per = []
        with torch.no_grad():
            pass
        for lb in TRAIN_LABELS:
            d, r = cfgs[lb], refs[lb]
            dev = device
            x = torch.from_numpy(r["x"]).double().to(dev)
            ur = torch.from_numpy(r["u"]).double().to(dev)
            ma = torch.tensor([0.5, float(d["m2"])], dtype=torch.float64,
                              device=dev)
            xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                              device=dev)
            Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                              dtype=torch.float64, device=dev)
            St = torch.zeros((2, 3), dtype=torch.float64, device=dev)
            pv = param_vec(float(d["q"]), float(d["m2"]), dev)
            u = model(x, ma, xs, Ps, St, pv, float(d["kappa"]),
                      float(d["wmin"]), float(d["wmax"]), float(d["sq"]))
            per.append(((u - ur) ** 2).sum() / ((ur ** 2).sum() + 1e-30))
        return torch.stack(per).mean()

    def pde_loss(labels):
        """3 配置 PDE+Robin 快路径(在线算 u_g 导数; Δ 项对 x 自动微分)。"""
        dev = device
        l2s, lbs_ = [], []
        picks = rng.choice(len(labels), 3, replace=False)
        for i in picks:
            d = cfgs[labels[i]]
            q_eff = float(d["q"]); m2_eff = float(d["m2"])
            k = float(d["kappa"]); sqv = float(d["sq"]); sq = sqv ** 2
            ma = torch.tensor([0.5, m2_eff], dtype=torch.float64, device=dev)
            xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                              device=dev)
            Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                              dtype=torch.float64, device=dev)
            St = torch.zeros((2, 3), dtype=torch.float64, device=dev)
            pv = param_vec(q_eff, m2_eff, dev)
            idx = rng.choice(d["x_int"].shape[0], args.n_int_step, replace=False)
            xi = torch.from_numpy(
                np.ascontiguousarray(d["x_int"][idx])).double().to(dev)
            xi.requires_grad_(True)
            ug = physics.guide_u(xi, ma, xs, Ps, St)
            ps = physics.psi_sing(xi, ma, xs)
            kk = physics.bowen_york_KK(xi, ma, xs, Ps, St)
            u = model(xi, ma, xs, Ps, St, pv, k, float(d["wmin"]),
                      float(d["wmax"]), sqv)
            g1 = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
            lap = torch.zeros_like(u)
            for c in range(3):
                g2 = torch.autograd.grad(g1[:, c].sum(), xi,
                                         create_graph=True)[0]
                lap = lap + g2[:, c]
            psic = torch.clamp(ps + u, min=1e-4)
            R = k * lap + kk / (8.0 * psic ** 7)
            l2s.append((R ** 2).mean() / sq)
            xb = torch.from_numpy(d["x_bnd"]).double().to(dev)
            xb.requires_grad_(True)
            ub = model(xb, ma, xs, Ps, St, pv, k, float(d["wmin"]),
                       float(d["wmax"]), sqv)
            gb = torch.autograd.grad(ub.sum(), xb, create_graph=True)[0]
            robin = ((xb * gb).sum(dim=1) + ub) / R_MAX
            lbs_.append((robin ** 2).mean() * R_MAX ** 2 / sq)
        return torch.stack(l2s).mean(), torch.stack(lbs_).mean()

    def save_ckpt(step, final=False):
        ck = {"step": step, "fingerprint": fp, "model": model.state_dict(),
              "opt": opt.state_dict(), "sch": sch.state_dict(),
              "hist": hist, "ema": ema, "rng": rng.bit_generator.state}
        tmp = ck_p + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ck_p)
        if final:
            meta = {lb: {k: float(cfgs[lb][k]) for k in
                         ("q", "m1", "m2", "kappa", "sq", "wmin", "wmax")}
                    for lb in TRAIN_LABELS}
            meta.update(hmeta)
            torch.save({"variant": "opv3", "model_state": model.state_dict(),
                        "history": hist, "meta": meta,
                        "train_labels": TRAIN_LABELS,
                        "heldout_labels": [lb for _, lb in Q_HELDOUT],
                        "steps_done": step, "args": vars(args)},
                       os.path.join(ck_dir, "model.pt"))
            json.dump(hist, open(os.path.join(ck_dir, "history.json"), "w"))

    t0 = time.time()
    log_every = max(1, args.steps // 50)
    for s in range(step0 + 1, args.steps + 1):
        model.train()
        opt.zero_grad()
        lr_ = ref_loss_all()
        prog = s / args.steps
        if prog < args.pde_warm / args.steps:
            lam = 0.0
        else:
            lam = args.lambda_pde * min(1.0, (prog - args.pde_warm / args.steps)
                                        / max(1e-9, 1 - args.pde_warm / args.steps))
        if lam > 0 and s % args.pde_every == 0:
            l2, lbc = pde_loss(TRAIN_LABELS)
            total = ema_bal("L_ref", lr_) + lam * (ema_bal("L2", l2)
                                                   + ema_bal("LBC", lbc))
        else:
            l2 = lbc = torch.tensor(0.0, device=device)
            total = ema_bal("L_ref", lr_)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sch.step()
        hist["L_ref"].append(float(lr_))
        hist["L_pde"].append(float(l2))
        hist["L_bc"].append(float(lbc))
        hist["total"].append(float(total))
        if s % log_every == 0 or s == 1:
            log.info(f"[step {s:6d}/{args.steps}] L_ref={float(lr_):.4e} "
                     f"L_pde={float(l2):.3e} lam={lam:.3f} "
                     f"({time.time()-t0:.0f}s)")
        if s % 1000 == 0:
            save_ckpt(s)
    save_ckpt(args.steps, final=True)
    log.info(f"完成: {ck_dir} 用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
