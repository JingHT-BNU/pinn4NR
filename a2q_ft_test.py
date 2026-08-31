"""a2q_ft_test.py —— 决定性小实验:纯参考回归微调(champion 权重 → 15 配置全监督)。

动机(报告 §5.6 诊断):base q10(单配置监督)远场 L2RE 5.3e-3 —— ansatz 能表达
远场修正;champion(15 配置分摊,每步仅 3 配置有 ref 梯度)L_ref 全程平坦、
远场停在与引导同水平。本脚本剥离 PDE/边界/正则,仅做 15 配置 refsub 回归:
  - L_ref 跌破平台 → 平台是多损失优化干扰所致 → 改训练配方(all-15 ref);
  - L_ref 仍卡平台 → 共享 MLP 表示力不足 → 需要更大网络/每配置头。
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2q_model import BaselineAnsatz, param_vec
from a2q_train import TRAIN_LABELS, DATA_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
LR = 1e-4
PTS_PER_CFG = 10240   # refsub 全量(8192 球内 + 2048 球面)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(RUNS, "a2q_champion", "model.pt"),
                    map_location=dev)
    model = BaselineAnsatz().to(dev).double()
    model.load_state_dict(ck["model_state"])
    model.train()
    log_hdr = f"[ft_test] steps={STEPS} lr={LR} cfgs=15(all) pts/cfg={PTS_PER_CFG}"
    print(log_hdr)

    data = {}
    for lb in TRAIN_LABELS:
        z = np.load(os.path.join(DATA_DIR, f"refsub_{lb}.npz"))
        c = np.load(os.path.join(DATA_DIR, f"cfg_{lb}.npz"))
        x = torch.from_numpy(z["x"]).double().to(dev)
        u_ref = torch.from_numpy(z["u"]).double().to(dev)
        import physics
        ma = torch.tensor([0.5, float(c["m2"])], dtype=torch.float64, device=dev)
        xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64, device=dev)
        Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64,
                          device=dev)
        St = torch.zeros((2, 3), dtype=torch.float64, device=dev)
        with torch.no_grad():
            ugv = physics.guide_u(x, ma, xs, Ps, St)
        w = (ugv - float(c["wmin"])) / (float(c["wmax"]) - float(c["wmin"]) + 1e-8)
        data[lb] = dict(x=x, u_ref=u_ref, ug=ugv, w=w, k=float(c["kappa"]),
                        sq=float(c["sq"]), pv=param_vec(float(c["q"]), float(c["m2"]),
                                                        dev))

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR / 10)
    hist = []
    t0 = time.time()
    for s in range(1, STEPS + 1):
        opt.zero_grad()
        terms = []
        with torch.no_grad():
            pass
        per = []
        for lb in TRAIN_LABELS:
            d = data[lb]
            u, _, _ = model.forward_from_parts(d["x"], d["pv"], d["k"], d["ug"],
                                               d["w"], sq=d["sq"])
            per.append(((u - d["u_ref"]) ** 2).sum() / ((d["u_ref"] ** 2).sum() + 1e-30))
        loss = torch.stack(per).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sch.step()
        hist.append(float(loss))
        if s % 100 == 0 or s == 1:
            arr = np.array(hist[-100:])
            print(f"[step {s:5d}/{STEPS}] L_ref={loss.item():.4e} "
                  f"min100={arr.min():.4e} ({time.time() - t0:.0f}s)")
    # 每配置终值
    print("--- 每配置 refsub L2RE(微调后)---")
    model.eval()
    out = {}
    with torch.no_grad():
        for lb in TRAIN_LABELS:
            d = data[lb]
            u, _, _ = model.forward_from_parts(d["x"], d["pv"], d["k"], d["ug"],
                                               d["w"], sq=d["sq"])
            l2re = float(torch.linalg.norm(u - d["u_ref"])
                         / torch.linalg.norm(d["u_ref"]))
            out[lb] = l2re
            print(f"  {lb:<6} {l2re:.4e}")
    os.makedirs(os.path.join(RUNS, "a2q_ft_test"), exist_ok=True)
    meta = {lb: {k: float(np.load(os.path.join(DATA_DIR, f"cfg_{lb}.npz"))[k])
                 for k in ("q", "m1", "m2", "kappa", "sq", "wmin", "wmax")}
            for lb in TRAIN_LABELS}
    torch.save({"variant": "champion", "model_state": model.state_dict(),
                "history": {"L_ref": hist}, "meta": meta,
                "train_labels": TRAIN_LABELS, "heldout_labels": [],
                "steps_done": STEPS, "args": {"note": "ft_test from a2q_champion"}},
               os.path.join(RUNS, "a2q_ft_test", "model.pt"))
    print("saved: runs/a2q_ft_test/model.pt")


if __name__ == "__main__":
    main()
