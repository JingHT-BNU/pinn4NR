"""
multi_param_train.py —— 多参数参数化 PINN 训练
================================================

策略:
    1. 从 LHS 采样的 ~400 个配置中训练
    2. 每步随机采样 1 个配置 (mini-batch over 参数空间)
    3. 参考解监督 (仅 base case, 权重 1000→课程衰减)
    4. PDE 残差 + Robin 边界 (所有配置)
    5. guide_u / psi_sing / KK 在数据准备时预计算
    6. 网络在 double 精度下前向, float32 权重
"""

import argparse, json, logging, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import sample_ball, sample_sphere_surface
from logutil import setup_logging
import physics
from multi_param_model import (
    MultiParamGuidedPINN, compute_pde_residual, compute_robin_residual,
    normalize_params, denormalize_params, build_bbh_from_params,
    PARAM_NAMES, PARAM_LO, PARAM_HI, BASE_RAW, BASE_NORM,
)

log = logging.getLogger("paper.A3.multi_param_train")

KAPPA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "multi_param_kappa_cache.json")


def resolve_input_path(path, paper_root=None):
    """输入路径解析: 绝对路径原样返回; 相对路径优先按 CWD 解析,
    不存在时锚定到 paper/ 目录 (与运行时 CWD 无关)。"""
    if path is None or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    if paper_root is None:
        paper_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    anchored = os.path.join(paper_root, path)
    return anchored if os.path.exists(anchored) else path


# ── 数据准备 ─────────────────────────────────────────────────

def _sample_ball_shell(n: int, R: float, r_min: float, rng) -> np.ndarray:
    """球壳 [r_min, R] 内体积均匀采样 (n,3)。"""
    r = (r_min ** 3 + rng.random(n) * (R ** 3 - r_min ** 3)) ** (1.0 / 3.0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-300
    return (r[:, None] * v).astype(np.float32)


class MultiParamData:
    """为每个配置预计算采样点和物理量。

    guide_u, psi_sing, KK 在数据准备时计算并缓存为 numpy 数组,
    避免训练时重复计算 (因为这些量只依赖于固定的物理参数, 不依赖于网络)。

    tip_frac > 0 时启用针尖聚焦配点: 体积均匀采样下 r<tip_radius 球的体积
    占比 ~1e-4, 每步 1 万配点中期望 0 个落在奇点近场, PDE 梯度信号为零,
    引导解近场误差(谱参考解揭示的针尖 2.6~3.5× 偏差)无法被纠正。做法是把
    tip_frac 比例的配点替换到两奇点邻域球壳 [r_punc_min, tip_radius]
    (r_punc_min=0.02 以内是 guide_u 的浮点相消伪影区, 剔除)。
    """

    def __init__(self, configs: list, N_Omega: int, N_boundary: int,
                 R_max: float, seed: int = 42,
                 tip_frac: float = 0.0, tip_radius: float = 2.5,
                 r_punc_min: float = 0.02):
        self.data = []
        rng = np.random.default_rng(seed)

        for cfg_info in configs:
            label = cfg_info["label"]
            raw = np.array(cfg_info["raw_params"])
            norm = np.array(cfg_info["norm_params"])
            kappa = cfg_info["kappa"]
            masses, xs, Ps, Ss = build_bbh_from_params(raw)

            ma = torch.from_numpy(masses).double()
            xst = torch.from_numpy(xs).double()
            Pt = torch.from_numpy(Ps).double()
            St = torch.from_numpy(Ss).double()

            xi = sample_ball(N_Omega, R_max, rng).astype(np.float32)
            if tip_frac > 0.0:
                n_tip = int(N_Omega * tip_frac) // 2
                if n_tip > 0:
                    parts = [xi[:N_Omega - 2 * n_tip]]
                    for center in xs:
                        parts.append(_sample_ball_shell(n_tip, tip_radius,
                                                        r_punc_min, rng)
                                     + center[None, :].astype(np.float32))
                    xi = np.concatenate(parts).astype(np.float32)
            # 体/针分离掩膜(损失分开 EMA 平衡, 防止针尖大残差淹没 bulk)
            rp = np.min(np.linalg.norm(xi[:, None, :] - xs[None, :, :],
                                       axis=2), axis=1)
            tip_mask = (rp < tip_radius) if tip_frac > 0.0 else \
                np.zeros(len(xi), dtype=bool)
            xb = sample_sphere_surface(N_boundary, R_max, rng).astype(np.float32)

            def _phys(x_np):
                xt = torch.from_numpy(x_np).double()
                ps = physics.psi_sing(xt, ma, xst).numpy().astype(np.float32)
                kk = physics.bowen_york_KK(xt, ma, xst, Pt, St).numpy().astype(np.float32)
                ug = physics.guide_u(xt, ma, xst, Pt, St).numpy().astype(np.float32)
                return ps, kk, ug

            ps_i, kk_i, ug_i = _phys(xi)
            ps_b, kk_b, ug_b = _phys(xb)

            all_ug = np.concatenate([ug_i, ug_b])

            self.data.append(dict(
                label=label,
                raw_params=raw,
                norm_params=norm,
                kappa=kappa,
                masses=masses.astype(np.float32),
                xs=xs.astype(np.float32),
                Ps=Ps.astype(np.float32),
                Ss=Ss.astype(np.float32),
                x_int=xi, x_bnd=xb,
                ps_int=ps_i, kk_int=kk_i, ug_int=ug_i,
                ps_bnd=ps_b, kk_bnd=kk_b, ug_bnd=ug_b,
                tip_mask=tip_mask,
                u_min=float(all_ug.min()),
                u_max=float(all_ug.max()),
            ))

            if len(self.data) <= 3 or label == "train_0000":
                log.info(f"  {label}: κ={kappa:.4f}, ug∈[{all_ug.min():.4e},"
                         f"{all_ug.max():.4e}], 针尖配点 {int(tip_mask.sum())}")

        if tip_frac > 0.0:
            log.info(f"  针尖聚焦配点: {tip_frac:.0%} × N_Omega → 每奇点球壳 "
                     f"[{r_punc_min}, {tip_radius}] (体/针残差分开平衡)")

    def global_range(self):
        return min(d["u_min"] for d in self.data), max(d["u_max"] for d in self.data)

    def find_base_idx(self):
        """找到最接近 base case 的配置索引。"""
        best_idx, best_dist = 0, float("inf")
        for i, d in enumerate(self.data):
            dist = np.linalg.norm(d["norm_params"] - BASE_NORM)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx


# ── 训练器 ───────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, device, ncfg: int, R_max: float,
                 ref_x=None, ref_u=None, base_idx: int = 0,
                 lr: float = 3e-4, n_steps: int = 100000,
                 ema_alpha: float = 0.9, w_ref0: float = 167.0,
                 n_ref: int = 10000,
                 sup_refs=None,
                 ckpt_path=None, ckpt_every=500, ckpt_meta=None):
        self.model = model
        self.device = device
        self.ncfg = ncfg
        self.R_max = R_max
        self.base_idx = base_idx
        self.n_steps = n_steps
        self.w_ref0 = w_ref0
        self.lr0 = lr

        # 断点续训
        self.ckpt_path = ckpt_path
        self.ckpt_every = ckpt_every
        self.ckpt_meta = ckpt_meta or {}
        self.start_step = 0

        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=n_steps, eta_min=lr * 0.01)

        self.ema = {}
        self.hist = {"L2": [], "LBC": [], "L_ref": [], "L_sup": [], "total": [],
                     "L2b": [], "L2_tip": [], "LBCb": []}
        self.cfg_count = np.zeros(ncfg, dtype=int)
        self.rng = np.random.default_rng(0)

        # 按配置 EMA:不同配置的残差量级差数个数量级(κ 0.115~0.83),
        # 全局单 EMA 归一化会让 total 反复横跳。按配置各自维护 EMA。
        self.ema_l2 = np.full(ncfg, np.nan)
        self.ema_tip = np.full(ncfg, np.nan)
        self.ema_lbc = np.full(ncfg, np.nan)

        # 多配置谱参考监督: 非-base 的匹配训练配置, 针尖加密点集
        self.sup_refs = sup_refs or []

        # 参考解: 保存全量点,每步重新采样 n_ref 个(避免过拟合固定子样本)
        if ref_x is not None:
            self.ref_x_full = torch.from_numpy(np.ascontiguousarray(ref_x))
            self.ref_u_full = torch.from_numpy(np.ascontiguousarray(ref_u))
            self.n_ref = min(n_ref, ref_x.shape[0])
            log.info(f"[监督] 参考解全量 {ref_x.shape[0]} 点, 每步重采样 {self.n_ref} 点")
        else:
            self.ref_x_full = self.ref_u_full = None
            self.n_ref = 0

    def _sample_ref(self):
        """每步从全量参考解重新采样一批点(CPU → GPU)。"""
        idx = torch.randint(0, self.ref_x_full.shape[0], (self.n_ref,))
        rx = self.ref_x_full.index_select(0, idx).to(self.device).double()
        ru = self.ref_u_full.index_select(0, idx).to(self.device).double()
        return rx, ru

    # ── 断点续训 ─────────────────────────────────────────────

    def save_checkpoint(self):
        """原子写入检查点:模型+优化器+调度器+EMA+历史+随机数状态。"""
        if not self.ckpt_path:
            return
        tmp = self.ckpt_path + ".tmp"
        torch.save({
            "model_state": self.model.state_dict(),
            "opt_state": self.opt.state_dict(),
            "sch_state": self.sch.state_dict(),
            "step": len(self.hist["L2"]),
            "hist": self.hist,
            "cfg_count": self.cfg_count,
            "ema_l2": self.ema_l2,
            "ema_tip": self.ema_tip,
            "ema_lbc": self.ema_lbc,
            "torch_rng": torch.get_rng_state(),
            "cfg_rng": self.rng.bit_generator.state,
            "meta": dict(self.ckpt_meta, steps_total=self.n_steps),
        }, tmp)
        os.replace(tmp, self.ckpt_path)

    def try_resume(self):
        """尝试从检查点恢复。返回已完成步数(0=全新开始)。

        配置指纹(meta)不匹配时备份旧文件后重新开始;
        steps 与保存时不一致视为"延长训练",重新锚定余弦退火到新目标。
        """
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            return 0
        try:
            ckpt = torch.load(self.ckpt_path, map_location=self.device,
                              weights_only=False)
        except Exception as e:
            log.info(f"[resume] 检查点损坏({e}),从头开始")
            return 0

        meta = ckpt.get("meta", {})
        ignore = {"steps_total"}
        cur = {k: v for k, v in self.ckpt_meta.items() if k not in ignore}
        old = {k: v for k, v in meta.items() if k not in ignore}
        mismatch = [k for k in sorted(set(cur) | set(old))
                    if cur.get(k) != old.get(k)]
        if mismatch:
            bak = self.ckpt_path + ".incompatible.bak"
            os.replace(self.ckpt_path, bak)
            log.info(f"[resume] 检查点配置不匹配({','.join(mismatch)}),"
                  f"已备份至 {bak},从头开始")
            return 0

        done = int(ckpt["step"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.opt.load_state_dict(ckpt["opt_state"])
        # load_state_dict 会按各参数自身 dtype 转换状态张量(Linear 权重是
        # float32 → 其 exp_avg 被降为 float32);而训练循环在 model.double()
        # 下计算梯度(float64),dtype 不一致会让首次 opt.step() 崩溃。
        # 统一回 float64(与全新训练时惰性创建的 dtype 一致)。
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            for st_ in self.opt.state.values():
                for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    v = st_.get(k)
                    if isinstance(v, torch.Tensor):
                        st_[k] = v.to(device=dev, dtype=torch.float64)
                if isinstance(st_.get("step"), torch.Tensor):
                    st_["step"] = st_["step"].to(dev)
        saved_steps = meta.get("steps_total", done)
        if saved_steps == self.n_steps:
            self.sch.load_state_dict(ckpt["sch_state"])
        else:
            remaining = max(1, self.n_steps - done)
            self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=remaining, eta_min=self.lr0 * 0.01)
            log.info(f"[resume] 目标步数 {saved_steps}→{self.n_steps}: "
                  f"余弦退火对剩余 {remaining} 步重新锚定")

        self.hist = {k: list(v) for k, v in ckpt["hist"].items()}
        for k in ("L2", "LBC", "L_ref", "L_sup", "total", "L2b", "L2_tip", "LBCb"):
            self.hist.setdefault(k, [])
        self.cfg_count = np.asarray(ckpt["cfg_count"], dtype=int)
        self.ema_l2 = np.asarray(ckpt["ema_l2"], dtype=float)
        self.ema_tip = (np.asarray(ckpt["ema_tip"], dtype=float)
                        if "ema_tip" in ckpt else np.full(self.ncfg, np.nan))
        self.ema_lbc = np.asarray(ckpt["ema_lbc"], dtype=float)
        torch.set_rng_state(ckpt["torch_rng"].cpu().to(torch.uint8))
        try:
            self.rng.bit_generator.state = ckpt["cfg_rng"]
        except Exception:
            pass
        self.start_step = done

        last_lref = next((v for v in reversed(self.hist["L_ref"]) if v > 0), None)
        lrs_str = f"L_ref={last_lref:.3e}" if last_lref else "-"
        log.info(f"[resume] 从检查点继续: step {done}/{self.n_steps} "
              f"(上次 {lrs_str})")
        return done

    def _ema(self, name, loss):
        v = loss.item()
        if name not in self.ema:
            self.ema[name] = v
        else:
            self.ema[name] = 0.9 * self.ema[name] + 0.1 * v
        return loss / (self.ema[name] + 1e-12)

    def _fwd(self, d):
        """单配置前向: 返回 L2_bulk, L2_tip, LBC。

        体/针残差分开返回(训练里各自 EMA 平衡): 针尖配点处源项大数个量级,
        若混在一个 mean 里会把优化压力全部吸走, bulk 反而失守。"""
        device = self.device

        ma = torch.from_numpy(d["masses"]).double().to(device)
        xs = torch.from_numpy(d["xs"]).double().to(device)
        Ps = torch.from_numpy(d["Ps"]).double().to(device)
        Ss = torch.from_numpy(d["Ss"]).double().to(device)
        pn = torch.tensor(d["norm_params"], dtype=torch.float64,
                          device=device).unsqueeze(0)
        kappa = d["kappa"]

        xi = torch.from_numpy(d["x_int"]).double().to(device)
        xi.requires_grad_(True)
        ui = self.model(xi, ma, xs, Ps, Ss, pn, kappa)
        pde_r = compute_pde_residual(ui, xi, ma, xs, Ps, Ss)
        tmask = torch.from_numpy(d["tip_mask"]).to(device)
        if tmask.any():
            l2_tip = (pde_r[tmask] ** 2).mean()
            l2 = (pde_r[~tmask] ** 2).mean() if bool((~tmask).any()) \
                else l2_tip
        else:
            l2, l2_tip = (pde_r ** 2).mean(), torch.zeros_like(pde_r[:1]).sum()

        xb = torch.from_numpy(d["x_bnd"]).double().to(device)
        xb.requires_grad_(True)
        ub = self.model(xb, ma, xs, Ps, Ss, pn, kappa)
        rob_r = compute_robin_residual(ub, xb, self.R_max)
        lbc = (rob_r ** 2).mean()

        return l2, l2_tip, lbc

    def _sup_loss(self, s, d):
        """谱参考监督损失(非 base 配置, 针尖加密点集每步重采样)。"""
        device = self.device
        ma = torch.from_numpy(d["masses"]).double().to(device)
        xs = torch.from_numpy(d["xs"]).double().to(device)
        Ps = torch.from_numpy(d["Ps"]).double().to(device)
        Ss = torch.from_numpy(d["Ss"]).double().to(device)
        pn = torch.tensor(d["norm_params"], dtype=torch.float64,
                          device=device).unsqueeze(0)
        kappa = d["kappa"]
        idx = torch.randint(0, s["x"].shape[0], (self.n_ref,)).numpy()
        rx = torch.from_numpy(s["x"][idx]).to(device).double()
        ru = torch.from_numpy(s["u"][idx]).to(device).double()
        up = self.model(rx, ma, xs, Ps, Ss, pn, kappa)
        return torch.sum((up - ru) ** 2) / (torch.sum(ru ** 2) + 1e-30)

    def _compute_ref_loss(self, d):
        """计算参考解损失 (仅在 base case 配置时调用)。"""
        device = self.device
        ma = torch.from_numpy(d["masses"]).double().to(device)
        xs = torch.from_numpy(d["xs"]).double().to(device)
        Ps = torch.from_numpy(d["Ps"]).double().to(device)
        Ss = torch.from_numpy(d["Ss"]).double().to(device)
        pn = torch.tensor(d["norm_params"], dtype=torch.float64,
                          device=device).unsqueeze(0)
        kappa = d["kappa"]

        rx, ru = self._sample_ref()
        up = self.model(rx, ma, xs, Ps, Ss, pn, kappa)
        return torch.sum((up - ru) ** 2) / (torch.sum(ru ** 2) + 1e-30)

    def _bal(self, arr, idx, loss):
        """按配置 EMA 平衡:归一化到 O(1)。首次见到该配置时用原始值初始化。"""
        v = loss.item()
        if np.isnan(arr[idx]):
            arr[idx] = v
        else:
            arr[idx] = 0.9 * arr[idx] + 0.1 * v
        return loss / (arr[idx] + 1e-12)

    def step(self, data_list, step):
        self.opt.zero_grad()
        self.model.double()

        idx = self.rng.integers(0, self.ncfg)
        self.cfg_count[idx] += 1
        d = data_list[idx]

        l2, l2_tip, lbc = self._fwd(d)

        # 参考损失: 每步都在 base 配置上计算(强锚定, 修复稀疏监督问题)
        lr = torch.tensor(0.0, device=self.device)
        if self.ref_x_full is not None:
            lr = self._compute_ref_loss(data_list[self.base_idx])

        # 多配置谱参考监督: 每步随机抽 1 个有谱参考解的训练配置
        # (针尖加密点集), 把参考解信息从 base 推广到针尖偏差最大的配置
        ls = torch.tensor(0.0, device=self.device)
        if self.sup_refs:
            s = self.sup_refs[int(self.rng.integers(0, len(self.sup_refs)))]
            ls = self._sup_loss(s, data_list[s["cfg_idx"]])

        # 按配置 EMA 平衡(不同配置残差量级差数量级,平衡后同台竞争);
        # bulk 与针尖分开平衡, 针尖大残差不会淹没 bulk
        l2b = self._bal(self.ema_l2, idx, l2)
        ltib = self._bal(self.ema_tip, idx, l2_tip)
        lbcb = self._bal(self.ema_lbc, idx, lbc)

        # 课程衰减: 早期参考监督主导, 后期让 PDE/Robin 正则精调。
        # w_ref0 默认 200 ≈ A2 的有效平均权重(1000×1/6≈167, 每步都有监督时
        # 需要同量级才能保持与 A2 相同的监督强度;此前误用 1000/300=3.3,
        # 监督弱了 50 倍导致 L_ref 全程平坦)。
        prog = step / self.n_steps
        w_ref = self.w_ref0 * max(0.1, 1.0 - prog * 0.5)

        total = (l2b + ltib
                 + lbcb
                 + w_ref * (lr + ls))

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step()
        self.sch.step()
        self.model.float()

        out = {"L2": l2.item(), "L2_tip": l2_tip.item(), "LBC": lbc.item(),
               "L_ref": lr.item(), "L_sup": float(ls.item()), "total": total.item(),
               "L2b": l2b.item(), "LBCb": lbcb.item()}
        for k in out:
            self.hist[k].append(out[k])
        return out

    def train(self, data_list, n_steps, log_every=1000):
        t0 = time.time()
        ran = 0
        # 滚动窗口均值: 单步值取决于随机抽中的配置, 天然跨数量级横跳;
        # 均值才能反映真实趋势。原始单步值仍全量记入 history。
        from collections import deque
        win = {k: deque(maxlen=200) for k in
               ("L2", "LBC", "L_ref", "L_sup", "total", "L2b", "L2_tip", "LBCb")}
        for s in range(self.start_step + 1, n_steps + 1):
            o = self.step(data_list, s)
            ran += 1
            for k, dq in win.items():
                dq.append(o[k])
            if s % log_every == 0 or s == n_steps or s == self.start_step + 1:
                mean = {k: sum(dq) / len(dq) for k, dq in win.items()}
                prog = s / n_steps
                w_ref = self.w_ref0 * max(0.1, 1.0 - prog * 0.5)
                elapsed = time.time() - t0
                c_eff = (f" c={self.model.effective_c():.3f}"
                         if hasattr(self.model, "effective_c") else "")
                log.info(f"[step {s:6d}/{n_steps}] "
                         f"L2均={mean['L2']:.3e} L2tip均={mean['L2_tip']:.3e} "
                         f"LBC均={mean['LBC']:.3e} "
                         f"L_ref均={mean['L_ref']:.4e} L_sup均={mean['L_sup']:.4e} "
                         f"total均={mean['total']:.3e} "
                         f"w_ref={w_ref:.0f}{c_eff} ({elapsed:.0f}s)")
            if self.ckpt_path and (s % self.ckpt_every == 0 or s == n_steps):
                self.save_checkpoint()
                if s % max(self.ckpt_every * 10, log_every) == 0 or s == n_steps:
                    log.info(f"[ckpt] 已保存检查点 @ step {s}")
        if ran > 0:
            log.info(f"完成({self.start_step+1}→{n_steps}), "
                  f"本次用时{(time.time()-t0)/3600:.1f}h, "
                  f"累计配置采样分布: min={self.cfg_count.min()}, "
                  f"max={self.cfg_count.max()}, mean={self.cfg_count.mean():.0f}")
        return self.hist


# ── 评估工具 ─────────────────────────────────────────────────

def predict(model, x_np, raw_params, kappa, device, batch=8192):
    """用训练好的模型预测。自动适配模型精度(float32/double)。"""
    model.eval()
    # 检测模型精度:用 MLP 第一个 Linear 层的权重 dtype(兼容 float32/double 模型)
    dt = model.mlp.shared[0].weight.dtype
    masses, xs, Ps, Ss = build_bbh_from_params(raw_params)
    norm_p = normalize_params(raw_params)

    ma = torch.from_numpy(masses).to(dtype=dt, device=device)
    xst = torch.from_numpy(xs).to(dtype=dt, device=device)
    Pt = torch.from_numpy(Ps).to(dtype=dt, device=device)
    St = torch.from_numpy(Ss).to(dtype=dt, device=device)
    pn = torch.tensor(norm_p, dtype=dt, device=device).unsqueeze(0)

    out = []
    with torch.no_grad():
        for i in range(0, x_np.shape[0], batch):
            xb = torch.from_numpy(x_np[i:i+batch]).to(dtype=dt, device=device)
            out.append(model(xb, ma, xst, Pt, St, pn, kappa).cpu().numpy())
    return np.concatenate(out)


def l2re(up, ur):
    return float(np.sqrt(np.sum((up - ur) ** 2) / max(np.sum(ur ** 2), 1e-30)))


def build_sup_refs(refs_dir, train_data, base_idx, n_pts, device, R_max,
                   tip_radius=2.5, r_punc_min=0.02):
    """为非 base 的训练配置构建谱参考监督点集(针尖加密)。

    按 8 维参数匹配 tools/refs 下含 radial_coeffs 的谱参考解 npz; 监督点集 =
    60% 体积均匀(球 R_max) + 40% 两奇点邻域球壳 [r_punc_min, tip_radius]
    —— 让参考监督以远高于体积占比的密度覆盖引导解近场误差最大的针尖区。
    参考值用谱系数求值器一次性算好。目录缺失/无匹配返回 [](纯 PDE 训练)。
    """
    if not refs_dir or not os.path.isdir(refs_dir):
        return []
    tools_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    sys.path.insert(0, tools_dir)
    from spectral_reference import SpectralPunctureSolver
    rng = np.random.default_rng(123)
    out = []
    for fn in sorted(os.listdir(refs_dir)):
        if not fn.endswith(".npz"):
            continue
        path = os.path.join(refs_dir, fn)
        rd = np.load(path)
        if "raw" not in rd or "radial_coeffs" not in rd:
            continue
        raw = np.array(rd["raw"], dtype=np.float64)
        hit = None
        for i, d in enumerate(train_data.data):
            if i != base_idx and np.allclose(raw, d["raw_params"],
                                             rtol=0.0, atol=1e-9):
                hit = i
                break
        if hit is None:
            continue
        ev = SpectralPunctureSolver.from_coefficients(path, device=str(device))
        n_tip = int(n_pts * 0.4) // 2
        _, xs, _, _ = build_bbh_from_params(train_data.data[hit]["raw_params"])
        parts = [sample_ball(n_pts - 2 * n_tip, R_max, rng)]
        for center in xs:
            parts.append(_sample_ball_shell(n_tip, tip_radius, r_punc_min, rng)
                         + center[None, :].astype(np.float32))
        x = np.concatenate(parts).astype(np.float32)
        u = ev.evaluate(x, dtype=torch.float32).astype(np.float32)
        out.append({"cfg_idx": hit, "label": train_data.data[hit]["label"],
                    "x": x, "u": u})
        log.info(f"[sup-ref] {train_data.data[hit]['label']} ← {fn}: "
                 f"{len(x)} 点 (体积 {n_pts - 2 * n_tip} + 针尖 {2 * n_tip})")
    return out


# ── 主函数 ───────────────────────────────────────────────────

def main():
    setup_logging("A3", "multi_param_train")
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--N-Omega", type=int, default=10000)
    p.add_argument("--N-boundary", type=int, default=4000)
    p.add_argument("--exp-name", default="multi_param_a1")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--reference", default=None,
                   help="参考解 .npz 文件路径 (base case)")
    p.add_argument("--kappa-cache", default=None)
    p.add_argument("--n-ref", type=int, default=10000,
                   help="每步参考监督采样点数(从全量 47.7M 中重采样)")
    p.add_argument("--w-ref", type=float, default=None,
                   help="参考监督初始权重。默认 None = A2 等效强度(≈167)。")
    p.add_argument("--tip-frac", type=float, default=0.25,
                   help="针尖聚焦配点比例(替换到两奇点球壳 [0.02, --tip-radius])")
    p.add_argument("--tip-radius", type=float, default=2.5)
    p.add_argument("--sup-refs-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools", "refs"),
        help="谱参考解目录: 为匹配的非 base 训练配置构建参考监督(传空串关闭)")
    p.add_argument("--sup-n", type=int, default=1500000,
                   help="每配置谱参考监督点集大小(60%% 体积 + 40%% 针尖)")
    p.add_argument("--amp-mode", choices=["raw", "sigmoid"], default="sigmoid",
                   help="幅值 c 参数化: raw=v4 原样; sigmoid=c_max·σ(c_raw) 有界可学")
    p.add_argument("--c-max", type=float, default=1.0)
    p.add_argument("--c-init", type=float, default=0.2)
    p.add_argument("--ckpt-every", type=int, default=500,
                   help="每多少步保存一次断点检查点")
    p.add_argument("--no-resume", action="store_true",
                   help="忽略已有检查点,从头开始训练")
    p.add_argument("--hidden-layers", type=int, default=6)
    p.add_argument("--hidden-neurons", type=int, default=256)
    p.add_argument("--n-freq-coord", type=int, default=10)
    p.add_argument("--n-freq-param", type=int, default=12)
    args = p.parse_args()

    # 输入路径解析: 优先按 CWD 相对路径, 不存在则锚定到 paper/(与运行时 CWD 无关)
    paper_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for arg_name in ("reference", "kappa_cache", "sup_refs_dir"):
        val = getattr(args, arg_name)
        if val == "":                      # 显式传空串 = 关闭该项
            setattr(args, arg_name, None)
            continue
        setattr(args, arg_name, resolve_input_path(val, paper_root))

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto"
                          else "cpu")
    log.info(f"设备: {device}")

    # ── 加载 κ 缓存 ──
    cache_path = args.kappa_cache or KAPPA_CACHE
    log.info(f"\n加载 κ 缓存: {cache_path}")
    with open(cache_path) as f:
        cache = json.load(f)
    train_configs = cache["train"]
    val_configs = cache["val"]
    log.info(f"  训练: {len(train_configs)}, 验证: {len(val_configs)}")

    # ── 准备数据 ──
    R_max = cache["meta"]["R_max"]
    log.info(f"\n准备数据 (N_Omega={args.N_Omega}, N_boundary={args.N_boundary}, "
             f"tip_frac={args.tip_frac}, tip_radius={args.tip_radius})...")
    train_data = MultiParamData(train_configs, args.N_Omega, args.N_boundary, R_max,
                                tip_frac=args.tip_frac, tip_radius=args.tip_radius)
    gmin, gmax = train_data.global_range()
    log.info(f"全局 ug: [{gmin:.4e}, {gmax:.4e}]")

    base_idx = train_data.find_base_idx()
    log.info(f"Base case 索引: {base_idx} ({train_data.data[base_idx]['label']})")

    # ── 加载参考解 ──
    ref_x, ref_u = None, None
    if args.reference:
        ref = np.load(args.reference)
        ref_x, ref_u = ref["x_ref"], ref["u_ref"]
        log.info(f"参考解: {ref_x.shape[0]} 点")

    # ── 构建模型 ──
    log.info(f"\n构建模型 ({args.hidden_layers}×{args.hidden_neurons}, "
             f"FiLM, 正弦编码 {args.n_freq_coord}/{args.n_freq_param}, "
             f"amp={args.amp_mode}, c_max={args.c_max})...")
    model = MultiParamGuidedPINN(
        n_params=8, c_init=args.c_init,
        hidden_layers=args.hidden_layers,
        hidden_neurons=args.hidden_neurons,
        n_freq_coord=args.n_freq_coord,
        n_freq_param=args.n_freq_param,
        amp_mode=args.amp_mode, c_max=args.c_max,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"参数量: {n_params:,}")

    # ── 多配置谱参考监督点集 ──
    sup_refs = []
    if args.sup_refs_dir:
        sup_refs = build_sup_refs(args.sup_refs_dir, train_data, base_idx,
                                  args.sup_n, device, R_max,
                                  tip_radius=args.tip_radius)
        log.info(f"谱参考监督配置: {len(sup_refs)} 个"
                 if sup_refs else "谱参考监督: 无匹配 (纯 PDE + base 参考)")

    # ── 训练(支持断点续训) ──
    w_ref0 = args.w_ref if args.w_ref is not None else 167.0
    log.info(f"参考监督权重 w_ref0={w_ref0} (课程衰减至 {w_ref0*0.1:.0f})")

    # out-dir 相对路径锚定到 paper/ 目录(与运行时 CWD 无关,避免输出错位)
    paper_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(paper_root, args.out_dir)
        log.info(f"out-dir 锚定: {args.out_dir}")

    exp_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)
    ckpt_path = os.path.join(exp_dir, "checkpoint.pt")

    # 配置指纹:只有与检查点保存时完全一致才允许续训(steps 允许不同=延长训练)
    ckpt_meta = {
        "lr": args.lr, "N_Omega": args.N_Omega, "N_boundary": args.N_boundary,
        "n_ref": args.n_ref, "w_ref0": w_ref0,
        "ncfg": len(train_data.data), "base_idx": base_idx,
        "hidden_layers": args.hidden_layers,
        "hidden_neurons": args.hidden_neurons,
        "n_freq_coord": args.n_freq_coord, "n_freq_param": args.n_freq_param,
        "R_max": R_max, "cache_seed": cache["meta"].get("seed"),
        "amp_mode": args.amp_mode, "c_max": args.c_max, "c_init": args.c_init,
        "tip_frac": args.tip_frac, "tip_radius": args.tip_radius,
        "sup_labels": sorted(s["label"] for s in sup_refs),
        "win": "tanh_ug05+tip+csig+supref",
    }

    trainer = Trainer(
        model, device, ncfg=len(train_data.data), R_max=R_max,
        ref_x=ref_x, ref_u=ref_u, base_idx=base_idx,
        lr=args.lr, n_steps=args.steps, ema_alpha=0.9,
        w_ref0=w_ref0, n_ref=args.n_ref,
        sup_refs=sup_refs,
        ckpt_path=ckpt_path, ckpt_every=args.ckpt_every, ckpt_meta=ckpt_meta,
    )

    done = 0
    if not args.no_resume:
        done = trainer.try_resume()
    elif os.path.exists(ckpt_path):
        log.info(f"[resume] --no-resume: 忽略已有检查点 {ckpt_path}")

    if done >= args.steps:
        log.info(f"[resume] 检查点已完成全部 {args.steps} 步,跳过训练直接评估。"
              f"如需强制重训: --no-resume")
    else:
        log.info(f"\n训练 {args.steps} 步 (从 step {done+1} 开始)...")
    hist = trainer.train(train_data.data, n_steps=args.steps,
                         log_every=max(1, args.steps // 50))

    # ── 保存最终模型 ──
    torch.save({
        "model_state": model.state_dict(),
        "c": model.effective_c(),
        "amp_mode": args.amp_mode,
        "c_max": args.c_max,
        **({"c_raw": float(model.c_raw.item())}
           if args.amp_mode == "sigmoid" else {}),
        "u_scale": float(model.u_scale.item()),
        "history": hist,
        "n_params": 8,
        "param_names": PARAM_NAMES,
        "hidden_layers": args.hidden_layers,
        "hidden_neurons": args.hidden_neurons,
        "n_freq_coord": args.n_freq_coord,
        "n_freq_param": args.n_freq_param,
        "tip_frac": args.tip_frac,
        "tip_radius": args.tip_radius,
        "kappa_cache_meta": cache["meta"],
    }, os.path.join(exp_dir, "model.pt"))

    json.dump(hist, open(os.path.join(exp_dir, "history.json"), "w"), indent=2)

    # 保存配置信息
    config_info = {
        "train": [{"label": c["label"], "raw": c["raw_params"],
                    "norm": c["norm_params"], "kappa": c["kappa"]}
                  for c in train_configs],
        "val": [{"label": c["label"], "raw": c["raw_params"],
                 "norm": c["norm_params"], "kappa": c["kappa"]}
                for c in val_configs],
    }
    json.dump(config_info, open(os.path.join(exp_dir, "configs.json"), "w"), indent=2)

    # ── 快速评估 ──
    log.info(f"\n训练配置评估 (PDE 残差自检)...")
    rng = np.random.default_rng(42)
    for d in train_data.data[:5]:
        xc = sample_ball(3000, R_max, rng).astype(np.float32)
        up = predict(model, xc, d["raw_params"], d["kappa"], device)
        log.info(f"  {d['label']}: u∈[{up.min():.4e},{up.max():.4e}]")

    if ref_x is not None:
        bd = train_data.data[base_idx]
        up = predict(model, ref_x, bd["raw_params"], bd["kappa"], device)
        log.info(f"\n  Base L2RE = {l2re(up, ref_u):.4e}")

    log.info(f"\n完成: {exp_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
