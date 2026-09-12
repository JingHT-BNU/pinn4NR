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
"""a2q_model.py —— A2 单参数 q∈[1,10] 攻关:模型定义。

与原 A2(parametric_model.py)的关键差异:
  1. 窗口 w 逐配置归一化: w = (u_g − wmin_cfg)/(wmax_cfg − wmin_cfg)。
     原 A2 用全体配置的全局 min/max,轻质量配置的 w 被压到 ~1/(m2max/m2min)² ,
     修正幅度不足 —— 这是"轻 q 残差差 ~50×"诊断的方法学根源。
  2. 参数输入改为 [log10(q), m2/5](良好定标;q 跨 1 个量级,网络感知的是
     log 尺度)。n_params 仍为 2,复用 ParamConditionMLP。
  3. 新增 A2-1 神经算子 ansatz:
        u = κ·u_g·(1 + w·G_θ(x, p, [log1p(|u_g|/sq), w]))
     G 读入引导解局部特征(算子式),输出有界 ±3(末层零初始化,从引导解出发),
     无 tanh(h) 限幅与固定 c 的双重压缩,自由度远大于缩放因子。
"""
import os, sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import physics
from parametric_model import ParamConditionMLP, FiLM


class BaselineAnsatz(nn.Module):
    """A2-0:与原 A2 相同的缩放因子 ansatz,仅窗口改为逐配置归一化。"""

    def __init__(self, c_init=0.2, hidden_layers=4, hidden_neurons=128, n_freq=8):
        super().__init__()
        self.mlp = ParamConditionMLP(n_params=2, hidden_layers=hidden_layers,
                                     hidden_neurons=hidden_neurons, n_freq=n_freq)
        self.c = nn.Parameter(torch.tensor(float(c_init), dtype=torch.float64))

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        w = (ug - wmin) / (wmax - wmin + 1e-8)
        h = torch.tanh(self.mlp(x, params.to(x.dtype)).squeeze(-1))
        return kappa * ug * (1.0 + self.c.to(x.dtype) * w * h)

    def forward_from_parts(self, x, params, kappa, ug, w, sq=None, feats=None):
        """快路径:u_g/w 已预计算为常数张量。返回 (u, phi, psi),φ=w·ψ,
        ψ=c·tanh(h) 只经 MLP 反传;w 的 x-导数项由调用方用预计算 ∇u_g/Δu_g 补全。
        feats 仅为与算子 ansatz 的签名兼容(忽略)。"""
        h = torch.tanh(self.mlp(x, params.to(x.dtype)).squeeze(-1))
        psi = self.c.to(x.dtype) * h
        phi = w * psi
        return kappa * ug * (1.0 + phi), phi, psi


class OperatorMLP(nn.Module):
    """条件 MLP,额外读入逐点特征 feats (N,F)。末层零初始化。

    n_extra:coord 分支在 [log1p(|u_g|/sq), w] 之外的附加特征数(算子 v2 的
    邻域 patch 采样),v1 保持 0。"""

    def __init__(self, n_params=2, hidden_layers=4, hidden_neurons=128, n_freq=8,
                 n_extra=0):
        super().__init__()
        self.n_freq = n_freq
        coord_enc_dim = 3 + 3 * 2 * n_freq + 2 + n_extra  # 坐标编码 + 引导场特征
        param_enc_dim = n_params + n_params * 2 * n_freq
        H = hidden_neurons
        self.coord_net = nn.Sequential(
            nn.Linear(coord_enc_dim, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.param_net = nn.Sequential(
            nn.Linear(param_enc_dim, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.film = FiLM(H, H)
        layers, in_dim = [], H
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_dim, H), nn.SiLU()]
            in_dim = H
        layers.append(nn.Linear(in_dim, 1))
        self.shared = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.shared[-1].weight)
        nn.init.zeros_(self.shared[-1].bias)

    def _embed_coord(self, x):
        emb = [x]
        for i in range(self.n_freq):
            f = np.exp(i)
            emb.append(torch.sin(x * f))
            emb.append(torch.cos(x * f))
        return torch.cat(emb, dim=-1)

    def _embed_param(self, p):
        emb = [p]
        for i in range(self.n_freq):
            f = np.exp(i)
            emb.append(torch.sin(p * f))
            emb.append(torch.cos(p * f))
        return torch.cat(emb, dim=-1)

    def forward(self, x, params, feats):
        x_in = torch.cat([self._embed_coord(x), feats], dim=-1)
        x_enc = self.coord_net(x_in)
        if params.shape[0] == 1 and x.shape[0] > 1:
            params = params.expand(x.shape[0], -1)
        p_enc = self.param_net(self._embed_param(params))
        return self.shared(self.film(x_enc, p_enc))


class OperatorAnsatz(nn.Module):
    """A2-1:神经算子 ansatz(见模块 docstring)。"""

    def __init__(self, hidden_layers=4, hidden_neurons=128, n_freq=8, corr_max=3.0):
        super().__init__()
        self.mlp = OperatorMLP(n_params=2, hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons, n_freq=n_freq)
        self.corr_max = float(corr_max)

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        w = (ug - wmin) / (wmax - wmin + 1e-8)
        feats = torch.stack([torch.log1p(ug.abs() / sq), w], dim=-1)
        raw = self.mlp(x, params.to(x.dtype), feats).squeeze(-1)
        cm = self.corr_max
        corr = cm * torch.tanh(raw / cm)
        return kappa * ug * (1.0 + w * corr)

    def forward_from_parts(self, x, params, kappa, ug, w, sq=None):
        """快路径:同上,ug/w 为预计算常数;feats 由 ug/w 直接构造。
        返回 (u, phi, psi),ψ=corr 只经 MLP 反传。"""
        feats = torch.stack([torch.log1p(ug.abs() / sq), w], dim=-1)
        raw = self.mlp(x, params.to(x.dtype), feats).squeeze(-1)
        cm = self.corr_max
        psi = cm * torch.tanh(raw / cm)
        phi = w * psi
        return kappa * ug * (1.0 + phi), phi, psi


def fibonacci_dirs(n):
    """Fibonacci 球面均匀方向 (n,3),单位向量。"""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def patch_offsets(radii=(0.5, 1.5, 4.0), n_dirs=8):
    """算子 v2 的邻域 patch 采样偏移 (K,3):多半径球面方向。

    半径覆盖引导场形状的三个尺度:0.5(峰内/近奇点)、1.5(峰间谷)、4.0
    (整体包络;孔间距 6)。"""
    d = fibonacci_dirs(n_dirs)
    return np.concatenate([r * d for r in radii], axis=0)


class OperatorV2Ansatz(nn.Module):
    """A2-1 v2:真泛函输入的神经算子。

    v1 只读当前点数值 [log1p(|u_g|/sq), w],被"常数幅度重标定"平凡解满足
    (探针:|corr| p50=p90=p99≈0.21 恒定,±3 自由度仅用 7%)。v2 让每个查询点
    读入其邻域 patch 上的引导场采样 log1p(|u_g(x+r_i·d_j)|/sq),修正场得以
    表示"随引导场局部几何变化"的形状修正;15 个配置共享同一几何→修正映射。
    有界 ±corr_max、末层零初始化(从引导解出发),与 champion 同配方训练。
    """

    def __init__(self, hidden_layers=4, hidden_neurons=128, n_freq=8, corr_max=3.0,
                 radii=(0.5, 1.5, 4.0), n_dirs=8):
        super().__init__()
        offs = patch_offsets(radii, n_dirs)
        self.register_buffer("patch_off",
                             torch.tensor(offs, dtype=torch.float64),
                             persistent=False)
        self.mlp = OperatorMLP(n_params=2, hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons, n_freq=n_freq,
                               n_extra=offs.shape[0])
        self.corr_max = float(corr_max)

    def patch_feats(self, x, masses, xs, Ps, Ss, sq):
        """邻域引导场特征 (N,K):log1p(|u_g(x+off)|/sq)。仅 no-grad 调用。"""
        pts = (x.unsqueeze(1) + self.patch_off.to(x.dtype)).reshape(-1, 3)
        pug = physics.guide_u(pts, masses, xs, Ps, Ss).reshape(x.shape[0], -1)
        return torch.log1p(pug.abs() / sq)

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        w = (ug - wmin) / (wmax - wmin + 1e-8)
        feats = torch.stack([torch.log1p(ug.abs() / sq), w], dim=-1)
        pf = self.patch_feats(x, masses, xs, Ps, Ss, sq).to(ug.dtype)
        feats = torch.cat([feats, pf], dim=-1)
        raw = self.mlp(x, params.to(x.dtype), feats).squeeze(-1)
        cm = self.corr_max
        psi = cm * torch.tanh(raw / cm)
        return kappa * ug * (1.0 + w * psi)

    def forward_from_parts(self, x, params, kappa, ug, w, sq=None, feats=None):
        """快路径:ug/w 为预计算常数,feats 为 patch 特征 (N,K)(no-grad 预计算,
        不参与反传);逐点特征 [log1p(|u_g|/sq), w] 在此与 patch 拼接。
        返回 (u, phi, psi),ψ 只经 MLP 反传。"""
        if feats is None:
            raise ValueError("opv2 的 forward_from_parts 需要预计算 patch feats")
        pw = torch.stack([torch.log1p(ug.abs() / sq), w], dim=-1)
        raw = self.mlp(x, params.to(x.dtype), torch.cat([pw, feats], dim=-1)).squeeze(-1)
        cm = self.corr_max
        psi = cm * torch.tanh(raw / cm)
        phi = w * psi
        return kappa * ug * (1.0 + phi), phi, psi


class OperatorV3Ansatz(nn.Module):
    """A2-1 v3:DeepONet 式 branch-trunk 自由场修正算子(设计文档 §2)。

    u = κ·u_g + Δ_θ,  Δ_θ = Σ_k b_k(u_g; p) · t_k(x) · χ(x)
      - branch:输入 = 引导场在全局传感器点集(半径×Fibonacci 方向)上的
        log1p(|u_g|/sq) 采样 + 参数编码 [log10 q, m2/5],输出 p 个系数;
      - trunk:坐标正弦编码 → p 个基函数;
      - χ:奇点抑制窗 ∏_k (1−exp(−r_k²/rc²)),r_c=0.3;
      - branch 末层零初始化 → 从 κ·u_g 出发;Δ 为加性自由场,远场不失杠杆
        (针对 champion2 诊断的第③层误差:远场引导解形状误差)。
    """

    def __init__(self, hidden_layers=4, hidden_neurons=128, n_freq=8,
                 n_basis=128, radii=(0.5, 1.5, 4.0), n_dirs=8, rc=0.3):
        super().__init__()
        offs = patch_offsets(radii, n_dirs)
        self.register_buffer("patch_off", torch.tensor(offs, dtype=torch.float64),
                             persistent=False)
        self.n_freq = n_freq
        self.rc = float(rc)
        H = hidden_neurons
        # branch:传感器特征 (n_off) + 参数 2 → 系数 n_basis
        self.branch = nn.Sequential(nn.Linear(offs.shape[0] + 2, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.branch.append(nn.Linear(H, H))
            self.branch.append(nn.SiLU())
        self.branch.append(nn.Linear(H, n_basis))
        # trunk:坐标编码(3 + 3*2*n_freq)→ n_basis
        coord_dim = 3 + 3 * 2 * n_freq
        self.trunk = nn.Sequential(nn.Linear(coord_dim, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.trunk.append(nn.Linear(H, H))
            self.trunk.append(nn.SiLU())
        self.trunk.append(nn.Linear(H, n_basis))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def _embed_x(self, x):
        emb = [x]
        for i in range(self.n_freq):
            f = float(np.exp(i))
            emb.append(torch.sin(x * f))
            emb.append(torch.cos(x * f))
        return torch.cat(emb, dim=-1)

    def sensor_feats(self, x, masses, xs, Ps, Ss, sq):
        """全局传感器:在固定偏移点采样引导场(共享同一批查询点邻域——
        实际用查询点邻域 patch 作为传感器读数,保持真泛函输入)。"""
        pts = (x.unsqueeze(1) + self.patch_off.to(x.dtype)).reshape(-1, 3)
        pug = physics.guide_u(pts, masses, xs, Ps, Ss).reshape(x.shape[0], -1)
        return torch.log1p(pug.abs() / sq)

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        return kappa * ug + self.delta(x, masses, xs, Ps, Ss, params,
                                       kappa, ug, sq)

    def delta(self, x, masses, xs, Ps, Ss, params, kappa, ug, sq, feats=None):
        if feats is None:
            feats = self.sensor_feats(x, masses, xs, Ps, Ss, sq).to(ug.dtype)
        pin = params.to(x.dtype)
        if pin.shape[0] == 1 and x.shape[0] > 1:
            pin = pin.expand(x.shape[0], -1)
        b = self.branch(torch.cat([feats, pin], dim=-1))
        t = self.trunk(self._embed_x(x))
        d = (b * t).sum(-1, keepdim=False)
        r1 = (x - xs[0]).norm(dim=1)
        r2 = (x - xs[1]).norm(dim=1)
        chi = (1.0 - torch.exp(-r1 ** 2 / self.rc ** 2)) * \
              (1.0 - torch.exp(-r2 ** 2 / self.rc ** 2))
        return d * chi

    def forward_from_parts(self, x, params, kappa, ug, w, sq=None, feats=None):
        """快路径:ug/w 预计算。返回 (u, phi, psi),phi=Δ/(κ·u_g) 仅诊断用,
        psi=Δ。反传只经 branch/trunk。"""
        if feats is None:
            raise ValueError("opv3 的 forward_from_parts 需要预计算 patch feats")
        masses = None
        # 快路径无法访问 masses/xs → 由调用方在 feats 中给全;delta 需 xs 求 chi,
        # 这里以 x 直接近似:chi 由预计算 feats 之外的 cinfo 提供——简化为
        # 调用方通过 w 不再依赖。opv3 训练脚本统一走 forward()。
        raise NotImplementedError("opv3 训练请使用 forward 路径(见 a2q_train opv3)")


def fibonacci_dirs_v3(n):
    return fibonacci_dirs(n)


def _freq_embed(x, freqs):
    """截断频率正弦编码:频率表显式给定(默认 [1,2,4,8,16]),
    替代旧 e^i 指数频率(最高 e^7≈1096,是 opv3 修正场高频抖动的来源)。"""
    emb = [x]
    for f in freqs:
        emb.append(torch.sin(x * f))
        emb.append(torch.cos(x * f))
    return torch.cat(emb, dim=-1)


class OperatorV5Ansatz(nn.Module):
    """A2-1 v5:无窗自由场算子 u = κ·u_g + Δ_θ(x, p; u_g 特征)。

    与 opv3 的差别(2026-09-01 峰区回退诊断):
      1. 去掉奇点抑制窗 χ —— opv3 在 r≲0.5 被锁死在欠冲的引导解上,
         是峰高欠冲 ~18% 的直接原因;v5 的 Δ 处处激活(含峰区),
         配合 refsub_v2 近峰壳加密监督直接修峰;
      2. 频率截断(默认 [1,2,4,8,16])抑制 branch-trunk 高频抖动
         (opv3 误差曲线的高频噪声);
      3. 训练用纯 L_ref 监督(无 PDE 正则)。
    """

    def __init__(self, hidden_layers=4, hidden_neurons=128, n_basis=128,
                 radii=(0.5, 1.5, 4.0), n_dirs=8,
                 freqs=(1.0, 2.0, 4.0, 8.0, 16.0)):
        super().__init__()
        offs = patch_offsets(radii, n_dirs)
        self.register_buffer("patch_off",
                             torch.tensor(offs, dtype=torch.float64),
                             persistent=False)
        self.register_buffer("freq_buf",
                             torch.tensor(freqs, dtype=torch.float64),
                             persistent=False)
        H = hidden_neurons
        coord_dim = 3 + 3 * 2 * len(freqs)
        self.branch = nn.Sequential(nn.Linear(offs.shape[0] + 2, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.branch.append(nn.Linear(H, H))
            self.branch.append(nn.SiLU())
        self.branch.append(nn.Linear(H, n_basis))
        self.trunk = nn.Sequential(nn.Linear(coord_dim, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.trunk.append(nn.Linear(H, H))
            self.trunk.append(nn.SiLU())
        self.trunk.append(nn.Linear(H, n_basis))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # 只零初始化 branch 末层(Δ 从 κ·u_g 出发);trunk 末层必须非零,
        # 否则 b=0 且 t=0,双线性形式 Δ=b·t 处于全零鞍点,梯度恒为零
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def _embed_x(self, x):
        return _freq_embed(x, self.freq_buf.to(x.dtype))

    def sensor_feats(self, x, masses, xs, Ps, Ss, sq):
        pts = (x.unsqueeze(1) + self.patch_off.to(x.dtype)).reshape(-1, 3)
        pug = physics.guide_u(pts, masses, xs, Ps, Ss).reshape(x.shape[0], -1)
        return torch.log1p(pug.abs() / sq)

    def delta(self, x, masses, xs, Ps, Ss, params, ug, sq):
        feats = self.sensor_feats(x, masses, xs, Ps, Ss, sq).to(ug.dtype)
        pin = params.to(x.dtype)
        if pin.shape[0] == 1 and x.shape[0] > 1:
            pin = pin.expand(x.shape[0], -1)
        b = self.branch(torch.cat([feats, pin], dim=-1))
        t = self.trunk(self._embed_x(x))
        return (b * t).sum(-1)

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        return kappa * ug + self.delta(x, masses, xs, Ps, Ss, params, ug, sq)


class OperatorV4Ansatz(nn.Module):
    """A2-1 v4:混合 ansatz(champion2 峰区机制 + opv3 远场机制)。

        u = κ·u_g·(1 + w·ψ_N(x, p; [log1p(|u_g|/sq), w])) + Δ_F·χ_far

      - ψ_N:近场乘性修正(随 u_g 缩放 → 能修峰高,champion2 已证),
        有界 ±1(tanh),末层零初始化;
      - Δ_F:branch-trunk 加性自由场(管中远场形状,opv3 已证),
        χ_far = σ((ρ−0.8)/0.25),ρ=min(r1,r2) —— 近峰关闭(由 ψ_N 负责),
        中远场全开;
      - 全部频率截断 [1,2,4,8,16](光滑化)。
    """

    def __init__(self, hidden_layers=4, hidden_neurons=128, n_basis=128,
                 radii=(0.5, 1.5, 4.0), n_dirs=8,
                 freqs=(1.0, 2.0, 4.0, 8.0, 16.0),
                 near_cut=0.8, near_width=0.25):
        super().__init__()
        offs = patch_offsets(radii, n_dirs)
        self.register_buffer("patch_off",
                             torch.tensor(offs, dtype=torch.float64),
                             persistent=False)
        self.register_buffer("freq_buf",
                             torch.tensor(freqs, dtype=torch.float64),
                             persistent=False)
        self.near_cut = float(near_cut)
        self.near_width = float(near_width)
        H = hidden_neurons
        coord_dim = 3 + 3 * 2 * len(freqs)
        # branch/trunk(远场自由场)
        self.branch = nn.Sequential(nn.Linear(offs.shape[0] + 2, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.branch.append(nn.Linear(H, H))
            self.branch.append(nn.SiLU())
        self.branch.append(nn.Linear(H, n_basis))
        self.trunk = nn.Sequential(nn.Linear(coord_dim, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.trunk.append(nn.Linear(H, H))
            self.trunk.append(nn.SiLU())
        self.trunk.append(nn.Linear(H, n_basis))
        # near MLP:coord 编码(3+3·2F) + feats(2) + param 编码(2+2·2F)→ 标量
        pin_dim = 2 + 2 * 2 * len(freqs)
        self.near = nn.Sequential(
            nn.Linear(coord_dim + 2 + pin_dim, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.near.append(nn.Linear(H, H))
            self.near.append(nn.SiLU())
        self.near.append(nn.Linear(H, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # branch/trunk 只零初始化 branch(同 V5 鞍点论证);near 为普通链,
        # 末层零初始化即可
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def _embed_x(self, x):
        return _freq_embed(x, self.freq_buf.to(x.dtype))

    def _embed_p(self, p):
        return _freq_embed(p, self.freq_buf.to(p.dtype))

    def sensor_feats(self, x, masses, xs, Ps, Ss, sq):
        pts = (x.unsqueeze(1) + self.patch_off.to(x.dtype)).reshape(-1, 3)
        pug = physics.guide_u(pts, masses, xs, Ps, Ss).reshape(x.shape[0], -1)
        return torch.log1p(pug.abs() / sq)

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        w = (ug - wmin) / (wmax - wmin + 1e-8)
        feats_patch = self.sensor_feats(x, masses, xs, Ps, Ss, sq).to(ug.dtype)
        feats_pt = torch.stack([torch.log1p(ug.abs() / sq), w], dim=-1)
        pin = params.to(x.dtype)
        if pin.shape[0] == 1 and x.shape[0] > 1:
            pin = pin.expand(x.shape[0], -1)
        psi = torch.tanh(self.near(torch.cat(
            [self._embed_x(x), feats_pt, self._embed_p(pin)],
            dim=-1)).squeeze(-1))
        b = self.branch(torch.cat([feats_patch, pin], dim=-1))
        t = self.trunk(self._embed_x(x))
        dF = (b * t).sum(-1)
        r1 = (x - xs[0]).norm(dim=1)
        r2 = (x - xs[1]).norm(dim=1)
        rho = torch.minimum(r1, r2)
        chi_far = torch.sigmoid((rho - self.near_cut) / self.near_width)
        return kappa * ug * (1.0 + w * psi) + dF * chi_far


class OperatorV6Ansatz(nn.Module):
    """A2-1 v6:v4 + 远场乘性通道 m_F(2026-09-02 R6)。

    R5(v4c)诊断:远场加权后 global −20%(q10 首过)但 q1 仍 2.14e-2,
    剩余误差仍集中于 far——κ*_spec 是全局单一尺度,引导解远场形状误差
    意味着峰区最优 κ ≠ 远场最优 κ,加性 Δ_F 吸收"比例型"远场偏差效率低。

        u = κ·u_g·(1 + w·ψ_N + χ_far·m_F) + Δ_F·χ_far

    m_F:低频(freqs_far=(0.5,1,2))有界 tanh MLP,末层零初始化,χ_far 门控
    —— 近峰机械关断(峰区由 ψ_N 负责),远场给出随位置平滑变化的
    有效 κ 修正,直接吸收"κ 折衷 + 引导解形状"的比例型误差。
    """

    def __init__(self, hidden_layers=4, hidden_neurons=128, n_basis=128,
                 radii=(0.5, 1.5, 4.0), n_dirs=8,
                 freqs=(1.0, 2.0, 4.0, 8.0, 16.0),
                 freqs_far=(0.5, 1.0, 2.0),
                 near_cut=0.8, near_width=0.25):
        super().__init__()
        offs = patch_offsets(radii, n_dirs)
        self.register_buffer("patch_off",
                             torch.tensor(offs, dtype=torch.float64),
                             persistent=False)
        self.register_buffer("freq_buf",
                             torch.tensor(freqs, dtype=torch.float64),
                             persistent=False)
        self.register_buffer("freq_far_buf",
                             torch.tensor(freqs_far, dtype=torch.float64),
                             persistent=False)
        self.near_cut = float(near_cut)
        self.near_width = float(near_width)
        H = hidden_neurons
        coord_dim = 3 + 3 * 2 * len(freqs)
        coord_dim_far = 3 + 3 * 2 * len(freqs_far)
        pin_dim = 2 + 2 * 2 * len(freqs)
        pin_dim_far = 2 + 2 * 2 * len(freqs_far)
        self.branch = nn.Sequential(nn.Linear(offs.shape[0] + 2, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.branch.append(nn.Linear(H, H))
            self.branch.append(nn.SiLU())
        self.branch.append(nn.Linear(H, n_basis))
        self.trunk = nn.Sequential(nn.Linear(coord_dim, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.trunk.append(nn.Linear(H, H))
            self.trunk.append(nn.SiLU())
        self.trunk.append(nn.Linear(H, n_basis))
        self.near = nn.Sequential(
            nn.Linear(coord_dim + 2 + pin_dim, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.near.append(nn.Linear(H, H))
            self.near.append(nn.SiLU())
        self.near.append(nn.Linear(H, 1))
        self.mfar = nn.Sequential(
            nn.Linear(coord_dim_far + 2 + pin_dim_far, H), nn.SiLU())
        for _ in range(hidden_layers - 1):
            self.mfar.append(nn.Linear(H, H))
            self.mfar.append(nn.SiLU())
        self.mfar.append(nn.Linear(H, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # branch/trunk 只零初始化 branch(双线性鞍点论证);near/mfar 为普通
        # 链,末层零初始化(模型从 κ·u_g 精确出发)
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)
        nn.init.zeros_(self.near[-1].weight)
        nn.init.zeros_(self.near[-1].bias)
        nn.init.zeros_(self.mfar[-1].weight)
        nn.init.zeros_(self.mfar[-1].bias)

    def _embed_x(self, x, freq_buf=None):
        fb = self.freq_buf if freq_buf is None else freq_buf
        return _freq_embed(x, fb.to(x.dtype))

    def _embed_p(self, p, freq_buf=None):
        fb = self.freq_buf if freq_buf is None else freq_buf
        return _freq_embed(p, fb.to(p.dtype))

    def sensor_feats(self, x, masses, xs, Ps, Ss, sq):
        pts = (x.unsqueeze(1) + self.patch_off.to(x.dtype)).reshape(-1, 3)
        pug = physics.guide_u(pts, masses, xs, Ps, Ss).reshape(x.shape[0], -1)
        return torch.log1p(pug.abs() / sq)

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        w = (ug - wmin) / (wmax - wmin + 1e-8)
        feats_patch = self.sensor_feats(x, masses, xs, Ps, Ss, sq).to(ug.dtype)
        feats_pt = torch.stack([torch.log1p(ug.abs() / sq), w], dim=-1)
        pin = params.to(x.dtype)
        if pin.shape[0] == 1 and x.shape[0] > 1:
            pin = pin.expand(x.shape[0], -1)
        psi = torch.tanh(self.near(torch.cat(
            [self._embed_x(x), feats_pt, self._embed_p(pin)],
            dim=-1)).squeeze(-1))
        b = self.branch(torch.cat([feats_patch, pin], dim=-1))
        t = self.trunk(self._embed_x(x))
        dF = (b * t).sum(-1)
        r1 = (x - xs[0]).norm(dim=1)
        r2 = (x - xs[1]).norm(dim=1)
        rho = torch.minimum(r1, r2)
        chi_far = torch.sigmoid((rho - self.near_cut) / self.near_width)
        mF = torch.tanh(self.mfar(torch.cat(
            [self._embed_x(x, self.freq_far_buf), feats_pt,
             self._embed_p(pin, self.freq_far_buf)],
            dim=-1)).squeeze(-1))
        return kappa * ug * (1.0 + w * psi + chi_far * mF) + dF * chi_far


class OperatorV7Ansatz(OperatorV6Ansatz):
    """A2-1 v7:v6 + 零初始化"针尖通道"(2026-09-12)。

    动机(needle_profile.py 实测, tq100/v6l):
    - 峰比值 h(q=100) = 0.87,模型在小黑洞针尖(x=+3, d<0.4)低估 18%;
    - 300 步纯针尖微调后仅 0.819 -> 0.830 即平台 —— **容量问题**而非采样;
    - 近场通道频率编码最高 16,而针尖特征宽度 ~0.2-0.4(需 freq ~30+);
      且 psi_N 为 tanh 有界乘性,修尖锐增量峰的动态范围受限。

    结构:tip 通道与 near 同构但独立参数,自带高频编码 (8,16,32,64),
    输出 tanh 有界,以乘性方式叠加进近场修正项 1 + w*(psi_N + g_tip)。
    末层零初始化 => g_tip = 0,初始模型与 v6 **逐位等价**,可无损热启动
    (v6 state_dict 以 strict=False 载入,tip 参数保持零初始化)。
    """

    def __init__(self, hidden_layers=4, hidden_neurons=128, n_basis=128,
                 radii=(0.5, 1.5, 4.0), n_dirs=8,
                 freqs=(1.0, 2.0, 4.0, 8.0, 16.0),
                 freqs_far=(0.5, 1.0, 2.0),
                 freqs_tip=(8.0, 16.0, 32.0, 64.0),
                 near_cut=0.8, near_width=0.25, tip_neurons=64):
        super().__init__(hidden_layers=hidden_layers,
                         hidden_neurons=hidden_neurons, n_basis=n_basis,
                         radii=radii, n_dirs=n_dirs, freqs=freqs,
                         freqs_far=freqs_far, near_cut=near_cut,
                         near_width=near_width)
        self.register_buffer("freq_tip_buf",
                             torch.tensor(freqs_tip, dtype=torch.float64),
                             persistent=False)
        coord_dim_tip = 3 + 3 * 2 * len(freqs_tip)
        pin_dim_tip = 2 + 2 * 2 * len(freqs_tip)
        self.tip = nn.Sequential(
            nn.Linear(coord_dim_tip + 2 + pin_dim_tip, tip_neurons), nn.SiLU(),
            nn.Linear(tip_neurons, tip_neurons), nn.SiLU(),
            nn.Linear(tip_neurons, 1))
        for m in self.tip.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.tip[-1].weight)
        nn.init.zeros_(self.tip[-1].bias)

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        w = (ug - wmin) / (wmax - wmin + 1e-8)
        feats_pt = torch.stack([torch.log1p(ug.abs() / sq), w], dim=-1)
        pin = params.to(x.dtype)
        if pin.shape[0] == 1 and x.shape[0] > 1:
            pin = pin.expand(x.shape[0], -1)
        psi = torch.tanh(self.near(torch.cat(
            [self._embed_x(x), feats_pt, self._embed_p(pin)],
            dim=-1)).squeeze(-1))
        g_tip = torch.tanh(self.tip(torch.cat(
            [self._embed_x(x, self.freq_tip_buf), feats_pt,
             self._embed_p(pin, self.freq_tip_buf)],
            dim=-1)).squeeze(-1))
        b = self.branch(torch.cat([self.sensor_feats(
            x, masses, xs, Ps, Ss, sq).to(ug.dtype), pin], dim=-1))
        t = self.trunk(self._embed_x(x))
        dF = (b * t).sum(-1)
        r1 = (x - xs[0]).norm(dim=1)
        r2 = (x - xs[1]).norm(dim=1)
        rho = torch.minimum(r1, r2)
        chi_far = torch.sigmoid((rho - self.near_cut) / self.near_width)
        mF = torch.tanh(self.mfar(torch.cat(
            [self._embed_x(x, self.freq_far_buf), feats_pt,
             self._embed_p(pin, self.freq_far_buf)],
            dim=-1)).squeeze(-1))
        return (kappa * ug * (1.0 + w * (psi + g_tip) + chi_far * mF)
                + dF * chi_far)


class OperatorV8Ansatz(OperatorV6Ansatz):
    """A2-1 v8:v6 + 可学习高斯针尖凸起(2026-09-12)。

    依据(gaussian_tip_fit.py 实测, tq100/v6l):
    - 针尖亏损 D = u_ref - u_model 在小黑洞近域被
      a*exp(-(r/0.158)^2) + c 解释 98.2% 的方差;
    - opv7 的 MLP tip 通道因乘性链条( kappa*u_g ~ 1e-6 )梯度条件差,
      3000 步后 g_tip rms 仅 0.0033, 未激活;
    - 高斯振幅的梯度路径是直接的( d u/d a = exp(-(r/sigma)^2) = O(1) ),
      条件数与振幅尺度无关。

    结构:u += A_1*exp(-(r_1/s_1)^2) + A_2*exp(-(r_2/s_2)^2)
    其中 A_i = amp_scale*kappa*sq*tanh(raw_a_i)(零初始化 => 初始与 v6 逐位
    等价), s_i = sigma0*exp(tanh(raw_s_i)*ln4) (初值 sigma0=0.16, 即实测
    亏损宽度)。raw 由 tiny head(pin -> 16 -> 4) 从配置参数生成, 使凸起
    随 q 连续变化(大 q 亏损大, 中段 q 亏损小)。

    v8a 教训:amp 锚点须放大 —— tq100 所需凸起 8.4e-7 = 2.3*kappa*sq,
    v8a 的 tanh 范围(+/-1*kappa*sq)不够;且 Adam 每步只走 ~lr,
    tip_head 需单独参数组提高学习率(见 a2q_train_v4.py)。
    """

    def __init__(self, hidden_layers=4, hidden_neurons=128, n_basis=128,
                 radii=(0.5, 1.5, 4.0), n_dirs=8,
                 freqs=(1.0, 2.0, 4.0, 8.0, 16.0),
                 freqs_far=(0.5, 1.0, 2.0),
                 near_cut=0.8, near_width=0.25,
                 sigma0=0.16, sigma_span=4.0, head_hidden=16,
                 amp_scale=8.0):
        super().__init__(hidden_layers=hidden_layers,
                         hidden_neurons=hidden_neurons, n_basis=n_basis,
                         radii=radii, n_dirs=n_dirs, freqs=freqs,
                         freqs_far=freqs_far, near_cut=near_cut,
                         near_width=near_width)
        # param_vec = [log10(q), m2/5] 已是对数尺度编码, head 直接吃 2 维
        self.tip_head = nn.Sequential(
            nn.Linear(2, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, 4))
        for m in self.tip_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.tip_head[-1].weight)
        nn.init.zeros_(self.tip_head[-1].bias)
        self.sigma0 = float(sigma0)
        self.log_sigma_span = float(np.log(sigma_span))
        # tanh=1 时凸起最大 amp_scale*kappa*sq。tq100 实测亏损 8.4e-7 =
        # 2.3*kappa*sq, 故 amp_scale=8 给足余量且仍低于 u_tip(4.6e-6)
        self.amp_scale = float(amp_scale)

    def forward(self, x, masses, xs, Ps, Ss, params, kappa, wmin, wmax, sq):
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(x.dtype)
        w = (ug - wmin) / (wmax - wmin + 1e-8)
        feats_pt = torch.stack([torch.log1p(ug.abs() / sq), w], dim=-1)
        pin = params.to(x.dtype)
        if pin.shape[0] == 1 and x.shape[0] > 1:
            pin = pin.expand(x.shape[0], -1)
        psi = torch.tanh(self.near(torch.cat(
            [self._embed_x(x), feats_pt, self._embed_p(pin)],
            dim=-1)).squeeze(-1))
        b = self.branch(torch.cat([self.sensor_feats(
            x, masses, xs, Ps, Ss, sq).to(ug.dtype), pin], dim=-1))
        t = self.trunk(self._embed_x(x))
        dF = (b * t).sum(-1)
        r1 = (x - xs[0]).norm(dim=1)
        r2 = (x - xs[1]).norm(dim=1)
        rho = torch.minimum(r1, r2)
        chi_far = torch.sigmoid((rho - self.near_cut) / self.near_width)
        mF = torch.tanh(self.mfar(torch.cat(
            [self._embed_x(x, self.freq_far_buf), feats_pt,
             self._embed_p(pin, self.freq_far_buf)],
            dim=-1)).squeeze(-1))
        raw = self.tip_head(pin)
        amp1 = (kappa * sq * self.amp_scale * torch.tanh(raw[..., 0]))
        amp2 = (kappa * sq * self.amp_scale * torch.tanh(raw[..., 1]))
        s1 = self.sigma0 * torch.exp(
            torch.tanh(raw[..., 2]) * self.log_sigma_span)
        s2 = self.sigma0 * torch.exp(
            torch.tanh(raw[..., 3]) * self.log_sigma_span)
        bump = (amp1 * torch.exp(-(r1 / s1) ** 2)
                + amp2 * torch.exp(-(r2 / s2) ** 2))
        return (kappa * ug * (1.0 + w * psi + chi_far * mF) + dF * chi_far
                + bump)


def fibonacci_dirs_v4(n):
    return fibonacci_dirs(n)


def make_model(variant, device, hidden_layers=4, hidden_neurons=128,
               n_basis=128):
    # champion 用基线 ansatz(容量已证够用,瓶颈在监督稀释与 κ,见报告 §5.3/5.4)
    if variant == "operator":
        model = OperatorAnsatz(hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons)
    elif variant == "opv2":
        model = OperatorV2Ansatz(hidden_layers=hidden_layers,
                                 hidden_neurons=hidden_neurons)
    elif variant == "opv3":
        model = OperatorV3Ansatz(hidden_layers=hidden_layers,
                                 hidden_neurons=hidden_neurons,
                                 n_basis=n_basis)
    elif variant == "opv4":
        model = OperatorV4Ansatz(hidden_layers=hidden_layers,
                                 hidden_neurons=hidden_neurons,
                                 n_basis=n_basis)
    elif variant == "opv5":
        model = OperatorV5Ansatz(hidden_layers=hidden_layers,
                                 hidden_neurons=hidden_neurons,
                                 n_basis=n_basis)
    elif variant == "opv6":
        model = OperatorV6Ansatz(hidden_layers=hidden_layers,
                                 hidden_neurons=hidden_neurons,
                                 n_basis=n_basis)
    elif variant == "opv7":
        model = OperatorV7Ansatz(hidden_layers=hidden_layers,
                                 hidden_neurons=hidden_neurons,
                                 n_basis=n_basis)
    elif variant == "opv8":
        model = OperatorV8Ansatz(hidden_layers=hidden_layers,
                                 hidden_neurons=hidden_neurons,
                                 n_basis=n_basis)
    elif variant == "c2":
        model = BaselineAnsatz(hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons)
    else:
        model = BaselineAnsatz(hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons)
    return model.to(device)


def param_vec(q, m2, device, dtype=torch.float64):
    """参数编码 [log10(q), m2/5],shape (1,2)。"""
    return torch.tensor([[np.log10(q), m2 / 5.0]], dtype=dtype, device=device)


@torch.no_grad()
def predict_a2q(model, x, cinfo, device, chunk=16384):
    """批量预测。cinfo: 含 q,m2,kappa,sq,wmin,wmax 的 dict。"""
    model.eval()
    ma = torch.tensor([0.5, cinfo["m2"]], dtype=torch.float64, device=device)
    xs = torch.tensor([3.0, 0.0, 0.0, -3.0, 0.0, 0.0], dtype=torch.float64,
                      device=device).reshape(2, 3)
    Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64,
                      device=device)
    Ss = torch.zeros((2, 3), dtype=torch.float64, device=device)
    p = param_vec(cinfo["q"], cinfo["m2"], device)
    k = float(cinfo["kappa"])
    out = []
    for i in range(0, x.shape[0], chunk):
        out.append(model(torch.from_numpy(x[i:i + chunk]).float().to(device),
                         ma, xs, Ps, Ss, p, k,
                         float(cinfo["wmin"]), float(cinfo["wmax"]),
                         float(cinfo["sq"])).cpu().numpy())
    return np.concatenate(out)


def load_run(run_dir, device):
    """从 run 目录加载 model.pt → (model, meta)。"""
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    a = ckpt.get("args", {})
    model = make_model(ckpt["variant"], device,
                       hidden_neurons=a.get("hidden_neurons", 128),
                       n_basis=a.get("n_basis", 128))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
