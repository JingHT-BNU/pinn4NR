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


def make_model(variant, device):
    # champion 用基线 ansatz(容量已证够用,瓶颈在监督稀释与 κ,见报告 §5.3/5.4)
    if variant == "operator":
        model = OperatorAnsatz()
    elif variant == "opv2":
        model = OperatorV2Ansatz()
    else:
        model = BaselineAnsatz()
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
    model = make_model(ckpt["variant"], device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
