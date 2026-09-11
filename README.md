# pinn4NR

用物理信息神经网络（PINN）求解 **Bowen–York 双黑洞（BBH）初始数据** 的 Hamilton 约束方程，
并把单配置求解器逐步推进到**参数化求解器**（一个网络覆盖整个参数族）。

物理设定：求解共形因子 ψ 的光滑余量 `u = ψ − ψ_sing`（ψ_sing 为奇性解析主部）。
网络不直接学 u，而是学**解析引导解 u_g 的修正**：

```
u_θ = κ · u_g · (1 + w·ψ_θ) + Δ_F · χ_far      # A2 最终形式（opv6）
```

其中 κ 为全局尺度（QMC 预计算），w/χ_far 为窗函数与远场门控，ψ_θ/Δ_F 为网络输出。

---

## 1. 目录结构

本仓库由原来 15 个版本分支合并整理而来，**全部内容集中在单一 `main` 分支**，按"共享库 / 工具 / 子项目"分层：

```
pinn4NR/
├── README.md               本文件：结构说明 + 全部产物重新生成办法
├── requirements.txt
│
├── core/                   公共库（A1/A2/A3 共用，跨项目 import）
│   ├── physics.py          ψ_sing / KK / u_g / solve_kappa / Hamilton 残差
│   ├── data.py             配点采样 + 解析量缓存 + 参考解加载
│   ├── config.py           全部超参（dataclass）
│   ├── logutil.py          统一日志（控制台+文件，UTC+8）
│   └── parametric_model.py FiLM 条件调制基础模块
│
├── tools/                  参考解工具链（公共）
│   ├── spectral_reference.py   自研球谐谱参考解求解器（L=48 定案）
│   ├── make_reference.py        TwoPunctures 参考解生成
│   ├── make_refs_batch.py / make_refs_a2.py   批量参考解
│   ├── verify_ref_tip.py        针尖精度复核（自收敛 + FD 残差认证）
│   └── validate_with_tp.py      与 TwoPuncturesC 互证
│
├── a1_single/              A1：单配置（base, q=1）论文复现
├── a2_q/                   A2：单参数 q 参数化 PINN（主线，已收官）
│   ├── a2q_model.py        ansatz 库：baseline / operator / opv2…opv6 / c2
│   ├── a2q_train*.py       三个世代的训练器
│   ├── a2q_eval2.py        新口径评估器；a2q_gates.py 门槛；a2q_rough.py 粗糙度
│   ├── diagnostics/        探针与诊断脚本
│   ├── data_pipeline/      κ 预计算、参考解生成、数据集构建
│   ├── versions/           各历史版本特有文件（opv3 / noise / rar）
│   └── legacy/             早期 parametric_* 版本（已被 a2q_* 取代）
└── a3_multiparam/          A3：八维全参数（质量+位置+动量+自旋）参数化 PINN
```

> **路径自适应**：所有 `core/` 与 `tools/` 之外的脚本顶部插入了一段简短的
> `path bootstrap`，会自动把仓库根的 `core/` 与 `tools/` 加入 `sys.path`。
> 因此目录整理后的扁平 import（`import physics`、`from config import …`）依然可用，
> 脚本可以从任意工作目录启动。

---

## 2. 环境与硬件纪律

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # torch(CUDA) / numpy / matplotlib / scipy
```

- 开发环境：Windows + RTX 4070 Laptop 16 GB。
- **GPU 任务必须串行**：桌面已占约 10 GB，并发 GPU 任务会触发 CUDA sysmem 回退导致全局减速。
- 谱方法求值 `chunk ≤ 32768`（262144 曾导致 9.4 GB OOM）。
- 日志统一走 `core/logutil.py`，输出至 `logs/<项目>/`，时间戳为 UTC+8。

---

## 3. 三个子项目状态

| 项目 | 内容 | 状态 | 关键结果 |
|---|---|---|---|
| **A1** `a1_single/` | 单配置（q=1）PINN 复现 | 已完成 | 旧口径 L2RE ≈ 6.7e-3；新口径基准 T = 9.50e-3 |
| **A2** `a2_q/` | 单参数 q ∈ [1,10] 参数化 | **已收官** | opv6 / v6a：45/45 配置通过 G1/G2/G3，global 3.5–4.5e-3 |
| **A3** `a3_multiparam/` | 八维全参数空间 PINN | 进行中 | v5：base L2RE 0.52%（瓶颈在针尖修正） |

---

## 4. 重新生成全部产物

> **本仓库不含任何参考解、κ 缓存、训练数据集与模型权重** —— 它们体积大且完全可复现。
> 以下为从零重建的完整链路，均在**仓库根目录**执行，产物统一落在被 gitignore 的 `data/`。

### 4.1 κ 缓存（全局尺度）

```bash
# A2：单参数 q 的 κ（快）
python a2_q/data_pipeline/precompute_kappa.py
#   → data/kappa_cache.json

# A3：400 个 LHS 配置的 κ（约 40 分钟，2M Sobol 点 × 2 seed）
python a3_multiparam/multi_param_precompute.py --n-train 300 --n-val 100 --seed 42
#   → data/multi_param_kappa_cache.json
```

> A3 的 κ 有约 2–3% 的 QMC 噪声，网络可在训练中部分补偿；提高积分点数可减小噪声。

### 4.2 谱参考解（L=48 定案）

```bash
# 单个配置（参数为 m+, m-, x+, x-, P+_y, P-_y, S+_z, S-_z）
python tools/spectral_reference.py --params "0.5,0.5,3,-3,0.2,-0.2,0,0" \
       --out tools/refs/ref_base.npz

# A2 批量：15 个新配置 → data/refs/a2v2（约 8–10 分钟/配置，自动跳过已完成）
python a2_q/data_pipeline/make_refs_v2.py --all

# A3 批量：6 个代表配置
python tools/make_refs_batch.py
```

> **分辨率请用 L=48（径向 512 / 球谐 48）**。更高分辨率（N_r 768 / L 64+）默认延拓
> + Picard 迭代**不收敛**（实测 res_max=82），请勿使用。
> 需要更高精度时用 `tools/verify_ref_tip.py` 做自收敛与 FD 残差认证。

### 4.3 训练数据集

```bash
# v2：refsub 后的监督数据集 → data/datasets/a2q_data_v2
python a2_q/data_pipeline/post_refs_v2.py

# v3：在 v2 基础上增加远场 r0>5 加权（权重占比 29%→67%）→ data/datasets/a2q_data_v3
python a2_q/data_pipeline/post_refsub_v3.py
```

A2 最终（v6a）使用的是 **v3** 数据集。

### 4.4 训练模型

**A2 冠军模型（opv6 / v6a）—— 约 12000 步 × 2560 点/配置，4070 上 3.5–4 小时：**

```bash
python a2_q/a2q_train_v4.py --variant opv6 --exp-name a2q_v6a \
       --data-dir data/datasets/a2q_data_v3 \
       --steps 12000 --pts-per-cfg 2560 --lr 1e-4 --hidden-neurons 256
#   → data/runs/a2/a2q_v6a/{model.pt, ckpt.pt, history.json, configs.json}
```

训练器带**配置指纹 + 断点续训**：中途中断后用**完全相同的命令**重跑即可自动续训。
其他 ansatz 只需换 `--variant`（`operator` / `opv2` / `opv3` / `opv4` / `opv5` / `c2`）。

**A3 八维模型（约 1 秒/步，50000 步）：**

```bash
python a3_multiparam/multi_param_train.py --exp-name multi_param_v5 --steps 50000 \
       --hidden-layers 8 --reference <path-to-reference_u.npz>
```

> A3 的 base 参考解 `reference_u.npz`（47.7M 点）由 `tools/make_reference.py`
> 从 TwoPunctures 转换生成，不在仓库内；也可用 `tools/spectral_reference.py` 自行生成。

### 4.5 评估与门槛

```bash
python a2_q/a2q_eval2.py --run data/runs/a2/a2q_v6a     # 新口径全指标 → eval2.json + figs/
python a2_q/a2q_gates.py --runs data/runs/a2/a2q_v6a    # G1/G2/G3 门槛判定
python a2_q/a2q_rough.py --run data/runs/a2/a2q_v6a     # 远场粗糙度（二阶导符号翻转计数）
python a3_multiparam/multi_param_eval.py --exp runs/multi_param_v5
```

**A2 通过标准**（`data/gates_config.json`，基于 A1 base 新口径校准 T=9.495e-3）：
G1 global ≤ 1.25T；G2 峰高比 h ∈ [0.95,1.05] 且分区 pk ≤ 1.4e-2；
G3 谷深比 d ∈ [0.95,1.05] 且 vy ≤ 2.3e-2。训练配置与 5 个零样本配置同门槛。

---

## 5. A2 版本谱系与已排除的杠杆

完整的实战记录见 `a2_q/README.md`。这里只列结论，避免重复踩坑：

| 版本 | ansatz | 结局 |
|---|---|---|
| baseline / c2 | `u = κu_g(1+w·ψ)` 乘性 | 峰区好，远场缺自由度 |
| operator / opv2 | 加性算子（branch–trunk） | 体积改善，峰区回退 |
| opv3 | 加性 + 奇点窗 χ | 体积突破，峰区 h 0.74–0.86 |
| opv4 | 近乘性 + 远场加性 Δ_F·χ_far | 峰区全过，卡 global |
| opv5 | 无窗自由场 | 淘汰（h 0.80–0.91） |
| **opv6** | opv4 **+ 远场乘性通道 m_F** | **最终方案，45/45 全过** |

三条**已被排除**的杠杆（实验证伪，勿重复投入）：

1. **扩大网络容量 / 增加训练步数无效**（R3：参数量 ×2.5，global 纹丝不动）。
2. **纯 PDE 残差训练对参数化 PINN 无效**（A3 v1：残差对全局 κ 不敏感；
   必须参考监督 + 准确定标 κ）。
3. **θ 噪声增强、RAR 主动采样**。均为负结果。

---

## 6. 后续方向

- **A2**：补充**物理型指标**（独立于参考解）—— Hamilton 约束残差场、ADM 质量渐近多极
  拟合、对称性等变、u(q) 正则性，用于证明 `m_F` 是物理修正而非对参考解的过拟合。
- **A3**：瓶颈在针尖。P1 把窗函数的固定绝对尺度改为**逐配置 u_scale**（迁移 A2 的
  `m_F` 经验：单一全局尺度无法覆盖局部结构不同的区域）；P2 让 `tanh(h)` 能逐奇点取值。
