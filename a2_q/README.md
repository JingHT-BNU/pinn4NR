# A2 —— 单参数 q 参数化 PINN

目标：训练**一个**网络，使其在质量比 `q = m2/m1 ∈ [1, 10]` 上任意取值时，
都能达到 A1 单配置 solve 的精度水平。

物理设定：`m1 = 0.5`，`m2 = 0.5q`，孔位 `x± = (±3, 0, 0)`，动量 `P± = (0, ±0.2, 0)`，无自旋。
训练集 45 个配置，其中 5 个（q15 / q25 / q50 / q74 / q86）为零样本 held-out。

**当前状态：2026-09-02 收官，Final 方案 opv6 / v6a 通过全部门槛。**

---

## 1. 文件速查

| 文件 | 作用 |
|---|---|
| `a2q_model.py` | ansatz 库，`make_model(variant, …)` 分发全部版本 |
| `a2q_train.py` | 第一世代训练器（baseline / champion / operator / opv2） |
| `a2q_train_v4.py` | **统一训练器**，支持 opv4 / opv5 / opv6 / c2（纯 L_ref 监督 + 光滑化正则） |
| `versions/opv3/a2q_train_opv3.py` | opv3 世代训练器（历史） |
| `a2q_eval2.py` | 新口径评估：global / peak / ring / valley / far 分区 L2RE + 峰高比 + 谷深比 |
| `a2q_gates.py` | 门槛判定（读 `data/gates_config.json`） |
| `a2q_rough.py` | 粗糙度诊断：射线二阶导符号翻转计数 |
| `a2q_ft_test.py` | 微调 / finetune 测试 |
| `data_pipeline/` | κ 预计算、参考解生成、refsub 数据集构建 |
| `diagnostics/` | κ 探针、算子探针、 Kopf-to-head 对比、分辨率收敛核查 |
| `legacy/` | 早期 `parametric_*` 版本，已被 `a2q_*` 取代 |

### 为什么需要新评估口径

旧官方口径（81³ 网格 / [-30,30]³ / rcut=0.3）格距 0.75，**对峰区几乎失明**。
新口径：窗口 [-10,10]³、N=161（快速档 81）、rcut=0.1，并显式区分
global / peak(r<1.5) / ring(r<0.5) / valley / far(r0>5) 分区指标。

A1 base 在新口径下的校准值 **T = 9.4954e-3**（峰高比 h=1.007，谷深比 d=0.989），
A2 门槛以它为基准标定。

---

## 2. 一键复现冠军模型

```bash
# 0) κ 缓存 + 参考解 + 数据集（只需做一次）
python a2_q/data_pipeline/precompute_kappa.py
python a2_q/data_pipeline/make_refs_v2.py --all              # ~8-10 min/配置
python a2_q/data_pipeline/post_refs_v2.py                    # → data/datasets/a2q_data_v2
python a2_q/data_pipeline/post_refsub_v3.py                  # → data/datasets/a2q_data_v3

# 1) 训练 Final 方案
python a2_q/a2q_train_v4.py --variant opv6 --exp-name a2q_v6a \
       --data-dir data/datasets/a2q_data_v3 \
       --steps 12000 --pts-per-cfg 2560 --lr 1e-4 --hidden-neurons 256

# 2) 评估 + 门槛 + 粗糙度
python a2_q/a2q_eval2.py --run data/runs/a2/a2q_v6a
python a2_q/a2q_gates.py --runs data/runs/a2/a2q_v6a
python a2_q/a2q_rough.py --run data/runs/a2/a2q_v6a
```

`a2q_train_v4.py` 带**配置指纹 + 断点续训**，中断后重跑同一条命令即自动续。

---

## 3. 版本谱系

| 版本 | ansatz 形式 | 结局 |
|---|---|---|
| `baseline` | 基线 ansatz | 参照基准 |
| `c2`（champion2） | `u = κu_g(1 + w·ψ)` 乘性 MLP | 峰区好；远场无自由度 |
| `operator` / `opv2` | branch–trunk 加性算子 | 体积改善，峰区回退 |
| `opv3` | 加性 Δ_θ + 奇点窗 χ | **体积突破**；峰区回退（h 0.74–0.86） |
| `opv4` | `u = κu_g(1+w·ψ_N) + Δ_F·χ_far` 混合 | 峰区全过，唯卡 global |
| `opv5` | 无窗加性 | 淘汰（h 0.80–0.91） |
| **`opv6`** | opv4 **+ 远场乘性通道 `m_F`** | **Final 方案，45/45 全过** |

### opv6 的关键设计（改代码前必读）

- **`m_F`**：低频截断编码 [0.5,1,2] + `tanh` 有界 + 末层零初始化，
  由 `χ_far = σ((ρ−0.8)/0.25)` 门控，**近峰机械关断**，专职修正远场。
  它修补的是结构性缺陷：κ\*_spec 是全局单一尺度，并不等于远场最优 κ。
- **频率截断编码** `_freq_embed([1,2,4,8,16])`：替代旧指数频率（最高约 1096，
  曾是 opv3 高频抖动的来源）。
- **双线性鞍点教训**：`branch·trunk` 的末层**不能同时零初始化**（梯度恒为零），
  只零初始化 branch；近场/远场 MLP 是普通链，末层零初始化即可。
- 参数编码 `[log10 q, m2/5]`；`m2 = 0.5q` 实际冗余，可作微优化剪掉。

### 六轮迭代的杠杆有效性排序

1. **`m_F` 远场乘性通道 = 突破点**：far 误差一击降 5–7 倍（2.44e-2 → 3.4e-3）。
2. **refsub_v3 远场加权**：global −20%（q10 首次过 G1），但收益递减。
3. **光滑化正则**（远场 Laplacian + 轴向凸性 hinge）：signflip 23–26 → 9.6–15。
4. **容量排除**：v4b 参数量 ×2.5，global 纹丝不动 → **容量/步数不是杠杆**。
5. 粗糙度量化（二阶导符号翻转计数）是"凸起/凹陷"的有效诊断。
6. 兜底降噪后处理（diffusion / Fourier）**未启用**（`m_F` 消除误差源后无必要）。

### 已排除的负结果

- **θ 噪声增强**（见 `versions/noise/SCHEME.md`）：3.23e-2 平台，无收益。
- **RAR 主动采样**（见 `versions/rar/SCHEME.md`）：3.20e-2 平台，无收益。
- lr 直接 5e-4 微调会**打碎**已收敛的解（L_ref 4e-5 → 1.06）；微调应用约 4e-5。

---

## 4. Final 结果（v6a）

- **45/45 配置（含 5 个零样本）全部通过 G1 / G2 / G3。**
- global L2RE **3.5–4.5e-3**，约为 A1 base 校准值的一半。
- 峰高比 h ∈ [0.989, 1.017]；可视化剖面肉眼不可分辨。
- 遗留：far 二阶导符号翻转仍有 9.6–15 次（未达个位数），但绝对振幅已低于视觉阈值。

---

## 5. 后续增强

优先补充**物理型指标**（独立于参考解，对论文尤其有价值）：

1. **Hamilton 约束残差场**——在 `u = ψ − ψ_sing` 光滑余量形式下报加权 RMS 与分布图，
   证明纯 L_ref 监督学出的仍是物理一致解；
2. **ADM 质量渐近多极拟合**——单极系数 vs 解析 M_ADM，偶极随 q 的连续性；
3. **对称性等变**——q=1 时的镜像不对称度；
4. **参数正则性** u(q) 曲线；
5. **演化适配性** case study。
