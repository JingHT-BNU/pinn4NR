# 多参数参数化 PINN (A3) 运行指南

在 8 维物理参数空间（质量 + 位置 + 动量 + 自旋）训练单个 PINN 网络，
覆盖 BBH 初始数据求解的完整参数子空间。

---

## 概述

| 项目 | 说明 |
|---|---|
| **参数空间** | 8 维：m+, m-, x+, x-, P+y, P-y, S+z, S-z |
| **参数范围** | m∈[0.25,3], x∈[2,8]/[-8,-2], P∈[-0.5,0.5], S∈[0,0.4] |
| **采样方法** | Latin Hypercube Sampling (LHS)，300 训练 + 100 验证配置 |
| **网络架构** | 6×256 条件 MLP + FiLM 调制 + 正弦位置编码 (10/12 频率) |
| **可训练参数** | ~726K |
| **训练策略** | 每步随机 1 配置 + 参考监督(base case) + 课程衰减 |
| **κ 预计算** | 已完成：`multi_param_kappa_cache.json`（400 配置，200k QMC） |

---

## 1. 文件说明

```
paper/A3_multi_param/
├── multi_param_model.py       # 模型定义（8 维参数 + FiLM + 正弦编码）
├── multi_param_precompute.py  # κ 预计算（LHS 采样 + QMC 积分）
├── multi_param_train.py       # 训练脚本
├── multi_param_eval.py        # 评估脚本（PDE 自检 + L2RE + 参数敏感性）
├── multi_param_viz.py         # 可视化（损失曲线 / 剖面 / 残差分布）
├── multi_param_experiments.md # 实验记录与方法合集（v1-v5 历史/谱参考解/针尖复核/v5 诊断）
└── multi_param_kappa_cache.json   # κ 缓存（已生成，400 配置）

paper/tools/reference_u.npz    # base case 参考解（47.7M 点，已有）
paper/tools/spectral_reference.py  # 谱方法参考解求解器(任意参数→参考解+谱系数)
paper/tools/make_refs_batch.py     # 6 个代表配置参考解批量生成(L=48/N_r=512)
paper/tools/verify_ref_tip.py      # 参考解针尖精度复核(自收敛+FD 残差认证)
paper/tools/refs/ref_<label>.npz   # 6 个 viz 代表配置的 L=48 参考解(含谱系数)
```

> **v5（针尖改进版，2026-08-29 实现，服务器 50000 步已完成）**：sigmoid 可学习幅值
> + 针尖聚焦配点 + 体/针分离平衡 + 多配置谱参考监督 + 8×256 加深。结果：base
> L2RE 1.65%→**0.52%**（c=0.654）；逐配置针尖诊断、遗留问题与后备方案见
> **`multi_param_experiments.md` §四/§五**。启动命令与 v4 相同（exp-name 换成
> multi_param_v5，`--hidden-layers 8`）。

---

## 2. 运行步骤

### 前提

- Windows + 已有 `.venv`（torch CUDA 版）
- GPU: RTX 4070 Ti SUPER 16GB
- κ 缓存已就绪：`A3_multi_param/multi_param_kappa_cache.json`

### 步骤 1：训练

打开终端（cmd 或 PowerShell），执行（**建议 exp-name 用 `multi_param_a2`**，与已失败的 v1 `multi_param_a1` 区分）：

```bash
cd D:\AIs\PINN
.venv\Scripts\python.exe -u paper\A3_multi_param\multi_param_train.py --steps 30000 --lr 3e-4 --N-Omega 10000 --N-boundary 4000 --exp-name multi_param_a2 --out-dir runs --reference paper\tools\reference_u.npz
```

**日志自动存档（无需重定向）：** 输出同时打印到控制台（带 `HH:MM:SS` 时间戳）
并写入 `paper\logs\A3\multi_param_train_<启动时间>.log`，每次运行生成新文件、
不覆盖历史日志；异常堆栈同样记录在日志文件末尾。回传审查时直接发送该日志文件。

**预计时间：** ~0.9 秒/步，30000 步约 7.5 小时（RTX 4070 Ti SUPER）。

> 注意：v3 修复版的关键行为——`L_ref` 每步都显示数值且应持续下降；
> 若 L_ref 长期不降（如 v1 的 6.35e-3 平坦），说明有问题，先停止并回报。

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--steps` | 30000 | 训练步数 |
| `--lr` | 3e-4 | 学习率（CosineAnnealing 衰减） |
| `--N-Omega` | 10000 | 每配置内部采样点数 |
| `--N-boundary` | 4000 | 每配置边界采样点数 |
| `--exp-name` | multi_param_a1 | 实验名（输出目录名） |
| `--out-dir` | runs | 输出根目录（相对 paper/） |
| `--reference` | 无 | base case 参考解 .npz 路径 |
| `--kappa-cache` | 自动 | κ 缓存路径（默认同目录 json，**须为 version 2**） |
| `--n-ref` | 10000 | 每步参考监督采样点数（从全量 47.7M 重采样） |
| `--w-ref` | 自动(167) | 参考监督初始权重（课程衰减至 1/10） |
| `--hidden-layers` | 6 | 共享 MLP 层数 |
| `--hidden-neurons` | 256 | 每层神经元数 |
| `--n-freq-coord` | 10 | 坐标正弦编码频率数 |
| `--n-freq-param` | 12 | 参数正弦编码频率数 |

**预计时间：** ~1 秒/步，30000 步约 8 小时（RTX 4070 Ti SUPER）。

**如果时间紧张，可以减少步数：**
```bash
# ~2.5 小时，精度略降
.venv\Scripts\python.exe -u paper\A3_multi_param\multi_param_train.py --steps 10000 --exp-name multi_param_a2 --out-dir runs --reference paper\tools\reference_u.npz
```

> **v4 注意（2026-08-28）**：窗函数修复后训练配置指纹已变更——旧的
> multi_param_a2 检查点（窗函数 bug 时代训练的，L_ref 平坦无学习价值）会在
> 启动时自动备份为 `checkpoint.pt.incompatible.bak` 并从头训练，无需手动处理。

### 断点续训（随时中断，随时接续）

训练每 500 步（`--ckpt-every` 可调）自动保存检查点到
`paper/runs/<exp-name>/checkpoint.pt`（原子写入，含模型/优化器/调度器/EMA/
损失历史/随机数状态）。再次用**相同命令**启动时自动检测并续训：

```bash
# 中断后直接重跑同一条命令即可,输出会显示:
#   [resume] 从检查点继续: step N/30000 (上次 L_ref=...)
# 若已全部完成,则跳过训练直接评估。强制从头: 加 --no-resume
# 中途改 --steps(如 20000→30000)也可以: 余弦退火对剩余步数重新锚定
```

配置指纹校验：`lr / N_Omega / N_boundary / n_ref / w_ref / 网络结构 /
κ 缓存 seed` 任一改变都会拒绝续训并备份旧检查点为 `checkpoint.pt.incompatible.bak`
后从头开始，不会静默混用 incompatible 状态。

注意：输入路径（`--reference / --kappa-cache / --exp`）优先按当前工作目录的
相对路径解析，找不到时锚定到 `paper/` 目录；输出目录 `--out-dir` 锚定到
`paper/`。因此从 `D:\AIs\PINN` 或 `paper\` 目录运行、路径写成
`paper\tools\reference_u.npz` 或 `tools\reference_u.npz` 均可。

### 步骤 2：评估

训练完成后运行：

```bash
.venv\Scripts\python.exe -u paper\A3_multi_param\multi_param_eval.py --exp paper\runs\multi_param_a2 --reference paper\tools\reference_u.npz
```

### 步骤 3：可视化

```bash
.venv\Scripts\python.exe -u paper\A3_multi_param\multi_param_viz.py --exp paper\runs\multi_param_a2 --reference paper\tools\reference_u.npz
```

`--reference` 用于轴剖面图叠加谱方法参考解（仅 base 配置有参考解；其余配置以
引导解 u_g 作对照）。不加该参数也能生成图片，但剖面图无参考解曲线。

---

## 3. 训练输出解读

训练日志示例（v4 修复版实际输出，均为最近 200 步滚动均值）：
```
设备: cuda
加载 κ 缓存: D:\AIs\PINN\paper\A3_multi_param\multi_param_kappa_cache.json
  训练: 300, 验证: 100

准备数据 (N_Omega=10000, N_boundary=4000)...
  train_0000: κ=0.6350, ug∈[...]
  ...
全局 ug: [~7e-05, ~1.4]
Base case 索引: 0 (train_0000)
参考解: 47768505 点

构建模型 (6×256, FiLM, 正弦编码 10/12)...
参数量: 726,018
参考监督权重 w_ref0=167.0 (课程衰减至 17)

训练 30000 步...
[监督] 参考解全量 47768505 点, 每步重采样 10000 点
[step      1/30000] L2均=2.9e-11 LBC均=1.0e-11 L_ref均=6.4400e-03 total均=2.950e+00 w_ref=167 (1s)
[step    600/30000] L2均=... LBC均=... L_ref均=...(应明显下降) total均=... w_ref=161 (...)
...
```

**日志解读：**
- 所有数值均为**最近 200 步滚动均值**（单步值取决于随机抽中的配置，天然横跳；
  均值才能反映趋势。原始单步值全量记录在 history.json 供离线分析）
- `L2均`/`LBC均`: PDE/Robin 残差损失的滚动均值（当前窗口内随机配置混合）
- `L_ref均`: base case 参考监督损失（=L2RE²）的滚动均值，**必须持续下降**；
  若长期平坦（如 v3 的 6.35e-3 平坦 bug）说明有问题，先停止并回报
- `total均`: 总损失滚动均值；`w_ref`: 当前参考监督权重（课程衰减 167→17）
- 每隔 `steps/50` 步打印一行（默认每 600 步）

### 输出文件

训练完成后在 `paper/runs/multi_param_a2/` 下生成：

```
runs/multi_param_a2/
├── model.pt          # 模型权重 + 元数据
├── history.json      # 训练损失历史
├── configs.json      # 训练/验证配置列表
└── figs/             # (由 viz 脚本生成)
    ├── mp_loss_history.png
    ├── mp_param_sensitivity.png
    ├── mp_axis_profiles.png
    ├── mp_equatorial_difference.png   # 赤道面 (z=0) u_θ−参考解 差值图
    └── mp_pde_residual.png
```

---

## 4. 评估输出解读

评估脚本输出：
```
训练配置 PDE 残差自检:
  |R|_mean: 1.23e-07 (max=2.45e-07)

验证配置零样本 PDE 残差:
  |R|_mean: 1.45e-07 (max=3.12e-07)

Robin 边界残差:
  |R_B|_mean: 3.21e-04

Base case L2RE:
  L2RE = 0.00xx

  分区 L2RE:
    r<0.5       : 0.0xxxx (xxx pts)
    r[0.5,2]    : 0.0xxxx (xxx pts)
    r[2,10]     : 0.0xxxx (xxx pts)
    r>10        : 0.0xxxx (xxx pts)

参数敏感性 (u(0,0,0) vs 各参数):
  m+           : u(0)∈[x, x]
  ...
```

**关键指标：**
- `|R|_mean`: PDE 残差绝对值均值，越小越好。1e-7 量级为优秀
- `L2RE`: 与参考解的相对 L2 误差。base case 参考 A1 单配置结果 0.0067
- 训练 vs 验证 PDE 残差比值：接近 1 表示泛化良好

---

## 5. 需要你提供给我的输出

训练 + 评估完成后，请把以下内容发给我：

1. **训练日志文件**：`paper\logs\A3\multi_param_train_<启动时间>.log`（控制台输出已自动存档于此，无需截图复制）

2. **评估输出**（终端全文）

3. **如果方便，也可以发：**
   - `paper/runs/multi_param_a2/history.json` 的最后几行
   - 可视化生成的 4 张 PNG 图

---

## 6. κ 缓存(v2, 已生成)

当前缓存为 **version 2**：2M Sobol 点 × 2 个独立 seed 取平均、固定种子可复现。
base 配置 κ=0.6350，与 A2 的 5M QMC 精确值 0.63849 相差 0.54%。
(v1 缓存因随机扰动 + 点数不足导致 κ 误差高达 ±10%，已废弃。)

如需重新生成（例如调整参数范围）：

```bash
.venv\Scripts\python.exe -u paper\A3_multi_param\multi_param_precompute.py --n-train 300 --n-val 100 --seed 42
```

默认 2M 体积点 ×2 seed + 50k 表面点，400 配置约 40 分钟。

**训练修复记录(2026-08-26 v3)：**
- 参考监督改为每步计算（旧版仅在被随机抽中 base 时计算,概率 1/300）
- 参考权重 w_ref=167（A2 有效强度;旧版误设 3.3,弱了 50 倍）
- 损失平衡改为按配置 EMA（消除 total 横跳）
- eval/viz dtype 统一 double;修复 base 配置选择 bug

**路径解析修复(2026-08-28)：**
- 修复 `--reference` 相对路径被双重锚定成 `paper\paper\tools\...` 导致
  FileNotFoundError（v3 首次启动即崩的原因）
- 输入路径（reference / kappa-cache / exp）现优先按 CWD 相对路径解析，
  找不到才锚定到 `paper/`；输出目录 `--out-dir` 仍锚定 `paper/`
- eval / viz 的 `--exp` 同样处理，任意工作目录运行均可
- 已用 2 步冒烟训练 + 完整 eval 端到端验证通过

**窗函数修复(2026-08-28 v4，根因级修复)——详见 `multi_param_experiments.md`：**
- **根因**：旧窗函数 W=(ug−u_min)/(u_max−u_min) 用全部 300 配置的全局 ug 范围
  ([7e-5, 1.45]，跨 4 个数量级)，对 base 等大多数配置 W≈0.001~0.016，
  修正头被压到 1% 量级 → 30000 步 L_ref 平坦在 6.35e-3（=纯基线误差），
  base L2RE 卡在 0.0798。A2 能学是因为它只有 6 个近邻配置，全局范围≈单配置范围
- **修复**：W = tanh(ug/0.05)，逐点确定、与配置统计无关、训练/评估严格一致；
  远场→0（Robin BC 自动保持），中场 O(0.4~1)（修正有 ~20% 的行动能力）
- **验证**：300 步冒烟 L_ref 均值 6.44e-3 → 5.44e-3 单调下降无横跳（旧版 0 进展）；
  base L2RE 0.0798 → 0.0728（仅 300 步）
- 日志改为**最近 200 步滚动均值**显示（单步值随随机配置天然横跳，均值才有趋势意义）；
  轴剖面图重构：仅 x 轴、每配置一个子图、叠加谱方法参考解（base）/引导解（其余）
- 训练配置指纹新增 `win` 键：旧检查点自动失效备份，从头训练

**剖面图削峰修复(2026-08-29)——详见 `multi_param_experiments.md`：**
- 根因：奇点邻域 r<0.35 掩膜过度防御（guide_u 相消伪影只在精确命中奇点 r→0 时
  出现，实测 r=0.008 已干净），且被删点被 matplotlib 直线跨越 → 峰顶呈"斜切"假象；
  模型实际在峰值区与谱方法参考解重合（误差 ~-1.5%）
- 修复：剖面采样 400→3001 点、掩膜 0.35→0.02、缺口置 NaN 断线、
  引导解改画 **κ·u_g 基线**（与模型/参考解同尺度可比）
- v4 最终结果：Base L2RE=1.65%（c=0.2→0.3351 健康增长），详见 multi_param_experiments.md §5

**谱方法参考解生成器 + 可视化升级(2026-08-29)——详见 `multi_param_experiments.md`：**
- 新增自研球谐谱方法求解器 `paper/tools/spectral_reference.py`（参数→参考解 npz，
  不依赖 GSL）；批量驱动 `paper/tools/make_refs_batch.py`，输出 `paper/tools/refs/ref_<label>.npz`
- viz：剖面图 dpi→1000 + 细线；全部 6 个子图叠加各自参考解（`--refs-dir` 默认
  `paper/tools/refs/`，按参数自动匹配）；新增赤道面 (z=0) 差值图
  `mp_equatorial_difference.png`（发散色标，奇点邻域掩膜）

**针尖复核 + 参考解套件升级到 L=48(2026-08-29)——详见 `multi_param_experiments.md` §七：**
- `paper/tools/verify_ref_tip.py` 两项独立检查：L32/L48 分辨率自收敛 + 与求解器离散
  无关的 FD 局部 PDE 残差认证
- 结论：L=32 在轻质量+自旋奇点针尖**欠收敛 −20~26%**（0059 峰 0.395→0.495、0164
  0.0226→0.0305），重质量奇点针尖已收敛；6 个参考解全部按 **L=48/N_r=512** 重新生成
  （base vs TwoPunctures L2RE=8.8e-4 → **6.35e-5**，比 PINN 误差好 ~260 倍）
- 模型针尖过冲为真实问题：修正后偏差 +2.1×(0059) / +2.6×(0164)——训练看不到针尖
  （体积均匀配点）+ 幅值上限不足
- npz 新增 `radial_coeffs` 谱系数：`SpectralPunctureSolver.from_coefficients()` 免重解
  任意分辨率求值；viz 参考解改画**光滑实线**（针尖加密采样），赤道差值图参考解稠密
  求值 501²

**v5 ansatz 升级(2026-08-29，已实现待训)——详见 `multi_param_experiments.md`：**
- `multi_param_model.py`：幅值 c 改 sigmoid 参数化 c=c_max·σ(c_raw)（上限 c_max=1.0
  可调，旧模型按 checkpoint 内 amp_mode 自动兼容）
- `multi_param_train.py`：针尖聚焦配点（25% → 两奇点球壳 [0.02,2.5]）、体/针残差
  分开 EMA 平衡、多配置谱参考监督（`--sup-refs-dir`，监督点集 40% 针尖加密）
- 30 步冒烟通过；训练命令与监控要点见该 md（由用户启动，预计 ~1.3 s/步）

**v5 结果审查 + viz 断点修复 + 日志时区(2026-08-30)——详见 `multi_param_experiments.md` §四/§五：**
- v5 服务器训练完成：base L2RE=0.52%（v4 1.65%），c=0.654（未顶死上限）；
  0000/0164/0256 针尖大幅修正；0154/0234/0059 受"W 固定绝对尺度"与"tanh(h) 配置级
  常数"两机制限制仍未覆盖——**定量诊断排除 κ 错误**（同一配置两奇点偏差方向相反）
- viz 轴剖面：模型/引导/参考三条曲线统一稠密采样（Δx≈0.0012，修 0256 尖峰旁
  假"断点"——粗采样折角伪影，引导函数本身光滑）
- `logutil.py` 时间戳统一为 UTC+8 北京时间（此前随服务器本地时区为 UTC，
  日志文件名与内容时间差 8 小时）

---

## 7. 架构说明（简要）

```
输入: x(3D 坐标) + params(8维物理参数)
  │
  ├─ x → 正弦编码(10 频率) → coord_net(2×256) → x_enc
  │
  ├─ params → 归一化[-1,1] → 正弦编码(12 频率) → param_net(2×256) → p_enc
  │
  ├─ FiLM 调制: x_enc * (1+γ(p_enc)) + β(p_enc) → x_mod
  │
  ├─ 共享 MLP: 6×256 SiLU → h_θ
  │
  └─ 硬约束 ansatz:
       u_θ = κ · u_g · [1 + c · W · tanh(h_θ)]
       (κ: 预计算查表, c: 可学习标量, W: 窗函数, u_g: 引导解)
```

与 A2 单参数版的主要区别：
- 参数 2 维 → 8 维（质量+位置+动量+自旋）
- 网络 4×128 → 6×256
- 正弦编码频率 8 → 10/12（坐标/参数分别）
- LHS 采样 10 配置 → 400 配置
- 参数量 112K → 726K
