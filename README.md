# PINN 求解双黑洞哈密顿约束方程

复现论文:**Solving Hamiltonian Constraint Equation with Physics-Informed Neural Networks**
(arXiv:2607.06002v1, 2026-07-07)

用物理信息神经网络(PINN)求解双黑洞(BBH)初始数据问题中的哈密顿约束方程
(Lichnerowicz 方程),实现论文全部核心技巧,并在此基础上进行参数化扩展研究。

> **版本控制策略**:本仓库以 `main` 分支承载共享基础设施与最新统一代码;
> 各研究方案(A1 复现、A2 单参数 q 各变体、A3 多参数)以**分支**管理
> (如 `a2-base`、`a2-champion2`、`a3-multi-param`),详见 §1.1。
> 大文件(参考解 npz、训练产物、κ 数据)不入库,由脚本再生成(§1.2)。

---

## 1. 目录结构

```
pinn4NR/
├── config.py          # 配置:物理参数(4 个算例)与训练超参数
├── physics.py         # 物理模块:Bowen-York 源项 / 引导解 / κ / 残差
├── data.py            # 数据获取:球内/球面采样、Sobol QMC 积分点
├── logutil.py         # 统一日志:控制台 + logs/<项目>/<脚本>_<时间戳>.log
├── requirements.txt   # 依赖清单
├── README.md          # 本文档
│
├── A1_reproduction/   # ── 论文官方复现(单配置;分支 a1-reproduction) ──
│   ├── main.py        #   主入口:数据 → 训练 → 评估 → 可视化
│   ├── model.py       #   PINN:3×64 SiLU MLP + 硬约束 ansatz(Eq.5)
│   ├── train.py       #   训练:Adam + 复合损失 + EMA 损失平衡
│   ├── evaluate.py    #   评估:L2RE(需参考解)或残差自检
│   └── visualize.py   #   可视化:损失曲线 / x 轴剖面 / 赤道面
│
├── A2_parametric/     # ── 参数化 PINN(单参数 q∈[1,10];分支 a2-*) ──
│   ├── parametric_*.py               #   旧 A2(q∈[0.5,2]):FiLM 条件 MLP + 参考监督
│   ├── a2q_*.py                      #   新 A2 攻关(q∈[1,10],18 配置):快路径训练
│   │                                 #   + κ QMC 噪声修复 + champion/champion2/opv2 变体
│   ├── a2q_runner.cmd                #   自动重启包装(断点续训)
│   └── kappa_cache.json              #   κ 查表缓存(旧 A2)
│
├── A3_multi_param/    # ── 多参数参数化 PINN(8 维;分支 a3-multi-param) ──
│   ├── multi_param_*.py              #   8D FiLM 条件 MLP + LHS 采样 + 训练/评估/可视化
│   ├── multi_param_README.md         #   多参数项目专用运行说明
│   └── multi_param_experiments.md    #   实验记录
│
├── analysis/          # 诊断脚本(κ 缓存验证等)
├── reports/           # 全部研究报告与文档(md)
└── tools/             # 参考解生成工具(谱方法求解器 + TwoPunctures 验证)
```

### 1.1 分支策略

| 分支 | 内容 | 状态 |
|------|------|------|
| `main` | 共享基础设施(physics/config/data/logutil)+ tools + reports | 活跃 |
| `a1-reproduction` | A1 论文官方复现(base/a1/b2/b4 四算例) | ✅ 完成 |
| `a2-parametric` | 旧 A2:q∈[0.5,2] 参数化初探 | ✅ 完成 |
| `a2-single-q` | 新 A2 攻关主线:q∈[1,10] 18 配置(含 base/champion 等变体) | ✅ 阶段完成 |
| `a2-opv2` | 算子 v2 试验(patch 真泛函输入;结论:ansatz 非瓶颈) | ✅ 已定案 |
| `a3-multi-param` | 8 维参数空间扩展 | 🔧 进行中 |

> 注:同一 `A2_parametric/` 目录在分支间内容不同——切到 `a2-single-q`
> 即为 q∈[1,10] 攻关代码,切到 `a2-parametric` 即为旧初探代码。

### 1.2 数据再生成(不入库的大文件)

| 数据 | 生成方式 |
|------|----------|
| 谱方法参考解 `tools/refs*/ref_*.npz` | `tools/make_refs_batch.py`(A2)或 `tools/spectral_reference.py` 单配置 |
| base 参考解 `tools/reference_u*.npz` | `tools/make_reference.py`(TwoPunctures 流程) |
| A2 训练数据 `A2_parametric/a2q_data/` | `a2q_prep.py` → `a2q_prep2.py` → `a2q_refsub.py` → `a2q_kappa2.py` |
| κ 缓存 | `precompute_kappa.py`(旧 A2)/ `multi_param_precompute.py`(A3) |
| 训练产物 `**/runs/` | `a2q_runner.cmd <variant> <exp> <steps>` 等 |

## 2. 项目组成说明

本项目包含三个递进的研究阶段:

| 项目 | 目标 | 参数空间 | 网络规模 | 状态 |
|------|------|----------|----------|------|
| **A1** | 论文官方复现 | 单配置 | 3×64 (~13K 参数) | ✅ 完成 |
| **A2** | 参数化初探(质量比 q) | 1D: q∈[0.5, 2.0] | 4×128 + FiLM (~112K) | ✅ 完成 |
| **A3** | 多参数扩展 | 8D: 质量+位置+动量+自旋 | 6×256 + FiLM (~726K) | 🔧 进行中 |

**共享基础设施**(`paper/` 根目录):
- `physics.py`: Bowen-York 源项、引导解 u_g、窗函数 W、κ 求解、PDE/Robin 残差
- `config.py`: BBHConfig 数据结构、TrainConfig 训练超参数
- `data.py`: 球内/球面采样、Sobol QMC 积分点
- `logutil.py`: 统一日志。所有脚本输出同时打印到控制台(`HH:MM:SS | 消息`)并
  写入 `paper/logs/<项目>/<脚本>_<启动时间戳>.log`(含完整日期与级别)。
  新脚本接入方式:模块级 `log = logging.getLogger("paper.<项目>.<脚本名>")`,
  在 `main()` 开头调用 `setup_logging("<项目>", "<脚本名>")`,入口用
  `try/except log.exception` 包裹使异常也进入日志文件。

## 3. 论文对应关系(快速导航)

| 论文章节/公式 | 本项目代码 |
|---|---|
| Eq.(1)-(4) 哈密顿约束方程 | `physics.py: pde_residual, psi_sing` |
| Eq.(2) Bowen-York 解 | `physics.py: bowen_york_KK` |
| Eq.(3) 奇点分解 ψ=1+Σm/(2r)+u | `physics.py: psi_sing` |
| Eq.(5) 引导式硬约束 ansatz | `A1_reproduction/model.py: GuidedPINN.forward` |
| Eq.(6) 引导解 u_g | `physics.py: guide_u`(Lousto-Zlochower 2008 公式) |
| Eq.(7) 窗函数 W | `physics.py: window_function` |
| Eq.(8) Robin 边界条件 | `physics.py: robin_residual` |
| Eq.(9)-(10) κ 定标 | `physics.py: solve_kappa` |
| Eq.(11)-(15) 复合损失 | `A1_reproduction/train.py: _compute_losses` |
| Eq.(16)-(17) EMA 损失平衡 | `A1_reproduction/train.py: _ema_balance` |
| Eq.(18) L2RE 评估 | `A1_reproduction/evaluate.py: compute_l2re` |
| §II.B.4 网络结构 | `config.py` / `A1_reproduction/model.py` |
| Table I 消融 / Table II 三算例 | `config.py: CASE_HYPER`, `BBHConfig.from_case` |

## 4. 快速开始

### 4.1 A1 论文复现(本机 CPU 冒烟验证)

```bash
.venv\Scripts\python.exe paper\A1_reproduction\main.py --smoke          # ~200 步,约 1-3 分钟
```

`--smoke` 使用小规模配置,仅验证流程正确性,**不是**论文精度。

### 4.2 服务器完整训练(GPU)

```bash
# A1: 四个算例(论文 Table II)
python paper/A1_reproduction/main.py --case base --exp-name base_full

# A2: 参数化训练(质量比 q)
python paper/A2_parametric/parametric_train.py --steps 15000 --exp-name parametric_a1

# A3: 多参数训练(8 维,详见 A3_multi_param/multi_param_README.md)
python paper/A3_multi_param/multi_param_train.py --steps 30000 --exp-name multi_param_a1
```

### 4.3 各子项目详细命令

- **A1**: 见 [§4.4](#44-a1-详细命令)
- **A2**: 见 [§5](#5-a2-参数化-pinn)
- **A3**: 见 `A3_multi_param/multi_param_README.md`

## 4.4 A1 详细命令

### 训练四个算例(论文 Table II)

```bash
# 等质量无自旋(基准,论文 §II.B.4):约 30 分钟 / 5000 步(L40 量级)
python paper/A1_reproduction/main.py --case base --exp-name base_full

# 等质量自旋(Table II 第 1 行)
python paper/A1_reproduction/main.py --case spin_eq --exp-name spin_eq_full

# 不等质量无自旋(Table II 第 2 行,论文精度最好: L2RE≈0.0077)
python paper/A1_reproduction/main.py --case uneq_nospin --exp-name uneq_nospin_full

# 不等质量自旋(Table II 第 3 行,10000 步)
python paper/A1_reproduction/main.py --case uneq_spin --exp-name uneq_spin_full
```

参数说明:
- `--steps N`  覆盖训练步数
- `--lr x`     覆盖学习率
- `--device cpu/cuda/auto`(默认 auto 自动检测)
- `--reference path.npz` 提供 TwoPunctures 参考解(见 §6)
- `--retrain`   强制重新训练

### 断点续用:已有模型自动跳过训练

若 `runs/<exp_name>/model.pt` 已存在,再次运行会**跳过训练,直接评估与可视化**。

```bash
# 第一次:完整训练
python paper/A1_reproduction/main.py --case base --exp-name base_full

# 第二次:跳过训练,直接评估+出图
python paper/A1_reproduction/main.py --case base --exp-name base_full --reference paper/tools/reference_u.npz

# 强制重训
python paper/A1_reproduction/main.py --case base --exp-name base_full --retrain
```

### 训练日志解读

```
[step  5000/5000] L2=9.2e-09 softLinf=1.4e-05 LBC=2.1e-11 total=1.96  (396s)
```
- `L2`/`softLinf`/`LBC`:原始损失分量(论文 Eq.13-15)
- `total`:EMA 平衡后的加权组合(各分量被归一化到 O(1))

## 5. A2 参数化 PINN

一个网络覆盖质量比 q∈[0.5,2.0](m1=0.5 固定)的初始数据求解。
架构:引导式硬约束 ansatz + FiLM 条件 MLP + 正弦位置编码;κ 查表(预计算)。

```bash
# 1. 预计算 κ(已生成 kappa_cache.json)
python paper/A2_parametric/precompute_kappa.py

# 2. 训练(参考监督 base + PDE 正则全 q,15000 步 ~60 min GPU)
python paper/A2_parametric/parametric_train.py --steps 15000 --exp-name parametric_a1

# 3. 严格审查(L2RE 双口径 + 全配置残差自检 + 物理合理性)
python paper/A2_parametric/parametric_eval.py

# 4. 可视化
python paper/A2_parametric/parametric_viz.py
```

**结果**(runs/parametric_a1):
- base L2RE=0.0096(全 47.7M 参考),优于论文 0.017
- 零样本 q(0.6/0.9/1.4/1.8)PDE 残差与训练配置同量级

**谱参考解套件已接入**(2026-08-30):`paper/tools/refs_a2/ref_a2_<q>.npz` 为全部
10 个剖面配置(训练 6 + 零样本 4)的 L=48 谱参考解(生成命令
`python -u paper/tools/make_refs_a2.py`,~8-10 min/配置;与 A3 同一套件,base 精度
L2RE=6.35e-5)。`parametric_viz.py` 默认 `--refs-dir paper/tools/refs_a2`,
逐参数面板自动匹配绘制模型/引导基线/谱方法参考解三条曲线。

详见 `reports/A2_parametric_report.md`。

### 5.1 泛化能力测试(脚本与图像已归档)

插值/外推泛化测试脚本与图像已于 2026-08-30 移至临时归档
`.qwen/tmp/archive_a2_generalization/`(可能随 tmp 清理),完整结论见
`reports/A2_generalization_report.md`。当前保留的泛化验证方式:零样本参数
(q06/q09/q14/q18)的 PDE/Robin 残差自检——`parametric_eval.py` 与训练日志自动输出,
零样本与训练配置同精度。

## 6. 参考解与 L2RE 评估

论文的 L2RE 是与 TwoPunctures 谱方法参考解对比。本项目支持两种评估:

1. **有参考解**(推荐):准备 `reference_u.npz`,然后 `--reference reference_u.npz`。

   **自动生成流程**(需要 gcc + GSL):
   ```bash
   cd paper/tools
   python make_reference.py --case base        # 生成 base 算例参考解
   python make_reference.py --case uneq_spin   # 生成最高配置参考解
   ```

2. **无参考解**:自动退化为残差自检(PDE 残差均值/最大值、边界残差)。

## 7. 输出文件说明

每次运行在 `paper/runs/<exp_name>/` 下:

| 文件 | 内容 |
|---|---|
| `model.pt`   | 模型权重 + 元数据(κ、c、u_min/u_max) |
| `data.npz`   | 采样点与解析量缓存 |
| `history.json` | 损失历史 |
| `metrics.json` | 评估指标 |
| `figs/*.png` | 可视化图表 |

## 8. 与论文的已知差异

1. **引导解归一化**:论文 [40] 的 u_P/u_J/u_c 公式已按原文实现;κ 量级可能略有差异(正常)。
2. **κ 积分点数**:论文用 5×10⁶;本项目默认 2×10⁵(精度略降但不影响最终结果)。
3. **采样点数量**:冒烟配置远小于论文;完整训练默认与论文一致。
4. **soft-L∞ 的 β 与权重**:按论文 Table I/II 各算例取值。
5. **EMA α**:论文未给出具体值,本项目取 0.9。

## 9. 常见问题

- **Q: 本机 Windows 上打印中文乱码?**
  A: 控制台编码问题,终端执行 `chcp 65001` 后再运行。
- **Q: 训练很慢?**
  A: 完整训练需 GPU;CPU 只建议跑 `--smoke`。
- **Q: L2RE 算不出来?**
  A: 需要 `--reference` 参考解 npz(§6);没有则用残差自检。

---

*项目整理:2026-08-26。环境:本机 Windows + venv(torch 2.13 CPU/CUDA)。*
