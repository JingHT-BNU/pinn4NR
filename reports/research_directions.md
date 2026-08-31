# 研究方向规划

> 基于 `ds.md` 的深度讨论，将"PINN 求解 BBH 哈密顿约束"从**复现**推进到**方法学创新**。
>
> **核心问题**：如何让神经网络生成的解不仅"近似满足"PDE，而是在结构上被约束为 PDE 的解？
> 即从 `soft constraint`（PDE as loss）走向 `hard constraint`（PDE as architecture / representation / projection）。

---

## 总览

`ds.md` 提出了一个清晰的方法学层次，将研究目标分为三个递进层级：

| 层级 | 名称 | 描述 | 状态 |
|------|------|------|------|
| Level 1 | **PINN Solver** | 给定一套 BBH 参数 $\lambda$，训练一个网络求解该配置的哈密顿约束 | ✅ A1 已完成（L2RE=0.0067） |
| Level 2 | **Parametric / Amortized PINN** | 物理参数作为网络输入，一次训练覆盖一族 BBH 参数 | ✅ A2 已完成（q∈[0.5,2.0] 原型） |
| Level 3 | **Constraint-Preserving Generator** | 网络的输出**天然**被限制在 PDE 的解流形上，而不只是通过 loss 近似 | 🎯 **本文档的核心方向** |

以下各方向均服务于从 Level 2 向 Level 3 的跨越。

---

## 方向 A：全局约束投影（Global Constraint Projection）

### 核心思想

原论文已用散度定理得到积分一致性条件（Eq.9-10）来确定全局尺度 $\kappa$。**更进一步**：在训练过程中，每隔若干步直接计算全局约束残差 $C[u_\theta]$，然后通过一个解析的 correction basis 将其**精确归零**。

### 数学表述

定义全局约束：

$$C[u] = \int_{\partial\Omega} \nabla u \cdot d\mathbf{S} + \frac{1}{8} \int_\Omega \psi^{-7} \bar{K}^{ij}\bar{K}_{ij}\, dV$$

网络输出 $u_{\text{raw}}$ 后，施加校正：

$$u_{\text{corr}} = u_{\text{raw}} + \alpha_\theta \, q(x)$$

其中 $q(x)$ 是预构造的 correction basis（如 $q(x)=1$ 或 $q(x)=u_g(x)$），使得 $C[q] \neq 0$。令：

$$\alpha_\theta = -\frac{C[u_{\text{raw}}]}{C[q]}$$

则 **$C[u_{\text{corr}}] = 0$ 精确成立**。

### 实现方案

1. **Correction basis 选择**：
   - 最简单：$q(x) = 1$（常数偏移），但 $C[q] = \oint_{\partial\Omega} \nabla 1 \cdot d\mathbf{S} = 0$，不适用
   - 推荐：$q(x) = u_g(x)$（引导解本身），$C[q] \neq 0$ 一般成立
   - 或：$q(x) = r^{-1}$（满足 Laplace 方程的基本解），$C[q] = -4\pi$

2. **集成方式**：
   - 在 `GuidedPINN.forward()` 之后增加一个 `global_projection()` 层
   - 每 N 步（如 100 步）用 QMC 积分重新计算 $C[u_{\text{raw}}]$ 和 $C[q]$
   - 校正系数 $\alpha_\theta$ 作为可学习参数或直接解析计算

3. **与现有代码的兼容性**：
   - 继承 `GuidedPINN`，增加 `projection` 模块
   - 训练时 loss 不再包含全局约束项（已被硬满足）

### 预期效果

- 全局积分一致性从"loss 优化"提升为"解析精确"
- 网络不再需要学习全局尺度修正，专注于局部残差
- 对参数化版本尤其重要：即使 $\kappa$ 预计算有误差，投影层自动补偿

### 风险与注意事项

- 全局投影精确并不意味着局部 PDE 残差为零
- QMC 积分精度影响 $C[u]$ 的计算精度，需保证积分收敛
- 校正 basis 的选择影响投影的"平滑性"——应避免引入新的高频误差

---

## 方向 B：局部 Newton 校正（Local Newton Correction）

### 核心思想

在全局投影之后，对局部 PDE 残差做一步 Newton 型校正。设当前解为 $u$，残差为 $R(u) = \mathcal{H}[u]$，在 $u$ 附近线性化：

$$\mathcal{H}[u + \delta u] \approx \mathcal{H}[u] + \mathcal{L}_u[\delta u]$$

其中 $\mathcal{L}_u$ 是 Hamiltonian 算子在 $u$ 处的 Fréchet 导数。令 $\mathcal{L}_u[\delta u] \approx -\mathcal{H}[u]$，则校正后的解 $\mathcal{H}[u+\delta u] \approx 0$。

### 数学表述

对于 Hamiltonian 约束：

$$\mathcal{H}[\psi] = \Delta\psi + \frac{1}{8}\psi^{-7}\bar{K}^2$$

在 $\psi_\theta$ 附近线性化：

$$\mathcal{L}_{\psi_\theta}[\delta\psi] = \Delta(\delta\psi) - \frac{7}{8}\psi_\theta^{-8}\bar{K}^2 \cdot \delta\psi$$

求解：

$$\mathcal{L}_{\psi_\theta}[\delta\psi] = -\mathcal{H}[\psi_\theta]$$

这是一个**线性椭圆 PDE**，可以用一个小型辅助网络 $\delta u_\phi(x)$ 来近似求解。

### 实现方案

1. **辅助校正网络**：
   - 小型 MLP（2 层 × 32 神经元），输入坐标 $x$，输出 $\delta u_\phi(x)$
   - 损失函数：$L_{\text{corr}} = \|\mathcal{L}_{\psi_\theta}[\delta u_\phi] + \mathcal{H}[\psi_\theta]\|^2$
   - 可在主网络训练完成后单独训练，或联合训练

2. **迭代校正**：
   - 单步校正：$u_1 = u_0 + \delta u_\phi$
   - 多步校正：重复上述过程，每次在当前解处重新线性化
   - 通常 1-2 步即可显著降低残差

3. **与方向 A 的组合**：
   - 完整流水线：$u_{\text{raw}} \xrightarrow{\text{全局投影}} u_0 \xrightarrow{\text{Newton 校正}} u_1$

### 预期效果

- 局部 PDE 残差显著降低（尤其在奇点附近的高残差区）
- 校正网络很小，额外计算开销可控
- 形成"粗解 + 精细校正"的层次化架构

### 风险与注意事项

- 线性化算子的计算涉及 $\psi_\theta^{-8}$，数值稳定性需注意
- 校正网络可能过拟合到特定残差模式，需正则化
- 多步校正的收敛性需实验验证

---

## 方向 C：自适应残差细化采样（Adaptive Residual Refinement）

### 核心思想

哈密顿约束的最大残差通常集中在 puncture 附近。原论文也观察到 spinning BBH 时最大误差在 puncture 附近。自适应采样根据当前残差分布动态调整 collocation points 的密度。

### 数学表述

采样概率密度：

$$p(x) \propto |R_\theta(x)|^\gamma + \epsilon$$

其中 $\gamma \in [0.5, 1.0]$ 控制聚焦程度，$\epsilon$ 是均匀背景（保证全域覆盖）。

### 实现方案

1. **采样流程**：
   - 每 $N_{\text{adapt}}$ 个 epoch（如 500 步）：
     - 在整个域上粗采样一批点（如 5000 点）
     - 计算每个点的 PDE 残差 $|R_\theta(x)|$
     - 构造残差加权分布 $p(x)$
     - 从 $p(x)$ 中重采样新的 collocation points
   - 替换或扩充当前训练点集

2. **与现有代码的兼容性**：
   - `config.py` 中已有 `adaptive_sampling` 和 `peak_weighting` 参数
   - 需扩展为动态自适应（当前是静态的奇点邻域加权）
   - 新增 `AdaptiveSampler` 类，管理采样点集的生命周期

3. **实现细节**：
   - 使用残差归一化：$w_i = |R_i|^\gamma / \max_j |R_j|^\gamma$
   - 混合策略：保留 70% 残差加权点 + 30% 均匀点
   - 奇点附近 r < 0.5 的区域强制保留一定比例的点

### 预期效果

- 奇点附近残差显著降低
- 计算资源集中在"难"区域，效率更高
- 与方向 A+B 组合时，自适应采样可针对校正后的残差分布

### 风险与注意事项

- A1 复现实验已发现：过度聚焦奇点邻域会欠加权 bulk 区，导致整体 L2RE 恶化
- 需平衡局部聚焦与全局覆盖——建议 $\gamma$ 不要过大，$\epsilon$ 保持一定比例
- 残差计算本身有成本，采样频率需权衡

---

## 方向 D：参数化约束保持生成器（Parametric Constraint-Preserving Generator）

### 核心思想

将方向 A+B+C 整合到参数化 PINN（A2 架构）中，构造一个**对任意 BBH 参数 $\lambda$ 都能生成约束保持解**的生成器。

### 数学表述

$$u_\theta(x; \lambda) = \Pi_{\text{global}}\big[u_{\text{raw}}(x; \lambda) + \delta u_\phi(x; \lambda)\big]$$

其中：
- $u_{\text{raw}}$：参数化引导 ansatz 的原始输出
- $\Pi_{\text{global}}$：全局约束投影（方向 A）
- $\delta u_\phi$：局部 Newton 校正（方向 B）
- $\lambda = (m_1, m_2, \mathbf{x}_1, \mathbf{x}_2, \mathbf{P}_1, \mathbf{P}_2, \mathbf{S}_1, \mathbf{S}_2)$

### 实现方案

1. **架构扩展**：
   - 在 A2 的 FiLM 条件 MLP 基础上，增加投影层和校正网络
   - 校正网络也接收 $\lambda$ 作为条件输入
   - $\kappa(\lambda)$ 仍用 QMC 预计算查表

2. **训练策略**：
   - 课程学习：先训练单参数版本（如仅 $m_2$ 变化），再扩展到多参数
   - 损失函数：$L = L_{\text{PDE}} + L_{\text{BC}} + L_{\text{corr}}$（全局投影已硬满足，不在 loss 中）
   - 参数空间采样：每步随机采样一个 $\lambda$ 配置

3. **验证协议**：
   - 插值测试：训练区间内的未见参数
   - 外推测试：训练区间外的参数
   - 关键指标：$C[u_\theta(\lambda_{\text{test}})] = 0$ 是否精确成立

### 预期效果

- 对任意测试参数，全局约束精确满足
- 局部残差在训练区间内保持低水平
- 论文故事从"PINN 能解一个 PDE"升级为"约束保持的初始数据生成器"
- 这正是 `ds.md` 所说的 "constraint-preserving neural generator for BBH initial data"

### 风险与注意事项

- 参数空间维度过高时，训练复杂度急剧上升
- 外推性能无法保证（但全局投影至少保证积分一致性）
- 需大量 QMC 预计算 $\kappa(\lambda)$，可考虑用代理模型替代

---

## 方向 E：Green 算子 / 积分方程路线（Green-Operator / Integral Equation Route）

### 核心思想

`ds.md` 中最具洞察力的方向之一：将 PDE 转化为等价的积分方程，让神经网络学习源项或调和部分，通过 Green 函数**解析满足 PDE 算子**。

### 数学表述

对于 Poisson 方程 $\nabla^2 u = f$：

$$u(\mathbf{x}) = -\frac{1}{4\pi} \int \frac{f(\mathbf{x}')}{|\mathbf{x} - \mathbf{x}'|} d^3x' + u_h(\mathbf{x})$$

其中 $\nabla^2 u_h = 0$。

对于 Hamiltonian 约束（非线性 Poisson 型）：

$$\Delta u = -S(u; \lambda)$$

等价积分方程：

$$u(\mathbf{x}) = \frac{1}{4\pi} \int \frac{S(u(\mathbf{x}'); \lambda)}{|\mathbf{x} - \mathbf{x}'|} d^3x' + u_h$$

这是一个**不动点问题**：$u = \mathcal{G}_\lambda[u]$。

### 实现方案

1. **第一阶段（3D Poisson 原型）**：
   - 网络不直接输出 $u$，而是输出源项 $f_\theta(x)$ 和调和部分 $u_{h,\theta}(x)$
   - 通过 Green 函数层：$u_\theta = G * f_\theta + u_{h,\theta}$
   - 此时 $\nabla^2 u_\theta = f_\theta$ **解析成立**
   - 训练目标：$f_\theta \to f_{\text{true}}$（或残差 $\to 0$）

2. **第二阶段（Hamiltonian constraint）**：
   - 构造 Green 算子层（需处理非线性源项 $S(u; \lambda)$）
   - 使用不动点迭代：$u^{(k+1)} = \mathcal{G}_\lambda[u^{(k)}]$
   - 网络学习迭代的初始猜测或校正项

3. **实现挑战**：
   - 3D Green 函数积分是 $O(N^2)$ 操作，需用快速多极子或随机积分近似
   - 非线性源项使不动点迭代的收敛性不确定
   - 边界条件处理比 PINN 更复杂

### 预期效果

- **线性 PDE 可解析满足**（Poisson 方程）
- 对非线性 PDE，至少保证线性主部的解析满足
- PINN: $u_\theta \xrightarrow{\nabla^2} \text{residual} \xrightarrow{\text{loss}} 0$
- Ours: $f_\theta \xrightarrow{\nabla^{-2}} u_\theta$, $\nabla^2 u_\theta = f_\theta$ **by construction**
- `ds.md` 的评价："我怀疑这非常接近曹老师最初给您的那个 intuition"

### 风险与注意事项

- **最激进的方向**，实现难度最大
- 3D 积分的高效计算是主要瓶颈
- 建议先做 3D Poisson 原型验证可行性，再推广到 Hamiltonian constraint
- 与 `ds.md` 中建议的 "3D Poisson → nonlinear Poisson → Hamiltonian constraint → parameterized BBH generator"  progression 一致

---

## 方向 F：Poisson 方程原型验证（3D Poisson Toy Model）

### 核心思想

在进入复杂的 Hamiltonian constraint 之前，先用 3D Poisson 方程作为 toy model，同时实现 vanilla PINN、hard-BC PINN、Green-operator 三条路线，建立"PDE-satisfying generator"的评估框架。

### 数学表述

$$\nabla^2 u(\mathbf{x}) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega \subset \mathbb{R}^3$$

选择已知解析解的 $f$（如 $f = -6$，解 $u = r^2$），便于精确评估。

### 实现方案

1. **三条路线对比**：
   - **路线 1（Vanilla PINN）**：标准 PINN，PDE residual 作为 loss
   - **路线 2（Hard-BC PINN）**：解析满足边界条件的 ansatz
   - **路线 3（Green-operator）**：$u_\theta = G * f_\theta + u_{h,\theta}$，PDE 解析满足

2. **评估指标**：
   - PDE 残差 $L_2$ 和 $L_\infty$
   - 解误差 $L_2$ 相对误差
   - 训练稳定性（loss 曲线、梯度范数）
   - 对未见源项 $f$ 的泛化能力

3. **代码结构**：
   - `poisson_model.py`：三种架构定义
   - `poisson_train.py`：训练脚本
   - `poisson_eval.py`：评估与对比

### 预期效果

- 明确"PDE 解析满足"在不同架构下的可实现程度
- 为 Hamiltonian constraint 的架构选择提供实验依据
- 形成可复用的评估框架

### 风险与注意事项

- Poisson 方程是线性的，结论不一定完全迁移到非线性 Hamiltonian constraint
- 但这是**最低风险**的起点，适合快速上手
- Green 函数积分在 3D 中的高效实现是关键技术挑战

---

## 关键方法学问题（`ds.md` 提炼）

1. **"PDE 解析满足"的确切定义是什么？**
   - 全局积分精确满足（方向 A）vs 局部逐点精确满足（方向 E）
   - 论文中应明确区分，避免过度宣称
   - `ds.md` 建议："stronger constraint enforcement"或"constraint-preserving representation"，而不是第一版就宣称"严格解析满足 Hamiltonian equation"

2. **soft constraint 与 hard constraint 的界限？**
   - 普通 PINN: $\|\mathcal{F}[u_\theta]\| \ll 1$（numerical satisfaction）
   - 目标: $\mathcal{F}[u_\theta] = 0$ by construction（mathematical enforcement）
   - 全局投影是"半 hard"：全局积分精确，局部仍为 soft
   - Green 算子是"全 hard"：PDE 解析满足，但实现受限

3. **参数化版本中，约束保持的保证来自哪里？**
   - 全局投影：来自解析构造，**与参数无关**
   - 局部残差：来自训练，依赖参数空间覆盖
   - 论文应清晰区分这两种"保证"

4. **不是"近似像解"，而是"被结构约束为解"**
   - `ds.md` 的核心论点：真正区分这篇工作与原论文的，不是"把更多参数加进 PINN"，而是
     "改变 hypothesis space 本身"——从 $\min_\theta \|\mathcal{F}[u_\theta]\|^2$ 转变为
     $u_\theta \in \mathcal{M}_{\text{PDE}}$ by construction

---

## 实施路线图

### 第一阶段（2-3 周）：Poisson 原型验证（方向 F）

**目标**：建立评估框架，验证三条路线的可行性。

- [ ] 实现 3D Poisson 的 vanilla PINN
- [ ] 实现 hard-BC PINN（解析满足边界条件）
- [ ] 实现 Green-operator PINN（PDE 解析满足）
- [ ] 对比三者的精度、稳定性、泛化能力
- [ ] 形成评估报告

### 第二阶段（3-4 周）：全局约束投影（方向 A）

**目标**：在现有 Hamiltonian constraint 代码上实现全局投影。

- [ ] 实现 correction basis $q(x)$ 的选择与 $C[q]$ 的预计算
- [ ] 实现全局投影层 $\Pi_{\text{global}}$
- [ ] 在 base 算例上验证：$C[u_{\text{corr}}] = 0$ 精确成立
- [ ] 对比投影前后的 PDE 残差分布
- [ ] 扩展到 spin_eq / uneq_nospin / uneq_spin 算例

### 第三阶段（2-3 周）：自适应残差细化（方向 C）

**目标**：实现动态自适应采样，降低奇点附近残差。

- [ ] 实现 `AdaptiveSampler` 类
- [ ] 集成到训练循环中
- [ ] 在 base 算例上验证残差分布改善
- [ ] 与方向 A 组合测试

### 第四阶段（3-4 周）：局部 Newton 校正（方向 B）

**目标**：实现 Newton 校正层，进一步降低局部残差。

- [ ] 实现 Hamiltonian 算子的 Fréchet 线性化
- [ ] 实现小型校正网络 $\delta u_\phi$
- [ ] 在 base 算例上验证单步/多步校正效果
- [ ] 与方向 A+C 组合测试（完整流水线）

### 第五阶段（4-6 周）：参数化约束保持生成器（方向 D）

**目标**：将 A+B+C 整合到参数化架构中。

- [ ] 扩展 A2 参数化 PINN，增加投影层和校正网络
- [ ] 在质量比 $q \in [0.5, 2.0]$ 上训练和测试
- [ ] 验证插值/外推性能
- [ ] 扩展到更多参数维度（自旋、动量）

### 第六阶段（长期探索）：Green 算子路线（方向 E）

**目标**：探索积分方程方法在 Hamiltonian constraint 上的可行性。

- [ ] 实现 3D Green 函数积分的近似计算
- [ ] 在 Poisson 方程上验证 Green-operator 路线
- [ ] 探索非线性 Hamiltonian constraint 的不动点迭代
- [ ] 与 PINN 路线对比

---

## 与现有代码的集成

所有方向都设计为对现有代码的**增量扩展**：

| 方向 | 新增文件 | 修改文件 | 独立可运行 |
|------|---------|---------|-----------|
| A | `constraint_projection.py` | `model.py`, `train.py` | 是 |
| B | `newton_correction.py` | `model.py`, `physics.py` | 是 |
| C | `adaptive_sampler.py` | `train.py`, `config.py` | 是 |
| D | `parametric_constrained.py` | `parametric_model.py` | 是 |
| E | `green_operator.py`, `poisson_*.py` | 无 | 是 |
| F | `poisson_*.py` | 无 | 是 |

每个方向都可以独立实现和验证，降低耦合风险。

---

## 附录：`ds.md` 关键引用

> **"PINN 的训练目标鼓励 $u_\theta$ 满足 PDE，但一般并不保证 $u_\theta$ 严格属于 PDE 的解空间。"**
> — 区分 numerical satisfaction 与 mathematical enforcement

> **"Can the Hamiltonian constraint itself be built into the solution representation, rather than merely penalized through a residual loss?"**
> — 核心研究问题

> **"PINN: $u_\theta \xrightarrow{\nabla^2} \text{residual} \xrightarrow{\text{loss}} 0$ 变成 ours: $f_\theta \xrightarrow{\nabla^{-2}} u_\theta$, $\nabla^2 u_\theta = f_\theta$ by construction."**
> — Green 算子路线的核心洞察

> **"我怀疑这非常接近曹老师最初给您的那个 intuition。"**
> — 关于 Poisson Green 函数路线

> **"不只是 'AI 猜一个 u，然后 loss 告诉它猜得像不像 PDE 的解'，而是 'AI 只能在满足 PDE 的函数空间里生成/搜索 u'。"**
> — Level 3 的精神内核

> **"从 physics-informed learning 走向 physics-constrained generation。"**
> — 最终论文的故事定位