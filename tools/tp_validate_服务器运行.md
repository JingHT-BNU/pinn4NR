# TwoPunctures 交叉验证 —— 服务器运行说明

(2026-08-30 打包。背景: 本机已完成 q20 交叉验证 **PASS**——非等质量 2:1 配置自研谱求解器
vs TwoPunctures 逐点 L2RE=1.63e-4; 0164(质量比 3.1+自旋) 差 17.5%, 已定位为**我们的 L=48
参考解在最难配置上欠收敛**(L48→L64 光质量尖端峰值 3.05e-2→3.62e-2, +18.6%, 方向指向 TP);
公式已逐行对照确认与 TP 完全一致。本包完成剩余两件事: ①L64-vs-TP 全网格量化; ②TP n96
自收敛检验(可选)。两份结果 JSON 回传后即可出最终验证报告。)

## 0. 解压与环境

```bash
tar -xzf tp_validate_20260830.tar.gz
cd tp_validate_20260830
```

- 任务①需要: python 环境 (torch + numpy, 服务器现有环境即可; 有 pandas 更快, 无则自动
  回退 np.loadtxt)。
- 任务②需要: gcc + GSL (生成 TwoPunctures 参考解的服务器已有; 检查 `gsl-config --version`)。

## 1. 任务① L64-vs-TP 全网格对比 (预计 5-10 分钟 GPU / 15-25 分钟 CPU)

```bash
python scripts/compare_l64_server.py
```

输入 data/a3_0164_n60eval.psi (TwoPunctures n60 evaluation 模式网格, 3,652,828 点,
已在本机生成) + data/ours_0164_L64.npz (自研 L64 系数, 本机已求解) +
data/ref_train_0164.npz (自研现行 L48 参考)。输出 `l64_vs_tp.json`。

## 2. 任务② TP n96 自收敛检验 (可选; 预计 n96 求解 1.5-3 小时 + dump 几分钟)

```bash
# 2a. 编译 (若服务器已有编译好的 TwoPuncturesC/dump_psi_grid 可跳过)
cd TwoPuncturesC && make && cd ..
gcc -O2 -std=c99 -I TwoPuncturesC/include -I TwoPuncturesC/src \
    paper/tools/dump_psi_grid.c TwoPuncturesC/lib/libTwoPunctures.a \
    -lgsl -lgslcblas -lm -o dump_psi_grid

# 2b. n96 求解 + Taylor 网格输出 (b=7.0419414295, 网格块与 n60 完全一致)
./dump_psi_grid --adaptive data/a3_0164_n96.par 7.0419414295 data/a3_0164_n96.psi \
    5.5419414295 8.5419414295 -1.5 1.5 -1.5 1.5 121 121 121 \
    -8.5419414295 -5.5419414295 -1.5 1.5 -1.5 1.5 121 121 121 \
    -15.0419414295 15.0419414295 -0.06 0.06 -0.06 0.06 321 5 5 \
    -31 31 -31 31 -31 31 61 61 61

# 2c. TP 自身 n96 vs n60 同网格对比
python scripts/compare_tp_n60_n96.py
```

输出 `tp_n96_vs_n60.json`。注意 dump 的进度/Newton 输出全部在 stderr, 可
`2>&1 | tee n96.log` 留档。

## 3. 结果判读 (供回传后核对)

- 若 `L2RE_L64_vs_TP` 显著小于 `L2RE_L48_vs_TP`(≈1.755e-1) 且
  `L2RE_L48_vs_L64` 与前者同量级 → "L48 欠收敛"解释成立, 修复 = 难配置(高质量比/
  高自旋)参考解升到 L=64 重生成;
- 若 TP `n96≈n60`(tp_n96_vs_n60.json 各区 <1e-3) → TP n60 已收敛, 上述结论闭合;
  若 n96 明显移动 → TP n60 也未收敛, 两边都要提分辨率后再判。
- 若 `light_tip_peak` 三值 (L48/L64/TP) 中 L64 与 TP 靠拢 → 尖端问题同上。

## 4. 包内清单

```
scripts/compare_l64_server.py    任务①对比脚本(自包含, 路径相对包根)
scripts/compare_tp_n60_n96.py    任务②对比脚本
data/a3_0164_n60eval.psi         TP n60 evaluation 模式网格 (283MB)
data/a3_0164_n60eval.par         其 par 文件(grid_setup_method=1)
data/a3_0164_n96.par             任务②用 par (npoints=96, Taylor)
data/ours_0164_L64.npz           自研 L64 谱系数 (本机两次求解, 尖峰值可复现)
data/ref_train_0164.npz          自研现行 L48 参考(待检对象)
data/tp_vs_ours.json             本机已完成对比的结果存档(q20/spineq/0164-n60)
paper/physics.py, paper/logutil.py, paper/tools/spectral_reference.py
paper/tools/dump_psi_grid.c, paper/tools/validate_with_tp.py
TwoPuncturesC/src,include,Makefile,README.md   (源码, 服务器上需重新 make)
```

注意: 平移映射(0164 奇点不在 ±b 对称位置, TP 强制对称)已在脚本内处理
(x = x' + C, C=-0.9261996821); 两脚本均带点数/网格一致性断言, 不会静默对比截断数据。
