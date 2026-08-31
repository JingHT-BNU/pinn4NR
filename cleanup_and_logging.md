# 项目清理与统一日志系统（2026-08-28）

## 一、项目清理

### 已删除

| 类别 | 内容 |
|---|---|
| 过时运行产物 | `paper/runs/base_full/`（A1 旧默认配置，L2RE=0.133，结论已固化进 A1 报告）、`paper/runs/multi_param_a1/`（A3 v1 失败运行，L2RE=0.2065，结论已固化进记忆/README）、`paper/runs/parametric_a1_v2/`（无文档引用的旧 A2 训练）、`paper/runs/multi_param_smoke/`、`paper/runs/multi_param_a2/`（昨晚崩溃残留的空 figs/，重跑自动重建） |
| 散落日志 | `paper/runs/precompute_v2.log`、`train_v2.log`、`train_v3.log`（结论已固化） |
| 一次性诊断脚本 | `paper/analysis/` 下 21 个（a1_diag_*×7、a1_train/eval/viz/peak/region/fair_ref×6、check_axis/check_kappa/check_kappa_base/debug_kappa_cache/kappa_convergence×5、parametric_region/region2/smoke×3），问题均已解决、结论已固化进 reports |
| 过时文档 | `paper/docs/memory.md`（08-14 旧交接文档，指向旧机器路径，已被 paper/README.md + reports 取代） |
| 根目录垃圾 | `package.json`、`package-lock.json`、`tmp_out.txt`、`__pycache__/`（根目录及 paper 各处）、`paper/A3_multi_param/multi_param_kappa_smoke.json`（已被 v2 缓存取代） |

### 归档移动

根目录早期 BBH 演化可视化脚本（与 PINN 项目无关）整体移入新分项目文件夹 **`BBH_viz/`**：
`bbh_data.py`、`plot_lapse_evo.py`、`plot_trajectories.py`、`timelapse_chi.py`、
`track_punctures.py`、`viz_bbh.py`、`data.md` 及 `figs/` 产物。
脚本内的绝对路径已同步改为 `D:\AIs\PINN\BBH_viz\figs\...`（BBH/ 演化数据路径不变）。

### 保留

- `paper/runs/`：`a1/E1+E2`、`base_a1`（A1 报告引用）、`parametric_a1`（A2 泛化报告主结果）
- `paper/analysis/verify_cache_v2.py`（κ v2 缓存验证，仍有复用价值）
- `paper/tools/` 全部（764MB 主参考解 + uniform101 变体被 parametric_eval 引用）

### 清理后结构

```
D:\AIs\PINN\
├── BBH\              # 演化 hdf5 数据
├── BBH_viz\          # 早期演化可视化（脚本+data.md+figs）
├── TwoPuncturesC\    # 参考解 C 工具
├── paper\            # PINN 研究主项目（A1/A2/A3 + logs + runs + reports + tools）
├── .venv\  .idea\
```

---

## 二、统一日志系统

### 设计

新增 **`paper/logutil.py`**。所有 A1/A2/A3 运行脚本（共 15 个）已接入：

- **控制台输出**：`HH:MM:SS | 消息`（每次输出带时间戳，长任务可看节奏）
- **文件输出**：`paper/logs/<项目>/<脚本>_<启动时间戳>.log`，
  格式 `YYYY-MM-DD HH:MM:SS | INFO | 消息`
- **每次运行生成新文件**，不覆盖历史日志；无需再手动 `> xxx.log 2>&1` 重定向
- **异常也入日志**：入口以 `try/except` 包裹，崩溃时完整 traceback 写入日志文件末尾
  （本次 A2 冒烟的除零异常已被正确记录，验证生效）

### 日志文件位置

| 项目 | 日志目录 |
|---|---|
| A1 论文复现 | `paper/logs/A1/main_<时间>.log` |
| A2 参数化 | `paper/logs/A2/parametric_train_<时间>.log` 等 |
| A3 多参数 | `paper/logs/A3/multi_param_train_<时间>.log` 等 |

训练 / eval / viz / precompute 均有独立日志文件。**回传审查时直接发送 logs/ 下对应文件即可。**

### 新脚本接入方式

```python
import logging
from logutil import setup_logging

log = logging.getLogger("paper.<项目>.<脚本名>")   # 模块级,与 setup 名字一致

def main():
    setup_logging("<项目>", "<脚本名>")            # main() 第一行
    log.info("...")
    ...

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
```

同一项目下所有模块 logger（`paper.<项目>.*`）自动汇聚到入口配置的 handler，
子模块只需定义模块级 logger，无需重复 setup。

---

## 三、验证结果

- 16 个文件（logutil + 15 脚本）py_compile 全部通过
- A1 冒烟（`main.py --case base --smoke`，200 步）：控制台带时间戳输出 + 日志文件生成 ✅
- A2 冒烟（`parametric_train.py --steps 2`）：通过 ✅
  （顺手修复 `--steps` 很小时 `log_every=0` 导致的 ZeroDivisionError）
- A3 冒烟（`multi_param_train.py --steps 2`）：通过，参考解 47.7M 点正常加载、
  out-dir 锚定正常 ✅
- 顺手修复 A1 `visualize.py` 缺失 `Tuple` 导入（import 即崩的潜在问题）
- 冒烟产生的 runs/smoke_logtest_* 目录已清理；logs/ 下保留 4 个验证日志样例

相关文档已同步：`paper/README.md`（目录树、logutil 说明）、
`paper/A3_multi_param/multi_param_README.md`（§2 去掉重定向写法、§5 回传日志文件）。
