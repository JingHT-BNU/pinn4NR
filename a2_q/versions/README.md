# a2_q/versions —— 各历史版本的**特有**文件

这里的每个子目录只放**该版本独有、未被主线继承**的文件。
绝大部分 A2 代码在各版本间是共享的（都放在 `../` 下），
版本的差异主要体现在 `--variant` 取值与少量专属脚本/数据管线。

| 目录 | 内容 | 说明 |
|---|---|---|
| `opv3/` | `a2q_train_opv3.py`、`make_refs_opv3.py`、`post_refs_opv3.py` | opv3 世代的训练器与配套数据准备脚本（加性算子 + 奇点窗） |
| `noise/` | `SCHEME.md` | θ 噪声增强方案记录：**负结果**（3.23e-2 平台） |
| `rar/` | `SCHEME.md` | RAR 主动采样方案记录：**负结果**（3.20e-2 平台） |

其余版本（baseline / c2 / operator / opv2 / opv4 / opv5 / opv6）没有专属文件，
统一由 `../a2q_model.py` 的 `make_model(variant, …)` 与 `../a2q_train_v4.py` 支持，
复现命令见 `../README.md` §3 的版本谱系表。

这些脚本顶部同样带有 `path bootstrap`，直接运行会自动定位到仓库根的 `core/` 与 `tools/`。
