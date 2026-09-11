"""
tools/convert_psi_grid.py —— 把 TwoPunctures 的 ψ 网格文本转换为参考解 npz
==========================================================================

用途:
    在服务器上用 TwoPuncturesC(或任何谱方法求解器)解出共形因子 ψ,
    输出为文本文件(每行: x y z psi),本脚本把它转换为评估用的
    reference_u.npz(含 x_ref 与 u_ref = ψ − 1 − Σ m/(2r))。

    支持自适应网格生成的 psi 文本:自动检测并去除重叠区域的重复点
    (保留首次出现的点,对应高分辨率块优先)。

用法:
    # 基本用法
    python tools/convert_psi_grid.py --psi psi_grid.txt \
        --m1 0.5 --m2 0.5 --x1 3 --x2 -3 \
        --out reference_u.npz

    # 启用去重(自适应网格推荐)
    python tools/convert_psi_grid.py --psi psi_grid.txt \
        --m1 0.5 --m2 0.5 --x1 3 --x2 -3 \
        --out reference_u.npz --dedup

文本格式(空格/逗号分隔均可):
    x y z psi
    ...

说明:
    - 参考解修正项 u = ψ − 1 − m1/(2r1) − m2/(2r2),其中 r_n 是到奇点 n 的距离
    - 生成的 npz 可直接传给 main.py 的 --reference 参数
    - --dedup: 对坐标做 1e-6 精度去重,保留首次出现(高分辨率优先)
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--psi", required=True, help="ψ 网格文本文件(每行 x y z psi)")
    p.add_argument("--m1", type=float, required=True, help="奇点 1 质量")
    p.add_argument("--m2", type=float, required=True, help="奇点 2 质量")
    p.add_argument("--x1", type=float, default=3.0, help="奇点 1 的 x 坐标")
    p.add_argument("--x2", type=float, default=-3.0, help="奇点 2 的 x 坐标")
    p.add_argument("--out", default="reference_u.npz", help="输出 npz 路径")
    p.add_argument("--dedup", action="store_true",
                   help="去除重叠区域的重复点(自适应网格推荐)")
    args = p.parse_args()

    # ---- 读取文本(自动处理逗号分隔) ----
    arr = np.loadtxt(args.psi)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("文本格式应为每行 4 列: x y z psi")
    n_raw = arr.shape[0]

    # ---- 去重:量化坐标到 1e-6 精度,保留首次出现(高分辨率块优先) ----
    if args.dedup and n_raw > 0:
        coords = arr[:, :3]
        key = np.round(coords / 1e-6).astype(np.int64)
        _, unique_idx = np.unique(key, axis=0, return_index=True)
        unique_idx.sort()
        arr = arr[unique_idx]
        if arr.shape[0] < n_raw:
            print(f"去重: {n_raw} → {arr.shape[0]} 点 "
                  f"(移除 {n_raw - arr.shape[0]} 个重叠点)")

    x_ref = arr[:, :3].astype(np.float32)
    psi = arr[:, 3]

    # ---- 计算修正项 u = ψ − 1 − Σ m/(2r) ----
    r1 = np.linalg.norm(x_ref - np.array([args.x1, 0, 0]), axis=1)
    r2 = np.linalg.norm(x_ref - np.array([args.x2, 0, 0]), axis=1)
    u_ref = (psi - 1.0 - args.m1 / (2.0 * r1) - args.m2 / (2.0 * r2)).astype(np.float32)

    np.savez(args.out, x_ref=x_ref, u_ref=u_ref)
    print(f"已保存参考解: {args.out} ({x_ref.shape[0]} 点)")
    print(f"  u_ref 范围: [{u_ref.min():.4e}, {u_ref.max():.4e}]")


if __name__ == "__main__":
    main()