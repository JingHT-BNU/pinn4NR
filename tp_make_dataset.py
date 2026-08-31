"""tp_make_dataset.py —— TwoPuncturesC 批量初始配置生产线(opv3 数据拓展)。

配置族:q ∈ [1,100](m1=0.5 固定,m2=0.5q,x=±3,P=(0,±0.2,0),S=0),
几何+log 混合间隔;轴对称利用:u(x,y,z)=u(x,ρ,0) → 只采子午面半平面
(y≥0, z=0)网格(点数 ~3× 低于 3D 网格,3D 场可由旋转重构)。

流程(每配置):
  1. 写 TP par 文件(质心坐标,b=3,格式与 validate_with_tp 一致);
  2. WSL 调 dump_psi_grid --adaptive(两奇点邻域块 + 子午面薄板 + 外层块);
  3. 转 u_ref 并保存 data/datasets/opv3/ref_<label>.npz
     (x_ref, u_ref;子午面 y≥0)。

多核并行:多个配置的 WSL 进程并发(TP 求解单线程;subprocess 池控制并发数)。
断点:已有 npz 的配置跳过。日志:logs/A2/tp_make_dataset.log。

用法:
  .venv/Scripts/python.exe tp_make_dataset.py [--labels q1.5,q2.5,...]
      [--nprocs 4] [--npoints 60]
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logutil import setup_logging

log = logging.getLogger("paper.A2.tp_make_dataset")

HERE = os.path.dirname(os.path.abspath(__file__))
WIN_WORK = os.path.join(HERE, "data", "tp_opv3")
WSL_TP = "/mnt/d/AIs/PINN/TwoPuncturesC"
WSL_WORK = "/mnt/d/AIs/PINN/pinn4NR/data/tp_opv3"
WSL_EXE = f"{WSL_WORK}/dump_psi_grid"
WSL = ["wsl", "-d", "Ubuntu-24.04", "--"]

# q 列表:q∈[1,10] 沿用已有几何间隔+补密;q∈(10,100] log 间隔
Q_LIST = [1.5, 2.5, 5.0, 1.1, 1.3, 1.6, 1.9, 2.2, 2.6, 3.0, 3.6, 4.2, 5.8, 7.0,
          8.0, 9.0,
          12, 14, 17, 20, 24, 30, 36, 45, 55, 68, 82, 100]
M1 = 0.5
RMAX = 30.0


def label_of(q):
    return f"tq{str(q).replace('.', 'p')}"


def write_par(q, npoints, path):
    m1, m2 = M1, M1 * q
    b = 3.0
    par = f"""# TwoPunctures par — opv3 dataset q={q}
par_b={b!r}
par_m_plus={m1!r}
par_m_minus={m2!r}
target_M_plus=0.5
target_M_minus=0.5
par_P_plus1=0.
par_P_plus2=0.2
par_P_plus3=0.
par_P_minus1=0.
par_P_minus2=-0.2
par_P_minus3=0.
par_S_plus1=0.
par_S_plus2=0.
par_S_plus3=0.
par_S_minus1=0.
par_S_minus2=0.
par_S_minus3=0.
center_offset1=0.
center_offset2=0.
center_offset3=0.
give_bare_mass=1
npoints_A={npoints}
npoints_B={npoints}
npoints_phi={npoints}
Newton_tol=1.0000000000000000e-10
Newton_maxit=5
TP_epsilon=0.
TP_Tiny=0.
TP_Extend_Radius=0.
adm_tol=1.0000000000000000e-10
solve_momentum_constraint=0
use_external_initial_guess=0
do_residuum_debug_output=0
do_initial_debug_output=0
do_solution_file_output=0
do_bam_file_output=0
grid_setup_method=0
initial_lapse=2
initial_lapse_psi_exponent=-2.0000000000000000e+00
conformal_state=1
swap_xz=0
multiply_old_lapse=0
verbose=1
"""
    with open(path, "w") as f:
        f.write(par)


def meridian_blocks(b, tip_res=0.025):
    """子午面(y≥0, z=0)自适应块:奇点邻域块 + 全域子午面块。
    z 方向厚度 3 层(−δ..δ),覆盖插值需要。"""
    blocks = []
    for s in (+1.0, -1.0):
        blocks += [f"{s*b-1.5:.4f}", f"{s*b+1.5:.4f}", "-1.5", "1.5",
                   "-0.05", "0.05", "121", "121", "5"]
    # 全域子午面: x [-30,30] × y [0,30] × z 薄层
    blocks += ["-30", "30", "0", "30", "-0.05", "0.05", "241", "241", "5"]
    return blocks


def run_one(q, npoints, exe_ready):
    lb = label_of(q)
    out_npz = os.path.join(WIN_WORK, f"ref_{lb}.npz")
    if os.path.exists(out_npz):
        return (lb, "skip", 0.0)
    t0 = time.time()
    os.makedirs(WIN_WORK, exist_ok=True)
    par_win = os.path.join(WIN_WORK, f"{lb}.par")
    psi_win = os.path.join(WIN_WORK, f"{lb}.psi")
    write_par(q, npoints, par_win)
    par_wsl = f"{WSL_WORK}/{lb}.par"
    psi_wsl = f"{WSL_WORK}/{lb}.psi"
    cmd = WSL + [WSL_EXE, "--adaptive", par_wsl, "3.0", psi_wsl]
    cmd += meridian_blocks(3.0)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(psi_win):
        with open(psi_win + ".err.log", "w", encoding="utf-8",
                  errors="replace") as f:
            f.write(r.stdout + r.stderr)
        return (lb, f"FAIL rc={r.returncode}", time.time() - t0)
    arr = np.loadtxt(psi_win)
    x_ref = arr[:, :3].astype(np.float32)
    psi = arr[:, 3]
    r1 = np.linalg.norm(x_ref - np.array([3.0, 0, 0]), axis=1)
    r2 = np.linalg.norm(x_ref - np.array([-3.0, 0, 0]), axis=1)
    u = (psi - 1.0 - M1 / (2.0 * r1) - M1 * q / (2.0 * r2)).astype(np.float32)
    np.savez(out_npz, x_ref=x_ref, u_ref=u, q=float(q), m1=M1, m2=M1 * q)
    os.remove(psi_win)
    return (lb, f"ok {len(x_ref)} pts", time.time() - t0)


def main():
    setup_logging("A2", "tp_make_dataset")
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None,
                    help="逗号分隔 q 值;缺省全部 Q_LIST")
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--npoints", type=int, default=60)
    args = ap.parse_args()
    qs = [float(x) for x in args.labels.split(",")] if args.labels else Q_LIST
    os.makedirs(WIN_WORK, exist_ok=True)
    # 确保 dump_psi_grid 在 WSL 侧可用(从 tp_work 复制)
    exe_win = os.path.join(WIN_WORK, "dump_psi_grid")
    if not os.path.exists(exe_win):
        src = r"D:\AIs\PINN\paper\tools\tp_work\dump_psi_grid"
        import shutil
        shutil.copy2(src, exe_win)
    # 大 q 配置峰值更大 → npoints 随 log q 提升
    log.info(f"opv3 数据生产: {len(qs)} 配置, 并发 {args.nprocs}, "
             f"npoints={args.npoints}")
    fails = 0
    with ThreadPoolExecutor(max_workers=args.nprocs) as ex:
        futs = {ex.submit(run_one, q, args.npoints, True): q for q in qs}
        for fut in as_completed(futs):
            lb, msg, el = fut.result()
            log.info(f"  {lb:<8} {msg} ({el:.0f}s)")
            if msg.startswith("FAIL"):
                fails += 1
    log.info(f"完成,失败 {fails}/{len(qs)}")
    json.dump({"q_list": qs, "npoints": args.npoints},
              open(os.path.join(WIN_WORK, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
