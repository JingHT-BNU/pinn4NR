# --- path bootstrap (added during repo restructure: shared code lives in core/ and tools/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while True:
    if _os.path.isdir(_os.path.join(_ROOT, "core")):
        break
    _parent = _os.path.dirname(_ROOT)
    if _parent == _ROOT:
        break
    _ROOT = _parent
for _sub in ("core", "tools"):
    _p = _os.path.join(_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---
# -*- coding: utf-8 -*-
"""make_refs_v2.py -- 批量生成 a2v2 参考解(2026-09-01 无人值守指令 #2/#4)。

配置集:q∈[1,20] 共 50 个 = 复用已有 35 个(18 个 q* + 17 个 tq*,其中
q∈[1,20] 的)+ 新解 15 个补空隙。旧参考解(L48)不删除;a2v2 目录只放
本脚本产出的新解(分辨率由 --n-r/--lmax 等给定,默认高分辨率
N_r=768/L=64/N_th=96/N_ph=192,由 check_ref_conv.py 的结论决定)。

并行:ThreadPoolExecutor + CLI 子进程(--parallel N),幂等跳过已有 npz。

分辨率决策(check_ref_conv.py 2026-09-01 结论):N_r=768/L=64 高分辨率在
默认延拓+阻尼 Picard 下不收敛(q10:res_max=82,res_rms=6.9e-2,76.7 min),
故沿用已验证收敛且 TP 互证 1.5e-4 的 L48 标准参数。

用法:
  python make_refs_v2.py --parallel 2            # 只解缺失的 15 个新配置
  python make_refs_v2.py --all --parallel 2      # 全部 50 个(重解已有)
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = r"D:\AIs\PINN"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
CLI = os.path.join(ROOT, "pinn4NR", "tools", "spectral_reference.py")
HERE = os.path.join(ROOT, "pinn4NR")
DST = os.path.join(_ROOT, "data", "refs", "a2v2")
M1 = 0.5

EXISTING_Q = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.9, 2.0, 2.2, 2.4,
              2.5, 2.6, 2.8, 3.0, 3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.4, 5.8,
              6.3, 7.0, 7.4, 8.0, 8.6, 9.0, 10.0, 12.0, 14.0, 17.0, 20.0]
NEW_Q = [1.45, 2.35, 2.9, 3.15, 4.4, 6.6, 7.7, 9.5, 11.0, 10.5, 13.0, 15.0,
         16.0, 18.0, 19.0]


def label_of(q):
    if q == 1.0:
        return "q10"
    if q == 10.0:
        return "q100"
    base = f"q{q:g}".replace(".", "p")
    if q in (1.2, 1.4, 1.5, 1.7, 2.0, 2.4, 2.5, 2.8, 3.3, 3.9, 4.6, 5.0,
             5.4, 6.3, 7.4, 8.6):
        return base  # 原始 q* 命名(q12..q100)
    return f"tq{str(q).replace('.', 'p')}"


def raw_of(q):
    return f"{M1},{M1*q},3,-3,0.2,-0.2,0,0"


def solve_one(args, q):
    lb = label_of(q)
    out = os.path.join(DST, f"ref_{lb}.npz")
    if os.path.exists(out):
        return lb, True, "exists"
    cmd = [PY, "-u", CLI, "--params", raw_of(q), "--out", out, "--label", lb,
           "--n-r", str(args.n_r), "--lmax", str(args.lmax),
           "--n-theta", str(args.n_theta), "--n-phi", str(args.n_phi),
           "--grid-n", "161", "--R", "10"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(out)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-2:])
    return lb, ok, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="重解全部 50 个")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--n-r", type=int, default=512)
    ap.add_argument("--lmax", type=int, default=48)
    ap.add_argument("--n-theta", type=int, default=72)
    ap.add_argument("--n-phi", type=int, default=144)
    args = ap.parse_args()
    os.makedirs(DST, exist_ok=True)
    qs = sorted(set(EXISTING_Q + NEW_Q)) if args.all else sorted(set(NEW_Q))
    print(f"目标 {len(qs)} 配置,并行 {args.parallel},分辨率 "
          f"N_r={args.n_r} L={args.lmax} Nth={args.n_theta} Nph={args.n_phi}",
          flush=True)
    t0 = time.time()
    ok, fail = [], []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for lb, good, msg in ex.map(lambda q: solve_one(args, q), qs):
            (ok if good else fail).append(lb)
            print(f"[{len(ok)+len(fail)}/{len(qs)}] {lb}: "
                  f"{'OK' if good else 'FAIL'} {msg} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"\n完成 {len(ok)}/{len(qs)}" + (f"; 失败: {fail}" if fail else "")
          + f"; 总耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
