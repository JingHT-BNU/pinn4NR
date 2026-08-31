"""make_refs_opv3.py —— 经 CLI 子进程批量生成 opv3 新配置谱参考解(已验证路径)。

与 make_refs_a2.py 同款:逐配置调 spectral_reference.py CLI(默认 N_r=512/L48/
N_th=72/N_ph=144),单个失败不中断。完成后顺手生成 refsub+cfg(κ 用 kappa_tp.json)。
幂等:已有 npz 跳过。删除损坏 npz 后重跑即可修复。
"""
import json
import os
import subprocess
import sys
import time

ROOT = r"D:\AIs\PINN"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
CLI = os.path.join(ROOT, "paper", "tools", "spectral_reference.py")
REPO = os.path.join(ROOT, "pinn4NR")
DST = os.path.join(REPO, "data", "refs", "a2")
DATA_DIR = os.path.join(REPO, "data", "datasets", "a2q_data")
os.makedirs(DST, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

Q_LIST = [1.5, 2.5, 5.0, 1.1, 1.3, 1.6, 1.9, 2.2, 2.6, 3.0, 3.6, 4.2, 5.8,
          7.0, 8.0, 9.0, 12, 14, 17, 20, 24, 30, 36, 45, 55, 68, 82, 100]
M1 = 0.5


def label_of(q):
    return f"tq{str(q).replace('.', 'p')}"


def raw_of(q):
    return f"{M1},{M1*q},3,-3,0.2,-0.2,0,0"


def main():
    t0 = time.time()
    ok, fail = [], []
    for i, q in enumerate(Q_LIST, 1):
        lb = label_of(q)
        out = os.path.join(DST, f"ref_{lb}.npz")
        if os.path.exists(out):
            print(f"[{i}/{len(Q_LIST)}] {lb} 已存在,跳过", flush=True)
            ok.append(lb)
            continue
        print(f"[{i}/{len(Q_LIST)}] {lb} (q={q:g}) 开始, "
              f"已耗时 {time.time()-t0:.0f}s", flush=True)
        r = subprocess.run([PY, "-u", CLI, "--params", raw_of(q), "--out", out,
                            "--label", lb],
                           capture_output=True, text=True)
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-3:])
        if r.returncode == 0 and os.path.exists(out):
            ok.append(lb)
            print(f"    OK {lb}: {os.path.getsize(out)/1e6:.1f} MB\n{tail}",
                  flush=True)
        else:
            fail.append(lb)
            print(f"    FAIL {lb} (rc={r.returncode})\n{tail}", flush=True)
    print(f"\n谱解完成: {len(ok)}/{len(Q_LIST)}"
          + (f"; 失败 {fail}" if fail else "")
          + f"; 总耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
