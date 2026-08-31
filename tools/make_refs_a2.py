"""为 A2 的 10 个 q 配置批量生成 L=48 谱参考解(todo 十:A2 接入谱参考解套件)。

参数固定: m1=0.5, xs=(3,-3), P=(0,±0.2,0), S=0;仅 m2 变化(q=m2/m1∈[0.5,2])。
逐个调用 spectral_reference.py CLI(默认 N_r=512/L=48/N_th=72/N_ph=144),
单个失败不中断,最后汇总。预计 ~8-10 min/配置。
"""
import os, subprocess, sys, time

ROOT = r"D:\AIs\PINN"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
CLI = os.path.join(ROOT, "paper", "tools", "spectral_reference.py")
DST = os.path.join(ROOT, "paper", "tools", "refs_a2")
os.makedirs(DST, exist_ok=True)

# (label, m2)
CONFIGS = [("q05", 0.25), ("q06", 0.3), ("q07", 0.35), ("q09", 0.45),
           ("q10", 0.5), ("q13", 0.65), ("q14", 0.7), ("q16", 0.8),
           ("q18", 0.9), ("q20", 1.0)]

ok, fail = [], []
t0 = time.time()
for i, (lb, m2) in enumerate(CONFIGS, 1):
    out = os.path.join(DST, f"ref_a2_{lb}.npz")
    if os.path.exists(out):
        print(f"[{i}/10] {lb} 已存在,跳过", flush=True)
        ok.append(lb)
        continue
    params = f"0.5,{m2},3,-3,0.2,-0.2,0,0"
    print(f"[{i}/10] {lb} (m2={m2}) 开始, 已耗时 {time.time()-t0:.0f}s", flush=True)
    r = subprocess.run([PY, "-u", CLI, "--params", params, "--out", out,
                        "--label", f"a2_{lb}"],
                       capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-3:])
    if r.returncode == 0 and os.path.exists(out):
        ok.append(lb)
        print(f"    OK {lb}: {os.path.getsize(out)/1e6:.1f} MB\n{tail}", flush=True)
    else:
        fail.append(lb)
        print(f"    FAIL {lb} (rc={r.returncode})\n{tail}", flush=True)

print(f"\n完成: 成功 {len(ok)}/{len(CONFIGS)} {ok}"
      + (f"; 失败 {fail}" if fail else "")
      + f"; 总耗时 {(time.time()-t0)/60:.1f} min", flush=True)
sys.exit(1 if fail else 0)
