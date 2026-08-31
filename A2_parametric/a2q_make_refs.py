"""a2q_make_refs.py —— 为 A2 单参数 q∈[1,10] 攻关批量生成 L=48 谱参考解。

参数固定: m1=0.5, xs=(+3,-3), P=(0,±0.2,0), S=0;q = m2/m1 ∈ [1,10],m2 = 0.5·q。
训练 15 配置 + 留出 3 配置;ref_a2_q10/q20 已存在,共需新解 16 个。
逐个调用 spectral_reference.py CLI(默认 N_r=512/L=48/N_th=72/N_ph=144),
单个失败不中断,最后汇总。预计 ~9-10 min/配置(后台运行)。
"""
import os, subprocess, sys, time

ROOT = r"D:\AIs\PINN"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
CLI = os.path.join(ROOT, "paper", "tools", "spectral_reference.py")
DST = os.path.join(ROOT, "paper", "tools", "refs_a2")
os.makedirs(DST, exist_ok=True)

# (label, m2) —— q = 2*m2
CONFIGS = [("q12", 0.6), ("q14", 0.7), ("q15", 0.75), ("q17", 0.85),
           ("q24", 1.2), ("q25", 1.25), ("q28", 1.4), ("q33", 1.65),
           ("q39", 1.95), ("q46", 2.3), ("q50", 2.5), ("q54", 2.7),
           ("q63", 3.15), ("q74", 3.7), ("q86", 4.3), ("q100", 5.0)]

ok, fail = [], []
t0 = time.time()
for i, (lb, m2) in enumerate(CONFIGS, 1):
    out = os.path.join(DST, f"ref_a2_{lb}.npz")
    if os.path.exists(out):
        print(f"[{i}/16] {lb} 已存在,跳过", flush=True)
        ok.append(lb)
        continue
    params = f"0.5,{m2},3,-3,0.2,-0.2,0,0"
    print(f"[{i}/16] {lb} (m2={m2}) 开始, 已耗时 {time.time()-t0:.0f}s", flush=True)
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
