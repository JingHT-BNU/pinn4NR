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
"""run_queue_v2.py -- 无人值守 GPU 串行队列(2026-09-01 夜间管线)。

16GB 显存机器:所有 GPU 任务必须串行,否则 CUDA 系统内存回退导致全局减速。
队列(每步幂等,已完成的跳过;单步失败不阻塞后续,结尾汇总):
  1. eval2 --a1                     (A1 阈值校准)
  2. post_refs_v2                   (50 配置 refsub_v2/cfg_v2/κ*_v2)
  3. train opv5  (a2q_v5,   15000 步) → eval2 grid-n 81
  4. train opv4  (a2q_v4,   15000 步) → eval2 grid-n 81
  5. train c2    (a2q_c2_50,15000 步) → eval2 grid-n 81
  6. a2q_gates 汇总
用法: python run_queue_v2.py [--steps N]
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RUNS = os.path.join(_ROOT, "data", "runs", "a2")


def sh(cmd, log):
    print(f"\n===== [{time.strftime('%H:%M:%S')}] {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
    print("\n".join(tail), flush=True)
    print(f"----- rc={r.returncode} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--ppc", type=int, default=2560)
    args = ap.parse_args()
    venv_py = PY
    results = {}

    def run(name, cmd, done_check):
        if done_check():
            results[name] = "skip(done)"
            return
        rc = sh(cmd, name)
        results[name] = "ok" if rc == 0 else f"FAIL rc={rc}"

    run("eval2_a1", [venv_py, "-u", "a2q_eval2.py", "--a1",
                     "data/runs/a1/base_a1", "--skip-fig"],
        lambda: os.path.exists(os.path.join(_ROOT, "data", "runs", "a1",
                                            "base_a1", "eval2_a1.json")))
    v2_dir = os.path.join(_ROOT, "data", "datasets", "a2q_data_v2")

    def post_done():
        return os.path.isdir(v2_dir) and len(
            [f for f in os.listdir(v2_dir) if f.startswith("cfg_")]) >= 60

    run("post_refs_v2", [venv_py, "-u", "post_refs_v2.py"], post_done)

    for variant, exp in (("opv5", "a2q_v5"), ("opv4", "a2q_v4"),
                         ("c2", "a2q_c2_50")):
        ed = os.path.join(RUNS, exp)
        run(f"train_{variant}", [venv_py, "-u", "a2q_train_v4.py",
                                 "--variant", variant, "--exp-name", exp,
                                 "--steps", str(args.steps),
                                 "--pts-per-cfg", str(args.ppc)],
            lambda ed=ed: os.path.exists(os.path.join(ed, "model.pt")))
        run(f"eval2_{variant}", [venv_py, "-u", "a2q_eval2.py", "--run",
                                 os.path.relpath(ed, HERE), "--grid-n", "81"],
            lambda ed=ed: os.path.exists(os.path.join(ed, "eval2.json")))

    sh([venv_py, "-u", "a2q_gates.py", "--runs",
        "a2q_v5,a2q_v4,a2q_c2_50"], "gates")
    print("\n===== 队列汇总 =====", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
