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
"""a2q_gates.py -- 最终目标门槛判定(读 eval2.json 输出 PASS/FAIL 矩阵)。

最终目标(用户 2026-09-01 无人值守指令):q∈[1,10] 任意值达到 A1 base 同水准,
可视化上模型解与参考解高度贴合,仅峰顶与两峰间极小值附近有些许差异。

门槛(G = gates_config.json,由 A1 base 新口径校准值填充):
  G1 global : 每个 [1,10] 测试配置 global ≤ G.global_factor × T
  G2 peak   : peak_height_ratio ∈ [0.95, 1.05] 且 peak_win_l2re ≤ G.peak_l2re
  G3 valley : valley_depth_ratio ∈ [0.95, 1.05] 且 valley_l2re ≤ G.valley_l2re
  G4 zero-shot: 留出配置同样满足 G1-G3(泛化,防背题)

用法: python a2q_gates.py --runs a2q_v5,a2q_v4,a2q_c2_50
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(_ROOT, "data", "runs", "a2")
CFG = os.path.join(_ROOT, "data", "gates_config.json")

DEFAULT = {
    "T_global": None,          # A1 base 新口径 global(校准后填)
    "global_factor": 1.5,
    "peak_l2re": 3.0e-3,
    "valley_l2re": 5.0e-3,
    "ratio_lo": 0.95,
    "ratio_hi": 1.05,
    "goal_q_range": [1.0, 10.0],
}


def gate_row(r, G, T):
    ok1 = r["global"] <= G["global_factor"] * T
    hr = r.get("peak_height_ratio")
    ok2 = (hr is not None and G["ratio_lo"] <= hr <= G["ratio_hi"]
           and r["peak_win_l2re"] <= G["peak_l2re"])
    dr = r.get("valley_depth_ratio")
    ok3 = (dr is not None and G["ratio_lo"] <= dr <= G["ratio_hi"]
           and r["valley_l2re"] <= G["valley_l2re"])
    return ok1, ok2, ok3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()
    G = dict(DEFAULT)
    if os.path.exists(CFG):
        with open(CFG, encoding="utf-8") as f:
            G.update(json.load(f))
    T = G["T_global"]
    if T is None:
        a1p = os.path.join(_ROOT, "data", "runs", "a1", "base_a1", "eval2_a1.json")
        if os.path.exists(a1p):
            T = json.load(open(a1p))["results"]["q10"]["global"]
            G.pop("_note", None)
            G["T_global"] = T
            with open(CFG, "w", encoding="utf-8") as f:
                json.dump(G, f, indent=1)
        else:
            raise SystemExit("无 A1 校准值(先跑 a2q_eval2 --a1)")
    lo_q, hi_q = G["goal_q_range"]
    print(f"T(A1 base new-metric global, q=1) = {T:.4e}; "
          f"G1≤{G['global_factor']*T:.3e} G2_pk≤{G['peak_l2re']:.1e} "
          f"h∈[{G['ratio_lo']},{G['ratio_hi']}] "
          f"G3_vy≤{G['valley_l2re']:.1e} d∈同上; 目标 q∈[{lo_q},{hi_q}]")
    all_pass = True
    for rn in args.runs.split(","):
        p = os.path.join(RUNS, rn, "eval2.json")
        if not os.path.exists(p):
            print(f"[{rn}] eval2.json 缺失")
            all_pass = False
            continue
        res = json.load(open(p))["results"]
        rows, n_fail = [], 0
        for lb, r in sorted(res.items(), key=lambda kv: kv[1]["q"]):
            if not (lo_q <= r["q"] <= hi_q):
                continue
            ok1, ok2, ok3 = gate_row(r, G, T)
            ok = ok1 and ok2 and ok3
            zero = r["group"] in ("heldout", "zero_shot")
            if zero and not ok:
                n_fail += 1
            rows.append(f"  {lb:<7} q={r['q']:<6g} {'0shot' if zero else 'train'}"
                        f" g={r['global']:.3e}{'+' if ok1 else '-'}"
                        f" pk={r['peak']:.2e} axw={r['peak_win_l2re']:.2e}"
                        f"{'+' if ok2 else '-'} h={r['peak_height_ratio']:.3f}"
                        f" vy={r['valley']:.2e}"
                        f" d={r['valley_depth_ratio']:.3f}"
                        f"{'+' if ok3 else '-'} {'PASS' if ok else 'FAIL'}")
        print(f"[{rn}] {'='*20} {sum(1 for x in rows if 'FAIL' not in x)}"
              f"/{len(rows)} 配置达标;零样本失败 {n_fail}")
        print("\n".join(rows))
        if n_fail:
            all_pass = False
        fails = [x for x in rows if "FAIL" in x and "0shot" not in x]
        if fails:
            all_pass = False
    print("OVERALL:", "PASS(达最终目标)" if all_pass else "NOT YET(继续迭代)")


if __name__ == "__main__":
    main()
