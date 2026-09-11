#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把多个 run 的全部评估口径汇总成一张对照表。

A2 目前有 5 套口径,分散在不同文件里:
  eval2.json   全局/peak/ring/valley/far 相对 L2 误差、峰高比、谷深比
  gates        45 配置达标判定
  rough.json   远场二阶导符号翻转次数
  physics.json Hamilton 残差(near/mid/outer/far)、ADM 质量、偶极
  history.json 训练损失轨迹

逐个打开比对既慢又容易记错区间。本脚本一次读入并横向打印,默认把
"目标范围 q∈[1,10]"与"全部 68 配置"分开统计 —— 这两者的上界差别很大
(3.5e-3 vs 7.9e-3),混在一起会误判。

用法::

    python a2_q/compare_runs.py --runs a2q_v6a,a2q_v6d,a2q_v6e
    python a2_q/compare_runs.py --runs a2q_v6a,a2q_v6f --json cmp.json
"""
# ---- path bootstrap ----------------------------------------------------
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE,):
    while True:
        if _os.path.isdir(_os.path.join(_p, "core")) and \
           _os.path.isdir(_os.path.join(_p, "a2_q")):
            break
        _np = _os.path.dirname(_p)
        if _np == _p:
            break
        _p = _np
    if _os.path.isdir(_os.path.join(_p, "core")):
        import sys as _sys
        for _sub in ("core", "tools", "a2_q"):
            _q = _os.path.join(_p, _sub)
            if _os.path.isdir(_q) and _q not in _sys.path:
                _sys.path.insert(0, _q)
        _ROOT = _p
        break
# ------------------------------------------------------------------------

import argparse
import json
import os
import statistics as st

RUNS_DIR = os.path.join(_ROOT, "data", "runs", "a2")
REGIONS = ("near", "mid", "outer", "far")
GATE_CFG = None
try:
    with open(os.path.join(_ROOT, "data", "gates_config.json"),
              encoding="utf-8") as _fh:
        GATE_CFG = json.load(_fh)
except Exception:                                             # noqa: BLE001
    GATE_CFG = None


def jload(p):
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                         # noqa: BLE001
        return None


def med(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return st.median(v) if v else float("nan")


def collect(run):
    d = os.path.join(RUNS_DIR, run)
    out = {"run": run, "dir": d}
    ev = jload(os.path.join(d, "eval2.json"))
    out["eval2"] = ev
    out["rough"] = jload(os.path.join(d, "rough.json"))
    out["physics"] = jload(os.path.join(d, "physics.json"))
    out["hist"] = jload(os.path.join(d, "history.json"))
    return out


def fmt(x, spec="%.3e"):
    try:
        if x != x:
            return "-"
        return spec % x
    except Exception:                                         # noqa: BLE001
        return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="逗号分隔的 run 名(相对 data/runs/a2/)")
    ap.add_argument("--q-lo", type=float, default=1.0)
    ap.add_argument("--q-hi", type=float, default=10.0)
    ap.add_argument("--json", default=None, help="把汇总结果另存为 json")
    args = ap.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    data = [collect(r) for r in runs]
    summary = {}

    # ---------- 1. eval2 ----------
    print("=" * 108)
    print("A. eval2 相对 L2 误差  (goal: q∈[%.3g,%.3g]; 全 68 配置另列)"
          % (args.q_lo, args.q_hi))
    print("-" * 108)
    hdr = ("%-12s %6s | %-11s %-11s %-11s | %-11s | %-9s %-9s"
           % ("run", "n", "global min", "global mean", "global max",
              "far max", "h min", "d max"))
    print(hdr)
    for d in data:
        ev = d["eval2"]
        if not ev:
            print("%-12s  (无 eval2.json)" % d["run"])
            continue
        res = ev["results"]
        sel = {k: v for k, v in res.items()
               if args.q_lo <= v["q"] <= args.q_hi}
        g = [v["global"] for v in sel.values()]
        fa = [v["far"] for v in sel.values()]
        h = [v["peak_height_ratio"] for v in sel.values()]
        dv = [v["valley_depth_ratio"] for v in sel.values()]
        print("%-12s %6d | %-11s %-11s %-11s | %-11s | %-9.4f %-9.4f"
              % (d["run"], len(sel), fmt(min(g)), fmt(sum(g) / max(len(g), 1)),
                 fmt(max(g)), fmt(max(fa)), min(h), max(dv)))
        gA = [v["global"] for v in res.values()]
        summary.setdefault("eval2", {})[d["run"]] = dict(
            n_goal=len(sel), global_min=min(g), global_mean=sum(g) / len(g),
            global_max=max(g), far_max=max(fa), h_min=min(h), d_max=max(dv),
            n_all=len(res), global_max_all=max(gA),
            h_min_all=min(v["peak_height_ratio"] for v in res.values()))
    print("  全 68 配置上界:")
    for d in data:
        ev = d["eval2"]
        if not ev:
            continue
        res = ev["results"]
        print("    %-12s global max %.4e   h min %.4f   far max %.4e"
              % (d["run"], max(v["global"] for v in res.values()),
                 min(v["peak_height_ratio"] for v in res.values()),
                 max(v["far"] for v in res.values())))

    # ---------- 2. 门槛 ----------
    print()
    print("=" * 108)
    print("B. 达标门槛(重新判定, 不依赖历史日志)")
    print("-" * 108)

    def gates(ev):
        """与 a2q_gates.py 的 gate_row 完全一致的判定。

        注意门槛不是 0.99~1.01:gates_config.json 里 ratio 区间是
        [0.95, 1.05],而且 G2/G3 还各带一个 l2re 上限。曾用 0.99~1.01
        自测,误报 36/45;官方判定是 45/45。
        """
        if not ev or not GATE_CFG:
            return None
        G, T = GATE_CFG, GATE_CFG["T_global"]
        res = ev["results"]
        sel = {k: v for k, v in res.items()
               if G["goal_q_range"][0] <= v["q"] <= G["goal_q_range"][1]}
        ho = {k: v for k, v in res.items() if v.get("group") != "train"}

        def row(r):
            ok1 = r["global"] <= G["global_factor"] * T
            ok2 = (G["ratio_lo"] <= r["peak_height_ratio"] <= G["ratio_hi"]
                   and r["peak_win_l2re"] <= G["peak_l2re"])
            ok3 = (G["ratio_lo"] <= r["valley_depth_ratio"] <= G["ratio_hi"]
                   and r["valley_l2re"] <= G["valley_l2re"])
            return ok1, ok2, ok3

        g1 = g2 = g3 = 0
        for v in sel.values():
            a, b, c = row(v)
            g1 += a
            g2 += b
            g3 += c
        hofail = sum(1 for v in ho.values() if not all(row(v)))
        return dict(G1=g1, G2=g2, G3=g3, n=len(sel), heldout_fail=hofail,
                    T_global=T, global_factor=G["global_factor"],
                    ratio=(G["ratio_lo"], G["ratio_hi"]))

    for d in data:
        g = gates(d["eval2"])
        if not g:
            continue
        print("%-12s G1 %2d/%2d   G2 %2d/%2d   G3 %2d/%2d   零样本失败 %d"
              % (d["run"], g["G1"], g["n"], g["G2"], g["n"], g["G3"],
                 g["n"], g["heldout_fail"]))
        summary.setdefault("gates", {})[d["run"]] = g

    # ---------- 3. 粗糙度 ----------
    print()
    print("=" * 108)
    print("C. 远场二阶导符号翻转次数(越少越光滑)")
    print("-" * 108)
    for d in data:
        r = d["rough"]
        if not r:
            print("%-12s  (无 rough.json)" % d["run"])
            continue
        print("%-12s %s" % (d["run"], json.dumps(r, ensure_ascii=False)[:150]))
        summary.setdefault("rough", {})[d["run"]] = r

    # ---------- 4. 物理型指标 ----------
    print()
    print("=" * 108)
    print("D. Hamilton 相对残差 ‖R‖/‖S‖ 与 ADM 质量(中位数, 逐配置)")
    print("-" * 108)
    hdr2 = "%-12s | " + " ".join(["%-11s"] * 4) + " | %-12s %-12s"
    print(hdr2 % (("run",) + REGIONS + ("ADM rel", "dip rel")))
    for d in data:
        ph = d["physics"]
        if not ph:
            print("%-12s  (无 physics.json)" % d["run"])
            continue
        cfgs = ph["configs"]
        row = []
        for reg in REGIONS:
            row.append(med([c["resid"][reg]["R_rel_rms"]
                            for c in cfgs.values() if reg in c["resid"]]))
        adm, dip = [], []
        for c in cfgs.values():
            rf = c.get("M_ADM_ref")
            if rf:
                adm.append(abs(c["M_ADM"] - rf) / rf)
            dr = c.get("dipole_abs_ref")
            if dr:
                dip.append(abs(c["dipole_abs"] - dr) / dr)
        print(hdr2 % ((d["run"],) + tuple(fmt(x) for x in row)
                      + (fmt(med(adm)), fmt(med(dip)))))
        summary.setdefault("physics", {})[d["run"]] = dict(
            regions={r: med([c["resid"][r]["R_rel_rms"]
                             for c in cfgs.values() if r in c["resid"]])
                     for r in REGIONS},
            adm_rel=med(adm), dip_rel=med(dip), n=len(cfgs))

    # ---------- 5. 损失轨迹 ----------
    print()
    print("=" * 108)
    print("E. 训练损失(五分位中位数)")
    print("-" * 108)
    for d in data:
        h = d["hist"]
        if not h:
            print("%-12s  (无 history.json)" % d["run"])
            continue
        msg = "%-12s " % d["run"]
        for key in ("L_ref", "l_pde", "l_lap"):
            v = h.get(key)
            if not v:
                continue
            n = len(v)
            q = [st.median(v[i * n // 5:(i + 1) * n // 5]) for i in range(5)]
            msg += " %s[%s] n=%d" % (key, " ".join("%.2e" % x for x in q), n)
        print(msg)
        summary.setdefault("hist", {})[d["run"]] = {
            k: len(v) for k, v in h.items() if isinstance(v, list)}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1,
                      default=float)
        print("\n汇总已写入 %s" % args.json)


if __name__ == "__main__":
    main()
