"""verify_cache_v2.py — 验证 v2 κ 缓存质量"""
import json
import numpy as np

cache = json.load(open("D:/AIs/PINN/paper/A3_multi_param/multi_param_kappa_cache.json"))
meta = cache["meta"]
print(f"version: {meta.get('version')}, n_vol: {meta['n_vol']}, seeds: {meta.get('n_kappa_seeds')}")

for split in ["train", "val"]:
    ks = np.array([e["kappa"] for e in cache[split]])
    spreads = np.array([e.get("kappa_spread", 0) for e in cache[split]])
    print(f"\n{split}: n={len(ks)}")
    print(f"  κ 范围: [{ks.min():.4f}, {ks.max():.4f}], 均值 {ks.mean():.4f}")
    print(f"  seed spread: mean={spreads.mean():.5f}, max={spreads.max():.5f}")

base = cache["train"][0]
is_base = np.allclose(base["raw_params"], [0.5,0.5,3.0,-3.0,0.2,-0.2,0.0,0.0])
print(f"\ntrain_0000 是 base: {is_base}, κ={base['kappa']:.6f} (A2 精确值 0.63849)")
print(f"  相对误差: {abs(base['kappa']-0.63849)/0.63849*100:.2f}%")
