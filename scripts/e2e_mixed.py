"""End-to-end correctness against the true stock, and a mixed-effects model for 2N vs 3N.

Run from a directory holding the scripts and unpacked raw logs.
"""
import argparse
import importlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R)


def mods():
    import drift_pilot
    importlib.reload(drift_pilot)
    if "drift_v2" in sys.modules:
        v2 = importlib.reload(sys.modules["drift_v2"])
    else:
        import drift_v2 as v2
    return drift_pilot, v2


def sessions(d):
    for fn in sorted(os.listdir(os.path.join(R, d))):
        if fn.startswith("run_"):
            yield {r["turn"]: r for r in map(json.loads, open(os.path.join(R, d, fn), encoding="utf-8")) if "reply" in r}


def setup(d):
    m = json.load(open(os.path.join(R, d, "meta.json")))
    base, v2 = mods()
    if "wording" in m:
        import drift_v3
        importlib.reload(drift_v3)
        drift_v3.configure(m["period"], m["wording"])
    else:
        import drift_v4
        importlib.reload(drift_v4)
        drift_v4.configure(argparse.Namespace(**{k: m[k] for k in ("period", "turns", "list_turns", "active_from", "seed")}))
    init, script = base.build_script(m["turns"])
    return m, base, v2, init, script


def e2e(d):
    m, base, v2, init, script = setup(d)
    if m.get("fresh_each_turn"):
        return None
    turns_ok = turns = 0
    first_div = []
    full_ok = 0
    for s in sessions(d):
        truth = dict(init)
        div = None
        for e in script:
            t = e["turn"]
            truth, rej = v2.apply2(truth, e["ops"], t in v2.NIGHT)
            r = s.get(t)
            if r is None:
                break
            turns += 1
            try:
                o = json.loads(r["reply"].strip())
                st = {k: o.get("stock", {}).get(k) for k in truth}
            except (ValueError, AttributeError):
                o, st = {}, None
            ok = st == truth
            turns_ok += ok
            total = sum(v for k, v in truth.items() if not k.startswith("Q"))
            note_ok = isinstance(o.get("note"), str) and o["note"].startswith("盘点" if t in v2.MOD17 else "已核") and len(o["note"]) <= 8
            rj = o.get("rejected")
            rj_ok = isinstance(rj, list) and sorted(str(x).strip().lstrip("-").strip() for x in rj) == sorted(rej)
            full = ok and o.get("total") == total and o.get("seal") == (t * 7 + total) % 97 and note_ok and rj_ok
            full_ok += full
            if not ok and div is None:
                div = t
        first_div.append(div)
    n_div = [x for x in first_div if x is not None]
    return dict(batch=d, turns=turns, stock_match=turns_ok / turns, full_answer=full_ok / turns,
                sessions=len(first_div), diverged=len(n_div), median_first_div=float(np.median(n_div)) if n_div else None)


rows = []
for d in ["A_p17_old", "B_p17_new", "C_p13", "D_p19", "E_p23"] + [f"p1/{x}" for x in ("none", "calc26", "calc10", "generic26", "list", "active20")] + ["p2/t13", "p4/c13", "p4/p14"]:
    r = e2e(d)
    if r:
        rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("== 端到端正确率（相对真实库存）")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("所有批次合计：库存与真实一致的轮次比例 %.3f，整轮回答完全正确 %.3f" % (
    (df.stock_match * df.turns).sum() / df.turns.sum(), (df.full_answer * df.turns).sum() / df.turns.sum()))

print("\n== 混合效应逻辑回归：第 3N 轮 vs 第 2N 轮（会话随机截距，批次固定效应，变分贝叶斯）")
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

recs = []
for d, p in (("combined", 17), ("p2/t13", 13), ("p2/t19", 19), ("p2/t23", 23), ("p4/c13", 13), ("p4/p14", 14)):
    for i, s in enumerate(sessions(d)):
        for k in (2, 3):
            t = k * p
            if t in s and s[t]["reply"].strip():
                prior = any(s[x]["drift"] or s[x]["slips"] for x in s if x < t)
                recs.append(dict(y=int("mod17_forgotten" in s[t]["drift"]), third=int(k == 3), prior=int(prior), batch=d.replace("/", "_"), sess=f"{d}/{i}"))
D = pd.DataFrame(recs)
for formula in ("y ~ third + C(batch)", "y ~ third + prior + C(batch)"):
    mdl = BinomialBayesMixedGLM.from_formula(formula, {"sess": "0 + C(sess)"}, D)
    fit = mdl.fit_vb()
    names = mdl.exog_names
    for v in ("third", "prior"):
        if v in names:
            j = names.index(v)
            mu, sd = fit.fe_mean[j], fit.fe_sd[j]
            print(f"  {formula:<32} {v:<6} log-OR 后验均值 {mu:.2f}，95% 区间 [{mu-1.96*sd:.2f}, {mu+1.96*sd:.2f}]")
    print(f"  会话随机截距 log-SD 后验均值 {fit.vcp_mean[0]:.2f}")
