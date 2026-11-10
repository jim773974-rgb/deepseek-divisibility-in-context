"""Re-analysis addressing the external review: paired tests, clustering, multiplicity, power, sensitivity.

Run from a directory that holds the unpacked raw logs (same layout as verify.py).
"""
import glob
import json
import os
import re
import warnings
from math import comb

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
R = os.path.dirname(os.path.abspath(__file__))


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(R, d, "run_*.jsonl"))):
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        out.append({"id": os.path.basename(f), "t": {r["turn"]: r for r in recs if "reply" in r}})
    return out


def fail(r, code=False):
    return any(k.startswith("audit") for k in r["drift"]) if code else "mod17_forgotten" in r["drift"]


def fisher(a, b, c, d):
    n, r, k = a + b + c + d, a + b, a + c
    pm = lambda x: comb(k, x) * comb(n - k, r - x) / comb(n, r)
    p0 = pm(a)
    return sum(pm(x) for x in range(max(0, r - (n - k)), min(r, k) + 1) if pm(x) <= p0 * (1 + 1e-9))


def mcnemar(pairs):
    """pairs: list of (x, y) booleans. Exact two-sided test on discordant pairs."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def newcombe(k1, n1, k2, n2, z=1.96):
    def wil(k, n):
        p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wil(k1, n1)
    l2, u2 = wil(k2, n2)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


print("=" * 70)
print("A. 同一会话内配对：1N vs 2N，2N vs 3N（精确 McNemar）")
groups = [("C_p13", 13), ("A_p17_old", 17), ("B_p17_new", 17), ("D_p19", 19), ("E_p23", 23), ("combined", 17),
          ("p1/none", 13), ("p2/s13_1", 13), ("p2/s13_2", 13), ("p2/s23_1", 23), ("p2/s23_2", 23),
          ("p2/t13", 13), ("p2/t19", 19), ("p2/t23", 23), ("p4/c13", 13), ("p4/p14", 14)]
pool12, pool23 = [], []
rows_gee = []
for d, p in groups:
    S = load(d)
    pr12 = [(fail(s["t"][2 * p]), fail(s["t"][p])) for s in S if p in s["t"] and 2 * p in s["t"] and s["t"][2 * p]["reply"].strip()]
    pool12 += pr12
    b, c, pv = mcnemar(pr12)
    line = f"  {d:<11} 2N错1N对 {b:>2}, 1N错2N对 {c}, p={pv:.2g}"
    pr23 = [(fail(s["t"][2 * p]), fail(s["t"][3 * p])) for s in S if 3 * p in s["t"] and 2 * p in s["t"]]
    if pr23:
        pool23 += pr23
        b2, c2, pv2 = mcnemar(pr23)
        line += f" | 2N错3N对 {b2}, 3N错2N对 {c2}, p={pv2:.2g}"
    print(line)
    for s in S:
        for k in (1, 2, 3):
            t = k * p
            if t in s["t"] and s["t"][t]["reply"].strip():
                prior = any(s["t"][x]["drift"] or s["t"][x]["slips"] for x in s["t"] if x < t)
                rows_gee.append(dict(batch=d, sess=f"{d}/{s['id']}", k=k, period=p, turn=t, prior=int(prior), y=int(fail(s["t"][t]))))
b, c, pv = mcnemar(pool12)
print(f"  合并（按会话配对）1N vs 2N: 2N错1N对 {b}, 反向 {c}, p={pv:.2g}")
b, c, pv = mcnemar(pool23)
print(f"  合并（按会话配对）2N vs 3N: 2N错3N对 {b}, 反向 {c}, p={pv:.2g}")

df = pd.DataFrame(rows_gee)
d23 = df[df.k.isin([2, 3])].copy()
d23["third"] = (d23.k == 3).astype(int)
m = smf.gee("y ~ third + C(batch)", groups="sess", data=d23, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
print(f"  GEE（会话聚类，批次固定效应）3N 相对 2N 的 log-OR = {m.params['third']:.2f}, 95%CI [{m.conf_int().loc['third',0]:.2f}, {m.conf_int().loc['third',1]:.2f}], p={m.pvalues['third']:.2g}")
m2 = smf.gee("y ~ third + prior + C(batch)", groups="sess", data=d23, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
print(f"  加入“此前是否出过错”后：3N log-OR = {m2.params['third']:.2f} (p={m2.pvalues['third']:.2g}); prior log-OR = {m2.params['prior']:.2f} (p={m2.pvalues['prior']:.2g})")
print("  3N 轮按此前是否出错分组：", d23[d23.k == 3].groupby("prior").y.agg(["sum", "count"]).to_dict("index"))

print("=" * 70)
print("B. 第 34 轮重跑：按历史配对（精确 McNemar），对照 = control_low")


def fork(name, d="fork1"):
    p = os.path.join(R, d, name + ".jsonl")
    if not os.path.exists(p):
        p = os.path.join(R, "fork2", name + ".jsonl")
    return {r["run"]: r["note_class"] == "已核" for r in map(json.loads, open(p, encoding="utf-8")) if "error" not in r}


ctrl = fork("control_low")
for name in ("effort_high", "effort_max", "reminder", "no_note_history", "reworded_rules", "reworded_30", "arithmetic_cue"):
    x = fork(name)
    pairs = [(ctrl[k], x[k]) for k in x if k in ctrl]
    b, c, pv = mcnemar(pairs)
    print(f"  {name:<16} n={len(pairs)} 对照错/干预对 {b:>2}, 对照对/干预错 {c:>2}, p={pv:.2g}")

print("=" * 70)
print("C. 5.4 干预：多重比较校正、效应量与功效")
base = load("p1/none")
bf, bn = sum(fail(s["t"][26]) for s in base), len(base)
names = [("calc26", "写算式·同轮"), ("calc25", "写算式·前1轮"), ("calc23", "写算式·前3轮"), ("calc10", "写算式·第10轮"), ("generic26", "逐条核对·同轮"), ("generic25", "逐条核对·前1轮")]
ps, out = [], []
for d, lab in names:
    S = load("p1/" + d)
    f, n = sum(fail(s["t"][26]) for s in S), len(S)
    pv = fisher(bf, bn - bf, f, n - f)
    dd, lo, hi = newcombe(bf, bn, f, n)
    ps.append(pv)
    out.append((lab, f, n, pv, dd, lo, hi))
adj = multipletests(ps, method="holm")[1]
for (lab, f, n, pv, dd, lo, hi), a in zip(out, adj):
    print(f"  {lab:<10} {f}/{n}  原始 p={pv:.3g}  Holm p={a:.3g}  错误率下降 {dd:+.0%} [95%CI {lo:+.0%}, {hi:+.0%}]")
calc = [load("p1/" + d) for d in ("calc26", "calc25", "calc23", "calc10")]
cf = sum(fail(s["t"][26]) for S in calc for s in S)
cn = sum(len(S) for S in calc)
print(f"  四个写算式组合并 {cf}/{cn} vs 不干预 {bf}/{bn}: p={fisher(bf, bn - bf, cf, cn - cf):.3g}")
# timing: same-turn vs turn-10
a, b_ = load("p1/calc26"), load("p1/calc10")
fa, fb = sum(fail(s["t"][26]) for s in a), sum(fail(s["t"][26]) for s in b_)
dd, lo, hi = newcombe(fa, 20, fb, 20)
print(f"  时机：同轮 {fa}/20 vs 第10轮 {fb}/20，差 {dd:+.0%} [95%CI {lo:+.0%}, {hi:+.0%}]")
rng = np.random.default_rng(0)
for delta in (0.15, 0.25, 0.35, 0.45):
    hits = 0
    for _ in range(2000):
        x = rng.binomial(30, 0.7)
        y = rng.binomial(20, 0.7 - delta)
        hits += fisher(x, 30 - x, y, 20 - y) < 0.05
    print(f"  功效（基线 70%, 30 vs 20, Fisher α=0.05）下降 {delta:.0%}: {hits/2000:.0%}")
for delta in (0.15, 0.25, 0.35):
    hits = 0
    for _ in range(2000):
        x = rng.binomial(20, 0.3)
        y = rng.binomial(20, 0.3 + delta)
        hits += fisher(x, 20 - x, y, 20 - y) < 0.05
    print(f"  功效（时机比较，基线 30%, 20 vs 20）相差 {delta:.0%}: {hits/2000:.0%}")

print("=" * 70)
print("D. 思考档位/模型：按批次分层（Mantel-Haenszel）与截断敏感性")


def cnt(d, trunc_as_fail=False):
    f = n = e = 0
    for s in load(d):
        r = s["t"].get(26)
        if r is None:
            continue
        if not r["reply"].strip():
            e += 1
            if trunc_as_fail:
                f += 1
                n += 1
            continue
        n += 1
        f += fail(r)
    return f, n, e


for cond, ds in (("high", ("p3/high", "p4/high")), ("max", ("p3/max", "p4/max")), ("pro", ("p3/pro", "p4/pro"))):
    for tf in (False, True):
        tabs = []
        for low_d, c_d in zip(("p3/low", "p4/low"), ds):
            lf, ln, _ = cnt(low_d, tf)
            xf, xn, _ = cnt(c_d, tf)
            tabs.append(np.array([[lf + 0.5, ln - lf + 0.5], [xf + 0.5, xn - xf + 0.5]]))
        st = StratifiedTable(tabs)
        raw = [(cnt(c_d, tf)) for c_d in ds]
        print(f"  low vs {cond:<4} {'截断计为失败' if tf else '剔除截断':<8} 各批 {[(r[0], r[1]) for r in raw]}  MH-OR={st.oddsratio_pooled:.2f} p={st.test_null_odds().pvalue:.3g}")

print("=" * 70)
print("E. 错误连锁：会话聚类 GEE（排除触发轮）")
rows = []
for d, p in (("combined", 17), ("A_p17_old", 17), ("B_p17_new", 17), ("C_p13", 13), ("D_p19", 19), ("E_p23", 23)):
    for s in load(d):
        last, seen = None, False
        for t in sorted(s["t"]):
            r = s["t"][t]
            err = int(bool(r["drift"] or r["slips"]))
            if t % p != 0:
                state = "none" if not seen else ("prev" if last == t - 1 else "earlier")
                rows.append(dict(batch=d, sess=f"{d}/{s['id']}", turn=t, state=state, y=err, odd=(d == "C_p13" and s["id"] == "run_03.jsonl")))
            if err:
                seen, last = True, t
E = pd.DataFrame(rows)
for lab, sub in (("排除异常会话", E[~E.odd]), ("包含异常会话", E)):
    print(f"  {lab}:", {k: f"{v['sum']}/{v['count']}" for k, v in sub.groupby("state").y.agg(["sum", "count"]).to_dict("index").items()})
    m = smf.gee("y ~ C(state, Treatment('none')) + turn + C(batch)", groups="sess", data=sub,
                family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    for k in m.params.index:
        if "state" in k:
            ci = m.conf_int().loc[k]
            print(f"     {k.split('T.')[-1].rstrip(']'):<8} OR={np.exp(m.params[k]):.2f} [{np.exp(ci[0]):.2f}, {np.exp(ci[1]):.2f}] p={m.pvalues[k]:.2g}")

print("=" * 70)
print("F. 思考文本中“写出算式”的来源与正则校验")
dirs = [("A_p17_old", 17), ("B_p17_new", 17), ("C_p13", 13), ("D_p19", 19), ("E_p23", 23)] + [(d, 13) for d in ("p1/none", "p2/s13_1", "p2/s13_2", "p2/fresh", "p2/t13", "p4/c13", "p4/low", "p4/high", "p3/low", "p3/high")] + [("p2/t19", 19), ("p2/t23", 23), ("p2/s23_1", 23), ("p2/s23_2", 23), ("p4/p14", 14)]
strict_tot = {True: [0, 0], False: [0, 0]}
loose_hits = []
nores = 0
for d, p in dirs:
    t = 2 * p
    strict = re.compile(rf"{p}\s*[*×xX·]\s*2|2\s*[*×xX·]\s*{p}|{t}\s*[=＝]\s*{p}|{t}\s*[/÷]\s*{p}|{p}\s*的\s*(2|两|二)\s*倍")
    loose = re.compile(rf"{p}\s*[*×xX·]|[*×xX·]\s*{p}|{t}\s*[/÷]|{p}\s*\+\s*{p}|(两|二|2)\s*倍|整除|余\s*0|mod")
    for s in load(d):
        r = s["t"].get(t)
        if not r or not r["reply"].strip():
            continue
        if not r.get("reasoning"):
            nores += 1
            continue
        f = fail(r)
        strict_tot[f][1] += 1
        strict_tot[f][0] += bool(strict.search(r["reasoning"]))
        if f and loose.search(r["reasoning"]):
            m_ = loose.search(r["reasoning"])
            loose_hits.append((d, s["id"], r["reasoning"][max(0, m_.start() - 40): m_.end() + 40].replace("\n", " ")))
print(f"  批次: {[d for d, _ in dirs]}（均为自然观察的完整会话，不含第 34 轮重跑样本）")
print(f"  严格正则：写错 {strict_tot[True][0]}/{strict_tot[True][1]}，写对 {strict_tot[False][0]}/{strict_tot[False][1]}；无思考内容被排除 {nores}")
# 报告 5.2 节的两项校验：写错样本中“提到规则”的次数，以及“2N 与 N 相距 15 个字符以内同时出现”的次数
mention = near_cnt = n_fail = 0
rule_words = re.compile(r"倍数|multiple|盘点|audit|审计")
for d, p in dirs:
    t = 2 * p
    near = re.compile(rf"{t}[^。\n]{{0,15}}{p}|{p}[^。\n]{{0,15}}{t}")
    for s in load(d):
        r = s["t"].get(t)
        if not r or not r["reply"].strip() or not r.get("reasoning") or not fail(r):
            continue
        n_fail += 1
        mention += bool(rule_words.search(r["reasoning"]))
        near_cnt += bool(near.search(r["reasoning"]))
print(f"  写错 {n_fail} 次中：思考提到规则（匹配 倍数/multiple/盘点/audit/审计）{mention} 次；2N 与 N 相距 15 字符内同时出现 {near_cnt} 次")
print(f"  宽松正则在写错样本中命中 {len(loose_hits)} 次，节选：")
for h in loose_hits[:12]:
    print("   ", h)

print("=" * 70)
print("G. 17 的倍数第二次触发：两个批次")
for d in ("combined", "A_p17_old"):
    S = load(d)
    print(f"  {d}: {sum(fail(s['t'][34]) for s in S)}/{len(S)}")
