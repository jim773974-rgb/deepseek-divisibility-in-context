"""Analysis of the fourth preregistered experiment, as specified in PREREG4.md.

Run from a directory holding the scripts and unpacked raw logs (conf4/ folder).
"""
import json
import os
from math import comb, sqrt

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

R = os.path.dirname(os.path.abspath(__file__))
ARMS = ["S0", "F26", "F22", "F14", "L"]
INSERT = {"F22": 22, "F14": 14}


def fisher(a, b, c, d):
    n, r, k = a + b + c + d, a + b, a + c
    pm = lambda x: comb(k, x) * comb(n - k, r - x) / comb(n, r)
    p0 = pm(a)
    return sum(pm(x) for x in range(max(0, r - (n - k)), min(r, k) + 1) if pm(x) <= p0 * (1 + 1e-9))


def newcombe(k1, n1, k2, n2, z=1.96):
    def wil(k, n):
        p = k / n; den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    p1, p2 = k1 / n1, k2 / n2; l1, u1 = wil(k1, n1); l2, u2 = wil(k2, n2)
    return p1 - p2, p1 - p2 - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), p1 - p2 + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def note(r):
    try:
        n = json.loads(r["reply"].strip()).get("note")
        return n if isinstance(n, str) else ""
    except (ValueError, AttributeError):
        return ""


S = {}
for arm in ARMS:
    sess, stops = [], []
    d = os.path.join(R, "conf4", arm)
    for fn in sorted(os.listdir(d)):
        recs = [json.loads(l) for l in open(os.path.join(d, fn), encoding="utf-8")]
        stops += [r for r in recs if "stopped" in r]
        sess.append({r["turn"]: r for r in recs if "reply" in r})
    S[arm] = (sess, stops)

T, C = {}, {}
print("== 各组概况")
for arm in ARMS:
    sess, stops = S[arm]
    reach = [s for s in sess if 26 in s]
    ok = [s for s in reach if s[26]["reply"].strip()]
    empty = len(reach) - len(ok)
    f = sum("mod17_forgotten" in s[26]["drift"] for s in ok)
    clean = [s for s in ok if not any(s[t]["drift"] or s[t]["slips"] for t in s if t < 26)]
    cf = sum("mod17_forgotten" in s[26]["drift"] for s in clean)
    t13 = [s for s in sess if 13 in s and s[13]["reply"].strip()]
    f13 = sum("mod17_forgotten" in s[13]["drift"] for s in t13)
    lo, hi = proportion_confint(f, len(ok), method="wilson")
    T[arm] = (f, len(ok), f + empty, len(ok) + empty); C[arm] = (cf, len(clean))
    print(f"  {arm}: 会话 {len(sess)}，到达第26轮 {len(reach)}（空回复 {empty}），中途停止 {len(stops)} {[ (x['turn'], x['stopped'][:30]) for x in stops]}")
    print(f"       第26轮错误 {f}/{len(ok)}（Wilson 95% {lo:.1%}–{hi:.1%}）；干净观察 {cf}/{len(clean)}；第13轮错误 {f13}/{len(t13)}")

for label, idx in (("主分析", 0), ("敏感性（空回复计为错误）", 2)):
    print(f"== 主要假设，{label}")
    hs = [("H9 F22 vs S0", "F22"), ("H10 F14 vs S0", "F14")]
    ps = [fisher(T[a][idx], T[a][idx + 1] - T[a][idx], T["S0"][idx], T["S0"][idx + 1] - T["S0"][idx]) for _, a in hs]
    for (name, a), p_, q in zip(hs, ps, multipletests(ps, method="holm")[1]):
        lower = T[a][idx] / T[a][idx + 1] < T["S0"][idx] / T["S0"][idx + 1]
        print(f"  {name}: {T[a][idx]}/{T[a][idx+1]} vs {T['S0'][idx]}/{T['S0'][idx+1]}，p={p_:.3g}，Holm {q:.3g}，"
              f"{'获得支持' if q < 0.05 and lower else '未获支持'}")
    print(f"== 次要比较，{label}")
    hs = [("F26 vs S0", "F26"), ("L vs S0", "L")]
    ps = [fisher(T[a][idx], T[a][idx + 1] - T[a][idx], T["S0"][idx], T["S0"][idx + 1] - T["S0"][idx]) for _, a in hs]
    for (name, a), p_, q in zip(hs, ps, multipletests(ps, method="holm")[1]):
        print(f"  {name}: {T[a][idx]}/{T[a][idx+1]} vs {T['S0'][idx]}/{T['S0'][idx+1]}，p={p_:.3g}，Holm {q:.3g}")

print("== 描述性比较（差值，Newcombe 95% CI）")
for a, b in (("F22", "F26"), ("F14", "F26"), ("F22", "S0"), ("F14", "S0"), ("F26", "S0"), ("L", "S0")):
    d, lo, hi = newcombe(T[a][0], T[a][1], T[b][0], T[b][1])
    print(f"  {a} − {b}: {d*100:+.1f} 个百分点（{lo*100:+.1f} 至 {hi*100:+.1f}）")

print("== 干净观察（次要）")
for arm in ARMS:
    print(f"  {arm}: {C[arm][0]}/{C[arm][1]}")

print("== 安全性：非目标轮误写“盘点”")
for arm in ARMS:
    sess, _ = S[arm]
    start = INSERT.get(arm, 14)
    n = fp = 0
    for s in sess:
        for t in range(start, 26):
            if t in s and t != 13 and s[t]["reply"].strip():
                n += 1; fp += note(s[t]).startswith("盘点")
    at = ""
    if arm in INSERT:
        k = INSERT[arm]; rows = [s[k] for s in sess if k in s and s[k]["reply"].strip()]
        at = f"；插入那一轮（第{k}轮）{sum(note(r).startswith('盘点') for r in rows)}/{len(rows)}"
    print(f"  {arm}: 第{start}–25轮误写 {fp}/{n}{at}")
