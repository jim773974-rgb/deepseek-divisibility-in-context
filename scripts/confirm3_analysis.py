"""Analysis of the third preregistered experiment, as specified in PREREG3.md.

Run from a directory holding the scripts and unpacked raw logs (conf/ and conf3/ folders).
"""
import json
import os
import re
from math import comb, sqrt

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

R = os.path.dirname(os.path.abspath(__file__))


def fisher(a, b, c, d):
    """Two-sided Fisher exact test (same as confirm_analysis.py)."""
    n, r, k = a + b + c + d, a + b, a + c
    pm = lambda x: comb(k, x) * comb(n - k, r - x) / comb(n, r)
    p0 = pm(a)
    return sum(pm(x) for x in range(max(0, r - (n - k)), min(r, k) + 1) if pm(x) <= p0 * (1 + 1e-9))


def note(r):
    try:
        n = json.loads(r["reply"].strip()).get("note")
        return n if isinstance(n, str) else ""
    except (ValueError, AttributeError):
        return ""


def load(path):
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    err = sum("error" in r for r in rs)
    ok = [r for r in rs if "reply" in r and r["reply"].strip()]
    empty = sum(1 for r in rs if "reply" in r and not r["reply"].strip())
    return ok, err, empty


def newcombe(k1, n1, k2, n2, z=1.96):
    def wil(k, n):
        p = k / n; den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    p1, p2 = k1 / n1, k2 / n2; l1, u1 = wil(k1, n1); l2, u2 = wil(k2, n2)
    return p1 - p2, p1 - p2 - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), p1 - p2 + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


T = {}
print("== 目标轮")
for arm, (p, t) in {"C0_13": (13, 26), "L_13": (13, 26), "F_13": (13, 26), "R_13": (13, 26),
                    "C0_9": (9, 36), "L_9": (9, 36), "F_9": (9, 36), "R_9": (9, 36)}.items():
    ok, err, empty = load(os.path.join(R, "conf3", arm, "calls.jsonl"))
    f = sum("mod17_forgotten" in r["drift"] for r in ok)
    k = t // p
    calc = re.compile(rf"{p}\s*[*×xX·]\s*{k}|{k}\s*[*×xX·]\s*{p}|{t}\s*[=＝]\s*{p}|{t}\s*[/÷]\s*{p}|{p}\s*的\s*{k}\s*倍")
    wrote = sum(bool(calc.search(r["reasoning"])) for r in ok)
    lo, hi = proportion_confint(f, len(ok), method="wilson")
    T[arm] = (f, len(ok), f + empty, len(ok) + empty)
    print(f"  {arm}: 错误 {f}/{len(ok)}（Wilson 95% {lo:.1%}–{hi:.1%}），空回复 {empty}（计为错误 {f+empty}/{len(ok)+empty}），API 报错 {err}；思考中写出算式 {wrote}/{len(ok)}")


def test(a, b, idx=0):
    x, y = T[a], T[b]
    if idx:
        x, y = (x[2], x[3]), (y[2], y[3])
    return fisher(x[0], x[1] - x[0], y[0], y[1] - y[0])


for label, idx in (("主分析", 0), ("敏感性（空回复计为错误）", 2)):
    print(f"== 主要假设，{label}")
    hs = [("H5 L_13 vs C0_13", "L_13", "C0_13"), ("H6 L_9 vs C0_9", "L_9", "C0_9"),
          ("H7 F_13 vs C0_13", "F_13", "C0_13"), ("H8 F_9 vs C0_9", "F_9", "C0_9")]
    ps = [test(a, b, idx) for _, a, b in hs]
    adj = multipletests(ps, method="holm")[1]
    for (name, a, b), p_, q in zip(hs, ps, adj):
        lower = T[a][idx] / T[a][idx + 1] < T[b][idx] / T[b][idx + 1]
        print(f"  {name}: {T[a][idx]}/{T[a][idx+1]} vs {T[b][idx]}/{T[b][idx+1]}，p={p_:.3g}，Holm {q:.3g}，"
              f"{'获得支持' if q < 0.05 and lower else '未获支持'}")
    print(f"== 次要分析，{label}")
    hs = [("R_13 vs C0_13", "R_13", "C0_13"), ("R_9 vs C0_9", "R_9", "C0_9")]
    ps = [test(a, b, idx) for _, a, b in hs]
    adj = multipletests(ps, method="holm")[1]
    for (name, a, b), p_, q in zip(hs, ps, adj):
        print(f"  {name}: {T[a][idx]}/{T[a][idx+1]} vs {T[b][idx]}/{T[b][idx+1]}，p={p_:.3g}，Holm {q:.3g}")

print("== 效应量（干预组 − 对照组）")
for a, b in (("L_13", "C0_13"), ("F_13", "C0_13"), ("R_13", "C0_13"), ("L_9", "C0_9"), ("F_9", "C0_9"), ("R_9", "C0_9")):
    d, lo, hi = newcombe(T[a][0], T[a][1], T[b][0], T[b][1])
    print(f"  {a} − {b}: {d*100:+.1f} 个百分点，Newcombe 95% CI {lo*100:+.1f} 至 {hi*100:+.1f}")

print("== 对照组与前两次同输入结果比较（时间漂移）")
for c, olds in (("C0_13", ("R6", "A6")), ("C0_9", ("R4", "A4"))):
    for o in olds:
        ok, _, _ = load(os.path.join(R, "conf", o, "calls.jsonl"))
        f = sum("mod17_forgotten" in r["drift"] for r in ok)
        print(f"  {c} {T[c][0]}/{T[c][1]} vs {o} {f}/{len(ok)}，p={fisher(T[c][0], T[c][1]-T[c][0], f, len(ok)-f):.3g}")

print("== 阴性对照（非目标轮误写“盘点”）")
for arm in ("NL_13", "NR_13", "NL_9", "NR_9"):
    ok, err, empty = load(os.path.join(R, "conf3", arm, "calls.jsonl"))
    fp = sum(note(r).startswith("盘点") for r in ok)
    lo, hi = proportion_confint(fp, len(ok), method="wilson")
    print(f"  {arm}: 误写 {fp}/{len(ok)}（Wilson 95% {lo:.1%}–{hi:.1%}），空回复 {empty}，API 报错 {err}")
