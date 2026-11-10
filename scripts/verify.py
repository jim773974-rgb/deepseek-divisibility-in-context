"""Recompute every number used in the report directly from raw logs. Independent of earlier scripts."""
import glob
import json
import os
import re
from math import comb, sqrt

R = os.path.dirname(os.path.abspath(__file__))


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(R, d, "run_*.jsonl"))):
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        out.append({"id": os.path.basename(f), "turns": {r["turn"]: r for r in recs if "reply" in r},
                    "stopped": [r for r in recs if "stopped" in r]})
    return out


def fisher(a, b, c, d):
    n, r, k = a + b + c + d, a + b, a + c
    pm = lambda x: comb(k, x) * comb(n - k, r - x) / comb(n, r)
    p0 = pm(a)
    return sum(pm(x) for x in range(max(0, r - (n - k)), min(r, k) + 1) if pm(x) <= p0 * (1 + 1e-9))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def periodic_fail(r, kind="ledger"):
    if kind == "code":
        return any(k.startswith("audit") for k in r["drift"])
    return "mod17_forgotten" in r["drift"]


def trig(d, t, kind="ledger", valid_only=True):
    S = load(d)
    rows = [s["turns"][t] for s in S if t in s["turns"]]
    if valid_only:
        rows = [r for r in rows if r["reply"].strip()]
    f = sum(periodic_fail(r, kind) for r in rows)
    clean_n = clean_f = 0
    for s in S:
        if t not in s["turns"] or not s["turns"][t]["reply"].strip():
            continue
        prior = any(s["turns"][x]["drift"] or s["turns"][x]["slips"] for x in s["turns"] if x < t)
        if not prior:
            clean_n += 1
            clean_f += periodic_fail(s["turns"][t], kind)
    return f, len(rows), clean_f, clean_n, len(S), sum(1 for s in S if s["stopped"])


def show(label, d, t, kind="ledger"):
    f, n, cf, cn, ns, stopped = trig(d, t, kind)
    lo, hi = wilson(f, n)
    print(f"{label:<34} t{t:<3} {f:>3}/{n:<3} ({f/n:5.1%}, CI {lo:.0%}-{hi:.0%}) clean {cf}/{cn}  sessions {ns} stopped {stopped}")
    return f, n


print("== 1. 跨数字：第一次 vs 第二次触发（第一批 drift_v3）")
P = {}
for d, p in (("C_p13", 13), ("A_p17_old", 17), ("D_p19", 19), ("E_p23", 23)):
    P[p] = (show(f"period {p} first", d, p), show(f"period {p} second", d, 2 * p))
print("== 17 原始 60 轮 30 次（combined）")
show("p17 combined", "combined", 17); show("p17 combined", "combined", 34); show("p17 combined", "combined", 51)

print("\n== 2. 措辞 A/B（同时运行，完整会话）")
a = show("old wording", "A_p17_old", 34); b = show("new wording", "B_p17_new", 34)
print("   Fisher p =", round(fisher(a[0], a[1] - a[0], b[0], b[1] - b[0]), 3))

print("\n== 3. 干预与计算需求（p1，同批运行）")
base = show("none", "p1/none", 26)
for lab, d in (("calc same turn", "p1/calc26"), ("calc before1", "p1/calc25"), ("calc before3", "p1/calc23"), ("calc turn10", "p1/calc10"),
               ("generic same", "p1/generic26"), ("generic before1", "p1/generic25"), ("active from 20", "p1/active20"), ("explicit list", "p1/list")):
    x = show(lab, d, 26)
    print(f"   vs none p = {fisher(base[0], base[1]-base[0], x[0], x[1]-x[0]):.3g}")

print("\n== 4. 上下文")
x = show("fresh each turn", "p2/fresh", 26)
print(f"   vs none p = {fisher(base[0], base[1]-base[0], x[0], x[1]-x[0]):.3g}")

print("\n== 5. 多脚本")
for d, p in (("p2/s13_1", 13), ("p2/s13_2", 13), ("p2/s23_1", 23), ("p2/s23_2", 23)):
    show(d, d, 2 * p)

print("\n== 6. 第三次触发")
sec = thr = [0, 0]
sec, thr = [0, 0], [0, 0]
for d, p in (("p2/t13", 13), ("p2/t19", 19), ("p2/t23", 23), ("p4/c13", 13), ("combined", 17)):
    s = show(d + " 2nd", d, 2 * p); t3 = show(d + " 3rd", d, 3 * p)
    sec = [sec[0] + s[0], sec[1] + s[1]]; thr = [thr[0] + t3[0], thr[1] + t3[1]]
print(f"   pooled 2nd {sec[0]}/{sec[1]} vs 3rd {thr[0]}/{thr[1]} p = {fisher(sec[0], sec[1]-sec[0], thr[0], thr[1]-thr[0]):.3g}")
c39 = show("t39 control (p4/c13)", "p4/c13", 39); a39 = show("t39 first-active (p4/a39)", "p4/a39", 39)
print(f"   p = {fisher(c39[0], c39[1]-c39[0], a39[0], a39[1]-a39[0]):.3g}")
S = load("p4/c13")
fails39 = [s for s in S if 39 in s["turns"] and periodic_fail(s["turns"][39])]
print("   t39 failures in sessions that had an earlier error:", sum(any(s["turns"][x]["drift"] or s["turns"][x]["slips"] for x in s["turns"] if x < 39) for s in fails39), "/", len(fails39))

print("\n== 7. 奇偶")
p13 = show("p13", "p4/c13", 26); p14 = show("p14", "p4/p14", 28); show("p14", "p4/p14", 42)
print(f"   p = {fisher(p13[0], p13[1]-p13[0], p14[0], p14[1]-p14[0]):.3g}")

print("\n== 8. 思考档位与模型（第 26 轮，两批合并，去掉空回复）")
pool = {}
for lab, ds in (("low", ["p3/low", "p4/low"]), ("high", ["p3/high", "p4/high"]), ("max", ["p3/max", "p4/max"]), ("pro", ["p3/pro", "p4/pro"])):
    f = n = empty = 0
    for d in ds:
        for s in load(d):
            if 26 in s["turns"]:
                r = s["turns"][26]
                if not r["reply"].strip():
                    empty += 1
                    continue
                n += 1; f += periodic_fail(r)
    pool[lab] = (f, n)
    print(f"   {lab}: {f}/{n} empty {empty}")
for k in ("high", "max", "pro"):
    print(f"   low vs {k} p = {fisher(pool['low'][0], pool['low'][1]-pool['low'][0], pool[k][0], pool[k][1]-pool[k][0]):.3g}")

print("\n== 9. 代码任务")
for t in (13, 26, 39):
    show("code", "code1", t, "code")
S = load("code1")
seg = {}
for s in S:
    for t, r in s["turns"].items():
        if t % 13 == 0:
            continue
        k = (t - 1) // 10
        seg.setdefault(k, [0, 0]); seg[k][1] += 1; seg[k][0] += bool(r["drift"])
print("   non-audit error rate by 10-turn block:", {f"{k*10+1}-{k*10+10}": f"{v[0]}/{v[1]}={v[0]/v[1]:.1%}" for k, v in sorted(seg.items())})

print("\n== 10. 思考中是否写出算式（第二次触发，所有保存了思考的完整会话批次）")
tot = {"fail": [0, 0], "ok": [0, 0]}
dirs = [("A_p17_old", 17), ("B_p17_new", 17), ("C_p13", 13), ("D_p19", 19), ("E_p23", 23)] + [(d, 13) for d in ("p1/none", "p2/s13_1", "p2/s13_2", "p2/fresh", "p2/t13", "p4/c13", "p4/low", "p4/high", "p3/low", "p3/high")] + [("p2/t19", 19), ("p2/t23", 23), ("p2/s23_1", 23), ("p2/s23_2", 23), ("p4/p14", 14)]
for d, p in dirs:
    t = 2 * p
    calc = re.compile(rf"{p}\s*[*×xX·]\s*2|2\s*[*×xX·]\s*{p}|{t}\s*[=＝]\s*{p}|{t}\s*[/÷]\s*{p}|{p}\s*的\s*(2|两|二)\s*倍")
    for s in load(d):
        r = s["turns"].get(t)
        if not r or not r["reply"].strip() or not r.get("reasoning"):
            continue
        k = "fail" if periodic_fail(r) else "ok"
        tot[k][1] += 1; tot[k][0] += bool(calc.search(r["reasoning"]))
print(f"   wrong: wrote calc {tot['fail'][0]}/{tot['fail'][1]};  right: wrote calc {tot['ok'][0]}/{tot['ok'][1]}")

print("\n== 11. 错误连锁（排除触发轮，记账任务完整会话批次）")
a = [0] * 6
for d, p in [("combined", 17), ("A_p17_old", 17), ("B_p17_new", 17), ("C_p13", 13), ("D_p19", 19), ("E_p23", 23)]:
    for s in load(d):
        if d == "C_p13" and s["id"] == "run_03.jsonl":
            continue
        last, seen = None, False
        for t in sorted(s["turns"]):
            r = s["turns"][t]
            err = bool(r["drift"] or r["slips"])
            if t % p != 0:
                i = 0 if not seen else (2 if last == t - 1 else 4)
                a[i] += 1; a[i + 1] += err
            if err:
                seen, last = True, t
print(f"   before any error {a[1]}/{a[0]}={a[1]/a[0]:.2%}; next turn after error {a[3]}/{a[2]}={a[3]/a[2]:.2%}; later {a[5]}/{a[4]}={a[5]/a[4]:.2%}")

print("\n== 12. 第 34 轮重跑")
for d in ("fork1", "fork2"):
    for fn in sorted(os.listdir(os.path.join(R, d))):
        rs = [json.loads(l) for l in open(os.path.join(R, d, fn), encoding="utf-8")]
        ok = [r for r in rs if "error" not in r]
        if fn.startswith("probe"):
            from collections import Counter
            print(f"   {fn[:-6]}: answers {dict(Counter(r['reply'].strip()[:6] for r in ok))}")
        else:
            print(f"   {fn[:-6]}: wrong {sum(r['note_class']=='已核' for r in ok)} right {sum(r['note_class']=='盘点' for r in ok)} other {sum(r['note_class'] not in ('已核','盘点') for r in ok)} of {len(ok)}")

print("\n== 13. 第一版简单任务（pilot1）")
S = load("pilot1")
print("   sessions", len(S), "turns", sum(len(s["turns"]) for s in S), "rule drift turns", sum(1 for s in S for r in s["turns"].values() if r["drift"]), "slip turns", sum(1 for s in S for r in s["turns"].values() if r["slips"]))
