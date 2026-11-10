"""Analysis of the preregistered confirmatory experiment, exactly as specified in PREREG.md.

Run from a directory holding the scripts and unpacked raw logs (conf/ folder).
"""
import glob
import json
import os
import re
from math import comb

from statsmodels.stats.multitest import multipletests

R = os.path.dirname(os.path.abspath(__file__))


def fisher(a, b, c, d):
    n, r, k = a + b + c + d, a + b, a + c
    pm = lambda x: comb(k, x) * comb(n - k, r - x) / comb(n, r)
    p0 = pm(a)
    return sum(pm(x) for x in range(max(0, r - (n - k)), min(r, k) + 1) if pm(x) <= p0 * (1 + 1e-9))


ARMS = {"A1": (36, 36), "A2": (18, 36), "A3": (12, 36), "A4": (9, 36), "A5": (26, 26), "A6": (13, 26)}
A = {}
print("== 实验 A")
for d, (p, t) in ARMS.items():
    rs = [json.loads(l) for l in open(os.path.join(R, "conf", d, "calls.jsonl"), encoding="utf-8")]
    api_err = sum("error" in r for r in rs)
    empty = [r for r in rs if "reply" in r and not r["reply"].strip()]
    ok = [r for r in rs if "reply" in r and r["reply"].strip()]
    f = [r for r in ok if "mod17_forgotten" in r["drift"]]
    g = [r for r in ok if "mod17_forgotten" not in r["drift"]]
    k = t // p
    calc = re.compile(rf"{p}\s*[*×xX·]\s*{k}|{k}\s*[*×xX·]\s*{p}|{t}\s*[=＝]\s*{p}|{t}\s*[/÷]\s*{p}|{p}\s*的\s*{k}\s*倍")
    neg = re.compile(rf"(不是|并非|不为|not)[^。\n,，]{{0,6}}({p}|倍数|multiple)|{t}[^。\n]{{0,6}}(不是|not)")
    A[d] = (len(f), len(ok), len(f) + len(empty), len(ok) + len(empty))
    print(f"  {d} N={p} 第{t}轮 k={k}: 错误 {len(f)}/{len(ok)}（截断计为错误 {len(f)+len(empty)}/{len(ok)+len(empty)}），API 报错 {api_err}；"
          f"写错时写出算式 {sum(bool(calc.search(r['reasoning'])) for r in f)}/{len(f)}，直接否定 {sum(bool(neg.search(r['reasoning'])) for r in f)}/{len(f)}；"
          f"写对时写出算式 {sum(bool(calc.search(r['reasoning'])) for r in g)}/{len(g)}")

h1 = fisher(A["A2"][0], A["A2"][1] - A["A2"][0], A["A1"][0], A["A1"][1] - A["A1"][0])
h2 = fisher(A["A3"][0], A["A3"][1] - A["A3"][0], A["A2"][0], A["A2"][1] - A["A2"][0])
adj = multipletests([h1, h2], method="holm")[1]
print(f"  H1（A2 vs A1）p={h1:.3g}，Holm {adj[0]:.3g}；H2（A3 vs A2）p={h2:.3g}，Holm {adj[1]:.3g}")
print(f"  次要：A4 vs A2 p={fisher(A['A4'][0], A['A4'][1]-A['A4'][0], A['A2'][0], A['A2'][1]-A['A2'][0]):.3g}；"
      f"A6 vs A5 p={fisher(A['A6'][0], A['A6'][1]-A['A6'][0], A['A5'][0], A['A5'][1]-A['A5'][0]):.3g}")
k1 = (A["A1"][0] + A["A5"][0], A["A1"][1] + A["A5"][1])
k2 = (sum(A[x][0] for x in ("A2", "A3", "A4", "A6")), sum(A[x][1] for x in ("A2", "A3", "A4", "A6")))
print(f"  k=1 合计 {k1[0]}/{k1[1]}，k≥2 合计 {k2[0]}/{k2[1]}，p={fisher(k1[0], k1[1]-k1[0], k2[0], k2[1]-k2[0]):.3g}")

print("== 实验 B")
ps = []
def load_b(names):
    out = []
    for b in names:
        for f in sorted(glob.glob(os.path.join(R, "conf", b, "run_*.jsonl"))):
            out.append({r["turn"]: r for r in map(json.loads, open(f, encoding="utf-8")) if "reply" in r})
    return out


for label, names in (("第一轮（未完成）", ["{}"]), ("补跑", ["{}b"]), ("合并", ["{}", "{}b"])):
  print(f" -- 实验 B {label}")
  ps = []
  for b, a, p in (("B1", "A1", 36), ("B2", "A2", 18), ("B3", "A3", 12)):
    S = load_b([n.format(b) for n in names])
    rows = [s[36] for s in S if 36 in s and s[36]["reply"].strip()]
    f = sum("mod17_forgotten" in r["drift"] for r in rows)
    clean = [s for s in S if 36 in s and not any(s[x]["drift"] or s[x]["slips"] for x in s if x < 36)]
    cf = sum("mod17_forgotten" in s[36]["drift"] for s in clean)
    pv = fisher(f, len(rows) - f, A[a][0], A[a][1] - A[a][0])
    ps.append(pv)
    stops = sorted(max(s) for s in S if 36 not in s)
    earlier = "；".join(f"第{t}轮 {sum('mod17_forgotten' in s[t]['drift'] for s in S if t in s)}/{sum(1 for s in S if t in s)}" for t in range(p, 36, p))
    print(f"  {b} N={p}: 会话 {len(S)}，到达第36轮 {len(rows)}，第36轮错误 {f}/{len(rows)}，干净观察 {cf}/{len(clean)}，"
          f"对应 {a} {A[a][0]}/{A[a][1]}，原始 p={pv:.3g}；未到第36轮的会话停在 {stops}" + (f"；{earlier}" if earlier else ""))
  print("  Holm 校正 p:", [round(float(x), 4) for x in multipletests(ps, method="holm")[1]])

print("== 第二次预登记（PREREG2.md）")
D = {}
for d in ("D1", "D2", "D3"):
    fp = os.path.join(R, "conf", d, "calls.jsonl")
    if not os.path.exists(fp):
        print("  尚未运行", d); continue
    rs = [json.loads(l) for l in open(fp, encoding="utf-8")]
    from collections import Counter
    c = Counter(r.get("verdict", "API报错") for r in rs)
    D[d] = (c["否"], c["是"] + c["否"])
    print(f"  {d} {json.load(open(os.path.join(R, 'conf', d, 'meta.json'), encoding="utf-8"))['direct']}  {dict(c)}  错误（答否）{c['否']}/{c['是'] + c['否']}")
RR = {}
for d in ("R4", "R6", "NC1", "NC2"):
    fp = os.path.join(R, "conf", d, "calls.jsonl")
    if not os.path.exists(fp):
        print("  尚未运行", d); continue
    rs = [json.loads(l) for l in open(fp, encoding="utf-8")]
    ok = [r for r in rs if "reply" in r and r["reply"].strip()]
    def note(r):
        try:
            n = json.loads(r["reply"].strip()).get("note")
            return n if isinstance(n, str) else ""
        except (ValueError, AttributeError):
            return ""
    if d.startswith("NC"):
        fpos = sum(note(r).startswith("盘点") for r in ok)
        RR[d] = (fpos, len(ok))
        from statsmodels.stats.proportion import proportion_confint
        lo, hi = proportion_confint(fpos, len(ok), method="wilson")
        print(f"  {d} 阴性对照：误写“盘点” {fpos}/{len(ok)}（Wilson 95% {lo:.1%}–{hi:.1%}），API 报错 {sum('error' in r for r in rs)}，空回复 {len(rs) - len(ok) - sum('error' in r for r in rs)}")
    else:
        f = sum("mod17_forgotten" in r["drift"] for r in ok)
        RR[d] = (f, len(ok))
        print(f"  {d} 任务中重跑：错误 {f}/{len(ok)}，API 报错 {sum('error' in r for r in rs)}")
if all(k in D for k in ("D1", "D2")) and all(k in RR for k in ("R4", "R6")):
    h3 = fisher(D["D1"][0], D["D1"][1] - D["D1"][0], RR["R6"][0], RR["R6"][1] - RR["R6"][0])
    h4 = fisher(D["D2"][0], D["D2"][1] - D["D2"][0], RR["R4"][0], RR["R4"][1] - RR["R4"][0])
    adj = multipletests([h3, h4], method="holm")[1]
    print(f"  H3（D1 vs R6）p={h3:.3g}，Holm {adj[0]:.3g}；H4（D2 vs R4）p={h4:.3g}，Holm {adj[1]:.3g}")
    def newcombe(k1, n1, k2, n2, z=1.96):
        from math import sqrt
        def wil(k, n):
            p = k / n; den = 1 + z * z / n
            c = (p + z * z / (2 * n)) / den; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            return c - h, c + h
        p1, p2 = k1 / n1, k2 / n2; l1, u1 = wil(k1, n1); l2, u2 = wil(k2, n2)
        return p1 - p2, p1 - p2 - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), p1 - p2 + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    for lab, a_, b_ in (("任务−直接问 26/13", RR["R6"], D["D1"]), ("任务−直接问 36/9", RR["R4"], D["D2"]),
                        ("R6−A6（跨时间）", RR["R6"], A["A6"][:2]), ("R4−A4（跨时间）", RR["R4"], A["A4"][:2])):
        d_, lo, hi = newcombe(a_[0], a_[1], b_[0], b_[1])
        print(f"  效应量 {lab}: {d_*100:+.1f} 个百分点，Newcombe 95% CI {lo*100:+.1f} 至 {hi*100:+.1f}")
    print(f"  跨时间参考：R6 {RR['R6'][0]}/{RR['R6'][1]} vs A6 {A['A6'][0]}/{A['A6'][1]} p={fisher(RR['R6'][0], RR['R6'][1]-RR['R6'][0], A['A6'][0], A['A6'][1]-A['A6'][0]):.3g}；"
          f"R4 {RR['R4'][0]}/{RR['R4'][1]} vs A4 {A['A4'][0]}/{A['A4'][1]} p={fisher(RR['R4'][0], RR['R4'][1]-RR['R4'][0], A['A4'][0], A['A4'][1]-A['A4'][0]):.3g}")

print("== 审查建议的补充分析")
tot = {True: [0, 0], False: [0, 0]}
for d, (p, t) in ARMS.items():
    k = t // p
    if k < 2:
        continue
    calc = re.compile(rf"{p}\s*[*×xX·]\s*{k}|{k}\s*[*×xX·]\s*{p}|{t}\s*[=＝]\s*{p}|{t}\s*[/÷]\s*{p}|{p}\s*的\s*{k}\s*倍")
    for r in map(json.loads, open(os.path.join(R, "conf", d, "calls.jsonl"), encoding="utf-8")):
        c = bool(calc.search(r["reasoning"]))
        tot[c][1] += 1
        tot[c][0] += "mod17_forgotten" in r["drift"]
from statsmodels.stats.proportion import proportion_confint
lo, hi = proportion_confint(0, tot[True][1], method="wilson")
print(f"  实验 A，k≥2 四组合并：写出算式 {tot[True][0]}/{tot[True][1]} 错误（95% 上限 {hi:.1%}）；未写算式 {tot[False][0]}/{tot[False][1]} 错误")
