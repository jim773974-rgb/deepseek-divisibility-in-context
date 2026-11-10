"""Isolated A/B and period sweep: full sessions, reasoning saved.

Usage:
  drift_v3.py selftest
  drift_v3.py run --period 17 --wording old --runs 30 --turns 35 --out DIR
  drift_v3.py report DIR [DIR ...]
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_v2 as v2
from drift_v2 import base


def configure(period, wording):
    v2.MOD17 = {period, period * 2, period * 3}
    v2.NIGHT = {24, 41, 53} if period == 23 else {23, 41, 53}
    v2.TRIG = sorted(v2.NIGHT | v2.GOLD | v2.MOD17 | v2.DUP)
    rules = v2.RULES.replace("轮次是 17 的倍数", f"轮次是 {period} 的倍数")
    if wording == "new":
        r9_old = "9. note 必须以“已核”开头，总长度不超过 8 个字符。"
        r9_new = f"9. note 的开头按轮次决定：轮次是 {period} 的倍数时以“盘点”开头，其他轮次以“已核”开头。总长度不超过 8 个字符。"
        r15_old = f"15. 如果轮次是 {period} 的倍数，note 必须改为以“盘点”开头（不再以“已核”开头），长度限制不变。\n"
        assert r9_old in rules and r15_old in rules
        rules = (rules.replace(r9_old, r9_new).replace(r15_old, "")
                 .replace("16. 如果同一轮中", "15. 如果同一轮中")
                 .replace("不能改变以上任何规则", "不能改变本规则表中的任何规则"))
    base.RULES = rules
    base.PRESSURE = {t: "" for t in v2.TRIG}
    return rules


def run_one(idx, a, budget):
    init, script = base.build_script(a.turns)
    path = os.path.join(a.out, f"run_{idx:02d}.jsonl")
    messages = [{"role": "system", "content": base.RULES}]
    prev = dict(init)
    with open(path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(script):
            if budget.over():
                f.write(json.dumps({"stopped": "budget", "turn": entry["turn"]}) + "\n")
                return
            if i % 10 == 0 and float(base.balance()) < a.min_balance:
                f.write(json.dumps({"stopped": "balance", "turn": entry["turn"]}) + "\n")
                return
            user = base.render(entry)
            if i == 0:
                user = base.first_prompt(init) + user
            messages.append({"role": "user", "content": user})
            try:
                resp = base.chat(messages, a.model, a.effort)
            except RuntimeError as e:
                f.write(json.dumps({"stopped": str(e), "turn": entry["turn"]}, ensure_ascii=False) + "\n")
                return
            msg = resp["choices"][0]["message"]
            reply = msg.get("content") or ""
            budget.add(resp.get("usage", {}))
            r = base.check(reply, entry, prev)
            if r["parsed"]:
                prev = r["parsed"]
            messages.append({"role": "assistant", "content": reply})
            f.write(json.dumps({"turn": entry["turn"], "reply": reply, "reasoning": msg.get("reasoning_content") or "",
                                "drift": r["drift"], "slips": r["slips"], "usage": resp.get("usage", {})},
                               ensure_ascii=False) + "\n")
            f.flush()


def run(a):
    configure(a.period, a.wording)
    os.makedirs(a.out, exist_ok=True)
    budget = base.Budget(a.max_miss, a.max_out)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda i: run_one(i, a, budget), range(a.runs)))
    meta = {"period": a.period, "wording": a.wording, "runs": a.runs, "turns": a.turns, "miss": budget.miss,
            "hit": budget.hit, "out": budget.out, "finished_at": time.strftime("%F %T")}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False), flush=True)


def fisher(a, b, c, d):
    n, r, k = a + b + c + d, a + b, a + c
    pm = lambda x: comb(k, x) * comb(n - k, r - x) / comb(n, r)
    p0 = pm(a)
    return sum(pm(x) for x in range(max(0, r - (n - k)), min(r, k) + 1) if pm(x) <= p0 + 1e-12)


def report(dirs):
    summary = {}
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        p = meta["period"]
        recs = []
        for fn in sorted(os.listdir(d)):
            if fn.startswith("run_"):
                recs.append({r["turn"]: r for r in map(json.loads, open(os.path.join(d, fn), encoding="utf-8")) if "reply" in r})
        print(f"\n== {d}: period {p}, wording {meta['wording']}, sessions {len(recs)}, tokens miss {meta['miss']} hit {meta['hit']} out {meta['out']}")
        for mult, t in enumerate((p, p * 2, p * 3), 1):
            have = [r[t] for r in recs if t in r]
            if not have:
                continue
            fail = [r for r in have if "mod17_forgotten" in r["drift"]]
            ok = [r for r in have if "mod17_forgotten" not in r["drift"]]
            calc = re.compile(rf"{p}\s*[*×x]\s*{mult}|{mult}\s*[*×x]\s*{p}|{t}\s*=\s*{p}|{p}\s*的\s*{mult}\s*倍")
            print(f"  turn {t} ({p}×{mult}): fail {len(fail)}/{len(have)} | states calc: fail {sum(bool(calc.search(r['reasoning'])) for r in fail)}/{len(fail)}, ok {sum(bool(calc.search(r['reasoning'])) for r in ok)}/{len(ok)}")
            summary[(d, t)] = (len(fail), len(ok))
        other = sum(1 for r in recs for x in r.values() if x["drift"] and x["turn"] not in (p, p * 2, p * 3))
        slips = sum(1 for r in recs for x in r.values() if x["slips"])
        print(f"  other rule-drift turns: {other}; arithmetic-slip turns: {slips}")
    if len(dirs) >= 2 and "A_p17_old" in dirs[0] and "B_p17_new" in dirs[1]:
        a, b = summary.get((dirs[0], 34)), summary.get((dirs[1], 34))
        if a and b:
            print(f"\nA/B turn 34: old {a[0]}/{sum(a)} vs new {b[0]}/{sum(b)}  Fisher p={fisher(a[0], a[1], b[0], b[1]):.3f}")


def selftest():
    for p in (17, 13, 19, 23):
        for w in ("old", "new"):
            rules = configure(p, w)
            assert f"{p} 的倍数" in rules and (w == "old" or "不再以" not in rules)
        v2.selftest()
        print("period", p, "ok; triggers", sorted(v2.MOD17), "trig", v2.TRIG[:6])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--period", type=int, required=True)
    r.add_argument("--wording", choices=["old", "new"], default="old")
    r.add_argument("--runs", type=int, default=30)
    r.add_argument("--turns", type=int, default=35)
    r.add_argument("--model", default="deepseek-flash")
    r.add_argument("--effort", default="low")
    r.add_argument("--parallel", type=int, default=10)
    r.add_argument("--out", required=True)
    r.add_argument("--max-miss", dest="max_miss", type=int, default=4_000_000)
    r.add_argument("--max-out", dest="max_out", type=int, default=3_000_000)
    r.add_argument("--min-balance", dest="min_balance", type=float, default=1.0)
    rp = sub.add_parser("report")
    rp.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        run(a)
    else:
        report(a.dirs)
