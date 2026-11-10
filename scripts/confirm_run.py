"""Confirmatory experiment A: single fresh calls at a fixed target turn, rule period N varied.

Every call: system = rules with period N; user = current true stock + the target turn's order sheet.
Across arms only the number N in rule 15 differs; the order sheet and stock are identical.

Usage:
  confirm_run.py check
  confirm_run.py run --period 18 --turn 36 --n 100 --out DIR
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_v4 as v4
from drift_v4 import base, v2


def build(period, turn):
    ns = argparse.Namespace(period=period, turns=turn, list_turns="", active_from=0, seed=0)
    trig = v4.configure(ns)
    init, script = base.build_script(turn)
    truth = dict(init)
    for e in script[:-1]:
        truth = v2.apply2(truth, e["ops"], e["turn"] in v2.NIGHT)[0]
    e = script[-1]
    user = "当前库存：" + json.dumps(truth, ensure_ascii=False) + "\n\n" + base.render(e)
    return base.RULES, user, e, truth, sorted(trig)


def one(args):
    i, a, system, user, e, truth = args
    try:
        resp = base.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], "deepseek-flash", "low")
    except RuntimeError as ex:
        return {"i": i, "error": str(ex)}
    msg = resp["choices"][0]["message"]
    reply = msg.get("content") or ""
    g = base.check(reply, e, truth)
    return {"i": i, "turn": e["turn"], "period": a.period, "reply": reply, "reasoning": msg.get("reasoning_content") or "",
            "drift": g["drift"], "slips": g["slips"], "usage": resp.get("usage", {}),
            "finish": resp["choices"][0].get("finish_reason")}


def direct_one(args):
    i, q = args
    try:
        resp = base.chat([{"role": "user", "content": q}], "deepseek-flash", "low")
    except RuntimeError as ex:
        return {"i": i, "error": str(ex)}
    msg = resp["choices"][0]["message"]
    reply = msg.get("content") or ""
    s = reply.strip().lstrip("“\"'「*").strip()
    verdict = "是" if s.startswith("是") else "否" if s.startswith("否") else "无法判定"
    return {"i": i, "reply": reply, "reasoning": msg.get("reasoning_content") or "", "verdict": verdict,
            "usage": resp.get("usage", {}), "finish": resp["choices"][0].get("finish_reason")}


def run_direct(a):
    os.makedirs(a.out, exist_ok=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(direct_one, [(i, a.direct) for i in range(a.n)]))
    with open(os.path.join(a.out, "calls.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"direct": a.direct, "n": a.n, "finished_at": time.strftime("%F %T")},
              open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    from collections import Counter
    print(a.direct, dict(Counter(r.get("verdict", "API报错") for r in res)), flush=True)


def run(a):
    if a.direct:
        return run_direct(a)
    system, user, e, truth, trig = build(a.period, a.turn)
    os.makedirs(a.out, exist_ok=True)
    if float(base.balance()) < a.min_balance:
        print("balance too low"); return
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(one, [(i, a, system, user, e, truth) for i in range(a.n)]))
    with open(os.path.join(a.out, "calls.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"period": a.period, "turn": a.turn, "n": a.n, "triggers": trig, "system": system, "user": user,
               "finished_at": time.strftime("%F %T")}, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    ok = [r for r in res if "error" not in r]
    print(a.period, a.turn, "periodic errors", sum("mod17_forgotten" in r["drift"] for r in ok), "/", len(ok), "api errors", len(res) - len(ok), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--period", type=int, default=0)
    r.add_argument("--turn", type=int, default=0)
    r.add_argument("--direct", default="")
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--parallel", type=int, default=10)
    r.add_argument("--out", required=True)
    r.add_argument("--min-balance", dest="min_balance", type=float, default=1.5)
    a = ap.parse_args()
    if a.cmd == "check":
        users = {}
        for p, t in ((36, 36), (18, 36), (12, 36), (9, 36), (26, 26), (13, 26)):
            s, u, e, truth, trig = build(p, t)
            users.setdefault(t, set()).add(u)
            rule = [l for l in s.splitlines() if l.startswith("15.")][0]
            good = {"turn": t, "stock": v2.apply2(truth, e["ops"], t in v2.NIGHT)[0], "rejected": v2.apply2(truth, e["ops"], t in v2.NIGHT)[1], "note": "盘点"}
            good["total"] = sum(v for k, v in good["stock"].items() if not k.startswith("Q"))
            good["seal"] = (t * 7 + good["total"]) % 97
            g = base.check(json.dumps(good, ensure_ascii=False), e, truth)
            bad = base.check(json.dumps(dict(good, note="已核"), ensure_ascii=False), e, truth)
            print(p, t, trig, rule[:30], "| correct graded:", g["drift"], g["slips"], "| wrong note graded:", bad["drift"])
        print("identical user message per target turn:", {t: len(v) == 1 for t, v in users.items()})
    else:
        run(a)
