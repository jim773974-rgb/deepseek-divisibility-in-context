"""General harness for the follow-up experiments. Every session is independent and starts fresh.

Usage:
  drift_v4.py selftest
  drift_v4.py run --name NAME --period 13 --turns 27 --runs 30 [options] --out DIR
  drift_v4.py report DIR [DIR ...]
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_v2 as v2
from drift_v2 import base

CALC = "\n（提醒：判断轮次是否满足规则条件时，请写出算式验算，不要凭印象。）"
GENERIC = "\n（提醒：请在输出前逐条核对全部规则。）"
TEXTS = {"calc": CALC, "generic": GENERIC}
OLD15 = "15. 如果轮次是 17 的倍数，note 必须改为以“盘点”开头（不再以“已核”开头），长度限制不变。"


def configure(a):
    p = a.period
    if a.list_turns:
        trig = {int(x) for x in a.list_turns.split(",")}
        cond = "如果轮次是 " + " 或 ".join(str(x) for x in sorted(trig))
    elif a.active_from:
        trig = {n for n in range(a.active_from, a.turns + 1) if n % p == 0}
        cond = f"从第 {a.active_from} 轮开始，如果轮次是 {p} 的倍数"
    else:
        trig = {n for n in range(1, a.turns + 1) if n % p == 0}
        cond = f"如果轮次是 {p} 的倍数"
    assert OLD15 in v2.RULES
    rules = v2.RULES.replace(OLD15, f"15. {cond}，note 必须改为以“盘点”开头（不再以“已核”开头），长度限制不变。")
    v2.MOD17 = trig
    v2.NIGHT = {x + 1 if x in trig else x for x in {23, 41, 53}}
    v2.GOLD = {x + 1 if x in trig else x for x in {31, 47, 58}}
    v2.DUP = {x + 1 if x in trig else x for x in {29, 44, 56}}
    v2.TRIG = sorted(v2.NIGHT | v2.GOLD | v2.MOD17 | v2.DUP)
    base.RULES = rules
    base.PRESSURE = {t: "" for t in v2.TRIG}
    base.SEED = 20261109 + a.seed
    return trig


def parse_intervene(s):
    out = {}
    if s:
        for part in s.split(","):
            t, name = part.split(":")
            out[int(t)] = TEXTS[name]
    return out


def run_one(idx, a, budget, intervene):
    init, script = base.build_script(a.turns)
    path = os.path.join(a.out, f"run_{idx:02d}.jsonl")
    messages = [{"role": "system", "content": base.RULES}]
    prev = dict(init)
    true_stock = dict(init)
    with open(path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(script):
            if budget.over():
                f.write(json.dumps({"stopped": "budget", "turn": entry["turn"]}) + "\n")
                return
            if i % 8 == 0 and float(base.balance()) < a.min_balance:
                f.write(json.dumps({"stopped": "balance", "turn": entry["turn"]}) + "\n")
                return
            t = entry["turn"]
            user = base.render(entry) + intervene.get(t, "")
            if a.fresh_each_turn:
                user = "当前库存：" + json.dumps(true_stock, ensure_ascii=False) + "\n\n" + user
                call = [{"role": "system", "content": base.RULES}, {"role": "user", "content": user}]
            else:
                if i == 0:
                    user = base.first_prompt(init) + user
                messages.append({"role": "user", "content": user})
                call = messages
            try:
                resp = base.chat(call, a.model, a.effort)
            except RuntimeError as e:
                f.write(json.dumps({"stopped": str(e), "turn": t}, ensure_ascii=False) + "\n")
                return
            msg = resp["choices"][0]["message"]
            reply = msg.get("content") or ""
            budget.add(resp.get("usage", {}))
            ref = true_stock if a.fresh_each_turn else prev
            r = base.check(reply, entry, ref)
            if r["parsed"]:
                prev = r["parsed"]
            true_stock = v2.apply2(true_stock, entry["ops"], t in v2.NIGHT)[0]
            if not a.fresh_each_turn:
                messages.append({"role": "assistant", "content": reply})
            f.write(json.dumps({"turn": t, "reply": reply, "reasoning": msg.get("reasoning_content") or "",
                                "drift": r["drift"], "slips": r["slips"], "usage": resp.get("usage", {}),
                                "finish": resp["choices"][0].get("finish_reason")}, ensure_ascii=False) + "\n")
            f.flush()


def run(a):
    trig = configure(a)
    intervene = parse_intervene(a.intervene)
    os.makedirs(a.out, exist_ok=True)
    budget = base.Budget(a.max_miss, a.max_out)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda i: run_one(i, a, budget, intervene), range(a.runs)))
    meta = dict(vars(a), triggers=sorted(trig), miss=budget.miss, hit=budget.hit, out=budget.out,
                finished_at=time.strftime("%F %T"))
    meta.pop("cmd", None)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(a.name, "done miss", budget.miss, "hit", budget.hit, "out", budget.out, flush=True)


def load(d):
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("run_"):
            recs = [json.loads(l) for l in open(os.path.join(d, fn), encoding="utf-8")]
            out.append({r["turn"]: r for r in recs if "reply" in r})
    return out


def report(dirs):
    print("name | model/effort | trigger: fail/all (clean fail/clean n) | states calc fail/ok | other drift turns | slip turns")
    for d in dirs:
        m = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        sess = load(d)
        parts, calc_f, calc_o, nf, no = [], 0, 0, 0, 0
        for t in m["triggers"]:
            fail = n = cf = cn = 0
            for s in sess:
                if t not in s:
                    continue
                n += 1
                bad = "mod17_forgotten" in s[t]["drift"]
                fail += bad
                prior = any(s[x]["drift"] or s[x]["slips"] for x in s if x < t)
                if not prior:
                    cn += 1
                    cf += bad
                r = s[t].get("reasoning", "")
                p = m["period"]
                k = t // p if t % p == 0 else 0
                hit = bool(re.search(rf"{p}\s*[*×x]\s*{k}|{k}\s*[*×x]\s*{p}|{t}\s*=\s*{p}|{t}\s*[/÷]\s*{p}", r)) if k else False
                if bad:
                    nf += 1
                    calc_f += hit
                else:
                    no += 1
                    calc_o += hit
            parts.append(f"t{t}: {fail}/{n} ({cf}/{cn})")
        trig = set(m["triggers"])
        other = sum(1 for s in sess for r in s.values() if r["drift"] and r["turn"] not in trig)
        slip = sum(1 for s in sess for r in s.values() if r["slips"])
        print(m["name"], "|", m["model"], m["effort"], "seed", m["seed"], "|", "; ".join(parts), f"| calc {calc_f}/{nf}, {calc_o}/{no} | {other} | {slip} | out {m['out']}")


def selftest():
    for kw in ({"period": 13}, {"period": 13, "active_from": 20}, {"period": 13, "list_turns": "13,26"}):
        ns = argparse.Namespace(period=kw["period"], turns=60, list_turns=kw.get("list_turns", ""),
                                active_from=kw.get("active_from", 0), seed=0)
        trig = configure(ns)
        v2.selftest()
        print(kw, "triggers", sorted(trig)[:6], "ok; rule:", [l for l in base.RULES.splitlines() if l.startswith("15.")][0][:40])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--period", type=int, default=13)
    r.add_argument("--turns", type=int, default=27)
    r.add_argument("--runs", type=int, default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--list-turns", dest="list_turns", default="")
    r.add_argument("--active-from", dest="active_from", type=int, default=0)
    r.add_argument("--intervene", default="")
    r.add_argument("--fresh-each-turn", dest="fresh_each_turn", action="store_true")
    r.add_argument("--model", default="deepseek-flash")
    r.add_argument("--effort", default="low")
    r.add_argument("--parallel", type=int, default=8)
    r.add_argument("--out", required=True)
    r.add_argument("--max-miss", dest="max_miss", type=int, default=6_000_000)
    r.add_argument("--max-out", dest="max_out", type=int, default=5_000_000)
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
