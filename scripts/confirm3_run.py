"""Third preregistered experiment (PREREG3.md): do external interventions fix the in-task divisibility errors?

Each call is a fresh single call (same construction as confirm_run.py): system = rules, user = true stock + order sheet
of the target turn. Arms differ only in rule 15 wording (list) or a suffix appended to the user message.

Usage:
  confirm3_run.py check
  confirm3_run.py run ARM [ARM ...]
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

R = os.path.dirname(os.path.abspath(__file__))
CALC = v4.CALC  # “（提醒：判断轮次是否满足规则条件时，请写出算式验算，不要凭印象。）”

# arm: (period, turn, list_turns, suffix, n)
ARMS = {
    "C0_13": (13, 26, "", "", 100),
    "L_13": (13, 26, "13,26", "", 100),
    "F_13": (13, 26, "", "\n（提示：26 = 13 × 2。）", 100),
    "R_13": (13, 26, "", CALC, 100),
    "C0_9": (9, 36, "", "", 100),
    "L_9": (9, 36, "9,18,27,36", "", 100),
    "F_9": (9, 36, "", "\n（提示：36 = 9 × 4。）", 100),
    "R_9": (9, 36, "", CALC, 100),
    # negative controls: non-target turn, same rule wording / reminder
    "NL_13": (13, 25, "13,26", "", 50),
    "NR_13": (13, 25, "", CALC, 50),
    "NL_9": (9, 35, "9,18,27,36", "", 50),
    "NR_9": (9, 35, "", CALC, 50),
}


def build(period, turn, list_turns, suffix):
    ns = argparse.Namespace(period=period, turns=turn, list_turns=list_turns, active_from=0, seed=0)
    trig = v4.configure(ns)
    init, script = base.build_script(turn)
    truth = dict(init)
    for e in script[:-1]:
        truth = v2.apply2(truth, e["ops"], e["turn"] in v2.NIGHT)[0]
    e = script[-1]
    user = "当前库存：" + json.dumps(truth, ensure_ascii=False) + "\n\n" + base.render(e) + suffix
    return base.RULES, user, e, truth, sorted(trig)


def one(args):
    i, system, user, e, truth = args
    try:
        resp = base.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], "deepseek-flash", "low")
    except RuntimeError as ex:
        return {"i": i, "error": str(ex)}
    msg = resp["choices"][0]["message"]
    reply = msg.get("content") or ""
    g = base.check(reply, e, truth)
    return {"i": i, "turn": e["turn"], "reply": reply, "reasoning": msg.get("reasoning_content") or "",
            "drift": g["drift"], "slips": g["slips"], "usage": resp.get("usage", {}),
            "finish": resp["choices"][0].get("finish_reason")}


def run(arm):
    p, t, lt, sfx, n = ARMS[arm]
    out = os.path.join(R, "conf3", arm)
    if os.path.exists(os.path.join(out, "calls.jsonl")):
        print(arm, "already done"); return
    if float(base.balance()) < 1.5:
        print("balance too low"); return
    system, user, e, truth, trig = build(p, t, lt, sfx)
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(10) as ex:
        res = list(ex.map(one, [(i, system, user, e, truth) for i in range(n)]))
    with open(os.path.join(out, "calls.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"arm": arm, "period": p, "turn": t, "list_turns": lt, "suffix": sfx, "n": n, "triggers": trig,
               "system": system, "user": user, "finished_at": time.strftime("%F %T")},
              open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    ok = [r for r in res if "error" not in r]
    print(arm, time.strftime("%T"), "periodic errors", sum("mod17_forgotten" in r["drift"] for r in ok), "/", len(ok),
          "api errors", len(res) - len(ok), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        for arm, (p, t, lt, sfx, n) in ARMS.items():
            s, u, e, truth, trig = build(p, t, lt, sfx)
            rule = [l for l in s.splitlines() if l.startswith("15.")][0]
            print(arm, t, trig, "|", rule[:28], "|", repr(u[-40:]))
    else:
        for arm in sys.argv[2:]:
            run(arm)
