"""Fourth preregistered experiment (PREREG4.md): does inserting the hint EARLIER work as well as at the error turn?

Full multi-turn sessions (fresh session each, history kept), N=13, turns 1..26, deepseek-flash low.
All five arms run interleaved in one thread pool so that they share the same time window.
Arms differ only in the system rule 15 wording (L) or one suffix appended to one turn's user message.

Usage:
  confirm4_run.py check
  confirm4_run.py run
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_v4 as v4
from drift_v4 import base, v2

R = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(R, "conf4")
TURNS, PERIOD, RUNS = 26, 13, 30
NOW = "\n（提示：26 = 13 × 2。）"
AHEAD = "\n（提示：第 26 轮的轮次 26 = 13 × 2，是 13 的倍数。）"

# arm: (list_turns, {turn: suffix})
ARMS = {
    "S0": ("", {}),
    "F26": ("", {26: NOW}),
    "F22": ("", {22: AHEAD}),
    "F14": ("", {14: AHEAD}),
    "L": ("13,26", {}),
}


def system_for(list_turns):
    v4.configure(argparse.Namespace(period=PERIOD, turns=TURNS, list_turns=list_turns, active_from=0, seed=0))
    return base.RULES


SYSTEMS = {arm: system_for(lt) for arm, (lt, _) in ARMS.items()}
# configure() also sets the shared schedule (trigger turns, night/gold/dup turns); it is identical for all arms
# because the trigger set is {13, 26} either way. Leave it in the period-rule state.
system_for("")
assert v2.MOD17 == {13, 26}


def run_one(job):
    arm, idx = job
    system, inter = SYSTEMS[arm], ARMS[arm][1]
    init, script = base.build_script(TURNS)
    os.makedirs(os.path.join(OUT, arm), exist_ok=True)
    path = os.path.join(OUT, arm, f"run_{idx:02d}.jsonl")
    messages = [{"role": "system", "content": system}]
    prev = dict(init)
    with open(path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(script):
            t = entry["turn"]
            if i % 8 == 0 and float(base.balance()) < 1.5:
                f.write(json.dumps({"stopped": "balance", "turn": t}) + "\n"); return
            user = base.render(entry) + inter.get(t, "")
            if i == 0:
                user = base.first_prompt(init) + user
            messages.append({"role": "user", "content": user})
            try:
                resp = base.chat(messages, "deepseek-flash", "low")
            except RuntimeError as e:
                f.write(json.dumps({"stopped": str(e), "turn": t}, ensure_ascii=False) + "\n"); return
            msg = resp["choices"][0]["message"]
            reply = msg.get("content") or ""
            r = base.check(reply, entry, prev)
            if r["parsed"]:
                prev = r["parsed"]
            messages.append({"role": "assistant", "content": reply})
            f.write(json.dumps({"turn": t, "reply": reply, "reasoning": msg.get("reasoning_content") or "",
                                "drift": r["drift"], "slips": r["slips"], "usage": resp.get("usage", {}),
                                "finish": resp["choices"][0].get("finish_reason")}, ensure_ascii=False) + "\n")
            f.flush()


def main():
    if os.path.exists(os.path.join(OUT, "meta.json")):
        print("already done"); return
    jobs = [(arm, i) for arm in ARMS for i in range(RUNS)]
    random.Random(20261110).shuffle(jobs)
    start = time.strftime("%F %T")
    with ThreadPoolExecutor(30) as ex:
        list(ex.map(run_one, jobs))
    json.dump({"arms": {a: {"list_turns": lt, "intervene": {str(k): v for k, v in iv.items()}, "system": SYSTEMS[a]}
                        for a, (lt, iv) in ARMS.items()},
               "turns": TURNS, "period": PERIOD, "runs": RUNS, "started_at": start, "finished_at": time.strftime("%F %T")},
              open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("done", time.strftime("%T"), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        init, script = base.build_script(TURNS)
        base_users = [base.render(e) for e in script]
        for arm, (lt, iv) in ARMS.items():
            rule = [l for l in SYSTEMS[arm].splitlines() if l.startswith("15.")][0]
            print(arm, "|", rule[:30], "|", {t: s.strip() for t, s in iv.items()})
        print("trigger turns", sorted(v2.MOD17), "night", sorted(v2.NIGHT), "gold", sorted(v2.GOLD), "dup", sorted(v2.DUP))
        print("systems identical except rule 15:",
              len({"\n".join(l for l in s.splitlines() if not l.startswith("15.")) for s in SYSTEMS.values()}) == 1)
    else:
        main()
