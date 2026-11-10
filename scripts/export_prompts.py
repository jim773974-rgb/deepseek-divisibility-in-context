"""Rebuild the exact input messages of every batch from the scripts and each batch's settings,
then re-grade the logged replies against them to prove the rebuild matches what was sent.

Usage: export_prompts.py OUT_DIR   (run from a directory holding the scripts and unpacked raw logs)
"""
import argparse
import importlib
import json
import os
import sys

R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R)


def fresh_modules():
    import drift_pilot
    importlib.reload(drift_pilot)
    if "drift_v2" in sys.modules:
        drift_v2 = importlib.reload(sys.modules["drift_v2"])
    else:
        import drift_v2
    return drift_pilot, drift_v2


def logged(d):
    out = []
    for fn in sorted(os.listdir(os.path.join(R, d))):
        if fn.startswith("run_"):
            out.append([json.loads(l) for l in open(os.path.join(R, d, fn), encoding="utf-8")])
    return out


def regrade(d, base, check, script, init, ref_true=None):
    """Re-run the grader on logged replies using rebuilt inputs; count label mismatches."""
    mism = total = 0
    for recs in logged(d):
        prev = dict(init)
        truth = dict(init)
        for r in recs:
            if "reply" not in r:
                continue
            e = script[r["turn"] - 1]
            ref = truth if ref_true else prev
            g = check(r["reply"], e, ref)
            total += 1
            mism += (sorted(g["drift"]) != sorted(r["drift"])) or (sorted(g["slips"]) != sorted(r["slips"]))
            if g["parsed"]:
                prev = g["parsed"]
            if ref_true:
                truth = ref_true(truth, e)
    return mism, total


def export(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    report = []

    def write(name, system, turns, note=""):
        with open(os.path.join(out_dir, name.replace("/", "_") + ".json"), "w", encoding="utf-8") as f:
            json.dump({"batch": name, "note": note, "system": system, "turns": turns}, f, ensure_ascii=False, indent=1)

    # pilot1
    base, _ = None, None
    import drift_pilot as base
    importlib.reload(base)
    init, script = base.build_script(40)
    turns = [{"turn": e["turn"], "user": (base.first_prompt(init) if i == 0 else "") + base.render(e)} for i, e in enumerate(script)]
    write("pilot1", base.RULES, turns, "6 个会话输入完全相同")
    report.append(("pilot1",) + regrade("pilot1", base, base.check, script, init))

    # combined (drift_v2, period 17, 60 turns)
    base, v2 = fresh_modules()
    init, script = v2.build_script(60)
    turns = [{"turn": e["turn"], "user": (base.first_prompt(init) if i == 0 else "") + v2.render(e)} for i, e in enumerate(script)]
    write("combined", v2.RULES, turns, "30 个会话输入完全相同")
    report.append(("combined",) + regrade("combined", base, v2.check2, script, init))

    # drift_v3 batches
    for d in ("A_p17_old", "B_p17_new", "C_p13", "D_p19", "E_p23"):
        base, v2 = fresh_modules()
        import drift_v3
        importlib.reload(drift_v3)
        m = json.load(open(os.path.join(R, d, "meta.json")))
        drift_v3.configure(m["period"], m["wording"])
        init, script = base.build_script(m["turns"])
        turns = [{"turn": e["turn"], "user": (base.first_prompt(init) if i == 0 else "") + base.render(e)} for i, e in enumerate(script)]
        write(d, base.RULES, turns, "同批会话输入完全相同")
        report.append((d,) + regrade(d, base, base.check, script, init))

    # drift_v4 batches
    for root in ("p1", "p2", "p3", "p4"):
        for sub in sorted(os.listdir(os.path.join(R, root))):
            d = f"{root}/{sub}"
            base, v2 = fresh_modules()
            import drift_v4
            importlib.reload(drift_v4)
            m = json.load(open(os.path.join(R, d, "meta.json")))
            ns = argparse.Namespace(**{k: m[k] for k in ("period", "turns", "list_turns", "active_from", "seed")})
            drift_v4.configure(ns)
            iv = drift_v4.parse_intervene(m["intervene"])
            init, script = base.build_script(m["turns"])
            truth = dict(init)
            turns = []
            for i, e in enumerate(script):
                t = e["turn"]
                u = base.render(e) + iv.get(t, "")
                if m["fresh_each_turn"]:
                    u = "当前库存：" + json.dumps(truth, ensure_ascii=False) + "\n\n" + u
                elif i == 0:
                    u = base.first_prompt(init) + u
                turns.append({"turn": t, "user": u})
                truth = v2.apply2(truth, e["ops"], t in v2.NIGHT)[0]
            note = "每轮都是新会话，只有 system 和这一条 user 消息" if m["fresh_each_turn"] else "同批会话输入完全相同"
            write(d, base.RULES, turns, f"{note}；模型 {m['model']}，思考档位 {m['effort']}")
            step = (lambda tr, e, v2=v2: v2.apply2(tr, e["ops"], e["turn"] in v2.NIGHT)[0]) if m["fresh_each_turn"] else None
            report.append((d,) + regrade(d, base, base.check, script, init, step))

    # code task
    import code_task
    importlib.reload(code_task)
    turns = [{"turn": n, "user": code_task.user_msg(n)} for n in range(1, 41)]
    write("code1", code_task.SYSTEM, turns, "20 个会话输入完全相同")
    mism = total = 0
    for recs in logged("code1"):
        for r in recs:
            if "reply" in r:
                g = code_task.check(r["reply"], r["turn"])
                total += 1
                mism += sorted(g["drift"]) != sorted(r["drift"])
    report.append(("code1", mism, total))

    # turn-34 resampling: describe how each condition's input was built
    base, v2 = fresh_modules()
    import fork_34
    importlib.reload(fork_34)
    conds = {k: {kk: vv for kk, vv in v.items() if kk != "n"} for k, v in fork_34.CONDS.items()}
    with open(os.path.join(out_dir, "fork_turn34_conditions.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "历史 = combined 批次对应会话第 1–33 轮的 user 消息和模型回复；最后一条 user 消息按条件替换或追加。",
                   "turn34_user": v2.render(fork_34.SCRIPT[33]), "conditions": conds, "reworded_rules_system": fork_34.REWORDED},
                  f, ensure_ascii=False, indent=1)

    print("batch | 重新判分与日志不一致的轮数 / 总轮数")
    for row in report:
        print(*row)


if __name__ == "__main__":
    export(sys.argv[1])
