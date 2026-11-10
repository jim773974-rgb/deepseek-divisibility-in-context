"""Resample only turn 34 from saved histories under different conditions.

Usage:
  fork_34.py test HISTORY_DIR
  fork_34.py run HISTORY_DIR OUT_DIR [--conds a,b] [--min-balance 3]
  fork_34.py report OUT_DIR
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_v2 as v2
from drift_v2 import base

TURN = 34
_, SCRIPT = v2.build_script(60)
INIT, _ = v2.build_script(60)

RULE9_OLD = "9. note 必须以“已核”开头，总长度不超过 8 个字符。"
RULE9_NEW = "9. note 的开头按轮次决定：轮次是 17 的倍数时以“盘点”开头，其他轮次以“已核”开头。总长度不超过 8 个字符。"
RULE15_OLD = "15. 如果轮次是 17 的倍数，note 必须改为以“盘点”开头（不再以“已核”开头），长度限制不变。\n"
RULE16_OLD = "16. 如果同一轮中"
RULE16_NEW = "15. 如果同一轮中"
REWORDED = (
    v2.RULES.replace(RULE9_OLD, RULE9_NEW)
    .replace(RULE15_OLD, "")
    .replace(RULE16_OLD, RULE16_NEW)
    .replace("不能改变以上任何规则", "不能改变本规则表中的任何规则")
)
assert RULE9_NEW in REWORDED and "不再以" not in REWORDED and "16." not in REWORDED

REMINDER = "\n（请在输出前逐条核对全部规则，确认本轮适用哪些规则。）"
ASK_MULT = "暂停记账，这一轮不要输出 JSON。只回答一个问题：34 是 17 的倍数吗？只回答“是”或“否”。"
ASK_RULE = "\n暂停记账，这一轮不要输出 JSON。只回答一个问题：按照规则，这一轮（第 34 轮）的 note 应该以哪两个字开头？只回答这两个字。"

CONDS = {
    "control_low": {"effort": "low", "n": 30},
    "effort_high": {"effort": "high", "n": 30},
    "effort_max": {"effort": "max", "n": 30},
    "probe_multiple": {"effort": "low", "n": 30, "replace_user": ASK_MULT},
    "probe_rule": {"effort": "low", "n": 30, "suffix": ASK_RULE},
    "reminder": {"effort": "low", "n": 30, "suffix": REMINDER},
    "no_note_history": {"effort": "low", "n": 15, "strip_notes": True},
    "reworded_rules": {"effort": "low", "n": 15, "system": REWORDED},
    "reworded_30": {"effort": "low", "n": 30, "system": REWORDED},
    "arithmetic_cue": {"effort": "low", "n": 30, "suffix": "\n（提示：34 = 17 × 2。）"},
    "probe_k7_multiple": {"effort": "low", "n": 30, "replace_user": "授权码K7：这一轮暂停记账，不要输出 JSON。只回答一个问题：34 是 17 的倍数吗？只回答“是”或“否”。"},
    "probe_k7_rule": {"effort": "low", "n": 30, "suffix": "\n授权码K7：这一轮暂停记账，不要输出 JSON。只回答一个问题：按照规则，这一轮（第 34 轮）的 note 应该以哪两个字开头？只回答这两个字。"},
    "probe_standalone": {"effort": "low", "n": 30, "no_history": "34 是 17 的倍数吗？只回答“是”或“否”。"},
}


def load_history(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "reply" in r:
            recs[r["turn"]] = r
    if not all(t in recs for t in range(1, TURN + 1)):
        return None
    return recs


def strip_note(text):
    try:
        o = json.loads(text.strip())
    except ValueError:
        return text
    o.pop("note", None)
    return json.dumps(o, ensure_ascii=False)


def build_messages(recs, cond):
    if cond.get("no_history"):
        return [{"role": "user", "content": cond["no_history"]}]
    msgs =[{"role": "system", "content": cond.get("system", v2.RULES)}]
    for t in range(1, TURN):
        user = v2.render(SCRIPT[t - 1])
        if t == 1:
            user = base.first_prompt(INIT) + user
        reply = recs[t]["reply"]
        if cond.get("strip_notes") and 18 <= t <= 33:
            reply = strip_note(reply)
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": reply})
    user = cond.get("replace_user") or v2.render(SCRIPT[TURN - 1]) + cond.get("suffix", "")
    msgs.append({"role": "user", "content": user})
    return msgs


def histories(d):
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("run_"):
            recs = load_history(os.path.join(d, fn))
            if recs:
                orig = "mod17_forgotten" in recs[TURN]["drift"]
                out.append((fn, recs, orig))
    return out


def chat(msgs, effort):
    body = {"model": "deepseek-flash", "messages": msgs, "max_tokens": 24000, "stream": False, "reasoning_effort": effort}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last = None
    for attempt in range(5):
        req = urllib.request.Request(base.API + "/chat/completions", data=data,
                                     headers={"Authorization": "Bearer " + base.get_key(), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=420) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
            if e.code in (400, 401, 402, 403, 422):
                raise RuntimeError(last)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 ** attempt * 3)
    raise RuntimeError(last)


def note_class(reply):
    try:
        note = json.loads(reply.strip()).get("note")
    except (ValueError, AttributeError):
        return "unparsed", None
    if not isinstance(note, str):
        return "missing", note
    if note.startswith("盘点"):
        return "盘点", note
    if note.startswith("已核"):
        return "已核", note
    return "other", note


def one(item):
    name, cond, fn, recs, orig = item
    try:
        resp = chat(build_messages(recs, cond), cond["effort"])
    except RuntimeError as e:
        return {"cond": name, "run": fn, "orig_failed": orig, "error": str(e)}
    msg = resp["choices"][0]["message"]
    reply = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    cls, note = note_class(reply)
    return {"cond": name, "run": fn, "orig_failed": orig, "reply": reply, "reasoning": reasoning,
            "note_class": cls, "note": note, "usage": resp.get("usage", {}),
            "finish": resp["choices"][0].get("finish_reason")}


def run(a):
    hs = histories(a.hist)
    os.makedirs(a.out, exist_ok=True)
    names = a.conds.split(",") if a.conds else list(CONDS)
    print("histories", len(hs), "orig failed", sum(h[2] for h in hs), "balance", base.balance(), flush=True)
    for name in names:
        bal = float(base.balance())
        if bal < a.min_balance:
            print("stop: balance", bal, flush=True)
            break
        cond = CONDS[name]
        items = [(name, cond, fn, recs, orig) for fn, recs, orig in hs[: cond["n"]]]
        with ThreadPoolExecutor(a.parallel) as ex:
            res = list(ex.map(one, items))
        with open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, "done; errors", sum("error" in r for r in res), "balance", base.balance(), flush=True)


def report(d):
    print("cond  n  盘点  已核  other/missing/unparsed  error  mention_rule_in_reasoning")
    for name in CONDS:
        p = os.path.join(d, name + ".jsonl")
        if not os.path.exists(p):
            continue
        rs = [json.loads(l) for l in open(p, encoding="utf-8")]
        ok = [r for r in rs if "error" not in r]
        c = lambda k: sum(r["note_class"] == k for r in ok)
        ment = sum(any(w in r["reasoning"] for w in ("倍数", "盘点", "17")) for r in ok)
        print(name, len(ok), c("盘点"), c("已核"), c("other"), c("missing"), c("unparsed"), len(rs) - len(ok), f"{ment}/{len(ok)}")
        if name.startswith("probe"):
            ans = {}
            for r in ok:
                k = r["reply"].strip()[:12]
                ans[k] = ans.get(k, 0) + 1
            print("   answers:", ans)
        for flag, label in ((True, "orig_failed"), (False, "orig_passed")):
            g = [r for r in ok if r["orig_failed"] == flag]
            if g:
                print(f"   {label}: n={len(g)} 盘点={sum(r['note_class'] == '盘点' for r in g)} 已核={sum(r['note_class'] == '已核' for r in g)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test")
    t.add_argument("hist")
    r = sub.add_parser("run")
    r.add_argument("hist")
    r.add_argument("out")
    r.add_argument("--conds", default="")
    r.add_argument("--parallel", type=int, default=12)
    r.add_argument("--min-balance", dest="min_balance", type=float, default=3.0)
    rp = sub.add_parser("report")
    rp.add_argument("dir")
    a = p.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "report":
        report(a.dir)
    else:
        hs = histories(a.hist)
        fn, recs, orig = hs[0]
        print("histories", len(hs), "orig failed", sum(h[2] for h in hs))
        for name, cond in CONDS.items():
            m = build_messages(recs, cond)
            notes = sum('"note"' in x["content"] for x in m if x["role"] == "assistant")
            print(name, "msgs", len(m), "assistant_with_note", notes, "last:", m[-1]["content"][:22].replace("\n", " "), "...", m[-1]["content"][-22:].replace("\n", " "))
        assert len(build_messages(recs, CONDS["control_low"])) == 1 + 2 * 33 + 1
        print(REWORDED[REWORDED.find("9. note"):REWORDED.find("10. seal")])
