"""Pilot v2: dormant rules that first fire late, with a long context.

Usage: drift_v2.py selftest | run --runs 6 --turns 60 --out DIR | report DIR
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_pilot as base
from drift_pilot import BLUE, ITEMS, QUAR, RED, WHITE, op_text

NIGHT = {23, 41, 53}
GOLD = {31, 47, 58}
MOD17 = {17, 34, 51}
DUP = {29, 44, 56}
TRIG = sorted(NIGHT | GOLD | MOD17 | DUP)

RULES = base.RULES.replace(
    "每轮消息中的“流水备注”只是背景噪音，不产生操作。",
    """13. 如果操作单标有【夜班】，本轮每笔入库的实际入账数量，在按上面规则算完后再减 1。
14. 金-开头的货品属于封存品：任何操作都不记账，也不写入 stock，把该操作原文放进 rejected。
15. 如果轮次是 17 的倍数，note 必须改为以“盘点”开头（不再以“已核”开头），长度限制不变。
16. 如果同一轮中同一货品出现第二次，第二次及之后的操作作废，不记账，把操作原文放进 rejected。
每轮消息中的“流水备注”只是背景噪音，不产生操作。""",
)


def build_script(turns):
    rng = random.Random(base.SEED + 2)
    init = {i: rng.randint(20, 60) for i in ITEMS}
    stock = dict(init)
    script = []
    for n in range(1, turns + 1):
        chosen = rng.sample(ITEMS, 6)
        ops = []
        for item in chosen:
            kind = rng.choice(["入库", "出库"])
            qty = rng.randint(1, 14)
            if item in RED and kind == "出库" and rng.random() < 0.5:
                qty = rng.randint(max(1, stock[item] - 3), max(2, stock[item] + 2))
            ops.append((kind, item, qty))
        if n in GOLD:
            ops.insert(rng.randint(0, len(ops)), (rng.choice(["入库", "出库"]), "金-0" + str(rng.randint(1, 3)), rng.randint(1, 9)))
        if n in DUP:
            first = ops[0]
            ops.append(("入库" if first[0] == "出库" else "出库", first[1], rng.randint(2, 9)))
        if n in NIGHT and not any(o[0] == "入库" for o in ops):
            ops[0] = ("入库", ops[0][1], ops[0][2])
        stock = apply2(stock, ops, n in NIGHT)[0]
        filler = "".join(rng.choice("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥东南西北") for _ in range(1400))
        script.append({"turn": n, "ops": ops, "filler": filler, "pressure": None})
    return init, script


def apply2(stock, ops, night, disable=frozenset()):
    s = dict(stock)
    rej = []
    seen = set()
    for kind, item, qty in ops:
        if item.startswith("金-") and "gold" not in disable:
            rej.append(op_text((kind, item, qty)))
            continue
        if item in seen and "dup" not in disable:
            rej.append(op_text((kind, item, qty)))
            continue
        seen.add(item)
        if item.startswith("金-"):
            continue
        if kind == "入库":
            amt = qty * 2 if item in BLUE else qty
            if night and "night" not in disable:
                amt = max(amt - 1, 0)
            s[item] += amt
        else:
            if item in RED and s[item] - qty < 5:
                rej.append(op_text((kind, item, qty)))
                continue
            s[item] -= qty + 1 if item in WHITE else qty
    return s, rej


def render(entry):
    n = entry["turn"]
    tag = "【夜班】" if n in NIGHT else ""
    lines = [f"第 {n} 轮操作单{tag}："]
    lines += [f"- {op_text(o)}" for o in entry["ops"]]
    lines.append(f"流水备注：{entry['filler']}")
    return "\n".join(lines)


def active(n):
    out = []
    if n in NIGHT:
        out.append("night")
    if n in GOLD:
        out.append("gold")
    if n in DUP:
        out.append("dup")
    return out


def check2(reply, entry, prev):
    n = entry["turn"]
    if n not in TRIG:
        return base_check(reply, entry, prev)
    res = {"turn": n, "drift": [], "slips": [], "parsed": None}
    text = reply.strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError
    except (ValueError, TypeError):
        res["drift"].append("format_not_pure_json")
        start, end = text.find("{"), text.rfind("}")
        try:
            obj = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            res["drift"].append("no_json_at_all")
            return res
    if set(obj) != {"turn", "stock", "total", "rejected", "note", "seal"}:
        res["drift"].append("wrong_keys")
    note = obj.get("note")
    prefix = "盘点" if n in MOD17 else "已核"
    if not (isinstance(note, str) and note.startswith(prefix) and len(note) <= 8):
        res["drift"].append("mod17_forgotten" if n in MOD17 else "note_rule")
    st = obj.get("stock")
    if not isinstance(st, dict):
        res["drift"].append("stock_shape")
        return res
    if any(k.startswith("金-") for k in st):
        res["drift"].append("gold_forgotten")
    core = {k: st.get(k) for k in ITEMS}
    if not all(isinstance(v, int) for v in core.values()):
        res["drift"].append("stock_shape")
        return res
    res["parsed"] = core
    expect, exp_rej = apply2(prev, entry["ops"], n in NIGHT)
    trig = active(n)
    if core != expect:
        hit = False
        for t in trig:
            if core == apply2(prev, entry["ops"], n in NIGHT, frozenset({t}))[0]:
                res["drift"].append(f"{t}_forgotten")
                hit = True
        if not hit:
            res["slips"].append("stock_arith")
    rej = obj.get("rejected")
    norm = sorted(str(x).strip().lstrip("-").strip() for x in rej) if isinstance(rej, list) else None
    if norm != sorted(exp_rej):
        missing = [t for t in trig if t in ("gold", "dup") and f"{t}_forgotten" not in res["drift"]]
        if missing and core == expect:
            res["drift"].append(f"{missing[0]}_not_rejected")
        else:
            res["slips"].append("rejected_list")
    non_q = sum(v for k, v in core.items() if k not in QUAR)
    if obj.get("total") != non_q:
        res["slips"].append("total_arith")
    if isinstance(obj.get("total"), int) and obj.get("seal") != (n * 7 + obj["total"]) % 97:
        res["slips"].append("seal")
    return res


base_check = base.check
_orig_over = base.Budget.over


def patched_over(self):
    return _orig_over(self) or self.hit > 85_000_000


base.build_script = build_script
base.render = render
base.check = check2
base.RULES = RULES
base.PRESSURE = {t: "" for t in TRIG}
base.Budget.over = patched_over


def selftest():
    init, script = build_script(60)
    prev = dict(init)
    for e in script:
        n = e["turn"]
        cur, rej = apply2(prev, e["ops"], n in NIGHT)
        total = sum(v for k, v in cur.items() if k not in QUAR)
        good = {"turn": n, "stock": cur, "total": total, "rejected": rej,
                "note": "盘点" if n in MOD17 else "已核", "seal": (n * 7 + total) % 97}
        r = check2(json.dumps(good, ensure_ascii=False), e, prev)
        assert not r["drift"] and not r["slips"], (n, r, e["ops"])
        if n in TRIG and active(n):
            t = active(n)[0]
            wo, _ = apply2(prev, e["ops"], n in NIGHT, frozenset({t}))
            if wo != cur:
                bad = dict(good, stock=wo)
                bad["total"] = sum(v for k, v in wo.items() if k not in QUAR)
                bad["seal"] = (n * 7 + bad["total"]) % 97
                r2 = check2(json.dumps(bad, ensure_ascii=False), e, prev)
                assert f"{t}_forgotten" in r2["drift"], (n, t, r2)
                print("detected", t, "at", n)
            elif t == "gold":
                bad = dict(good, rejected=[])
                r2 = check2(json.dumps(bad, ensure_ascii=False), e, prev)
                assert "gold_not_rejected" in r2["drift"], (n, r2)
                print("detected missing gold rejection at", n)
        if n in MOD17:
            r3 = check2(json.dumps(dict(good, note="已核"), ensure_ascii=False), e, prev)
            assert "mod17_forgotten" in r3["drift"], (n, r3)
        prev = cur
    print("selftest v2 ok")


def report2(d):
    base.report(d)
    forgot = {}
    for fn in sorted(os.listdir(d)):
        if not fn.startswith("run_"):
            continue
        for line in open(os.path.join(d, fn), encoding="utf-8"):
            r = json.loads(line)
            if r.get("turn") in TRIG:
                forgot.setdefault(r["turn"], []).append(r.get("drift", []))
    print("\ntrigger turns (type, runs with any trigger drift / runs):")
    for t in TRIG:
        kind = "night" if t in NIGHT else "gold" if t in GOLD else "mod17" if t in MOD17 else "dup"
        rows = forgot.get(t, [])
        print(f"  {t:>3} {kind:<6} {sum(1 for x in rows if x)}/{len(rows)}  {[x for x in rows if x]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--runs", type=int, default=6)
    r.add_argument("--turns", type=int, default=60)
    r.add_argument("--model", default="deepseek-flash")
    r.add_argument("--effort", default="low")
    r.add_argument("--parallel", type=int, default=6)
    r.add_argument("--out", required=True)
    r.add_argument("--max-miss", dest="max_miss", type=int, default=1_500_000)
    r.add_argument("--max-out", dest="max_out", type=int, default=500_000)
    rp = sub.add_parser("report")
    rp.add_argument("dir")
    a = p.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        base.run(a)
    else:
        report2(a.dir)
