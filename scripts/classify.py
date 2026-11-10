"""Split every observed error into separate sample groups.

clean = the first error of any kind in that session, so no earlier error sits in its context.
contaminated = at least one earlier error was already in the context.

Usage: classify.py OUT_PREFIX LABEL=DIR [LABEL=DIR ...]
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

CATEGORY = {
    "mod17_forgotten": "周期规则判断错",
    "blue_not_doubled": "常规规则漏用", "white_no_loss": "常规规则漏用", "red_not_rejected": "常规规则漏用",
    "total_includes_quarantine": "常规规则漏用",
    "night_forgotten": "事件规则漏用", "gold_forgotten": "事件规则漏用", "gold_not_rejected": "事件规则漏用",
    "dup_forgotten": "事件规则漏用", "dup_not_rejected": "事件规则漏用",
    "note_rule": "格式崩坏", "format_not_pure_json": "格式崩坏", "no_json_at_all": "格式崩坏",
    "wrong_keys": "格式崩坏", "stock_shape": "格式崩坏",
    "stock_arith": "计算失误", "total_arith": "计算失误", "seal": "计算失误", "rejected_list": "计算失误",
    "untouched_changed": "计算失误", "turn_number": "计算失误",
}


def sessions(d):
    for fn in sorted(os.listdir(d)):
        if fn.startswith("run_"):
            recs = [json.loads(l) for l in open(os.path.join(d, fn), encoding="utf-8")]
            yield fn[:-6], [r for r in recs if "reply" in r]


def main():
    prefix = sys.argv[1]
    rows, per_batch = [], {}
    for arg in sys.argv[2:]:
        label, d = arg.split("=", 1)
        n_sessions = n_turns = 0
        at_risk, clean_events, all_events = Counter(), Counter(), Counter()
        for sid, recs in sessions(d):
            n_sessions += 1
            n_turns += len(recs)
            prior = Counter()
            for r in recs:
                labels = r["drift"] + r["slips"]
                t = r["turn"]
                is_clean = sum(prior.values()) == 0
                if is_clean:
                    at_risk[t] += 1
                for k in labels:
                    cat = CATEGORY.get(k, k)
                    rows.append({
                        "批次": label, "会话": sid, "轮次": t, "错误类别": cat, "错误细类": k,
                        "样本组": "干净样本" if is_clean else "受污染样本",
                        "此前错误数": sum(prior.values()),
                        "此前同类错误数": sum(v for kk, v in prior.items() if CATEGORY.get(kk, kk) == cat),
                        "模型回复节选": r["reply"].replace("\n", " ")[:120],
                    })
                    all_events[t] += 1
                if is_clean and labels:
                    clean_events[t] += 1
                prior.update(labels)
        per_batch[label] = dict(sessions=n_sessions, turns=n_turns, at_risk=at_risk, clean=clean_events, all=all_events)

    with open(prefix + "_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("batch | sessions turns | error rows | clean rows | contaminated rows")
    for label, b in per_batch.items():
        c = [r for r in rows if r["批次"] == label]
        print(label, b["sessions"], b["turns"], len(c), sum(r["样本组"] == "干净样本" for r in c), sum(r["样本组"] != "干净样本" for r in c))
    print("\ncategory x group")
    tab = defaultdict(Counter)
    for r in rows:
        tab[(r["批次"], r["错误类别"])][r["样本组"]] += 1
    for k, v in sorted(tab.items()):
        print(k, dict(v))
    print("\nturns with errors: clean first errors / sessions still clean, vs all errors")
    for label, b in per_batch.items():
        turns = sorted(b["all"])
        print(label, [(t, f"{b['clean'][t]}/{b['at_risk'][t]}", b["all"][t]) for t in turns])


if __name__ == "__main__":
    main()
