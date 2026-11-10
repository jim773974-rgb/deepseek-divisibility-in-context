"""Count saved periodic-rule error records (each record once) and check whether any was
labeled wrong only because the note was longer than 8 characters.

Run from a directory holding the scripts and unpacked raw logs.
"""
import glob
import json
import os

R = os.path.dirname(os.path.abspath(__file__))
LEDGER = ["pilot1", "combined", "A_p17_old", "B_p17_new", "C_p13", "D_p19", "E_p23"] + \
    [d for d in sorted(glob.glob(os.path.join(R, "p[1-4]", "*"))) for d in [os.path.relpath(d, R)]] + \
    [os.path.relpath(d, R) for d in sorted(glob.glob(os.path.join(R, "conf", "*")))]


def note_of(reply):
    try:
        n = json.loads(reply.strip()).get("note")
        return n if isinstance(n, str) else ""
    except (ValueError, AttributeError):
        return ""


total = length_only = 0
for d in LEDGER:
    for f in sorted(glob.glob(os.path.join(R, d, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if "mod17_forgotten" in r.get("drift", []):
                total += 1
                length_only += note_of(r["reply"]).startswith("盘点")
print(f"记账任务（探索性实验和验证实验的完整日志）：周期规则错误记录 {total} 条，其中以“盘点”开头、仅因超过 8 个字符被判错的 {length_only} 条")

fork_wrong = 0
for f in sorted(glob.glob(os.path.join(R, "fork*", "*.jsonl"))):
    if os.path.basename(f).startswith("probe"):
        continue  # 提问组（含作废的第一版提问）不是记账判断，不计入
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        fork_wrong += r.get("note_class") == "已核"
print(f"第 34 轮重跑（不含提问组）：写成“已核”的记录 {fork_wrong} 条（按 note 开头判定，不涉及长度规则）")

code = 0
for f in sorted(glob.glob(os.path.join(R, "code1", "run_*.jsonl"))):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        code += any(k.startswith("audit") for k in r.get("drift", []))
print(f"代码任务：违反 audit 规则的记录 {code} 条（每条记录只计一次，不涉及长度规则）")
