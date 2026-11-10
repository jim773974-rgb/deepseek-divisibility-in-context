"""Realistic long task: build a small Python inventory library over many turns.

Each turn asks for one new function. Standing conventions are stated once in the system prompt.
Every reply is checked with the ast module, so judging is automatic.

Usage:
  code_task.py selftest
  code_task.py run --runs 12 --turns 30 --out DIR
  code_task.py report DIR
"""
import argparse
import ast
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_pilot as base

SYSTEM = """你在协助开发一个 Python 库存管理库。整个会话都要遵守下面的团队规范：
1. 每次回复只包含一个 Python 代码块，代码块外不能有任何文字。
2. 代码块里只写本轮要求的那一个函数，不要重复之前的函数，也不要写示例或测试。
3. 函数必须有完整的类型注解（所有参数和返回值）。
4. 函数必须有 docstring，且 docstring 的第一行必须以中文动词开头并以句号结尾。
5. 不得使用 print。
6. 不得使用 global 语句。
7. 函数体内出现的错误必须抛出 InventoryError（已在别处定义），不得抛出其他异常类型。
8. 函数名必须以 inv_ 开头并使用 snake_case。
9. 不得使用 lambda。
10. 函数体不超过 15 行（不含 docstring）。
11. 如果轮次是 13 的倍数，函数名必须改用 audit_ 开头，且 docstring 第一行必须以“审计”开头。
"""

TASKS = [
    "按货品编号查询库存数量", "入库指定数量", "出库指定数量，库存不足时报错", "返回库存为零的货品编号列表",
    "计算全部库存总量", "按前缀筛选货品", "合并两个库存字典", "把库存数量全部乘以系数",
    "找出库存最多的货品", "找出库存最少的货品", "按数量降序返回货品列表", "统计库存低于阈值的货品数量",
    "校验货品编号格式是否合法", "把库存字典转换成 CSV 字符串", "从 CSV 字符串解析库存字典", "计算两次盘点之间的差异",
    "批量入库", "批量出库，任一失败则全部不生效", "按百分比预留库存", "返回库存分布的直方图字典",
    "计算平均库存", "计算库存中位数", "按货品分类汇总", "重命名货品编号",
    "删除库存为零的货品", "把库存拆分到多个仓库", "把多个仓库的库存汇总", "检测重复的货品编号",
    "为库存生成简单校验和", "按入库时间清理过期批次", "计算批次加权平均成本", "估算可售天数",
    "生成补货建议", "按供应商分组", "标记滞销货品", "计算周转率",
    "按区域汇总库存", "限制单货品最大库存", "返回最近一次变动的货品", "生成库存快照",
]
FILLER = "（背景：本库已在线上运行，近期变更记录较多，请专注当前函数。）"


def user_msg(n):
    return f"第 {n} 轮：请实现函数，功能是：{TASKS[(n - 1) % len(TASKS)]}。{FILLER}"


def check(reply, n):
    bad, tech = [], []
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, flags=re.S)
    outside = re.sub(r"```(?:python|py)?\s*\n.*?```", "", reply, flags=re.S).strip()
    if len(m) != 1:
        bad.append("not_one_code_block")
        return {"drift": bad, "slips": ["no_parse"] if not m else tech}
    if outside:
        bad.append("text_outside_code")
    try:
        tree = ast.parse(m[0])
    except SyntaxError:
        return {"drift": bad, "slips": ["syntax_error"]}
    funcs = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(funcs) != 1 or len(tree.body) != len(funcs):
        bad.append("extra_top_level_code")
    if not funcs:
        return {"drift": bad, "slips": tech}
    f = funcs[0]
    audit_turn = n % 13 == 0
    if audit_turn:
        if not f.name.startswith("audit_"):
            bad.append("audit_name_forgotten")
    elif not f.name.startswith("inv_"):
        bad.append("inv_prefix_rule")
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", f.name):
        bad.append("snake_case")
    args = f.args.posonlyargs + f.args.args + f.args.kwonlyargs
    if any(a.annotation is None for a in args if a.arg != "self") or f.returns is None:
        bad.append("missing_type_hints")
    doc = ast.get_docstring(f)
    if not doc:
        bad.append("missing_docstring")
    else:
        first = doc.strip().splitlines()[0].strip()
        if not first.endswith("。"):
            bad.append("doc_no_period")
        if audit_turn and not first.startswith("审计"):
            bad.append("audit_doc_forgotten")
        if not re.match(r"[一-鿿]", first):
            bad.append("doc_not_chinese")
    body = f.body[1:] if doc else f.body
    if body:
        if body[-1].end_lineno - body[0].lineno + 1 > 15:
            bad.append("too_long")
    for x in ast.walk(f):
        if isinstance(x, ast.Call) and getattr(x.func, "id", "") == "print":
            bad.append("uses_print")
        if isinstance(x, ast.Global):
            bad.append("uses_global")
        if isinstance(x, ast.Lambda):
            bad.append("uses_lambda")
        if isinstance(x, ast.Raise) and x.exc is not None:
            exc = x.exc.func if isinstance(x.exc, ast.Call) else x.exc
            if getattr(exc, "id", None) != "InventoryError":
                bad.append("wrong_exception")
    return {"drift": sorted(set(bad)), "slips": tech}


def selftest():
    good = '```python\ndef inv_q(a: int) -> int:\n    """查询库存。"""\n    if a < 0:\n        raise InventoryError("x")\n    return a\n```'
    assert check(good, 1) == {"drift": [], "slips": []}, check(good, 1)
    audit = good.replace("inv_q", "audit_q").replace("查询库存", "审计库存")
    assert check(audit, 13) == {"drift": [], "slips": []}
    def has(text, n, label):
        r = check(text, n)
        assert label in r["drift"], (label, r)
    has(good, 13, "audit_name_forgotten")
    has(good.replace("inv_q", "audit_q"), 13, "audit_doc_forgotten")
    has(audit, 1, "inv_prefix_rule")
    has(good.replace("a: int", "a"), 1, "missing_type_hints")
    has(good.replace('"""查询库存。"""', '"""查询库存"""'), 1, "doc_no_period")
    has(good.replace('"""查询库存。"""', '"""Query stock."""'), 1, "doc_not_chinese")
    has(good.replace('    """查询库存。"""\n', ""), 1, "missing_docstring")
    has(good.replace("InventoryError", "ValueError"), 1, "wrong_exception")
    has(good.replace("return a", "print(a)\n    return a"), 1, "uses_print")
    has(good.replace("return a", "global z\n    return a"), 1, "uses_global")
    has(good.replace("return a", "return (lambda x: x)(a)"), 1, "uses_lambda")
    has("好的\n" + good, 1, "text_outside_code")
    has(good + "\n" + good, 1, "not_one_code_block")
    has(good.replace("```\n", "```\n", 1).replace("def inv_q", "x = 1\ndef inv_q"), 1, "extra_top_level_code")
    long_body = "\n".join(f"    v{i} = {i}" for i in range(16))
    has(good.replace("    if a < 0:", long_body + "\n    if a < 0:"), 1, "too_long")
    print("selftest ok; tasks", len(TASKS))


def run_one(idx, a):
    path = os.path.join(a.out, f"run_{idx:02d}.jsonl")
    messages = [{"role": "system", "content": SYSTEM}]
    with open(path, "w", encoding="utf-8") as f:
        for n in range(1, a.turns + 1):
            if n % 8 == 1 and float(base.balance()) < a.min_balance:
                f.write(json.dumps({"stopped": "balance", "turn": n}) + "\n")
                return
            messages.append({"role": "user", "content": user_msg(n)})
            try:
                resp = base.chat(messages, a.model, a.effort)
            except RuntimeError as e:
                f.write(json.dumps({"stopped": str(e), "turn": n}, ensure_ascii=False) + "\n")
                return
            msg = resp["choices"][0]["message"]
            reply = msg.get("content") or ""
            r = check(reply, n)
            messages.append({"role": "assistant", "content": reply})
            f.write(json.dumps({"turn": n, "reply": reply, "reasoning": msg.get("reasoning_content") or "",
                                "drift": r["drift"], "slips": r["slips"], "usage": resp.get("usage", {})},
                               ensure_ascii=False) + "\n")
            f.flush()


def run(a):
    os.makedirs(a.out, exist_ok=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda i: run_one(i, a), range(a.runs)))
    json.dump(vars(a), open(os.path.join(a.out, "meta.json"), "w"))


def report(d):
    sess = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("run_"):
            sess.append([json.loads(l) for l in open(os.path.join(d, fn), encoding="utf-8")])
    ok = [[r for r in s if "reply" in r] for s in sess]
    print("sessions", len(sess), "turns", sum(len(s) for s in ok), "stopped", sum(any("stopped" in r for r in s) for s in sess))
    per = {}
    kinds = {}
    for s in ok:
        for r in s:
            per.setdefault(r["turn"], [0, 0])
            per[r["turn"]][1] += 1
            per[r["turn"]][0] += bool(r["drift"])
            for k in r["drift"]:
                kinds.setdefault(k, []).append(r["turn"])
    print("turn fail/n:", " ".join(f"{t}:{v[0]}/{v[1]}" for t, v in sorted(per.items())))
    for k, v in sorted(kinds.items(), key=lambda x: -len(x[1])):
        print(f"  {k}: {len(v)} turns={sorted(v)[:24]}")
    for t in (13, 26, 39):
        rows = [r for s in ok for r in s if r["turn"] == t]
        if rows:
            bad = sum(any(k.startswith("audit") for k in r["drift"]) for r in rows)
            print(f"  audit rule at {t}: {bad}/{len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--runs", type=int, default=12)
    r.add_argument("--turns", type=int, default=30)
    r.add_argument("--model", default="deepseek-flash")
    r.add_argument("--effort", default="low")
    r.add_argument("--parallel", type=int, default=6)
    r.add_argument("--out", required=True)
    r.add_argument("--min-balance", dest="min_balance", type=float, default=1.0)
    rp = sub.add_parser("report")
    rp.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        run(a)
    else:
        report(a.dir)
