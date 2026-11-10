import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.stats.proportion import proportion_confint
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f.replace(" ", "") and "Regular" in f:
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name(); break
arms = [("C0", "不干预"), ("R", "提醒写出算式"), ("F", "本轮给出算式"), ("L", "规则写成列表")]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, dpi=200)
for ax, (suf, title) in zip(axes, (("13", "第 26 轮，N=13（26 是 13 的倍数？）"), ("9", "第 36 轮，N=9（36 是 9 的倍数？）"))):
    xs, ys, lo, hi = [], [], [], []
    for a, lab in arms:
        rs = [json.loads(l) for l in open(f"conf3/{a}_{suf}/calls.jsonl", encoding="utf-8")]
        k = sum("mod17_forgotten" in r["drift"] for r in rs); n = len(rs)
        l, h = proportion_confint(k, n, method="wilson")
        xs.append(lab); ys.append(k / n * 100); lo.append(k / n * 100 - l * 100); hi.append(h * 100 - k / n * 100)
    bars = ax.bar(xs, ys, yerr=[lo, hi], capsize=4, color=["#9aa5b1", "#e0a458", "#4c9f70", "#4c9f70"])
    for b, y, h in zip(bars, ys, hi):
        ax.text(b.get_x() + b.get_width() / 2, y + h + 1.5, f"{y:.0f}%", ha="center", fontsize=9)
    ax.set_title(title, fontsize=10); ax.set_ylim(0, 70); ax.tick_params(axis="x", labelsize=8.5)
axes[0].set_ylabel("错误率（%），每组 100 次")
fig.suptitle("第三次预登记验证：在出错点从外部干预（误差线为 Wilson 95% CI）", fontsize=11)
fig.text(0.5, 0.005, "数据：data/raw_logs.zip 中 conf3/；非目标轮阴性对照均为 0/50", ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig("fig3_validated_interventions.png")
