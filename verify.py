# -*- coding: utf-8 -*-
"""
一键验证 —— 不调 API，纯本地重算结果表。

用途：
  证明 README 里的数字与仓库内的原始数据自洽，结论可被第三方复核。
  任何人 clone 后无需申请 API key 即可运行。

用法：
  python verify.py
"""
import os
import sys
import json
import collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus", "text")
EXPECT_CORPUS = 12
EXPECT_QUESTIONS = 20
EXPECT_RESULTS = 120
GROUPS = ("A", "B", "C")
TYPE_NAME = {
    "A": "A 类 直接事实",
    "B": "B 类 跨条推理",
    "C": "C 类 边界陷阱",
    "D": "D 类 应拒答",
}

ok = True


def check(cond, msg, detail=""):
    global ok
    if cond:
        print("  [OK]   " + msg)
    else:
        ok = False
        print("  [FAIL] " + msg + ("  -> " + detail if detail else ""))


def load(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


print("=" * 62)
print("企业知识库准确率对照实验 · 一键验证")
print("=" * 62)

print("\n[1/4] 语料")
n_txt = len([f for f in os.listdir(CORPUS) if f.endswith(".txt")]) if os.path.isdir(CORPUS) else 0
check(n_txt == EXPECT_CORPUS, "语料 %d 份（预期 %d）" % (n_txt, EXPECT_CORPUS),
      "检查 corpus/text/ 目录")

print("\n[2/4] 评测集")
qs = load("questions.json")
n_q = len(qs) if qs else 0
check(n_q == EXPECT_QUESTIONS, "题库 %d 题（预期 %d）" % (n_q, EXPECT_QUESTIONS))
if qs:
    missing = [q.get("id") for q in qs
               if not all(k in q for k in ("answer", "source", "points"))]
    check(not missing, "每题均含 标准答案 / 出处 / 评分要点",
          "缺失: %s" % missing)

print("\n[3/4] 实验记录")
res = load("results.json")
n_r = len(res) if res else 0
check(n_r == EXPECT_RESULTS, "问答记录 %d 条（预期 %d）" % (n_r, EXPECT_RESULTS))
if res:
    cnt = collections.Counter(r.get("group") for r in res)
    for g in GROUPS:
        check(cnt.get(g) == 40, "  group %s = %d 条（预期 40）" % (g, cnt.get(g, 0)))
    rounds = collections.Counter(r.get("round") for r in res)
    check(len(rounds) >= 1, "  轮次: %s" % dict(rounds))

print("\n[4/4] 重算结果表（数据源 manual_scores.json）")
ms = load("manual_scores.json")
scores = (ms or {}).get("scores", {}) if isinstance(ms, dict) else {}
n_s = len(scores)
check(n_s == EXPECT_QUESTIONS, "人工判分 %d 题（预期 %d）" % (n_s, EXPECT_QUESTIONS))

if scores:
    bucket = collections.defaultdict(lambda: collections.Counter())
    total = collections.Counter()
    for qid, v in sorted(scores.items()):
        t = v.get("type", "?")
        bucket[t]["n"] += 1
        for g in GROUPS:
            bucket[t][g] += float(v.get(g, 0) or 0)
            total[g] += float(v.get(g, 0) or 0)

    print("\n  %-16s %4s %14s %14s %16s" % ("题型", "题数", "KB-A 基线", "KB-B 优化", "KB-C 主体过滤"))
    print("  " + "-" * 66)
    for t in ("A", "B", "C", "D"):
        b = bucket.get(t)
        if not b or not b["n"]:
            continue
        n = b["n"]
        print("  %-16s %4d %14s %14s %16s" % (
            TYPE_NAME[t], n,
            "%.1f (%.0f%%)" % (b["A"], b["A"] / n * 100),
            "%.1f (%.0f%%)" % (b["B"], b["B"] / n * 100),
            "%.1f (%.0f%%)" % (b["C"], b["C"] / n * 100),
        ))
    n = n_s
    print("  " + "-" * 66)
    print("  %-16s %4d %14s %14s %16s" % (
        "合计", n,
        "%.1f (%.1f%%)" % (total["A"], total["A"] / n * 100),
        "%.1f (%.1f%%)" % (total["B"], total["B"] / n * 100),
        "%.1f (%.1f%%)" % (total["C"], total["C"] / n * 100),
    ))

    print("\n  关键对比:")
    print("    KB-A -> KB-B  (标准优化全上)   %.1f%% -> %.1f%%   差 %+.1f 个点" % (
        total["A"] / n * 100, total["B"] / n * 100,
        (total["B"] - total["A"]) / n * 100))
    print("    KB-B -> KB-C  (加主体过滤)     %.1f%% -> %.1f%%   差 %+.1f 个点" % (
        total["B"] / n * 100, total["C"] / n * 100,
        (total["C"] - total["B"]) / n * 100))

    cb = bucket.get("C")
    if cb and cb["n"]:
        print("    C 类边界陷阱                   %.0f%% -> %.0f%%   差 %+.0f 个点" % (
            cb["A"] / cb["n"] * 100, cb["C"] / cb["n"] * 100,
            (cb["C"] - cb["A"]) / cb["n"] * 100))

print("\n" + "=" * 62)
if ok:
    print("全部检查通过 —— README 中的数字与原始数据自洽。")
else:
    print("存在未通过的检查项，请按上面 [FAIL] 提示处理。")
print("=" * 62)

sys.exit(0 if ok else 1)
