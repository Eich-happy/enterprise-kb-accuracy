# -*- coding: utf-8 -*-
"""
实验结果自动初判（人工复核为辅）

判分逻辑：
  A/B 类（直接事实 / 跨文档综合）：参考答案的要点词命中率 ≥60% 记 1 分，30–60% 记 0.5 分
  C   类（边界陷阱）：出现别的公司的版本（trap 词）→ 0 分；命中正确表述 → 1 分
  D   类（应拒答）：明确说"无法回答/未找到/未涉及"→ 1 分；出现具体数字等编造痕迹 → 0 分

输出：per_question 明细 + 分类汇总 + 标记为「需人工复核」的条目
用法： python score.py            # 读 results.json
       python score.py results_mock.json
"""
import os, sys, json, re, io
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else "results.json"
data = json.load(open(os.path.join(OUT, src), encoding="utf-8"))

REFUSE = ["无法回答", "未找到", "未涉及", "没有找到", "无法找到", "无此", "不包含",
          "未规定", "未作", "未就此", "没有相关规定", "无法从", "资料中未", "未检索到", "不能回答"]
FABRICATE = [r"\d+\s*%", r"\d+\s*天", r"\d+\s*日", r"\d+\s*万元", r"\d+\s*亿元",
             r"\d+\s*年", r"\d+\s*个"]


def has(text, words):
    t = text.replace(" ", "")
    return [w for w in words if w.replace(" ", "") in t]


def judge(q, ans):
    """返回 (得分, 判定理由, 是否需人工复核)"""
    a = ans.replace(" ", "")
    if a.startswith("【") and ("失败" in a or "mock" in a):
        return None, "无效回答", True

    if q["type"] in ("A", "B"):
        pts = q.get("points", [])
        hit = has(ans, pts)
        rate = len(hit) / len(pts) if pts else 0
        if rate >= 0.6:
            return 1.0, f"命中 {len(hit)}/{len(pts)} 要点", rate < 0.8
        if rate >= 0.3:
            return 0.5, f"部分命中 {len(hit)}/{len(pts)}", True
        return 0.0, f"命中 {len(hit)}/{len(pts)} 要点", True

    if q["type"] == "C":
        tr = has(ans, q.get("trap", []))
        ok = has(ans, q.get("points", []))
        if tr:
            return 0.0, f"张冠李戴（出现他司表述：{'、'.join(tr[:3])}）", False
        if ok:
            return 1.0, f"正确区分主体（{ '、'.join(ok[:2]) }）", True
        return 0.5, "未见陷阱词，但也未明确给出本公司规定，需人工看", True

    if q["type"] == "D":
        rf = has(ans, REFUSE)
        fb = [p for p in FABRICATE if re.search(p, ans)]
        if rf and not fb:
            return 1.0, "明确拒答", False
        if fb:
            return 0.0, f"疑似编造（出现具体数值：{fb[:3]}）", True
        if rf:
            return 0.5, "有拒答表述但同时出现数字，需人工看", True
        return 0.0, "既未拒答也未给数字，需人工看", True

    return None, "未知题型", True


# 载入题库，取回 points / trap 等判分依据（results.json 里只带了 ref_answer）
QMAP = {}
try:
    _qf = os.path.join(OUT, "questions.json")
    QMAP = {q["id"]: q for q in json.load(open(_qf, encoding="utf-8"))}
except Exception as e:
    print("警告：无法读取 questions.json，判分要点将缺失 ->", e)

# 按题聚合（两轮取平均）
qs = {}
for r in data:
    d = qs.setdefault(r["qid"], {"type": r["type"], "question": r["question"],
                                 "ref": r["ref_answer"], "A": [], "B": []})
    src_q = QMAP.get(r["qid"], {})
    for f in ("points", "trap"):
        if f in src_q:
            d[f] = src_q[f]
    d[r["group"]].append(r)

rows = []
for qid in sorted(qs):
    q = qs[qid]
    line = {"qid": qid, "type": q["type"], "question": q["question"][:34],
            "ref": q["ref"][:60]}
    for g in ("A", "B"):
        scores, notes = [], []
        for r in q[g]:
            s, note, need = judge(q, r["answer"])
            if s is not None:
                scores.append(s)
            notes.append(note)
        line[g] = round(sum(scores) / len(scores), 2) if scores else None
        line[g + "_note"] = " | ".join(dict.fromkeys(notes))
        line[g + "_ans"] = q[g][0]["answer"][:200] if q[g] else ""
    rows.append(line)

# 分类汇总
print("=" * 72)
print("逐题初判（满分 1.0，两轮平均）")
print("=" * 72)
print(f"{'题号':<6}{'类':<4}{'题目':<26}{'A基线':<8}{'B优化':<8}差值")
print("-" * 72)
for r in rows:
    a, b = r["A"], r["B"]
    diff = "" if (a is None or b is None) else ("%+.2f" % (b - a))
    print(f"{r['qid']:<6}{r['type']:<4}{r['question']:<26}"
          f"{('%.2f' % a) if a is not None else '-':<8}"
          f"{('%.2f' % b) if b is not None else '-':<8}{diff}")

print()
print("=" * 72)
print("分类汇总")
print("=" * 72)
cat = {}
for r in rows:
    c = cat.setdefault(r["type"], {"A": [], "B": [], "n": 0})
    c["n"] += 1
    if r["A"] is not None:
        c["A"].append(r["A"])
    if r["B"] is not None:
        c["B"].append(r["B"])
print(f"{'类别':<10}{'题数':<6}{'A基线':<10}{'B优化':<10}提升")
print("-" * 52)
tot_a = tot_b = 0
for t in sorted(cat):
    c = cat[t]
    sa = sum(c["A"]) / len(c["A"]) if c["A"] else 0
    sb = sum(c["B"]) / len(c["B"]) if c["B"] else 0
    tot_a += sa
    tot_b += sb
    print(f"{t+'类':<10}{c['n']:<6}{'%.2f' % sa:<10}{'%.2f' % sb:<10}{'%+.2f' % (sb - sa)}")
print("-" * 52)
print(f"{'合计(各类满分4/2/5/3)':<10}{len(rows):<6}{'%.2f' % tot_a:<10}{'%.2f' % tot_b:<10}{'%+.2f' % (tot_b - tot_a)}")

print()
print("=" * 72)
print("需人工复核的条目（自动判分可能不准，请重点看这些）")
print("=" * 72)
for r in rows:
    for g in ("A", "B"):
        note = r[g + "_note"]
        if "需人工" in note or "部分命中" in note:
            print(f"\n[{r['qid']} · {g}组] {r['question']}")
            print(f"  参考答案：{r['ref']}")
            print(f"  模型回答：{r[g+'_ans'][:180]}")
            print(f"  初判：{r[g]} 分（{note}）")

out_path = os.path.join(OUT, "scores.json")
json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n明细已存:", out_path)
