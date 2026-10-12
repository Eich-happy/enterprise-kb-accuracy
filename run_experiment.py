# -*- coding: utf-8 -*-
"""
作品 2 · 企业知识库准确率对照实验

对照设计（三组，语料完全相同，只有分块 / 检索 / 提示词不同）：
  KB-A 基线：固定 500 字分块 | 纯向量检索 TopK=3 | 最简提示词
  KB-B 优化：父子分块(父650/子200) | 向量+BM25混合 TopK=6 → Rerank→3 | 结构化提示词
  KB-C 主体过滤：KB-B 的全部配置 + 检索层主体过滤（allowed={'A'}，只让目标公司文档进入上下文）
三边共用：同一个 Embedding 模型(bge-m3)、同一个 LLM(Qwen2.5-7B)、temperature=0

用法：
  set SILICONFLOW_API_KEY=sk-xxx
  python run_experiment.py            # 跑全量 20题 x 3组 x 2轮 = 120 次
  python run_experiment.py --dry      # 只做分块与检索，不调 LLM（自检用）
  python run_experiment.py --rounds 1 # 只跑 1 轮（60 次）
  python run_experiment.py --mock     # 离线冒烟，不调 API，验证流程可通
"""
import os, sys, json, re, time, io, argparse, math

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import requests
import jieba
from rank_bm25 import BM25Okapi

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus", "text")
OUT = BASE
os.makedirs(OUT, exist_ok=True)

API = "https://api.siliconflow.cn/v1"
EMB_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

COMPANY = "百川畅银"
OTHERS = ["嘉益股份", "西宁特钢"]

# ---------------- 提示词 ----------------
PROMPT_A = """请仅根据以下资料回答问题。

资料：
{context}

问题：{question}

回答："""

PROMPT_B = """你是河南百川畅银环保能源股份有限公司（下称「本公司」）内部制度问答助手。

【回答规则】
1. 只能依据下方《参考资料》作答，不得使用你自己的知识或推测。
2. 参考资料中可能同时存在其他公司（嘉益股份、西宁特钢）的同名制度。若检索到的内容不属于本公司，该内容不得作为答案依据。
3. 本公司制度未规定的事项，必须明确回答「本公司现有制度未就此作出规定」，不得引用其他公司的制度来填补。
4. 参考资料中没有答案时，必须明确说明「依据现有制度资料无法回答该问题」，禁止猜测或编造任何数字、期限、条款号。
5. 回答时注明所依据的制度名称与条款号。
6. 回答控制在 150 字以内，直接给结论。

【参考资料】
{context}

【问题】
{question}

【回答】"""


# ---------------- 加载语料 ----------------
def load_docs():
    docs = {}
    for fn in sorted(os.listdir(CORPUS)):
        if not fn.endswith(".txt"):
            continue
        did = fn[:-4]
        raw = open(os.path.join(CORPUS, fn), encoding="utf-8").read()
        docs[did] = re.sub(r"[ \t]+", " ", raw)
    return docs


# ---------------- 分块 ----------------
CLAUSE_RE = re.compile(r"(第[一二三四五六七八九十百零〇\d]+条)")


def split_clauses(text):
    """按 第X条 切成条款单元；切不动则退回按段落"""
    parts = CLAUSE_RE.split(text)
    if len(parts) < 3:
        return [p.strip() for p in re.split(r"\n{1,}", text) if len(p.strip()) > 20]
    units, buf = [], parts[0].strip()
    for i in range(1, len(parts), 2):
        head, body = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
        seg = (buf + head + body).strip()
        if len(seg) > 15:
            units.append(seg)
        buf = ""
    if buf.strip() and len(buf.strip()) > 15:
        units.append(buf.strip())
    return [u for u in units if len(u) > 15]


def chunk_baseline(docs, size=500):
    """KB-A：固定字数切分，不按条款、不重叠"""
    out = []
    for did, txt in docs.items():
        t = re.sub(r"\s+", "", txt)
        for i in range(0, len(t), size):
            seg = t[i:i + size]
            if len(seg) >= 60:
                out.append({"id": f"{did}#A{len(out)}", "doc": did, "text": seg, "parent": seg})
    return out


def chunk_parent_child(docs, parent_size=650, child_size=200, overlap=50):
    """KB-B：子块=单条款(用于检索)，父块=以子块为中心向两侧扩展到 parent_size(用于生成)"""
    out = []
    for did, txt in docs.items():
        flat = re.sub(r"\s+", "", txt)
        units = split_clauses(flat)
        if not units:
            units = [flat[i:i + child_size] for i in range(0, len(flat), child_size)]
        # 记录每个 unit 在 flat 中的起始位置
        pos, spans = 0, []
        for u in units:
            k = flat.find(u, pos)
            if k < 0:
                k = pos
            spans.append((k, k + len(u)))
            pos = k + len(u)
        for idx, u in enumerate(units):
            child = u if len(u) <= child_size * 2 else u[:child_size * 2]
            # 父块：从本条款起向两侧扩展
            s, e = spans[idx]
            while e - s < parent_size:
                grew = False
                if idx > 0 and s > spans[idx - 1][0]:
                    s = spans[idx - 1][0]
                    grew = True
                if idx + 1 < len(units) and e < spans[idx + 1][1]:
                    e = spans[idx + 1][1]
                    grew = True
                if not grew:
                    break
            # 再向外补 overlap 个字符保证衔接
            s = max(0, s - overlap // 2)
            e = min(len(flat), e + overlap // 2)
            parent = flat[s:e]
            out.append({"id": f"{did}#B{len(out)}", "doc": did, "text": child, "parent": parent})
    return out


# ---------------- API ----------------
MOCK = False  # 离线冒烟测试开关：不调 API，用随机向量/假回答跑通全流程


def _headers(key):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def embed(texts, key, batch=25, retry=3):
    if MOCK:
        rng = np.random.RandomState(42)
        v = rng.rand(len(texts), 1024).astype("float32")
        # 让相同文本得到相同向量，便于观察流程
        return v + np.arange(len(texts), dtype="float32").reshape(-1, 1) * 1e-6
    vecs = []
    for i in range(0, len(texts), batch):
        sub = texts[i:i + batch]
        for attempt in range(retry):
            try:
                r = requests.post(API + "/embeddings", headers=_headers(key),
                                  json={"model": EMB_MODEL, "input": sub, "encoding_format": "float"},
                                  timeout=90)
                if r.status_code == 200:
                    d = sorted(r.json()["data"], key=lambda x: x["index"])
                    vecs.extend([x["embedding"] for x in d])
                    break
                print(f"  embed {r.status_code}: {r.text[:150]}")
            except Exception as e:
                print(f"  embed err: {e}")
            time.sleep(2 + attempt * 2)
        else:
            raise RuntimeError("embedding 连续失败")
        if (i // batch) % 5 == 0:
            print(f"  embedding {len(vecs)}/{len(texts)}")
    return np.array(vecs, dtype="float32")


def rerank(query, docs_text, key, top_n=3, retry=3):
    if MOCK:
        return [(i, 1.0) for i in range(min(top_n, len(docs_text)))]
    for attempt in range(retry):
        try:
            r = requests.post(API + "/rerank", headers=_headers(key),
                              json={"model": RERANK_MODEL, "query": query,
                                    "documents": docs_text, "top_n": top_n,
                                    "return_documents": False}, timeout=90)
            if r.status_code == 200:
                res = r.json().get("results", [])
                return [(x["index"], x["relevance_score"]) for x in res]
            print(f"  rerank {r.status_code}: {r.text[:150]}")
        except Exception as e:
            print(f"  rerank err: {e}")
        time.sleep(2 + attempt * 2)
    return [(i, 0.0) for i in range(min(top_n, len(docs_text)))]


def chat(prompt, key, temperature=0.0, max_tokens=500, retry=3):
    if MOCK:
        return "【mock 回答】prompt 长度 %d 字" % len(prompt)
    for attempt in range(retry):
        try:
            r = requests.post(API + "/chat/completions", headers=_headers(key),
                              json={"model": LLM_MODEL,
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": temperature, "max_tokens": max_tokens},
                              timeout=120)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            print(f"  chat {r.status_code}: {r.text[:150]}")
        except Exception as e:
            print(f"  chat err: {e}")
        time.sleep(3 + attempt * 3)
    return "【API 调用失败】"


# ---------------- 检索 ----------------
def norm(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1e-9
    return v / n


def retrieve(q, chunks, emb_mat, key, mode, bm25=None, topk=3, cand=6, allowed=None):
    """mode: 'vector' | 'hybrid'（hybrid = 向量+BM25 融合后 rerank）
    allowed: 允许的 doc 前缀集合，如 {'A'} —— 用于 KB-C 的检索层主体过滤"""
    qv = embed([q], key)[0:1]
    qv = norm(qv)
    sims = (emb_mat @ qv[0]).astype("float32")

    if allowed:
        mask = np.array([any(c["doc"].startswith(p) for p in allowed) for c in chunks])
        # 不允许的块直接打出极低的相似分（等同过滤，且不破坏索引对应关系）
        sims = np.where(mask, sims, -10.0).astype("float32")

    if mode == "vector":
        idx = np.argsort(-sims)[:topk]
        return [(int(i), float(sims[i])) for i in idx], [chunks[int(i)]["parent"] for i in idx]

    # hybrid：向量候选 + BM25 候选 合并去重
    bm = bm25.get_scores(list(jieba.cut(q)))
    bm = (bm - bm.min()) / (bm.max() - bm.min() + 1e-9)
    if allowed:
        bm = np.where(
            np.array([any(c["doc"].startswith(p) for p in allowed) for c in chunks]),
            bm, -10.0)
    fused = 0.5 * ((sims - sims.min()) / (sims.max() - sims.min() + 1e-9)) + 0.5 * bm
    cand_idx = list(np.argsort(-fused)[:cand])
    # rerank 用父块文本
    rr = rerank(q, [chunks[i]["parent"] for i in cand_idx], key, top_n=topk)
    picked = [cand_idx[i] for i, _ in rr if i < len(cand_idx)]
    if not picked:
        picked = cand_idx[:topk]
    return [(int(i), float(fused[i])) for i in picked], [chunks[int(i)]["parent"] for i in picked]


def build_context(parents, chunks_meta):
    buf = []
    for p, meta in zip(parents, chunks_meta):
        buf.append(f"[{meta}] {p}")
    return "\n\n".join(buf)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("SILICONFLOW_API_KEY", ""))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--dry", action="store_true", help="只做分块与检索，不调 LLM")
    ap.add_argument("--mock", action="store_true", help="离线冒烟：不调 API，验证流程可跑通")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    args = ap.parse_args()

    global MOCK
    MOCK = args.mock
    key = (args.key or "").strip()
    if not key and not MOCK:
        print("缺少 API key。用法： set SILICONFLOW_API_KEY=sk-xxx  或 --key sk-xxx")
        print("（想先验证流程可跑通，加 --mock）")
        sys.exit(1)
    if MOCK:
        print("!!! MOCK 模式：不会调用任何 API，结果仅供流程验证 !!!")

    print("=" * 60)
    print("加载语料 ...")
    docs = load_docs()
    print(f"  {len(docs)} 份文档")

    print("分块 ...")
    ch_a = chunk_baseline(docs, size=500)
    ch_b = chunk_parent_child(docs)
    print(f"  KB-A 基线块数: {len(ch_a)}")
    print(f"  KB-B 父子块数: {len(ch_b)}（子块检索 / 父块生成）")

    # embedding（子块用于检索；A 组本身就是检索块）
    suffix = "_mock" if MOCK else ""
    cache_a = os.path.join(OUT, f"emb_A{suffix}.npy")
    cache_b = os.path.join(OUT, f"emb_B{suffix}.npy")
    print("向量化 KB-A ...")
    if os.path.exists(cache_a):
        ea = np.load(cache_a)
        print(f"  命中缓存 {ea.shape}")
    else:
        ea = norm(embed([c["text"] for c in ch_a], key))
        np.save(cache_a, ea)
    print("向量化 KB-B ...")
    if os.path.exists(cache_b):
        eb = np.load(cache_b)
        print(f"  命中缓存 {eb.shape}")
    else:
        eb = norm(embed([c["text"] for c in ch_b], key))
        np.save(cache_b, eb)

    bm25_b = BM25Okapi([list(jieba.cut(c["text"])) for c in ch_b])

    qs = json.load(open(os.path.join(OUT, "questions.json"), encoding="utf-8"))
    print(f"\n题库 {len(qs)} 题")

    res_path = os.path.join(OUT, f"results{suffix}.json")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else []

    done = {(r["qid"], r["group"], r["round"]) for r in results}

    todo = qs if not args.limit else qs[:args.limit]

    for rnd in range(1, args.rounds + 1):
        for q in todo:
            for grp in ("A", "B", "C"):
                if (q["id"], grp, rnd) in done:
                    continue
                if grp == "A":
                    picked, parents = retrieve(q["question"], ch_a, ea, key, "vector", topk=3)
                    ctx = build_context(parents, [ch_a[i]["doc"] for i, _ in picked])
                    prompt = PROMPT_A.format(context=ctx, question=q["question"])
                    srcs = [ch_a[i]["doc"] for i, _ in picked]
                elif grp == "B":
                    picked, parents = retrieve(q["question"], ch_b, eb, key, "hybrid",
                                               bm25=bm25_b, topk=3, cand=6)
                    ctx = build_context(parents, [ch_b[i]["doc"] for i, _ in picked])
                    prompt = PROMPT_B.format(context=ctx, question=q["question"])
                    srcs = [ch_b[i]["doc"] for i, _ in picked]
                else:
                    # KB-C：与 KB-B 唯一差别 —— 检索层只保留本公司（A 组）文档
                    picked, parents = retrieve(q["question"], ch_b, eb, key, "hybrid",
                                               bm25=bm25_b, topk=3, cand=6,
                                               allowed={"A"})
                    ctx = build_context(parents, [ch_b[i]["doc"] for i, _ in picked])
                    prompt = PROMPT_B.format(context=ctx, question=q["question"])
                    srcs = [ch_b[i]["doc"] for i, _ in picked]

                if args.dry:
                    ans = "【dry-run 未调用 LLM】"
                else:
                    ans = chat(prompt, key)

                results.append({
                    "qid": q["id"], "type": q["type"], "group": grp, "round": rnd,
                    "question": q["question"], "answer": ans,
                    "sources": srcs,
                    "ref_answer": q["answer"],
                })
                json.dump(results, open(res_path, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
                tail = ans.replace("\n", " ")[:70]
                print(f"  {q['id']} [{grp}] r{rnd} <- {','.join(results[-1]['sources'])}")
                print(f"      {tail}")
                if not args.dry:
                    time.sleep(0.4)

    print("\n完成。结果:", res_path, f"（{len(results)} 条）")


if __name__ == "__main__":
    main()
