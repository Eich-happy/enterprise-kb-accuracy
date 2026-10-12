# -*- coding: utf-8 -*-
import io, sys, os, re, importlib.util
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

spec = importlib.util.spec_from_file_location('exp', os.path.join(os.path.dirname(__file__), 'run_experiment.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

docs = m.load_docs()
print('文档数:', len(docs), '| 总字数:', sum(len(v) for v in docs.values()))

a = m.chunk_baseline(docs, 500)
b = m.chunk_parent_child(docs)
print()
print('KB-A 基线块数: %d | 平均长度 %.0f 字' % (len(a), sum(len(c['text']) for c in a) / len(a)))
print('KB-B 子块数:   %d | 平均子块 %.0f 字 | 平均父块 %.0f 字'
      % (len(b), sum(len(c['text']) for c in b) / len(b), sum(len(c['parent']) for c in b) / len(b)))

# 各文档块数分布
from collections import Counter
ca = Counter(c['doc'] for c in a)
cb = Counter(c['doc'] for c in b)
print()
print('每份文档块数（A / B）：')
for d in sorted(docs):
    print('  %-4s A=%-4d B=%-4d' % (d, ca[d], cb[d]))

# 检查父块是否真的包含子块（父子关系正确性）
bad = 0
for c in b:
    core = re.sub(r'\s+', '', c['text'])[:60]
    if core and core not in re.sub(r'\s+', '', c['parent']):
        bad += 1
print()
print('父块未包含子块的异常数:', bad, '/', len(b))

print()
print('--- KB-A 样例块（固定切分，注意是否切断条款）---')
print(a[3]['text'][:230])
print()
print('--- KB-B 子块样例（按条款切）---')
print(b[5]['text'][:230])
print()
print('--- KB-B 对应父块（含上下文，用于生成）---')
print(b[5]['parent'][:420])

# 关键检索词在各库的分布（确认干扰项真的混进来了）
print()
for kw in ['嘉益', '西宁']:
    na = sum(1 for c in a if kw in c['text'])
    nb = sum(1 for c in b if kw in c['text'])
    print('干扰项 %s: KB-A 命中块 %d | KB-B 命中块 %d' % (kw, na, nb))
