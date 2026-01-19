# Locomo 评估完整报告

**生成时间**: 2026-01-19 23:21:09

---

## 1. 总体概览

| 指标 | 数值 |
|------|------|
| 用户对数量 | 10 |
| 总问题数 | 1540 |
| 正确数 | 1238 |
| 错误数 | 302 |
| **正确率 (Accuracy)** | **80.39%** |

---

## 2. 记忆检索 Recall@K

衡量检索到的前 K 个记忆是否包含回答问题所需的全部 evidence。

| K | 命中数 | Recall@K |
|---|--------|----------|
| 1 | 806 | 52.47% |
| 3 | 1067 | 69.47% |
| 5 | 1177 | 76.63% |
| 10 | 1208 | 78.65% |
| 12 | 1208 | 78.65% |

*有效问题数（有 evidence 标注）: 1536*

### 各用户对 Recall@K

| 用户对 | 有效问题数 | R@1 | R@3 | R@5 | R@10 |
|--------|-----------|-----|-----|-----|------|
| 0__Caroline__Melanie | 150 | 50.0% | 70.0% | 76.0% | 77.3% |
| 1__Jon__Gina | 81 | 50.6% | 69.1% | 77.8% | 80.2% |
| 2__John__Maria | 152 | 55.3% | 73.7% | 82.9% | 85.5% |
| 3__Joanna__Nate | 199 | 50.3% | 63.8% | 72.4% | 74.9% |
| 4__Tim__John | 178 | 58.4% | 71.3% | 79.2% | 79.8% |
| 5__Audrey__Andrew | 123 | 42.3% | 57.7% | 65.9% | 67.5% |
| 6__James__John | 150 | 66.7% | 84.0% | 88.7% | 90.0% |
| 7__Deborah__Jolene | 191 | 59.7% | 73.8% | 80.1% | 82.2% |
| 8__Evan__Sam | 156 | 41.7% | 64.7% | 71.8% | 73.7% |
| 9__Calvin__Dave | 156 | 45.5% | 64.7% | 70.5% | 74.4% |

---

## 3. 错误类型分布

对错误进行分类分析，识别系统瓶颈。

| 错误类型 | 数量 | 占错误比例 | 占总体比例 |
|----------|------|-----------|-----------|
| 记忆抽取问题 | 0 | 0.0% | 0.0% |
| 检索问题 | 0 | 0.0% | 0.0% |
| 推理/其他问题 | 302 | 100.0% | 19.6% |

**错误类型说明**:
- **记忆抽取问题**: 关键信息未被抽取成记忆，导致无法回答
- **检索问题**: 记忆已抽取但检索时未被召回
- **推理/其他问题**: 记忆已检索到但 LLM 回答错误（理解/推理问题）

---

## 4. 分类别统计 (Category)

按问题类型分析正确率。

| Category | 总数 | 正确 | 错误 | 正确率 |
|----------|------|------|------|--------|
| Single-hop | 841 | 765 | 76 | 91.0% |
| Temporal | 321 | 247 | 74 | 76.9% |
| Multi-hop | 282 | 171 | 111 | 60.6% |
| Open-domain | 96 | 55 | 41 | 57.3% |

---

## 5. 各用户对统计

| 用户对 | 总数 | 正确 | 错误 | 正确率 | 抽取错误 | 检索错误 | 推理错误 |
|--------|------|------|------|--------|---------|---------|---------|
| 0__Caroline__Melanie | 152 | 123 | 29 | 80.9% | 0 | 0 | 29 |
| 1__Jon__Gina | 81 | 72 | 9 | 88.9% | 0 | 0 | 9 |
| 2__John__Maria | 152 | 126 | 26 | 82.9% | 0 | 0 | 26 |
| 3__Joanna__Nate | 199 | 147 | 52 | 73.9% | 0 | 0 | 52 |
| 4__Tim__John | 178 | 134 | 44 | 75.3% | 0 | 0 | 44 |
| 5__Audrey__Andrew | 123 | 95 | 28 | 77.2% | 0 | 0 | 28 |
| 6__James__John | 150 | 131 | 19 | 87.3% | 0 | 0 | 19 |
| 7__Deborah__Jolene | 191 | 156 | 35 | 81.7% | 0 | 0 | 35 |
| 8__Evan__Sam | 156 | 123 | 33 | 78.8% | 0 | 0 | 33 |
| 9__Calvin__Dave | 158 | 131 | 27 | 82.9% | 0 | 0 | 27 |

---

## 6. Badcase 样例

### 6.1 记忆抽取问题样例

*无*

### 6.2 检索问题样例

*无*

### 6.3 推理问题样例

**[0__Caroline__Melanie] Q5: When did Caroline have a picnic?**
- 预期答案: The week before 6 July 2023
- 预测答案: There is no mention of Caroline having a picnic in the provided conversations.
- Judge 评价: The ground truth answer provides a specific timeframe ('The week before 6 July 2023'), while the predicted answer denies any mention of the event. The...

**[0__Caroline__Melanie] Q19: What instruments does Melanie play?**
- 预期答案: clarinet and violin
- 预测答案: Melanie plays the clarinet.
- Judge 评价: The ground truth answer states that Melanie plays clarinet AND violin. The predicted answer only mentions clarinet, omitting the key fact about violin...

**[0__Caroline__Melanie] Q25: What items has Melanie bought?**
- 预期答案: Figurines, shoes
- 预测答案: Melanie has bought running shoes (mentioned on 2023-07-12).
- Judge 评价: The predicted answer only mentions shoes (specifically running shoes), but fails to mention figurines, which is a key entity from the ground truth ans...

---

## 7. 结论与建议

### 主要发现

- **LLM 推理能力有限** (100.0% 的错误)：可尝试更强的模型或优化 prompt

### Recall 分析

- Recall@1: 52.47%, Recall@5: 76.63%, Recall@10: 78.65%

---

*详细的 badcase 数据保存在 `all_badcases.json`*
