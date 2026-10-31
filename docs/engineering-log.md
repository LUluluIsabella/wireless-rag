# Wireless-RAG 工程问题与改进日志

> **当前状态（2026-10-31）**：RAG 与 Agent 核心开发完成，CLI、批量评测和 Streamlit 双模式已跑通。Agent 单元测试 16/16 通过；真实评测 8 条完成，状态符合 6/7、工具要求满足 6/6。1 条无效引用保留为待复核记录。本轮 Git 提交与推送尚未确认。

## 问题与技术设计索引

| ID | 模块 | 内容 | 状态 |
|---|---|---|---|
| ISSUE-001 | Chunking | Token 超限与动态窗口修复 | 基础测试通过 |
| TECH-001 | Retrieval | Dense + BM25 + RRF 混合检索 | 已实现并完成基线评测 |
| ISSUE-002 | PDF Parsing | 表格与双栏解析导致检索噪声 | Frozen |
| TECH-002 | Agent | 工具调用、预算、证据管理与异常处理 | 核心开发完成，16项单元测试通过 |
| TECH-003 | Streamlit | RAG / Agent 双模式及轨迹下载 | 用户手动验收通过 |
| ISSUE-004 | Agent Citation | 批量评测出现1条 invalid_citation | 保留记录，具体原因待补 |
| ISSUE-003 | Hybrid Retrieval 排名退化 | dev_005 的正确证据由 Dense Rank 4 降至 Hybrid Rank 10，导致 Hit@5 失败；已分析可能原因，暂缓优化。 | Open / Deferred |
---

## ISSUE-001：Chunk Token 数超过设定上限

**模块：** `src/chunking.py`

**状态：** 基础测试通过

### 1. 问题

项目设置 `CHUNK_SIZE=300`、`CHUNK_OVERLAP=50`，但测试时发现部分 Chunk 超过 300 tokens。

### 2. 原因

Tokenizer 的 Encode/Decode 不一定严格可逆。原始 Token 窗口长度符合要求，不代表解码后重新编码的长度仍然符合要求。

### 3. 解决方案

- 解码后重新编码，检查实际 Token 数。
- 超限时执行 `end -= 1`，动态收缩窗口。
- 通过 `max(start + 1, end - overlap)` 保证窗口前进，避免死循环。

### 4. 验证与改进

运行 `python -m scripts.test_chunking`，基础检查通过。

后续考虑句子感知分块，减少技术术语和句子被截断的情况。

---

## TECH-001：Hybrid Retrieval 混合检索设计

**模块：** `src/retriever.py`

**阶段：** 第 10 节

**状态：** 已实现并完成基线评测（以下保留原始设计）

### 1. 设计动机

Wireless-RAG 使用英文 Wi-Fi 6 技术文档，支持中文提问。

仅使用 Dense Retrieval 存在两个潜在问题：

- 对 `802.11ax`、`26-tone RU` 等精确术语匹配不稳定。
- 语义相关不代表检索片段包含正确的技术参数。

因此，引入 BM25 关键词检索，与 Dense 形成互补。

### 2. 核心原理

| 方法 | 作用 | 局限 |
|---|---|---|
| Dense（E5 + FAISS） | 语义匹配、跨语言检索 | 精确术语匹配可能不稳定 |
| BM25 | 关键词、协议编号精确匹配 | 缺乏深层语义理解 |
| RRF | 根据排名融合两路结果 | 不保证相关性一定提升 |

**RRF 公式：**

\[
\operatorname{RRF}(d)=\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)}
\]

其中 \(k=60\)，未出现在某一路结果中的文档不贡献该路分数。

RRF 不直接相加 Dense 与 BM25 分数，而是根据排名融合，避免不同评分尺度带来的问题。

### 3. 实现方案

```text
User Query
    ├── Dense Retrieval → Top20
    └── BM25 Retrieval  → Top20
                  ↓
             RRF Fusion
                  ↓
          chunk_id 去重
                  ↓
             Final Top5
                  ↓
            LLM Context
```

原设计接口示意（混合检索现已实现）：

```python
dense_search(question, top_k=20)
bm25_search(question, top_k=20)
rrf_fuse(dense_results, bm25_results, k=60)
hybrid_search(question, top_k=5)
```

BM25 增加 Wi-Fi 领域分词规则，保留 `MU-MIMO`、`802.11ax` 等技术术语，并考虑中英文术语映射。

### 4. 评测方案

对比 Dense、BM25、Hybrid 三种检索模式。

| 指标 | 含义 |
|---|---|
| Hit@5 | 前 5 条是否命中正确证据 |
| MRR@5 | 首个正确证据的排名质量 |
| Latency | 检索耗时 |

分别测试语义问题、精确术语问题和中文查询英文文档的场景。

### 5. 工程总结

Hybrid Retrieval 的核心在于**多路召回互补，而不是单纯增加检索器数量**。

Dense 负责语义匹配，BM25 补充精确关键词检索，RRF 负责排名融合。最终是否有效，需要通过固定测试集验证。

---

## ISSUE-002：PDF 复杂表格与双栏解析导致检索噪声

**所属模块：** `src/ingest.py`、`src/chunking.py`

**发现阶段：** 第 10 节 Hybrid Retrieval 测试

**状态：** 暂缓优化（Frozen），不阻塞 RAG 主流程

### 1. 问题现象

在测试问题“26-tone RU 在 Wi-Fi 6 中有什么作用？”时，发现部分检索结果存在：

- **表格碎片：** `RU52 NS AID1...` 等缺少行列关系的文本。
- **双栏交错：** 左右栏内容被错误拼接，破坏原始语义。
- **低质量 Chunk：** 封面、页眉及纯数字内容进入 Dense Top5。

### 2. 原因分析

PyMuPDF 提取的是 PDF 文本及坐标信息，并不能保证复杂排版和表格结构被正确还原。

表格二维关系丢失后，碎片化文本被正常分块并进行 Embedding，可能干扰语义检索结果。

### 3. 已完成的改进

- 使用文本块坐标优化 PDF 阅读顺序。
- 增加重复页眉页脚识别。
- 在 Chunking 阶段加入低质量文本过滤。
- 临时排除 Spirent PDF 第 16 页的复杂表格。
- 重新生成 Chunk 并测试 Hybrid Retrieval。

**运行结果：**

- 生成 Chunk：152 个。
- Cisco：47 个；ZTE：56 个；Spirent：49 个。
- 过滤后的疑似低质量 Chunk：0（仅针对现有规则）。
- Hybrid Top1：`zte_wifi6_p014_c001`，能够解释 RU 的基本作用。
- 部分表格碎片及双栏交错问题仍然存在。

### 4. 暂缓优化的原因

当前 Hybrid Retrieval 已能召回有价值的技术证据，复杂表格问题暂未阻塞基础问答功能。

考虑到项目以完成 RAG MVP 为优先目标，暂不继续投入复杂表格结构化解析。

**注意：** 当前结果不能证明表格噪声对整体检索性能影响很小，仍需要后续评测验证。

### 5. 后续改进方向

- 根据 Hit@5、MRR@5 和失败案例分析检索质量。
- 必要时使用 `page.find_tables()` 或版面分析提取结构化表格。
- 为 Chunk 增加 `content_type`，区分正文与表格。
- 对正文和表格使用不同的索引或检索策略。

**重新启动优化的条件：** 如果评测发现表格碎片持续挤占 Top5、导致证据命中失败或影响答案准确性，再恢复该问题的优化工作。

### 6. 面试可能提问

**Q1：为什么不直接删除所有表格？**

表格可能包含重要的协议参数，例如 RU 分配数量。直接删除会损失有效证据，因此应保留原始 PDF，并在后续按需进行结构化处理。

**Q2：为什么当前选择暂缓优化？**

因为项目优先目标是完成可运行、可评测的 RAG 闭环。当前已能检索到相关解释性证据，复杂表格解析的优先级低于问答生成与系统评测。

**Q3：如何判断未来是否需要继续优化？**

通过固定测试集的 Hit@5、MRR@5、引用支持情况和失败案例，判断表格噪声是否明显影响检索和生成质量。

### 7. 工程经验总结

**RAG 优化应由实际瓶颈和评测结果驱动，而不是无限追求数据预处理的完美。**

对于不阻塞 MVP 的问题，可以记录已知限制和恢复优化的条件，优先完成端到端系统，再依据实验结果决定后续投入。

## ISSUE-003：Hybrid Retrieval 相关证据排名退化

- **日期：** 2026-10-23
- **模块：** `src/retriever.py`、`scripts/evaluate.py`
- **状态：** Open（已分析，暂缓优化）
- **优先级：** Low
- **相关阶段：** Stage 12 — RAG Evaluation

### 1. 问题描述

在 Dev Set 检索评测中，Hybrid Retrieval 整体获得更高的 MRR@5，但在个别问题上出现正确证据排名下降的情况。

初步评测结果：

| Method | Hit@5 | MRR@5 |
|---|---:|---:|
| Dense | 0.8750 | 0.4729 |
| BM25 | 0.8750 | 0.6979 |
| Hybrid (RRF) | 0.8750 | 0.7917 |

虽然 Hybrid 整体排序效果较好，但并不保证所有问题都得到改善。

### 2. 失败案例

**Question ID：** `dev_005`

**Question：**

> OFDMA 为什么适合高密度 Wi-Fi 场景？

**Gold Chunks：**

```python
[
    "zte_wifi6_p013_c001",
    "cisco_wifi6_p006_c002"
]
```

检索结果：

| Method | 首个 Gold Rank | Hit@5 |
|---|---:|---:|
| Dense | 4 | 1 |
| BM25 | Top 5 未命中 | 0 |
| Hybrid | 10 | 0 |

正确证据 `cisco_wifi6_p006_c002` 在 Dense 中排名第 4，但经过 RRF 融合后下降至第 10 位，无法进入当前 Generator 使用的 Top 5 Context。

### 3. 原因分析

**（1）BM25 关键词匹配偏差**

BM25 返回的 Top 5 全部来自 Spirent OFDMA 白皮书。

该文档包含大量 OFDMA、Wi-Fi 等技术术语，而查询中的“高密度”尚未进行英文术语扩展，可能导致 BM25 更倾向于召回关键词匹配度高、但证据支持不足的片段。

**（2）RRF 融合导致排名下降**

当前使用：

\[
RRF(d)=\sum_r\frac{1}{60+\operatorname{rank}_r(d)}
\]

RRF 根据多路检索排名累计分数，不直接判断文本能否支持答案。

某些在 Dense 和 BM25 中均有一定排名的 Chunks，可能超过只在 Dense 中排名较高的正确证据。

**（3）存在低信息量 Chunk**

Hybrid Top 1：

```text
spirent_ofdma_p001_c001
```

主要包含白皮书标题，虽然关键词匹配度较高，但没有提供回答问题所需的技术机制。

说明当前仍存在 PDF 封面、目录或其他低信息量文本参与检索的问题，与 ISSUE-002 有关联。

**（4）Gold Annotation 可能存在遗漏**

当前只标注了两个 Gold Chunks。Hybrid Top 10 中还有部分与 OFDMA 多用户传输相关的内容，尚未逐一确认其证据支持程度。

因此，当前 Hit@5 结果属于基于候选 Gold 标注的初步评测结果。

### 4. 可选优化方案

后续可根据需要开展以下实验：

1. **Query Expansion**：增加“高密度”等中文技术概念的英文映射。
2. **Chunk Filtering**：过滤纯标题、目录和其他低信息量 Chunks。
3. **RRF Analysis**：分析 Dense/BM25 Top 20 排名及 RRF 融合分数。
4. **Gold Annotation Review**：补充遗漏的有效证据。
5. **Reranking**：必要时增加候选重排，提高证据相关性。

上述方案目前均未实施或验证，不应记录为已取得性能提升。

### 5. 当前处理决定

考虑到当前项目目标是完成可复现的 Wireless-RAG 系统，并进一步扩展 Agent Tool Calling，暂不针对单个 Dev 问题进行过度调参。

当前决定：

- 保留现有 Dense、BM25 和 RRF 实现。
- 保留失败案例及可选改进方向。
- 不修改 Baseline RRF 参数。
- Test Set 基线评测已完成；保留该 Dev 失败案例，后续依据证据审核决定是否优化。
- RAG 闭环与 Agent 扩展现已完成，当前推进交付收尾。

### 6. 工程经验

本案例说明：

- Hybrid Retrieval 并不保证每个查询都优于单一路检索。
- 检索命中率与正确证据排名需要分别评估。
- 低信息量 Chunk 可能干扰融合排序。
- 检索评测依赖可靠的 Gold Annotation。
- 在工程开发中，需要平衡局部优化收益与整体交付进度。

**结论：** Hybrid 在当前 Dev Set 上提升了整体 MRR@5，但存在个别排名退化案例。问题已记录，暂缓优化，优先完成 RAG 系统闭环。

**Status：Open / Deferred**


## 2026-10-31：第19阶段 Agent 开发与验收

### 完成范围

- 实现 WifiAgent 工具调用循环：模型决定调用 search_wifi_docs 或 get_pdf_page，程序校验参数、执行工具并回传结果。
- 每次 run 独立管理消息、证据编号与轨迹；重复片段复用编号。
- 增加工具调用预算、模型请求轮数限制、时间预算检查及异常处理。请求前时间检查不代表同步工具可被强制中断。
- 完成 Agent CLI、批量评测入口及 Streamlit RAG / Agent 双模式。
- 网页展示答案、引用证据、工具轨迹和统计信息；使用 session_state 保留结果，支持下载 JSON。
- 按用户反馈，CLI 与 Streamlit 已手动验证成功。截图可确认 Agent answered、2 次工具调用、2 轮模型请求、页面查询总耗时 12.94 秒。该单次页面耗时与批量评测平均耗时采用不同测量范围，不能直接比较。

### 自动测试

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agent.py -q
```

用户提交结果：**16 passed in 0.50s**。

覆盖正常工具循环、重复证据编号、无效及缺失引用、工具预算、非法工具及参数、工具异常、模型异常、最终 JSON 错误和请求前超时。使用 Mock，不调用真实 API；不评估真实检索质量及语义支持。

### 真实模型批量评测

- 数据集：evaluation/agent_dev.jsonl，共 8 条。
- 入口：scripts/evaluate_agent.py。
- 输出：evaluation/agent_dev_results.json。

| 指标 | 结果 |
|---|---:|
| 已完成用例 | 8 |
| answered | 5 |
| abstained | 2 |
| invalid_citation | 1 |
| 状态符合率 | 6/7，85.71% |
| 工具要求满足率 | 6/6，100% |
| 平均工具调用次数 | 2.25 |
| 平均运行耗时 | 8.12325 秒 |

口径：expected_answerable 为 null 的用例不计入状态符合率；required_tools 为空的用例不计入工具要求满足率。工具成功执行不代表选择最优或调用必要；引用编号有效不代表原文支持全部结论。执行失败不能视为正确拒答。

用户随后表示已完成指标核验并进入网页阶段，但未提供修订后的完整评测结果。因此上述数字保留为已观测批次；1 条 invalid_citation 不标记为已修复，其具体用例、原因与修复后重测结果尚无记录。不能据此声称 8/8 全部通过或得到答案准确率。

### 当前结论与交付事项

**Agent 核心功能开发完成，已完成基础自动测试、小规模真实评测及网页集成。**

- 当前范围是面向 Wi-Fi 6 文档的工具调用型 Agent；小规模评测不代表生产环境可靠性或全面抗注入能力。
- 待记录：无效引用用例的具体原因和处理结论、人工证据支持审核明细。
- 待完成或补充结果：最终全量 pytest、git diff --check、handoff 更新、Git 提交及推送。此前基线的提交状态不能作为本轮新增代码已提交的证明。