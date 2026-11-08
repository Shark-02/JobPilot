# JobPilot V1 — Knowledge & Retrieval

**状态：Baseline Candidate**  
**定位：原始职业资料、Source、检索与 RAG 能力**

---

## 1. 设计目标

Knowledge 系统只做一件事：

> **保存用户原始职业资料，并在 Agent 需要时提供可追溯、可检索的原始信息。**

Knowledge 是权威 Source，不是低可信数据。

用户上传的项目资料、实习资料、简历源文件、面试记录等，默认都可以直接支持用户职业事实，除非用户明确说明它只是参考资料或不属于自己。

Knowledge 不负责：

- 自动分析 JD；
- 自动生成 Memory；
- 自动生成画像；
- 自动总结所有文档；
- 自动 RAG；
- 自动生成训练任务；
- 保存 Agent 的每次派生输出。

---

## 2. 总体架构

```text
          .md / .txt / .tex
                  │
                  ▼
        ┌──────────────────┐
        │ Document Ingest  │
        └────────┬─────────┘
                 │
        Parse / Normalize
                 │
                 ▼
             Document
                 │
              Sections
                 │
               Chunks
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
  Lexical Index        Vector Index
   pg_trgm              pgvector
       │                   │
       └─────────┬─────────┘
                 ▼
          Hybrid Retrieval
                 │
                 ▼
          Knowledge Modules
                 │
                 ▼
             SourceRefs
                 │
                 ▼
              Agent
```

---

## 3. Source 模型

任何用户提供的原始信息都可以成为 Source。

V1 SourceRef 统一表示：

```text
message://{message_id}

document://{document_id}/section/{section_id}

document://{document_id}/chunk/{chunk_id}
```

Agent通过统一：

```python
read_source(source_refs, max_tokens=None)
```

重新读取原始内容。

这使系统无需为 Job、Interview、Resume 等场景分别设计 `get_job`、`get_interview`、`get_resume_context`。

---

## 4. Document 支持范围

V1 正式支持：

```text
.md
.txt
.tex
```

V1 不做：

```text
.pdf
.docx
.pptx
.xlsx
OCR
图片理解
音视频解析
ZIP
代码仓库解析
```

---

## 5. Document 模型

```python
class KnowledgeDocument:
    id: UUID

    title: str
    filename: str
    format: MARKDOWN | TEXT | LATEX

    source_role: USER_CAREER_SOURCE | REFERENCE_SOURCE

    raw_text: str
    normalized_text: str

    sha256: str
    status: PROCESSING | READY | FAILED | ARCHIVED

    parser_version: str
    chunker_version: str

    metadata: dict

    created_at: datetime
    updated_at: datetime
```

默认：

```text
source_role = USER_CAREER_SOURCE
```

只有用户明确说明“这是学习资料 / 别人的资料 / 参考资料”等时，才使用 `REFERENCE_SOURCE`。

---

## 6. Source Authority

### 6.1 USER_CAREER_SOURCE

可以直接支撑：

- 用户做过什么；
- 项目采用了什么技术；
- 用户获得了什么结果；
- 用户原始简历中写了什么；
- 用户面试时实际回答了什么。

不要求先转换成 Memory。

### 6.2 REFERENCE_SOURCE

只能支持通用知识和参考信息，不能证明用户本人有相关经历。

---

## 7. Ingestion Pipeline

```text
Upload
↓
校验格式与大小
↓
读取 UTF-8 文本
↓
SHA-256 去重
↓
Parser
↓
Normalized Text
↓
Section Builder
↓
Chunker
↓
Lexical Index + Embedding
↓
READY
```

整个默认 ingestion：

```text
不调用 LLM
不自动写 Memory
不自动总结文档
```

---

## 8. Parser

### 8.1 Markdown

识别 Markdown heading hierarchy：

```text
# / ## / ### ...
```

保留：

- 标题层级；
- 普通段落；
- 列表；
- 代码块；
- 技术 token。

### 8.2 TXT

以空行和明显标题模式构造弱 Section。

没有明确结构时，整个文件作为 Root Section，再按段落切 Chunk。

### 8.3 LaTeX

`.tex` 不编译，不生成 PDF。

目标只是提取可搜索的职业文本和结构。

识别：

```latex
\section{}
\subsection{}
\subsubsection{}
```

以及简历常见宏参数中的自然语言。

V1 可使用 `pylatexenc` 作为基础文本解析器，并对少量自定义简历宏增加 conversion rule。

---

## 9. Document / Section / Chunk

```text
Document
   └── Section
         └── Chunk
```

### 9.1 Section

Section 是 Agent 理解的“资料模块”。

```python
class KnowledgeSection:
    id: UUID
    document_id: UUID
    parent_section_id: UUID | None

    heading: str | None
    level: int
    section_path: list[str]

    ordinal: int

    start_offset: int
    end_offset: int
```

### 9.2 Chunk

Chunk 是搜索单元。

```python
class KnowledgeChunk:
    id: UUID
    document_id: UUID
    section_id: UUID

    ordinal: int

    content: str
    retrieval_text: str

    start_offset: int
    end_offset: int

    token_count: int
    # 向量存于 KnowledgeChunkEmbedding(chunk_id, profile_id)，不属于实体列
```

---

## 10. Chunking

默认：

| 参数 | 值 |
|---|---:|
| target_chunk_tokens | 600 |
| min_chunk_tokens | 180 |
| max_chunk_tokens | 1000 |
| overlap | 0 |

策略：

```text
先按 Section
↓
Section <= max
→ 整体一个 Chunk

Section > max
→ 按 paragraph/list/code block 拆

同 Section 内过小相邻块
→ 合并
```

不跨越明显不同主题强行 merge。

---

## 11. Retrieval Text

`content` 保存原始块文本。

`retrieval_text` 用于检索：

```text
Document: 智能 Oncall 项目资料
Section: 并发与幂等 > Execution 推进

{chunk content}
```

V1 不为每个 Chunk 再调用 LLM 生成 contextual description。

---

## 12. Hybrid Retrieval

V1：

```text
lexical = pg_trgm / exact token match
semantic = pgvector
fusion = RRF
grouping = section-aware
reranker = disabled by default
```

流程：

```text
query
 │
 ├─ lexical candidates
 │
 └─ vector candidates
        ↓
      RRF
        ↓
Section regroup
        ↓
Module packing
        ↓
KnowledgeModule[]
```

精确技术词：

```text
uniq_case_no
event_id
SELECT FOR UPDATE
PydanticAI
```

应主要受 lexical retrieval 保证。

语义表达：

```text
“怎么避免多个执行器重复推进同一个任务”
```

应主要受 vector retrieval 支持。

---

## 13. KnowledgeModule

Agent 不直接接收大量孤立 Chunk。

Retriever 按 Section 整理后返回：

```python
class KnowledgeModule:
    document_id: UUID
    document_title: str

    section_id: UUID
    section_path: list[str]

    content: str

    source_refs: list[SourceRef]
    token_count: int
```

---

## 14. Agent 控制 Retrieval Budget

Agent可以自己指定希望拿多少模块和上下文。

```python
search_knowledge(
    query: str,
    document_ids: list[UUID] | None = None,
    max_modules: int | None = None,
    max_tokens: int | None = None
)
```

Runtime 只提供硬上限。

默认：

| 参数 | 默认 | 硬上限 |
|---|---:|---:|
| max_modules | 6 | 16 |
| max_tokens | 8k | 24k |
| lexical_candidates | 30 | 60 |
| vector_candidates | 30 | 60 |

Agent可以根据任务主动扩大或缩小检索范围。

---

## 15. Knowledge Tools

V1核心 Knowledge 能力：

```text
search_knowledge
read_source
list_documents   # 可选辅助
```

### search_knowledge

用于从文档库中找相关资料。

### read_source

统一读取：

- 历史用户消息；
- Document Section；
- Document Chunk。

当 Memory 指向某个历史 JD 用户消息时，Agent直接：

```text
read_source(message://...)
```

恢复完整 JD。

### list_documents

只用于用户模糊指代已上传资料时定位文件，不返回全文。

---

## 16. 不自动 RAG

请求流程：

```text
User
↓
Agent
↓
Agent自己判断是否需要 search_memory / search_knowledge
```

不存在：

```text
User
↓
middleware 自动检索
↓
一堆内容预塞 Context
↓
Agent
```

这样避免：

- 不必要检索；
- 重复上下文；
- 不可观察 RAG；
- 固定 Pipeline 限制 Agent。

---

## 17. Knowledge 与 Memory

首次任务：

```text
search_memory
→ 无足够信息

search_knowledge
→ 获取原始 Evidence
→ Agent直接完成任务
```

如果 Agent认为某部分信息未来会显著影响帮助方式，可以在同一 Turn 调用：

```text
update_user_model(...)
```

Memory 只是高密度长期投影。

当前任务不需要等待 Memory 写入才能使用 Knowledge。

---

## 18. Derived Output 不自动进入 Knowledge

以下内容默认只在 Conversation：

```text
JD分析
岗位匹配
简历草稿
简历修改稿
面试复盘
训练题
训练反馈
参考答案
```

不会自动建立 `KnowledgeDocument`。

如果用户明确说：

> 保存这版简历，以后作为我的主简历。

系统才把这段内容保存为一个新的 USER_CAREER_SOURCE。

---

## 19. 删除与归档

### ARCHIVED

- 原始数据保留；
- 默认不参加 `search_knowledge`；
- 已有 SourceRef 仍然可以解析。

### DELETE

删除：

- Document；
- Sections；
- Chunks；
- Embeddings；
- lexical index。

同时清理 MemorySource。

如果某 Memory 删除 Source 后不再有任何有效来源：

- Source-backed fact 应失效或删除；
- Agent assessment 若其全部行为证据被删除，也应失效。

具体事务规则在 Implementation Spec 定义。

---

## 20. Retrieval Eval

V1建立真实 Golden Set，至少覆盖：

- 精确技术词；
- 中文语义查询；
- 中英文混合；
- 项目限定；
- 跨项目问题；
- 模糊用户表达。

指标：

```text
Hit@K
Recall@K
MRR
Evidence Success Rate
```

只有真实 eval 证明必要时，才增加：

```text
cross-encoder reranker
LLM contextualization
multi-query
HyDE
query rewrite
```

---

## 21. Hard Invariants

1. Knowledge 是原始 Source，不是低可信草稿。
2. USER_CAREER_SOURCE 可以直接支持用户职业事实。
3. 上传默认只 Parse / Chunk / Index。
4. 上传不自动抽 Memory。
5. V1只支持 `.md/.txt/.tex`。
6. Knowledge 按 Document → Section → Chunk 组织。
7. Chunk 是检索单元，Section 是返回给 Agent 的主要模块单位。
8. RAG 只能由 Agent主动调用。
9. Agent控制检索规模，Runtime只提供硬上限。
10. Hybrid retrieval = lexical + vector + rank fusion。
11. 每个检索结果必须携带可读取的 SourceRef。
12. `read_source` 统一读取历史 Message 与 Document Source。
13. Agent派生输出默认不进入 Knowledge。
14. 用户明确长期保存的输出才可以转成新的 Source。
15. 检索复杂度只根据真实 Eval 增加。

## 2026-10-06 Embedding Profile 合约更新

Embedding 空间、单 active 写入、profile 隔离索引、显式 backfill/activate 与 coverage gate 以 [EMBEDDING_PROFILES.md](EMBEDDING_PROFILES.md) 为准。切换不修改 chunk、SourceRef、词法检索或 RRF。
