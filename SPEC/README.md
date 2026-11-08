# JobPilot V1 Spec

这一版 Spec 已从“Career Management System + Agent”收敛为：

> **Long-term Career Intelligence + Agent**

最终设计文档只保留三个核心设计面和一个最后实施面：

1. `00_PRODUCT_USER_MODEL.md`
   - 产品定位
   - Conversation / Memory / Knowledge 边界
   - 长期用户模型
   - 持久化原则
   - JD / Resume / Interview / Training 作为任务场景，而不是独立 Domain

2. `01_KNOWLEDGE_RETRIEVAL.md`
   - `.md/.txt/.tex` ingestion
   - Document / Section / Chunk
   - SourceRef
   - Hybrid Retrieval
   - `search_knowledge` / `read_source`

3. `02_AGENT_RUNTIME_TOOLS.md`
   - FastAPI
   - OpenAI-compatible SDK
   - Thin Agent Loop
   - Core Toolset
   - Context Engineering
   - AG-UI / assistant-ui
   - Runtime Guardrails

4. `03_IMPLEMENTATION_SPEC.md`
   - 实施 contract；当前代码状态仍以 `docs/JOBPILOT_TECHNICAL_OVERVIEW.md` 为准
   - 最终 Database Schema
   - API
   - Python package/module layout
   - Frontend layout
   - 配置
   - Migration
   - Golden Tests
   - Codex 实施顺序

[Context Lifecycle V1 合同](04_CONTEXT_LIFECYCLE_V2_DESIGN.md)（2026-10-17）按用户审查收敛：当前 Turn 的完整 interaction 留在 `AgentContext` 内存，deterministic eviction 只改变 active refs；增加精确 read、简化 Working Task State、tool-schema token accounting 与 provider error classification。无 Archive 表、migration 或崩溃后 resume；持久化、revision 和 Compactor recovery 暂缓。文件名沿用原 v2 设计稿名称，当前合同已同步到 `02` / `03`；已执行验证以技术总览为准。

## 核心架构

```text
                    User
                     │
                     ▼
               Career Agent
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Current     Memory     Knowledge
      Context                  Sources
          │          │          │
          └──────────┼──────────┘
                     ▼
             Agent Intelligence
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Current Task        Long-term Change
        Output            update_user_model
          │                     │
          ▼                     ▼
     Conversation              Memory
```

## 技术架构

```text
React + assistant-ui
        │
       AG-UI
        │
     FastAPI
        │
OpenAI-compatible SDK
        │
 Thin Agent Loop
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
search  search         update
memory  knowledge      user model
        │
   read_source
        │
        ▼
PostgreSQL + pgvector
```

## 核心原则

- Agent 是产品本体。
- 代码提供上下文、能力和硬约束，不替 Agent 编码复杂职业推理。
- Persist user knowledge and evidence, not every derivation.
- Conversation 是短期任务工作空间。
- Memory 是高密度长期用户认知。
- Knowledge 是权威原始 Source。
- JD / Resume / Interview / Training 是 Agent interaction，不是业务系统。
- Main Agent 自己理解 JD、生成 Resume、复盘 Interview、执行 Training。
- Main Agent 自己决定何时调用 `update_user_model`。
- V1 不默认使用第二个 LLM 做 JD Analyzer / Resume Composer / Interview Reviewer / Memory Reflection。
