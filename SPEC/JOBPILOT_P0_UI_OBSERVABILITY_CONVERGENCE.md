# JobPilot P0 增量修复：前端聊天收敛 + Knowledge 删除 + Agent Observability

状态：待实施  
目标：只修 P0 可用性与可观测性问题，不在本轮调 Agent 行为策略。  
原则：**不是补丁式修复，而是删除重复实现、收敛到单一状态源和成熟 UI/runtime。**

---

## 0. 本轮结论

当前已知 P0 问题：

1. Knowledge 页面没有删除入口，FAILED / READY / ARCHIVED 文档都无法从 UI 正常删除。
2. 聊天 UI 没有正确渲染 Markdown，模型输出中的 `**bold**` 等语法直接裸显。
3. 用户点击发送后，用户消息不会立即进入对话列表，而是等模型回复返回后才一起出现。
4. Assistant 最终回复会出现重复两条。
5. 当前 Tool Activity 只能看到极浅摘要，例如 `Found 8 source module(s)`，无法看到：
   - Tool 实际输入参数
   - 完整 Tool 输出
   - Retriever lexical/vector 候选
   - 各候选分数与排序
   - RRF / regroup / module assembly 过程
   - 最终到底把哪些模块返回给 Agent
6. 当前 Model Trace 无法完整查看每一轮真正发给模型的输入与模型返回内容。
7. `tool_call_trace.created_at` 在同一 Turn 内可能完全相同，不能画真实时间线。
8. 当前只能靠 SQL 手工还原 Agent 执行过程，不具备类似 Langfuse 的开发观测体验。

本轮暂不优先处理：

- 为什么某个 query 返回 Top-K 不相关结果
- Agent 为什么没有主动写 Memory
- Agent 为什么用了 10 次 Tool Call / 6 次 Model Call
- Memory broad query 为什么漏召回
- RRF / threshold / recall 参数调优
- Prompt 调优
- Eval 体系扩展

这些属于 P1。  
**P0 先让系统“可操作、可观察”，再基于真实 Trace 修行为。**

---

# 1. 核心架构原则

## 1.1 AG-UI 保留，不因前端 Bug 替换协议

AG-UI 是 Agent 与前端之间的事件协议，不是完整 UI 组件库。

当前：

- Markdown 不解析
- User Message 延迟出现
- Assistant Message 重复
- Tool Activity 生命周期混乱

这些更像是：

> 前端存在多套消息状态、手工 SSE/事件拼装、最终消息回填与流式消息同时 append 等问题。

因此：

**不要因为这些问题删除 AG-UI。**

应优先删除自维护的聊天状态与渲染胶水，让成熟 UI/runtime 负责：

- user message optimistic append
- assistant placeholder
- token streaming
- finalization
- message id 去重
- Markdown rendering
- auto scroll
- retry / error state
- tool call 生命周期
- conversation hydration

---

## 1.2 本轮禁止形成 legacy/new 双轨

不允许：

```text
旧 message store
+
新 message store

旧 SSE parser
+
新 Agent runtime

旧 ChatBubble
+
新 ChatBubble

旧 Markdown parser
+
组件库 Markdown renderer
```

新路径验收通过后，应真正删除旧实现。

Codex 在修改前必须先画出当前前端消息数据流：

```text
submit
→ local state?
→ POST / agent?
→ AG-UI events?
→ message store?
→ history reload?
→ render?
```

并明确指出：

- 谁负责创建 user message
- 谁负责创建 assistant message
- 谁消费 stream delta
- 谁消费 final message
- 谁负责 history hydration
- 哪些地方会 append 同一 assistant message

然后收敛成一个 owner。

---

# 2. Agent UI 组件选型：先审计，再替换

## 2.1 不继续手搓 Chat UI

本轮目标不是继续修：

- Markdown 正则
- 手工消息气泡
- 自己拼流式字符串
- 自己维护 pending assistant message
- 自己做 tool result accordion

而是选一套成熟 Agent UI / headless runtime，并让现有业务只保留必要的业务展示。

---

## 2.2 若当前前端是 React

优先保留 / 迁移到：

```text
assistant-ui
+
@assistant-ui/react-ag-ui
+
@ag-ui/client
```

原因：

- assistant-ui 的 React 路径成熟；
- 有现成 AG-UI runtime；
- 自带 Thread / Message / Composer / Tool UI；
- 支持 Markdown、streaming、tool call、history adapter；
- 当前不应再手写另一套消息状态机。

实施原则：

1. `AssistantRuntimeProvider` / AG-UI runtime 成为唯一 Chat runtime。
2. 删除自定义 stream parser 和重复 message store。
3. User message 发送后立即由 runtime 出现在 transcript。
4. Assistant streaming 更新同一 message，不在 FINISH 时 append 第二条。
5. Markdown 使用组件库默认 renderer，不自己解析。

---

## 2.3 若当前前端是 Vue 3

Codex 必须先做一个很小的兼容性 Spike，再二选一，不允许先大改。

### 首选候选 A：`@copilotkit/vue/v2`

适合条件：

- 当前后端已有标准 AG-UI endpoint；
- 能以很薄的方式接入，不引入第二套业务 Agent runtime；
- 不要求为了 Chat UI 重写后端。

优势：

- Vue 3 原生；
- 有 `CopilotChat` / `CopilotChatView`；
- 消息、running state、input、auto-scroll、attachments 等由组件/runtime 管；
- Assistant Message 支持 Markdown、Tool Calls、Action toolbar；
- AG-UI 原生；
- 有 Vue Tool UI / Generative UI；
- 有开发 Inspector 能查看 AG-UI events。

注意：

- 如果生产直连 self-managed AG-UI 会引入不合适的授权/商业依赖，或必须额外引入一个不必要的 Copilot Runtime 服务，则不要为了 UI 强行采用。
- 本项目已有后端 Runtime，不允许为了套组件再引入第二个“决定 Agent 如何执行”的 Runtime。

### 候选 B：`@tanstack/ai-vue/ui`

适合条件：

- 希望保持现有后端与传输层；
- 需要 Vue 原生、typed、headless 的 Chat UI；
- 愿意保留 JobPilot 自己的页面视觉，但不想自己维护消息/stream/markdown 基础设施。

优势：

- Vue 3 原生；
- streaming chat；
- typed message parts / tools / interrupts；
- `TextPart` 自带 Markdown 渲染；
- 可以自定义 Tool component；
- 更适合“保留当前后端，只替换前端聊天 runtime/UI 层”。

### 本轮不建议

#### assistant-ui Vue

当前 Vue binding 仍处于 preview，不能作为本轮 P0 的生产收敛依赖。

#### 非官方 `ant-design-x-vue` 作为核心 runtime

可作为视觉组件参考，但不要把社区 Vue 移植版当 JobPilot 的 Agent runtime 核心。
官方 Ant Design X 主线仍是 React。

---

## 2.4 选型验收门槛

无论 React / Vue，候选必须用同一个最小 fixture 通过：

1. 用户点击发送后 **100ms 内**看到自己的消息，不等模型返回。
2. Assistant 流式内容只对应 **一个 message id**。
3. final event 后不新增第二条 assistant message。
4. Markdown：
   - `**bold**`
   - 列表
   - 标题
   - fenced code block
   - inline code
   均正确渲染。
5. Tool Call：
   - pending
   - running
   - success
   - error
   均能显示。
6. 页面刷新后 history hydrate 不产生重复消息。
7. 同一个 server message id 只能渲染一次。
8. 不需要业务代码自己拼 token stream。

如果某候选无法满足，不进入主代码。

---

# 3. 聊天 Bug 的根治要求

## 3.1 用户消息必须 optimistic append

当前错误体验：

```text
用户发送问题
→ UI 没变化
→ 等模型回来
→ 用户消息 + Assistant 一起出现
```

正确行为：

```text
submit
→ 立即将 user message 放入唯一 message store
→ input clear
→ assistant run 开始
→ assistant placeholder / stream
```

禁止：

- 等后端返回完整 conversation 后再把 user message插入；
- 依赖最终 history refresh 才展示 user message。

---

## 3.2 Assistant 重复消息必须从状态所有权上解决

重点排查：

```text
AG-UI TEXT_MESSAGE_* streaming
```

与：

```text
run finished / final REST response / history refresh
```

是否都在：

```text
messages.push(...)
```

最终设计：

```text
message_id / server_message_id
```

是唯一身份。

状态变化：

```text
START
→ create assistant message once

CONTENT
→ append delta to same message

END
→ mark finalized

snapshot/history
→ upsert by id, never blind append
```

禁止用：

- 文本内容比较去重
- “如果最后一条一样就不加”
- setTimeout
- boolean `hasAddedFinal`
- 额外临时数组

这类都属于补丁。

---

## 3.3 Markdown 不自己手搓

Assistant 文本必须通过选定 UI 库自带的 Markdown renderer。

验收 fixture：

```md
这是 **粗体**。

- A
- B

`inline code`

```java
public static void main(String[] args) {}
```
```

不能再出现裸 `**`。

必须保证 raw HTML 默认不直接执行。

---

# 4. Knowledge 删除：P0

## 4.1 UI

每个资料卡提供明确操作：

```text
READY:
[归档] [删除]

ARCHIVED:
[恢复/重新启用（若已有）] [删除]

FAILED:
[查看错误] [删除]
```

删除必须二次确认。

文案示例：

> 删除后会移除该文档及其 Section、Chunk、Embedding。此操作不可撤销。

---

## 4.2 Backend

若不存在，新增：

```http
DELETE /api/documents/{document_id}
```

要求：

1. 事务内执行。
2. 删除 document 及其 section/chunk。
3. 尊重当前 FK / memory_source 约束。
4. Codex 必须先审计当前 FK，不得直接猜 cascade。
5. 如果 memory_source 阻止删除：
   - 明确处理依赖关系；
   - 不允许用关闭 FK / 原始 TRUNCATE 的方式绕过。
6. 返回结构化结果：

```json
{
  "deleted": true,
  "document_id": "...",
  "deleted_sections": 12,
  "deleted_chunks": 243
}
```

FAILED 文档也必须可删。

---

# 5. Agent Observability：P0

## 5.1 目标

实现一个 JobPilot 内建的 Langfuse-lite：

> 能从一次用户 Turn 开始，完整回放每一个 Model / Tool / Retrieval 步骤，并查看该步骤实际输入、实际输出、耗时、token、source refs 与错误。

它是 Agent Runtime 的基础设施，不是业务功能。

---

## 5.2 保留现有 Trace 表

继续保留：

```text
agent_turn
model_call_trace
tool_call_trace
```

它们负责：

- run overview
- 快速查询
- 聚合统计
- 状态 / token / latency summary

不要为了 Observability 删除已有表。

---

## 5.3 新增 `agent_observation`

新增一张专门保存完整调试 payload 的表。

建议最小 schema：

```sql
CREATE TABLE agent_observation (
    id UUID PRIMARY KEY,
    turn_id UUID NOT NULL
        REFERENCES agent_turn(id)
        ON DELETE CASCADE,

    parent_observation_id UUID NULL
        REFERENCES agent_observation(id)
        ON DELETE CASCADE,

    sequence INTEGER NOT NULL,

    kind VARCHAR(32) NOT NULL,
    -- MODEL
    -- TOOL
    -- RETRIEVAL
    -- FINAL
    -- ERROR

    name VARCHAR(255),

    status VARCHAR(32) NOT NULL,

    input_payload JSONB,
    output_payload JSONB,
    metadata JSONB,
    source_refs JSONB,

    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    latency_ms INTEGER,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_agent_observation_turn_sequence
ON agent_observation(turn_id, sequence);

CREATE INDEX idx_agent_observation_kind
ON agent_observation(kind);
```

字段名可根据当前 ORM 规范调整，但语义不要拆散成很多小表。

---

# 6. 必须记录什么

## 6.1 MODEL Observation

必须能看到：

```text
Provider
Model
Model Call Sequence
Started At
Finished At
Latency

Input:
- system prompt
- core profile
- recent conversation
- tool results included in this call
- available tool schemas
- current user message
- exact outbound message list

Output:
- raw assistant message
- tool_calls
- finish_reason
- provider usage

Token:
- input_tokens
- output_tokens
- total_tokens
```

注意：

- 记录“模型真正看到的内容”；
- 不记录或展示隐藏 chain-of-thought；
- provider 未返回 hidden reasoning 时不要人工构造；
- 如果模型返回公开 reasoning summary，可作为普通输出记录。

---

## 6.2 TOOL Observation

必须记录：

```text
tool_name
tool_call_id

input:
  exact arguments

output:
  exact result returned to Agent

status
error
latency
source_refs
```

当前 UI 里的：

```text
Found 8 source module(s)
```

只保留为 summary。

Debug 页面必须能展开完整结果。

---

## 6.3 RETRIEVAL Observation

这是本轮重点。

对于 `search_knowledge`，不能只看到最终：

```text
Found 8 source module(s)
```

必须记录：

### Query

```text
query
document_ids
max_modules
max_tokens
其它过滤条件
```

### Lexical stage

每个候选至少：

```text
chunk_id
document_id
section_id
score
rank
title/path
text preview
```

### Vector stage

每个候选至少：

```text
chunk_id
distance/similarity
rank
text preview
```

### Fusion / RRF

```text
chunk_id
lexical_rank
vector_rank
fusion_score
final_rank
```

### Module assembly

```text
module_id / section_id
included_chunk_ids
token_count
why included / rank
```

### Final result sent to Agent

必须记录 Tool 最终真正返回给模型的 modules。

这样以后才能回答：

> “为什么搜张三会返回 8 个 Agent 成本诊断模块？”

而不是先猜 threshold。

---

# 7. Observability API

建议新增只读 Debug API：

```http
GET /api/observability/turns
GET /api/observability/turns/{turn_id}
GET /api/observability/turns/{turn_id}/observations
GET /api/observability/observations/{observation_id}
```

列表需要：

```text
turn_id
conversation_id
user message preview
status
model calls
tool calls
duration
input/output tokens
started_at
```

详情按 `sequence` 返回。

不要为了前端页面再创建第二份 execution state。

---

# 8. 前端 Observability 看板

## 8.1 入口

在主导航增加：

```text
Observability
```

或：

```text
Developer → Observability
```

这是个人开发型产品，本轮优先直接可见。

---

## 8.2 Runs 页面

每行：

```text
时间
用户问题摘要
状态
Model Calls
Tool Calls
总耗时
Input Tokens
Output Tokens
```

例如：

```text
11:21:21
讲讲你知道的关于我的事情
COMPLETED
6 model · 10 tools · 40.8s
```

---

## 8.3 Turn Detail

固定结构：

```text
Overview
Timeline
Model Calls
Tool Calls
Retrieval
Raw
```

### Overview

展示：

- turn id
- conversation id
- status
- duration
- model/tool count
- token total
- error
- start/end

### Timeline

真实时间轴：

```text
Turn Started
↓
Model #1
↓
search_memory
↓
Model #2
↓
list_documents
↓
search_knowledge
...
↓
Final Answer
```

每个节点可以点击。

---

## 8.4 Observation Detail Drawer

统一 Drawer / Side Panel：

```text
Summary
Input
Output
Metadata
Sources
Raw JSON
```

不要给每种 Tool 手搓完全不同的详情页。

Retrieval 可以在统一 Drawer 中增加专门的 ranking table。

---

# 9. Replay 的定义

P0 的 “Replay” 是：

> 按 recorded timeline 回放历史执行过程。

不是重新执行 Agent。

支持：

```text
Play
Pause
Step Next
Step Previous
Speed 0.5x / 1x / 2x
Jump to observation
```

回放时只读取 observation。

禁止：

- 再次调用模型
- 再次执行工具
- 再次写 Memory
- 重放副作用

以后如需要 “Re-run from here”，必须创建新的 Turn，属于 P1/P2。

---

# 10. 时间语义修复

当前 trace 中存在同一 Turn 多条 tool trace 的 `created_at` 完全相同的问题。

P0 修复：

Model / Tool / Retrieval 都必须记录：

```text
started_at
finished_at
latency_ms
```

`created_at` 不再承担真实执行时间线含义。

Timeline 使用 `started_at` 排序，`sequence` 作为逻辑顺序兜底。

---

# 11. Observability 写入方式

## 11.1 不让业务代码到处手写 INSERT

实现一个统一：

```text
ObservationRecorder
```

职责仅：

```text
start(kind, name, ...)
finish(id, output, metadata, ...)
fail(id, error, ...)
```

Runtime / ModelClient / Tool executor / Retriever 在边界处调用。

不要在每个业务 Tool 内复制观测代码。

---

## 11.2 Retrieval instrumentation

Retriever 内部允许记录 debug snapshot，但不能让 debug 逻辑改变排序结果。

类似：

```text
retrieve(query)
  lexical()
  vector()
  fuse()
  assemble()
```

每层正常计算自己的结果。

最后把调试数据交给 recorder：

```text
recorder.recordRetrieval(...)
```

禁用 observability 时，结果必须完全一致。

---

# 12. Payload 与安全边界

本项目当前是个人本地 JobPilot，允许开发模式保存完整输入输出以便调试。

增加配置：

```text
OBSERVABILITY_ENABLED=true
OBSERVABILITY_CAPTURE_PAYLOAD=true
```

规则：

1. false 时 Agent 功能不受影响。
2. 不记录隐藏 chain-of-thought。
3. API Key / Authorization header 永远不落库。
4. exact model payload 可以记录，但必须剔除 credentials。
5. 原始文件二进制不复制进 observation。
6. source refs 可以完整记录。
7. 本轮不实现复杂采样平台和日志管道。

---

# 13. 前端消息状态收敛：删除清单

Codex 必须主动寻找并删除以下重复实现（实际名称以仓库为准）：

- 自定义 SSE token parser（如果 Agent UI runtime 已覆盖）
- 手工 `assistantBuffer`
- `pendingAssistantMessage`
- final response 再 append 一次 message 的逻辑
- submit 后依赖 reload history 才显示 user message 的逻辑
- 自定义 Markdown regex renderer
- 两套 conversation/message store
- Tool Activity 临时状态与 durable server message 的重复 store
- completion 后重新构造整个 transcript 的逻辑
- 为修 duplicate message 添加的临时 boolean / timeout / text-dedup hack

删除前必须通过引用搜索确认无其它消费者。

---

# 14. 实施顺序

## Phase 0：审计

输出一份短审计到本次实现报告：

1. 当前 frontend framework 与版本。
2. 当前 Chat UI / AG-UI 依赖。
3. 当前 message state owner。
4. user submit 路径。
5. assistant stream 路径。
6. duplicate assistant 的真实原因。
7. Markdown 当前由谁 render。
8. Knowledge delete 后端是否已有。
9. 现有三张 trace 表字段。
10. Retriever 内 lexical/vector/RRF 的真实实现位置。

没有审计，不直接改。

---

## Phase 1：聊天 UI 收敛

1. 完成候选组件 Spike。
2. 确定唯一 runtime。
3. 替换消息列表 / composer / markdown / streaming。
4. 删除旧状态与 parser。
5. 修复：
   - user message 延迟
   - assistant duplicate
   - markdown raw display

先完成 Chat Golden Test，再进入下一阶段。

---

## Phase 2：Knowledge Delete

1. Backend delete。
2. UI delete。
3. confirm。
4. READY / ARCHIVED / FAILED 全部验收。

---

## Phase 3：Observation Persistence

1. Migration。
2. ObservationRecorder。
3. model instrumentation。
4. tool instrumentation。
5. retrieval instrumentation。
6. 时间戳修复。

---

## Phase 4：Observability Dashboard

1. Runs。
2. Turn Detail。
3. Timeline。
4. Model detail。
5. Tool detail。
6. Retrieval detail。
7. Raw。
8. visual replay。

---

## Phase 5：删除旧代码 + 回归

不能以：

```text
oldChat + newChat feature flag
```

作为长期交付。

新路径验收后：

- 删除旧入口
- 删除重复 store
- 删除手工 parser
- 删除重复 CSS/component
- 删除废弃依赖

---

# 15. Chat Golden Tests

至少覆盖：

## Case A：立即显示 user message

发送：

```text
你好
```

断言：

- request 发出后立即看到 user message；
- 不依赖 assistant 返回。

## Case B：streaming assistant

服务端分 10 个 delta。

断言：

- 页面始终只有一个 assistant message；
- content 持续增长；
- final 后仍只有一个。

## Case C：Markdown

Assistant 返回：

```md
## 标题

这是 **粗体** 和 `code`。

- A
- B
```

断言按 Markdown 渲染。

## Case D：hydrate

刷新页面恢复历史。

断言：

- user/assistant 不重复；
- streaming 完成消息不重复。

## Case E：Tool

模拟：

```text
search_knowledge
RUNNING → SUCCESS
```

断言 Tool UI 生命周期正常。

---

# 16. Observability Golden Test

使用真实 Golden Case：

```text
讲讲你知道的关于我的事情
```

要求页面能看到：

```text
Turn
├── Model #1
├── search_memory
├── Model #2
├── list_documents
├── search_knowledge
│   ├── lexical candidates
│   ├── vector candidates
│   ├── fused ranking
│   ├── modules
│   └── exact result sent to Agent
├── ...
└── Final
```

对任意 `search_knowledge`：

必须能回答：

1. 输入 query 是什么？
2. lexical 找到了什么？
3. vector 找到了什么？
4. 每项 score/rank 是多少？
5. 如何 fuse？
6. 为什么组成这几个 module？
7. 最终返回给模型的内容是什么？

如果回答不了，本轮 Observability 不算完成。

---

# 17. Knowledge Delete Golden Test

创建三份 fixture：

```text
READY
ARCHIVED
FAILED
```

每份都必须：

```text
UI 点击 Delete
→ Confirm
→ 后端成功
→ 卡片消失
→ document 查不到
→ section/chunk 无孤儿
```

---

# 18. 本轮禁止做的事

为了避免再次膨胀，本轮明确禁止：

1. 不调 RRF threshold。
2. 不新增 reranker。
3. 不重写 Memory 策略。
4. 不强制 Agent 主动写 Memory。
5. 不限制 Tool Call 数来掩盖检索问题。
6. 不改 System Prompt 来解决前端 bug。
7. 不增加 JSONL 作为第二套 Trace。
8. 不实现新的 Agent workflow。
9. 不引入 Redis / Kafka / Celery 做 observability。
10. 不为了 UI 组件增加第二个业务 Agent runtime。
11. 不保留长期 old/new 双实现。
12. 不通过 setTimeout / bool flag / 文本比较修重复消息。

---

# 19. 最终交付物

Codex 完成后必须给出：

## 19.1 修改摘要

按：

```text
Frontend
Backend
DB
Observability
Deleted legacy
```

分类。

## 19.2 根因说明

必须明确解释：

- Markdown 为什么之前没解析；
- user message 为什么延迟；
- assistant 为什么重复；
- created_at 为什么全部相同。

不能只说“已修复”。

## 19.3 删除清单

列出真正删除的旧：

- component
- store
- parser
- helper
- dependency
- CSS

如果没有删除任何旧实现，需要解释为什么这不是补丁叠加。

## 19.4 验收结果

逐条报告：

```text
Chat Golden Tests
Knowledge Delete Tests
Observability Golden Test
Existing backend tests
Frontend build/typecheck/tests
```

## 19.5 暂不处理问题

保留为 P1：

```text
retrieval relevance / Top-K
Memory broad retrieval
主动 Memory 写入
Tool Call 轮次
Prompt / Agent 行为优化
Eval
```

不得在本轮顺手修改。

---

# 20. 完成定义

只有同时满足以下条件才算完成：

- [ ] Knowledge UI 可删除 READY / ARCHIVED / FAILED 文档
- [ ] User message 发送后立即显示
- [ ] Assistant 不再重复两条
- [ ] Markdown 正确渲染
- [ ] Chat 使用一个明确的 runtime/message state owner
- [ ] 不再手搓基础 Markdown / stream / message lifecycle
- [ ] 新增 Observability 入口
- [ ] 可查看历史 Turn
- [ ] 可查看完整 Model 输入输出
- [ ] 可查看完整 Tool 输入输出
- [ ] 可查看 Retrieval lexical/vector/fusion/module 明细
- [ ] 有真实 started_at / finished_at / latency
- [ ] 支持只读 Trace Replay
- [ ] 无第二套业务 Run 状态
- [ ] 无 JSONL 第二套 Trace
- [ ] 新路径通过后删除旧实现
- [ ] P1 行为问题未被“顺手打补丁”
