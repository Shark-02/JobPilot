# JobPilot V1 — Product & User Model

**状态：Baseline Candidate**  
**定位：产品定义、长期用户模型与持久化原则**

---

## 1. 产品定义

JobPilot 是一个 **Chat-first、长期记忆驱动的个人 Career Agent**。

产品主入口只有一个持续对话的 Agent。用户直接表达目标，Agent 结合当前对话、长期记忆和原始职业资料，自主决定是否需要检索、需要检索什么、是否需要更新长期用户模型，并直接完成当前任务。

JobPilot 的核心不是管理 Job、Resume、Interview、Training 等业务对象，而是：

> **持续积累对用户有长期价值的认知，使 Agent 越用越懂用户，并在新的求职任务中利用这些认知。**

JD 分析、简历生成、面试复盘、训练、项目问答等都只是 Agent 的任务场景，不是独立业务系统。

---

## 2. 产品架构

```text
                         User
                          │
                          ▼
                  ┌───────────────┐
                  │ Career Agent  │
                  └───────┬───────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         Current      Long-term    Knowledge
         Context       Memory       Sources
              │           │           │
              └───────────┼───────────┘
                          │
                          ▼
                  Agent Intelligence
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       Current-task Output      Durable User Change
       回答 / JD分析 /          update_user_model
       简历 / 训练 / 复盘              │
              │                       ▼
              ▼                    Memory
         Conversation
```

---

## 3. 核心产品原则

### 3.1 Agent 是产品本体

系统提供：

- 当前对话上下文；
- 少量核心用户画像；
- 长期记忆检索能力；
- 原始资料检索能力；
- 长期用户模型更新能力；
- 必要的技术约束。

Agent 负责：

- 理解用户真实意图；
- 判断是否需要历史信息；
- 决定搜索什么、搜索多少；
- 判断信息是否充分；
- 完成 JD 分析、简历生成、面试复盘、训练等任务；
- 判断哪些新信息值得长期记住；
- 判断已有长期认知是否需要更新。

代码不为这些任务建立固定工作流。

### 3.2 Persist user knowledge, not every derivation

默认长期保存：

```text
用户原始输入与对话
用户主动上传的职业资料
高密度长期 Memory
Memory 与原始 Source 的关系
```

默认不建立独立持久化业务对象：

```text
JD Analysis
Job Match
Resume Draft
Resume Revision
Interview Review
Training Plan
Training Task
Reference Answer
Artifact Version
```

这些结果默认存在于 Conversation 中。

只有当用户明确要求“保存”“设为长期基准”“以后继续使用”时，才把对应内容保存为长期 Source。

### 3.3 Current task output ≠ Long-term user model

Agent 本轮生成的内容，例如：

- 一版针对 JD 的简历；
- 一份岗位匹配分析；
- 一份面试复盘；
- 一组训练题；
- 一个参考答案；

不会因为生成过就自动成为长期事实或 Memory。

真正进入长期用户模型的是这些任务暴露出的、未来仍会影响 Agent 行为的信息，例如：

- 当前重点求职方向；
- 用户真实项目经历；
- 用户明确的表达偏好；
- 某知识点近期掌握不稳定；
- 某项目问题已经可以稳定说明；
- 用户正在持续准备某个岗位。

---

## 4. 信息分层

JobPilot 只保留三类长期信息和一类短期上下文。

```text
┌────────────────────────────────────┐
│ Conversation / Working Context     │
│ 当前对话与当前任务，短期工作空间      │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Core Profile View                  │
│ 从 CORE Memory 形成的极小常驻视图    │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Semantic Memory                    │
│ 高密度、结构化、长期用户认知          │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Knowledge Sources                  │
│ 用户原始资料与权威证据               │
└────────────────────────────────────┘
```

### 4.1 Conversation

Conversation 是短期工作空间。

它保存：

- 用户消息；
- Agent 最终回答；
- 当前 JD；
- 当前生成的简历；
- 当前训练过程；
- 当前面试复盘；
- 当前对话内需要继续引用的临时结果。

这些内容可以直接支持后续同一 Conversation 的继续交流，但不会自动成为长期 Memory。

### 4.2 Core Profile View

Core Profile 不是独立事实数据库。

它只是从 `importance=CORE` 的 ACTIVE Memory 中形成的小型上下文视图，用于让 Agent 在新 Conversation 开始时知道“这个用户大概是谁”。

典型包括：

- 当前身份；
- 教育背景；
- 当前求职方向；
- 核心经历；
- 主要技术方向；
- 重要长期偏好。

目标大小：**约 300–800 tokens，硬上限 1200 tokens**。

V1 不要求独立 `core_profile` 表；可以由 ContextBuilder 从 CORE Memory 动态组装或缓存。

### 4.3 Semantic Memory

Memory 是：

> **由 Agent 提炼并维护、未来可能显著改变其帮助方式的高密度长期用户信息。**

Memory 不是：

- 聊天全文；
- Document Chunk；
- 每次任务输出；
- 模型自由发挥的文案；
- 所有临时判断。

---

## 5. Memory 模型

### 5.1 MemoryItem

概念模型：

```python
class MemoryItem:
    id: UUID

    kind: MemoryKind
    content: str
    attributes: dict

    importance: CORE | NORMAL
    status: ACTIVE | SUPERSEDED | ARCHIVED

    valid_from: datetime | None
    valid_to: datetime | None

    created_at: datetime
    updated_at: datetime
```

`content` 是给模型直接阅读的高密度自然语言表达。  
`attributes` 只保存有实际用途的结构化补充，不要求每种 Memory 共享复杂固定 Schema。

### 5.2 MemoryKind

V1 保持少量通用类型：

```text
BACKGROUND
EXPERIENCE
ACHIEVEMENT
SKILL
CAREER_GOAL
PREFERENCE
LEARNING
OTHER
```

不建立大量细粒度 FactType。

### 5.3 示例：职业事实

```json
{
  "kind": "EXPERIENCE",
  "content": "Intelligent Oncall 在工作流推进中使用 CAS 与 SELECT FOR UPDATE 控制并发状态转换。",
  "attributes": {
    "project": "Intelligent Oncall",
    "topic": "concurrency",
    "mechanisms": ["CAS", "SELECT FOR UPDATE"]
  },
  "importance": "NORMAL"
}
```

### 5.4 示例：学习认知

```json
{
  "kind": "LEARNING",
  "content": "Kafka ACK 相关概念近期掌握不稳定，容易混淆 producer acknowledgement 与 consumer offset commit。",
  "attributes": {
    "topic": "Kafka ACK",
    "state": "unstable"
  },
  "importance": "NORMAL"
}
```

`state` 只是模型可选写入的描述字段，不由 Python 维护状态机，也不定义“答对几次才 stable”的固定规则。

---

## 6. Memory 的知识边界

### 6.1 两类 Memory

Memory 可以表达两类长期认知：

**A. Source-backed fact**

例如：

> 用户在 Oncall 项目中使用过 SELECT FOR UPDATE。

必须被用户消息或职业资料直接支持。

**B. Agent assessment**

例如：

> Kafka ACK 目前掌握不稳定。

这是 Agent 基于用户真实回答形成的长期判断，不是原始事实。它必须：

- 明确写成“当前表现/掌握情况”的判断；
- 指向真实的用户回答或面试记录；
- 不能被表述为无依据的绝对事实。

### 6.2 Source 必须存在

所有 Memory 至少有一个 SourceRef。

允许的主要来源：

```text
message://{message_id}
document://{document_id}/chunk/{chunk_id}
document://{document_id}/section/{section_id}
```

生成的 Resume、Agent Answer、JD Analysis 等不能作为证明用户事实的新 Source。

---

## 7. MemorySource

```python
class MemorySource:
    memory_id: UUID
    source_ref: SourceRef
    relation: SUPPORTS | CONTRADICTS
    excerpt: str | None
```

SourceRef 必须能被 `read_source` 再次读取。

---

## 8. Memory 更新模型

主 Agent 通过统一 Tool 更新用户模型：

```python
update_user_model(
    mutations: list[MemoryMutation]
)
```

支持：

```text
ADD
MERGE
SUPERSEDE
ARCHIVE
```

### 8.1 Agent 决定何时写

系统 Prompt 只给原则，不用代码规定复杂更新逻辑。

通常值得写入：

- 用户明确的长期职业事实；
- 用户明确的长期目标或偏好；
- 实际项目资料中被当前任务真正用到、且未来可能重复使用的重要信息；
- 面试或训练中表现出的、未来会改变训练策略的明显能力变化；
- 用户对已有长期信息的明确纠正。

通常不值得写入：

- 临时聊天内容；
- 当前任务生成结果；
- 普通知识问答；
- 一次性措辞；
- Agent 自己生成的总结；
- 对未来帮助方式没有明显影响的信息。

### 8.2 更新已有 Memory 前先检索

Prompt 要求 Agent 在修改相关长期认知前优先：

```text
search_memory
```

避免重复 ADD。

`update_user_model` 还应有最小数据卫生保护：

- SourceRef 必须存在；
- target memory 必须存在；
- 操作结构合法；
- 对明显重复内容可以返回 `POSSIBLE_DUPLICATE`，由 Agent 决定 MERGE 或取消。

不使用复杂阈值规则替 Agent 做语义判断。

### 8.3 用户纠正

例如：

> 我之前说错了，Oncall 这里不是 Redis 锁，是 SELECT FOR UPDATE。

Agent：

```text
search_memory(...)
↓
update_user_model(
  SUPERSEDE old memory,
  ADD/MERGE corrected memory
)
↓
Final answer
```

不需要额外 Memory Writer。

---

## 9. 不使用独立 Memory Reflection Pipeline

V1 不默认执行：

```text
Final Answer
↓
第二个 LLM 再读一遍
↓
Memory Reflection
```

主 Agent 已经拥有最完整的当前上下文，应在本轮 reasoning 中自行判断是否需要调用 `update_user_model`。

只有真实评测证明主 Agent 经常漏记重要信息或过度写入 Memory 时，才重新评估独立 Reflection。

---

## 10. Knowledge 与 Memory 的关系

```text
Knowledge / User Message
        │
        ├────────────→ 直接支持当前任务
        │
        └────────────→ Agent认为值得长期复用
                              │
                              ▼
                            Memory
```

Knowledge 本身就是权威 Source。

Memory 的价值是：

- 高密度；
- 跨会话易召回；
- 能表达 Agent 对用户形成的长期认知。

Memory 不是 Source 的“认证层”。

---

## 11. 典型任务模式

JD、Resume、Interview、Training 都是 Agent interaction，不是独立 Domain。

### 11.1 JD

用户直接粘贴 JD。

Agent已经能理解当前文本，不调用另一个 `analyze_jd` LLM。

需要用户历史信息时：

```text
search_memory
search_knowledge
```

如果用户表达：

> 接下来重点准备这个岗位。

Agent调用：

```text
update_user_model
```

记住当前长期目标，并把当前用户消息作为 SourceRef。

未来新 Conversation：

```text
search_memory("最近重点准备的岗位")
↓
read_source(message://...)
```

恢复完整 JD。

### 11.2 Resume

Agent根据：

- 当前 JD；
- 当前 Conversation；
- Memory；
- Knowledge；
- 用户已有简历 Source；

直接生成简历。

修改一句简历时，Agent直接修改当前 Conversation 中的文本。

默认不建立 ResumeVersion 或 ResumeClaim 表。

如果用户明确要求长期保存某版简历，则把它保存为新的 Knowledge Source，而不是创建完整 Resume Workflow 系统。

### 11.3 Interview

原始面试记录作为 Message 或 Knowledge Source 保存。

Agent直接复盘。

真正值得长期留下的是面试暴露的用户认知变化，由 Agent写入 Memory。

不建立 InterviewReview、InterviewQuestion 等业务表。

### 11.4 Training

Agent根据 Memory 和当前目标动态决定练什么。

训练题、参考答案、过程性点评都停留在 Conversation。

用户真实回答如果足以改变未来训练策略，Agent更新 LEARNING Memory。

不建立 TrainingTask、TrainingAttempt、LearningState 状态机。

---

## 12. 持久化边界

V1 产品长期核心数据：

```text
Conversation
Message

KnowledgeDocument
KnowledgeSection
KnowledgeChunk

MemoryItem
MemorySource
```

运维数据另行记录，但有 TTL。

默认不存在：

```text
Job
JobAnalysis
JobMatch

Resume
ResumeVersion
ResumeClaim

InterviewReview
TrainingTask
TrainingAttempt
LearningState
Artifact
```

---

## 13. Hard Invariants

1. JobPilot 是 Agent 产品，不是 Career CRUD 系统。
2. JD、Resume、Interview、Training 是任务场景，不是独立业务 Domain。
3. Conversation 是短期任务工作空间。
4. Knowledge 保存原始 Source。
5. Memory 保存高密度长期用户认知。
6. Core Profile 只是 CORE Memory 的小型视图。
7. Derived Output 默认只留在 Conversation。
8. 用户明确要求长期保存时，输出可以转为 Knowledge Source。
9. 主 Agent 自己决定何时更新长期用户模型。
10. V1 不默认使用独立 Memory Reflection LLM。
11. 所有 Memory 必须可追溯到真实 Source。
12. Agent 对学习状态的判断可以进入 Memory，但必须基于真实用户表现并表述为 assessment。
13. 不用固定状态机或硬编码答题次数替 Agent 判断用户掌握程度。
14. 代码负责数据完整性，Agent负责语义判断。
