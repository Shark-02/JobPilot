> 公开发布范围、私有材料及不依赖私有数据的验证命令见 [公开快照说明](PUBLIC_SNAPSHOT.md)。

# JobPilot

> 文档维护规则：任何代码、schema、配置、运行语义、评测、测试或 API 改动，必须在同一迭代同步更新相关 Markdown。当前实现以 [技术总览](docs/JOBPILOT_TECHNICAL_OVERVIEW.md) 为准；具体规则见 [AGENTS.md](AGENTS.md)。

JobPilot 是一个面向求职场景的“职业知识库 + 可控 Agent”原型。它不把简历、岗位、面试等需求拆成彼此割裂的应用，而是以一个可持续对话的 Career Agent 为入口，通过受约束的工具访问用户资料、知识库与用户模型。

项目当前实现的是 V1 范围：可追溯的对话、证据约束的长期记忆、文档知识库检索，以及支持流式文本与原生工具调用的 Agent 运行时。

## 已实现的能力

- 会话与消息：创建、查询、删除会话；支持用户、助手、系统消息及附件关联。
- 文档知识库：保存 Markdown/TXT/PDF/DOCX 来源，抽取文本、分段、切块、去重并建立检索索引。
- 混合检索：词法匹配与按 Active Embedding Profile 隔离的向量检索融合；当前检索调用需要该 profile 的可用嵌入凭据。
- 来源可追溯：检索结果和模型输入均使用 `SourceRef` 表示来源，可定位到消息、文档章节或文档切块。
- 用户长期记忆：Agent 只能通过 `update_user_model` 写入；每条记忆须附带用户消息或用户职业资料的证据来源。
- Agent 运行时：兼容 OpenAI 风格接口，支持流式输出、原生工具调用、多轮工具循环、超时控制、调用轨迹、稳定工作上下文 ref 与确定性 eviction；重启会中止遗留运行，不恢复 AgentTurn。
- 业务工具：`search_memory`、`update_user_model`、`search_knowledge`、`read_source`、`list_documents`、`save_source`；另有 turn-local 内部 `checkpoint_task_state` / `read_working_context`。
- 调试与审计：可查看单次运行、工具调用、模型调用摘要和模型出站组成；完整 observation payload 默认开启，egress preview 默认只存结构摘要。观测 best-effort，不保证工作上下文可恢复。
- 前端控制台：提供 Chat、Memory、Knowledge、Debug 四个页面，用于完成上述核心流程。

## 架构概览

```text
React/Vite 控制台
       │ SSE + HTTP
       ▼
FastAPI API
       │
       ├── Agent Runtime ── OpenAI-compatible LLM API
       │       └── 受控工具注册表
       │
       ├── Conversation / Memory / Knowledge 服务
       │
       └── PostgreSQL + pgvector + pg_trgm
               ├── 对话、运行与审计轨迹
               ├── 用户记忆及其证据来源
               └── 文档、章节、切块与检索索引
```

后端不会把全文资料直接塞入模型上下文。Agent 先看到会话、记忆摘要和附件元信息，再按需调用检索和读取工具；这也是资料可控、引用可审计的基础。

## 技术栈

| 层级 | 主要技术 |
| --- | --- |
| 后端 | Python、FastAPI、SQLAlchemy、Alembic、Pydantic Settings |
| Agent | OpenAI 兼容流式 API、原生 function tools、SSE |
| 数据库 | PostgreSQL 16、pgvector、pg_trgm |
| 检索 | 向量相似度、PostgreSQL 全文检索、trigram 模糊匹配 |
| 前端 | React、TypeScript、Vite、TanStack Query |
| 测试 | Pytest、前端生产构建验证、真实 PostgreSQL 与真实模型链路验证 |

## 快速开始

### 1. 前置条件

- Docker 与 Docker Compose
- Python 3.12 或更高版本
- Node.js 20 或更高版本
- 可用的 OpenAI 兼容模型 API；如需向量检索，还需要嵌入模型 API

新机器需要安装 `uv`；Python 和前端依赖分别由 `backend/uv.lock` 和 `frontend/package-lock.json` 锁定。本地虚拟环境、数据库、API 密钥和模型缓存不随仓库分发。

### 2. 配置环境变量

运行时**只读取项目根目录的 `.env`**。`.env.example` 仅是可复制的模板，不会被程序作为后备配置读取。

```bash
cp .env.example .env
chmod 600 .env
```

在 `.env` 中填写模型服务的地址、密钥和模型名。请勿提交 `.env`，它已被 Git 忽略。

最小配置项：

```dotenv
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_API_KEY=your-key
LLM_MODEL=your-chat-model
```

Context Lifecycle V1 可选配置（不新增 migration）：

```dotenv
# 不设置时使用 min(48000, int(hard_input_limit * 0.70))
# CONTEXT_EVICTION_TRIGGER_TOKENS=48000
RECENT_RAW_RUNTIME_KEEP_COUNT=1
WORKING_TASK_STATE_MAX_TOKENS=6000
```

完整 token budget 含 tool schemas；runtime keep count 最小为 1，并独立于 Conversation target/max。`CONTEXT_COMPACTION_MAX_TOKENS` 仅暂留 legacy 配置。

可选嵌入配置：

```dotenv
EMBEDDING_BASE_URL=https://your-openai-compatible-endpoint/v1
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=your-embedding-model
```

> `EMBEDDING_BASE_URL/MODEL/DIM` 仅用于首次 `0005_embedding_profiles` 迁移注册当前空间；迁移已有数据时必须保持原来的模型、地址和维度。迁移后在线模型由数据库唯一 Active Profile 决定，环境变量不能直接切换模型；API key 仍从环境读取。Knowledge/Memory 使用独立的 profile 向量表，支持不同维度与同维度不同模型隔离。

后续切换流程为显式 `register → backfill → retrieval/Eval 验证 → validate → activate`；只为 Active Profile 实时写入，历史 profile 不双写。Backfill 与 Activate 完全独立，缺少任何 eligible 向量或索引/验证不通过时拒绝切换。命令与边界见 [Embedding Profile 合约](SPEC/EMBEDDING_PROFILES.md)。2026-10-06 Phase B 的 Qwen ACTIVE / BGE READY 是历史状态，执行证据见 Phase B 报告（历史资料未包含在公开快照中）。2026-10-08 开工只读 DB inventory 显示 BGE 已 ACTIVE；本轮保留该前置状态，没有 activation。稳定注册定义是 `backend/config/embedding_profiles/bge-m3-school-v1.json`。identity v2 排除 endpoint、credential 环境变量名、batch/input admission metadata；0006 保留旧 Qwen UUID。

Backfill 前可运行 `python -m app.embedding compatibility --profile <name>`；全量扫描含 special tokens，任何超限在真实 API 调用前停止。`hf-json:<sha256>` 需要提供 hash 匹配的 `artifacts/tokenizers/<sha256>.json`（或 `EMBEDDING_TOKENIZER_DIR`），不自动下载/截断。学校 BGE-M3 Profile 使用独立的 base_url=https://myai.bupt.edu.cn/llm-gw/v1 和 SCHOOL_EMBEDDING_API_KEY 凭据，与 Agent/Judge 的 LLM_API_KEY 解耦。Embedding Profile 的运维配置通过稳定 JSON 定义及 app.embedding register 同步，Eval Corpus 使用 Source Profile 同步到隔离数据库。

### 3. 启动 PostgreSQL

```bash
docker compose up -d postgres
docker compose ps
```

默认连接信息见 `.env.example`；可以按部署环境修改 `DATABASE_URL`。

### 4. 启动后端并执行迁移

安装 [uv](https://docs.astral.sh/uv/) 并确保 `uv` 已加入 `PATH` 后，执行：

```bash
cd backend
uv sync --locked --extra dev
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

服务启动后可访问：

- OpenAPI：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

### 5. 启动前端

另开一个终端：

```bash
cd frontend
npm ci
npm run dev
```

默认访问地址为 <http://127.0.0.1:5173>。开发服务器会将 `/api` 请求代理到后端 `127.0.0.1:8000`。

## 推荐使用流程

1. 在 Knowledge 页面保存一份职位描述、项目材料或个人资料来源。
2. 在 Chat 页面新建会话，必要时为消息附加知识库文档。
3. 向 Agent 提问。它会按需搜索长期记忆与知识库，再读取具体来源。
4. 当 Agent 需要记录稳定、可复用的个人信息时，调用 `update_user_model`，并附上证据来源。
5. 在 Memory 页面核对已写入的长期记忆；在 Debug 页面核对本次工具调用与模型出站摘要。

## 核心 HTTP 接口

所有接口均带 `/api` 前缀，完整字段以 `/docs` 中的 OpenAPI 定义为准。

| 领域 | 方法与路径 | 用途 |
| --- | --- | --- |
| 健康检查 | `GET /health` | 服务与数据库连通性检查 |
| 会话 | `POST /api/conversations` | 新建会话 |
| 会话 | `GET /api/conversations` | 获取会话列表 |
| 会话 | `GET /api/conversations/{id}` | 读取 Conversation 与消息历史 |
| Agent | `POST /api/ag-ui/conversations/{id}` | 浏览器使用的 AG-UI SSE run |
| Agent | `POST /api/conversations/{id}/runs` | REST 创建 run，返回 events 地址 |
| Agent | `GET /api/turns/{id}/events` | REST run 的 SSE event stream |
| 文档 | `POST /api/documents` | 保存并入库文本来源 |
| 文档 | `GET /api/documents` | 查询知识库文档 |
| 记忆 | `GET /api/memories` | 查询可见的长期记忆 |
| Observability | `GET /api/observability/turns` | 查看 run、token 与按序 observation |
| 调试 | `GET /api/debug/turns/{turn_id}` | development-only 查看 Turn 与 egress trace |

Agent 流接口会依次发出如 `RUN_STARTED`、`TEXT_MESSAGE_CONTENT`、`TOOL_CALL_*`、`TEXT_MESSAGE_END`、`RUN_FINISHED` 或 `RUN_ERROR` 的事件。前端应以事件流为准，而不是假定一次请求只产生一个完整文本块。

## 证据与安全约束

- 记忆写入不接受助手消息或网页引用作为唯一证据；只能引用用户消息或用户职业资料来源。
- Context Lifecycle V1 在当前 `AgentContext` 内保留全部完整 interaction；压力只按 oldest-first 改变 active membership，不调用 Compactor、不删除原文。稳定 `ctx://<turn>/i/<seq>` 可由内部 `read_working_context` 精确读取；内存缺失时仅 best-effort 使用现有 MODEL/TOOL observations，无法重建则明确返回 `WORKING_CONTEXT_NOT_RECOVERABLE`。
- 每个 Agent 回合设有总超时、单工具超时和最大工具循环次数；同一个会话同时只能运行一个 Agent 回合。
- 服务器重启时，未完成的运行会被标记为已中止，避免留下伪运行中状态。
- `OBSERVABILITY_CAPTURE_PAYLOAD=true` 默认保存完整 model/tool observation payload；egress preview 默认只保存结构摘要、长度和来源。仅在本地受控排查时，才可将 `MODEL_EGRESS_PREVIEW_STORE_CONTENT=true` 打开以额外保存完整 preview。
- `.env` 包含密钥，只从本地读取，永不应提交、打印或写入调试轨迹。

## 开发验证

后端测试和前端生产构建：

```bash
cd backend
uv run --locked pytest tests/unit -q --tb=short
# 需要单独准备 canonical jobpilot_eval 数据库，见评测指南：
JOBPILOT_TEST_POSTGRES=1 uv run --locked pytest tests/integration

cd ../frontend
npm ci
npm run build
```

首次使用 tiktoken 的编码时可能需要联网下载官方编码文件；本机已有缓存不随仓库分发。2026-11-08 在私人资料齐全的原工作区补齐缓存并修正测试后，单元测试结果为 **683 passed**（10.55 秒），前端生产构建通过。该结果不是新机器端到端复现证明。上面的完整测试命令列出原工作区验证入口；公开快照缺少私人评测输入，依赖这些输入的测试不能原样运行通过。

评测案例、fixtures、检索案例与证据标注、私人 baseline、冻结真实语料、数据库快照、向量与私有运行 trace 不随公开快照分发，因此新克隆不能直接复现依赖这些输入的历史真实评测或分数；不能用自备材料替换后声称得到同一实验。评测流程和隔离要求见 [Evaluation Guide](docs/EVALUATION.md)。

本迭代已完成以下真实链路验证：

- PostgreSQL 容器、扩展与 Alembic 迁移实际执行成功。
- 会话、消息、文档入库、分块、词法检索、记忆证据校验和清理链路通过 HTTP 验证。
- 使用 `.env` 中实际配置的模型服务，完成了基础流式回复验证。
- 使用真实模型完成了知识库检索/读取、记忆检索/写入、跨会话记忆召回及 `save_source` 工具循环验证。
- 2026-10-17 Context Lifecycle V1 改为 deterministic eviction、内存精确 ref read 与 add-only Working Task State；完整 request 的 budget 包括 tool schemas，provider 终止错误保留 classification。本次没有新增表或 migration，已有 `0004_context_compaction` 是历史 trace schema。当前验证记录见 [技术总览](docs/JOBPILOT_TECHNICAL_OVERVIEW.md#15-testing)；真实 A1/Suite 尚未运行，仍需隔离 readiness 和隔离 fixture/corpus 的模型出站条件。

上述 PostgreSQL、HTTP 与基础模型链路是历史验收，不等同于本次 Context Lifecycle V1 的真实模型验收。历史命令和范围见 迭代报告（历史资料未包含在公开快照中）；本次代码、验证与限制以 [技术总览](docs/JOBPILOT_TECHNICAL_OVERVIEW.md) 为准。

## Eval：正式 baseline 前合同

Agent/Memory/Runtime 执行协议保持 `eval_contract_v3` / `direct_semantics_1`，18 个案例继续使用 authoring `schema_version: "2"`；Retrieval 的 9 个案例使用独立 `retrieval_contract_v1`，见下文。模型直接判断任务完成、事实支持与能力边界；代码计算数量、集合、顺序、局部改稿相等性、执行状态与最终结果。除 Agent `answer_grounding` 外，每个 semantic check 独立生成 `semantic:<check_id>` JudgeTask，criterion 与响应校验只包含该 check。`answer_grounding` 保持同一 compiled check、Gold、criterion、source-policy 和权威 Evidence 边界，按现有有效 answer spans 的原始顺序每四个切为一个 task，ID 为 `semantic:answer_grounding:shard:001` 起的稳定序号；每个 shard 只能引用自己的 spans。既有分类/条目 extraction 仍只规划一次；遇到 grounding shards 时，它使用完整 answer observation task，避免缩小已有 extraction 的 span 范围。来源权限、历史观察时点和 Evidence 裁剪不变。已删除通用量化抽取/核验、精确引用和独立抽取完整性评判。普通语义输出只需 ID、PASS/FAIL/UNKNOWN 和短理由，bullet/问题选择原始 span，由代码恢复原文。Memory 历史来源、目标条目和指定 fixture 的身份绑定保留。

默认并发 2、每请求 timeout 600 秒、输出预算 6144、有限传输重试，可选成功缓存；旧 claim 分片/批量配置已删除。`answer_grounding` 的逻辑结果采用 ALL：全部 shards PASS 才 PASS，任一 shard FAIL 即 FAIL；没有已建立 FAIL 时的 shard provider interruption 为 BLOCKED，schema/evaluator 错误仍为 ERROR。该 answer-span 切分不恢复 material-claim pipeline。所有 required check 满足才 PASS；合法 FAIL/UNKNOWN 是评分结果，截断/JSON/schema/capture 故障为 ERROR，provider interruption 为 BLOCKED。质量结果与未评分数量分开，正式 baseline 尚未冻结。实现及跨任务离线验证见 全局精简实施记录（历史资料未包含在公开快照中），配置和操作见 [Evaluation Guide](docs/EVALUATION.md)。

2026-10-16 Retrieval MVP 已实现：三个根 YAML 定义 53 个 stable EU、R1–R9 facts/proof alternatives 和原有 K=5/8000 预算。指标为 EvidenceRecall、MRR、FactRecall、CompleteSupport；正常输出 SCORED，R9 的普通指标为 null，不要求 raw 空返回。产品 Retrieval/Embedding 未修改。验证为 592 unit + 25 隔离 PostgreSQL integration，真实检索前的 oracle/来源审阅与 BGE 运行记录见 Retrieval MVP 实施记录（历史资料未包含在公开快照中）。此前设计附件为历史提案，完整 TREC 式扩展未实现。

通用 Judge-only Replay 和单任务 Probe 保留冻结来源、私有 trace、独立输出目录与请求身份验证。新版只接受 `eval_contract_v3` 冻结计划；旧 A1 v2 计划在请求前明确拒绝，不能自动迁移或沿用旧缓存。Probe 的 `--prepare-only` 只检查输入/任务，不创建客户端或输出目录；`--verify-extracted` 已删除。5 份 A1 专用脚本和历史报告保留为旧协议实验材料，不是当前操作入口。

用户随后运行九个代表案例，A6/F2 的三个 Judge 请求均完整成功；主要运行失败发生在 Agent/Runner 阶段。已修复跨事务身份、失败终态计数和条目标签/正文边界，新增跨任务回归后 468 单测、27/280 preflight 通过。当前优先使用完成案例的冻结观测做人工校准，原结果不改写、不立即重跑全部 Agent，详见 运行审查与复核命令（历史资料未包含在公开快照中）。

产品库 `jobpilot` 禁止任何 Eval reset/write。唯一 disposable Eval DB 是 `jobpilot_eval`；需要 fresh 环境时 reset 同一个 DB。同库 Profile A/B 使用显式 binding，不依赖产品 active pointer；缺完整冻结向量覆盖时保持 REGISTERED，完整覆盖后才 READY。旧 Phase B 的日期库和旧 runs 已清理/归档。

首次正式 baseline 沿用既定显式 profile `bge-m3-school-v1`。Corpus setup/backfill/check、代表案例、suite、full 和 variance 均显式绑定 BGE；Qwen 保留为历史/对照 profile。Eval binding 独立于产品 active pointer，不执行 activate。

```bash
cd backend
export TIKTOKEN_CACHE_DIR="$PWD/../artifacts/tiktoken-cache"  # 本地已校验的官方tokenizer cache
.venv-py313/bin/python -m app.evals.database setup
.venv-py313/bin/python -m app.evals.corpus setup --embedding-profile bge-m3-school-v1
.venv-py313/bin/python -m app.evals.database smoke
pytest tests/unit
JOBPILOT_TEST_POSTGRES=1 pytest tests/integration
# Direct Judge 的无真实 API / 无数据库回归
.venv-py313/bin/python -m pytest tests/unit -q
```

Corpus setup 验证六份冻结原文 SHA 并复制既有向量；新 JobPilot snapshot 仅在 Eval 本地 parse/chunk。缺向量时明确阻止真实 run，显式 Eval-only embedding backfill 需要独立成本授权。历史重构与清理报告中的无真实模型/embedding/Judge 验证仅说明其各自记录的范围；2026-10-11 用户已完成首轮真实 compact Judge-only 重放，因超时为 BLOCKED。正式 baseline 仍未生成。真实运行与人工审阅命令见 [Evaluation Guide](docs/EVALUATION.md)，实施与清理证据见 重构报告（历史资料未包含在公开快照中） 和 DB 清理报告（历史资料未包含在公开快照中）。

## 当前边界与待补项

- 仍需把规格中的 G1–G9 Golden Scenarios 整理成可重复执行的端到端测试套件，并固定断言与报告格式。
- 前端已具备核心页面和 SSE 消费逻辑；`assistant-ui` 相关依赖尚未进一步整合为其完整原生组件体系。
- PDF/DOCX 的复杂排版、表格和图像语义提取仍需在真实样本上继续加强。
- 生产部署还需要补齐鉴权、权限隔离、速率限制、观测告警、备份策略和密钥托管。
- 当前项目严格遵循 V1 的“一个 Career Agent + 一组受控工具”边界，不包含岗位爬取、自动投递、训练营、面试模拟等额外产品模块。

## 项目结构

```text
.
├── backend/
│   ├── app/                 # API、Agent、会话、记忆、知识库服务
│   ├── migrations/          # Alembic 数据库迁移
│   └── tests/               # 单元与契约测试
├── frontend/                # React/Vite 控制台
├── SPEC/                    # 产品与实现规格（项目范围的依据）
├── docs/iterations/         # 每轮实现与验收报告
├── docker-compose.yml       # 本地 PostgreSQL 服务
└── .env.example             # 环境变量模板
```

## 迭代记录

- Iteration 001（历史资料未包含在公开快照中）：V1 基础实现、代码审查修正与真实环境验证。
- [Jev 重排序实验（2026-10-27）](experiments/jev_ab/README.md)：合成小样本中，MRR 的提升来自一条查询，完整证据指标未改善，存在误排与较高串行延迟；证据不足以支持生产接入。报告区分第三方实测与官方模型调用，并说明复现条件。
