# JobPilot Eval Contract 与运行说明

更新时间：2026-10-20（Asia/Shanghai）。Agent/Memory/Runtime 保持 `eval_contract_v3` / Judge `direct_semantics_1`，其 18 个案例继续使用 authoring `schema_version: "2"`。Retrieval 的 R1–R9 已迁移到三个根 YAML 与独立 `retrieval_contract_v1`：stable EU、facts/proof alternatives、deterministic provenance、四项指标，正常输出 SCORED。模型 Judge 不参与 Retrieval 评分。最小实现、迁移与验收见 Retrieval MVP（历史资料未包含在公开快照中）。正式 full-suite baseline 尚未冻结。

## 执行合同

```text
Authoring Case → Strict Typed Dataset → Compiler → ExecutableEvalPlan
             → Observations → Parallel Judge Tasks → Strict JudgeResponse
             → Deterministic Checks → Aggregator → PASS / FAIL / BLOCKED / ERROR
```

`backend/app/evals/schema.py` 使用 RetrievalCase、九种 Agent task_type、MemoryCase、RuntimeFailureCase 的严格 discriminated union。未知字段、无 executor 的评分字段、未知 check、互斥 Gold、无效 source/pass/fault policy 在 dataset preflight 立即 ERROR。Case ID 只标识 case、artifact 和报告；非 Retrieval 执行语义由 setup、task_type、Gold、checks 和 policy 编译，Retrieval 编译 facts/answerability 到 metrics-only 路由。`compiled_plans.json` 与逐项 check result 记录 Gold path、compiled check ID、evaluator kind、expected、actual、evidence IDs、结果和失败/跳过原因。

Judge 只返回原子 `checks` 与受限 `extractions`，没有总体 verdict。required check 的 FAIL/UNKNOWN 由程序聚合为 FAIL；optional 子项只由显式 any/min group 聚合。自然语言分类、主题和事实支持由模型识别；项目集合、bullet 数/顺序、题数/追问、topic coverage 和来源身份由代码计算/验证。数字的单位、范围与项目/因果归属由语义 rubric 判断，不再由字符串守卫决定。不能用另一项 PASS 覆盖 grounding failure。

A1–A9 分别为 profile_summary、mechanism_mapping、project_deep_dive、jd_matching、resume_generation、local_revision、interview_preparation、interview_review、evidence_fusion。保留专有 Gold，避免把任务压成通用维度列表。A6 的最终稿必须只改变目标 bullet，其余文字、顺序和条数相同；目标事实的 span 必须指向实际目标 bullet。A5 的已知量化 Gold 是 optional claim，其他职业数字只有在实际可见证据支持、项目/指标/机制归属正确时才允许。

## Evidence 与 fixture

Eval 测 Agent 对本次冻结 corpus、用户消息、fixture 和 Memory 的忠实使用，不维护仓库外现实世界的绝对真相。当前状态使用 corpus snapshot truth；历史面试保留 event-time truth；学习状态 Memory 忠实记录用户自报 stable/unstable，不将 Java 知识正确性混入该 Memory rubric。

主 Agent 实际发送的 model request 由 eval-only collector 在 provider 调用入口完整采集，不依赖普通业务 trace 的 best-effort 持久化。collector 校验主模型调用数、工具执行数与 mutation 事件覆盖。没有 48,000 字符静默截断；缺失已实际可见的必要记录属于 harness ERROR。Agent 自己没有获得所需资料，属于正常质量 FAIL 的可能原因。

每项 evidence 有稳定 ID、origin_type、SourceRef、正文、Agent 可见性、grounding authority、turn/call/conversation/seed origin。按 exact SourceRef 与实际 excerpt identity 去重，保留被主 Agent 看过的原文。Knowledge module、read_source 原文与用户消息可作为权威事实。历史 assistant、模型生成 Memory、compacted state、UI summary 和执行元数据明确标为 derived/metadata，不作为一般事实兜底；corpus manifest 的 authoritative origin policy 同步约束可用来源。

Memory evaluator 独立读取实际绑定 source 的完整正文，用于检查持久化 Memory 的来源支持。它们是 evaluator-only `bound_sources`，只允许 bound-memory 检查引用；既不注入 Agent，也不作为普通 answer-grounding 的 Evidence。跨会话判定同时观察 B 的真实 search_memory result、B 最终 answer 与 source binding。

A8/M4 面试 fixture 只保留当时真实问答；分类摘要留在 Gold/evaluator annotation。A9 用户补充只注入一次，并保存唯一 message SourceRef；精确 6.00s→0.18s、其他条件一致、结果一致绑定该 source，且与 input_hash/AES-GCM 的另一条优化分开。

## 三套专用执行规则

Retrieval 只维护 `retrieval_evidence.yaml`、`retrieval_cases.yaml`、`retrieval_protocol.yaml`。53 个 EU 绑定冻结 document/snapshot/canonical/span provenance；R1–R8 共 43 required facts，proofs 外 ANY、内 ALL，query 的 facts 全部 required。原 corpus 与 query 不变；不再读取 anchors/group pass_policy 或旧 `cases/retrieval.yaml`。前 K=5 个实际 modules 的 source span union 完整覆盖 EU 后才计分，预算仍 8000 tokens。输出 EvidenceRecall、MRR、FactRecall、CompleteSupport，及 covered EU / missing facts / witnesses；不实现 pooling、graded nDCG、density、calibration 或 release statistics。

R9 显式 unanswerable，四项普通指标为 null；raw TopK 非空不判失败，abstention=NOT_EVALUATED。SCORED 只表示指标已计算，不把 CompleteSupport 自动转为 case PASS。纯 core/adapter/实际 Gold oracle 在真实检索之前通过，完整验证与运行结果见 本次实施记录（历史资料未包含在公开快照中）。历史 anchor/clean-negative run 不与新指标直接比较。

Memory 以实际 mutation event 为观察中心，记录 operation、是否 commit、目标 Memory、目标 turn、事务内 before/after、new SourceRef 和 source 正文。事务整体 commit 后才标记 success；rollback 只保留失败尝试。seed 操作不能满足 target-turn update；写入后 archive 不能伪装成未写。目标主题的 ACTIVE 数量、状态和演化从完整 Memory/topic extraction 聚合，其他主题不干扰 HashMap case；同一 Memory 可以合法覆盖多个主题。M5 接受有新 evidence 的 MERGE/SUPERSEDE；M6 拒绝 stable/unstable 冲突 ACTIVE 状态，并保留旧来源/历史状态。M4/M7 必须实际跨会话召回目标 Memory，并在 B 最终回答中正确使用。

Runtime fault 从 typed YAML 的 target_tools、trigger/occurrence、error_code、retryable、then 执行。工具 ledger 区分 injected、real、cache 与 duplicate guard。未触发声明故障为 ERROR，不能 PASS。Recovery 必须第一轮注入失败之后，同 turn 的目标工具发生真实成功、取得非空原文证据，最终任务完成；cache replay 不算真实 recovery。持续 evidence failure 必须按预算正常停止，明确无法取得证据，且没有未获得来源的项目细节。attempt 和单一最终答复限制真实执行。

## Direct Streaming Judge

`StructuredJudge.judge(plan=..., payload=...)` 保持 Runner 接口。除 Agent `answer_grounding` 外，Planner 把每个 semantic check 生成为一个独立 JudgeTask，任务 ID 固定为 `semantic:<check_id>`，其 criterion、响应校验和结果恢复只覆盖该 check；即使 `support_mode`、`observation_scope`、`target_step_id` 与裁剪后的 payload 完全相同也不合并。`answer_grounding` 仍是同一 compiled check，只把现有有效 answer spans 按原始顺序确定性切为最多四个 spans 的 shard，ID 为 `semantic:answer_grounding:shard:001` 起的稳定序号。每个 shard 只看到并校验自己的 spans，仍使用原有 source-policy、真实观察时点、目标 step 和完整可用权威 Evidence；Gold、criterion 和 Evidence 权威规则不变。project/classification/topic/bullet/question extraction 仍按既有规则只规划一次；当原附着目标是 grounding shard 时，Planner 改用一个带完整 answer view 的 observation task，以保持 extraction 的既有全量 span 范围。没有适用 semantic task 时保持独立 observation task。保留完整可用原文，不用 Gold 补造事实。

普通请求响应为 `{"items":[{"id":"c1","result":"FAIL","reason":"必要事实缺失"}]}`。只有需要受限观测时增加 `observations`；普通语义 answer_span_id/evidence_ids 可省略，提供时验证合法性。A6 目标条目、指定 fixture 和 bound-memory 标准仍须相应原 span/短来源 ID。模型不抄引用、bullet 或追问正文；选择实际 span IDs，程序恢复原始交付文字，随后计算数量、顺序与相等性。判断不存在/额外/重复条目属于模型识别任务，代码不按 Gold 个数截断观察。span 以原始非空行组织；未提供独立边界的同一行多条自然语言条目仍有表示局限，应在跨格式校准中观察，不增加通用字符 offset 协议。

Agent `answer_grounding` 判断整个实际答复的职业事实和额外量化主张，并消费允许额外有支持主张的政策。分片只改变该项 Judge 的执行范围和最终 ALL 合并，不改变其定义或政策。预设 material Gold 作为直接语义标准保留项目、指标、数值、机制、控制条件、required/conditional、指定 fixture 和禁止归因。Memory 的量化答复标准同样直接判语义。已删除两个 claim 模块、旧响应元组、候选/数字线索/位置审计和独立 extraction completeness 元评测；没有用空 material 列表兼容旧执行器，也没有恢复 material-claim pipeline。

Memory 仍使用目标 turn 的 committed after 或实际 recall snapshot，绑定当时可用 source 正文；后续同 ID 更新不能支持早期判断。A8 历史表现与技术参考答案分开，A9 只允许指定补充消息满足 exact fixture 来源合同。证据 ID 的合法性仅证明身份；语义正确性由模型判断，需跨任务人工校准。

每个请求 `stream=True`、SDK retries=0，由共享 scheduler 限制并发与 deadline；仅对允许的暂时传输错误有限重试。截断、坏 JSON/schema 不自动提高预算、不让模型修复判定、不重跑到 PASS。只有完整 stop 且校验通过才建立任务结果。`answer_grounding` shard 全部 PASS 才形成 PASS，任一已完成 FAIL 形成 FAIL；shard provider interruption 只有在没有已建立 FAIL 时才使整案 BLOCKED，schema/evaluator 错误仍为 ERROR。其他必需任务执行失败保持原有 ERROR/BLOCKED 行为，保留已完成结果及 trace。合法 FAIL/UNKNOWN 是原子判定；UNKNOWN 不能自动满足 required 项。

默认配置：

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `EVAL_JUDGE_BASE_URL` / `API_KEY` / `MODEL` | 回落主模型配置 | Judge provider |
| `EVAL_JUDGE_TEMPERATURE` | `0` | 请求温度，不保证完全消除波动 |
| `EVAL_JUDGE_TIMEOUT_SECONDS` | `600` | 每请求 deadline |
| `EVAL_JUDGE_MAX_OUTPUT_TOKENS` | `6144` | 原样传给 API 的生成预算 |
| `EVAL_JUDGE_CONCURRENCY` | `2` | 共享并发上限 |
| `EVAL_JUDGE_TRANSPORT_RETRIES` | `1` | 暂时传输失败最大额外尝试 |
| `EVAL_JUDGE_RETRY_BACKOFF_SECONDS` | `0.5` | 退避基数 |
| `EVAL_JUDGE_CACHE_DIR` | 未设置 | 成功任务本地缓存 |

`EVAL_JUDGE_CLAIM_SHARD_MAX_SPANS/MAX_CHARS/VERIFY_BATCH_SIZE` 已删除；旧环境值不再驱动流程。学校接口不能关闭推理的既有约束保持，不发送 thinking/reasoning_effort 开关。

telemetry 区分正文、推理流统计与 provider usage；缺 usage 保持未知，不据 max_tokens 推算实际账单。普通运行不把 reasoning_content 纳入判分/Evidence，私有诊断 trace 可保存原始流供取证。摘要记录任务/语义判定覆盖、PASS/FAIL/UNKNOWN 和执行错误；覆盖完整只表示协议执行完成，不证明语义完全正确。受控比较 v3 包含并发、预算、重试/退避、缓存是否启用和实现 hash，不泄漏缓存路径或凭据。

缓存身份包含 provider/model、版本、计划、原文和有效设置；只保存成功结果，加载重新校验。旧协议缓存不能使用。目录沿用真实资料的本地私有边界，不写数据库。

### Judge-only 冻结 Artifact Replay

`python -m app.evals.judge_replay` 是单 case 文件入口，复用当前 Judge/executor，不执行 Agent、数据库、corpus setup 或 Gold compiler。输入中的两份 compiled plan 必须一致并为 `eval_contract_v3`；Agent 已完成，实际可见 Evidence 采集完整，终态/来源观察和 task/source_policy 元数据可验证。旧 v2 计划在网络、输出目录和旧源码读取之前拒绝，不做隐式重编。需从新版运行取得现代冻结观测后重放，不能直接使用历史 A1 目录。

```bash
cd backend
TIKTOKEN_CACHE_DIR=../artifacts/tiktoken-cache PYTHONPATH=. \
.venv-py313/bin/python -m app.evals.judge_replay \
  --run-dir /absolute/path/current_v3_run \
  --case-id CASE_ID \
  --metadata /absolute/path/frozen_case_metadata.json \
  --output-dir /absolute/path/new_replay \
  --cache-dir /absolute/path/new_cache
```

每次 invocation 保存原输入、request/chunks/attempt/validator、Judge 输出、确定性结果与按声明 check/observation 的覆盖审计。output/cache 与冻结输入隔离；已有 output 需显式 --resume 且 manifest 相同。trace 含私有资料和原始 reasoning，用私有权限保存，凭据不写入。审核状态表示执行覆盖，不再输出数字/offset/候选完整性证明；执行故障与质量 FAIL 分开。

### 冻结输入单任务诊断

`python -m app.evals.judge_task_probe` 只执行显式选择的新版冻结任务；`--task-id` 可重复。默认 recorded_request 要求当前实现/消息/schema 与保存记录一致，`--use-current-tasks` 显式允许同一个新版评分合同上的当前提示实现；完整输入/来源、模型与 endpoint 身份仍核对。两种模式都拒绝 v2 scoring plan，不能将旧 claim task 改作新语义 task。

`--prepare-only` 只验证输入并生成内存任务说明，输出 TASK_PREPARED，客户端和输出目录均不创建。真实 Probe 固定单并发、零重试、无缓存；`--verify-extracted` 已删除。TASK_VALIDATED/TASK_FAILED 是任务执行结果，case_scoring=false，不生成整案 verdict。输入或实现改变使用新独立目录。历史 A1 专用命令见历史报告；当前工作覆盖不同任务类型。

2026-10-12 用户跨任务运行已有 A6/F2 的新版完整冻结观测；三个 Judge 请求均成功，其他七例未建立完整 Judge 机会。本轮修复 Runner 跨事务保留 UUID、失败终态恢复实际运行计数和已验证条目 ID 标签/正文分离；采集严格检查不放宽。468 unit tests、27/280 preflight 通过；原 artifact 不改写。A6 的局部改稿误判已离线复算修正，grounding 分歧保留为人工校准。当前优先使用这两个完成案例的冻结观测，暂不重复九个 Agent；一次 A6 Judge-only 命令及边界见 跨任务审查（历史资料未包含在公开快照中）。

## 唯一 canonical Eval DB

产品库是 `jobpilot`，评测库是可丢弃、可重建的 `jobpilot_eval`。所有 reset/drop/setup/migration/corpus 写入路径先按数据库名字拒绝产品目标；换 host 也不能绕过名字断言。无需日期库。profile A/B 使用同一 canonical infrastructure 与 explicit profile binding，不修改产品 Active Profile，不复制产品 Memory。

首次正式 baseline 的默认显式 profile 是 `bge-m3-school-v1`，与当前产品已 ACTIVE 的 BGE 配置一致。以下 corpus、A1、suite、full 和 variance 命令全部显式绑定 BGE；Qwen 保留为历史/对照 profile。Eval binding 独立于产品 active pointer，不需要 activate；首次 baseline 不使用省略 profile 的通用 CLI 默认解析。

从 `backend/` 运行，配置由应用读取未提交 `.env`；以下准备命令不调用 LLM/Embedding：

```bash
.venv-py313/bin/python -m app.evals.runner --preflight --plan-output /tmp/jobpilot_eval_compiled_plans.json
.venv-py313/bin/python -m app.evals.database inventory
.venv-py313/bin/python -m app.evals.database setup
.venv-py313/bin/python -m app.evals.corpus setup --embedding-profile bge-m3-school-v1
.venv-py313/bin/python -m app.evals.database smoke
.venv-py313/bin/python -m app.evals.corpus check --embedding-profile bge-m3-school-v1
```

需要 fresh environment 时重建同一个 canonical DB，先保存 inventory：

```bash
.venv-py313/bin/python -m app.evals.database reset --inventory-output /tmp/jobpilot_eval_before_reset.json
.venv-py313/bin/python -m app.evals.corpus setup --embedding-profile bge-m3-school-v1
```

Corpus setup 核对六份本地 frozen snapshot SHA。与 source DB 当前快照一致、READY 且 UUID/SourceRef/选定 profile coverage 一致时可复制原索引与向量；新 JobPilot snapshot 不在 source 库时，在 canonical Eval DB 从 frozen source 解析、分块，并生成稳定 identity。该 setup 不请求 embedding、不向产品库导入/归档/删除文档。新 snapshot 的缺失向量如实报告；run preflight 要求全部文档选定 profile 完整 coverage，不得以旧总览或其他 profile 的向量兜底。

仅在后续明确允许生成 embedding 时执行 `.venv-py313/bin/python -m app.evals.corpus backfill --embedding-profile bge-m3-school-v1`，完成后运行 `.venv-py313/bin/python -m app.evals.corpus check --embedding-profile bge-m3-school-v1`；backfill 会产生 provider 成本。没有完整 coverage 时，真实模型 Eval 尚不能开始。registered profile 不依赖原定义文件路径；BGE definition 在 `config/embedding_profiles/bge-m3-school-v1.json`，不得将技术注册/验证误当 activation。

## 验证与模型验收

2026-11-08 复核：私人资料齐全的原工作区在补齐 tiktoken 缓存并修正测试后，单元测试结果为 683 passed（10.55 秒）。首次使用编码文件可能需要网络，缓存不随仓库分发。公开快照缺少私人评测输入，依赖这些输入的测试不能原样运行。

含私人资料的评测 cases、fixtures、retrieval cases/evidence、baseline、冻结 corpus、数据库快照、向量和私有运行 trace 不随公开快照分发。依赖这些输入的测试和以下历史评测命令不能在新克隆中原样运行；历史数字也不能直接重现。须在合法授权和隔离条件下重新准备输入、模型与向量，并明确标识其与原实验的差异；自备数据的结果不等同原实验。

无成本验证：

```bash
# Mock Provider / frozen contract / 跨任务代码回归，不调用 Agent、LLM 或数据库
TIKTOKEN_CACHE_DIR=../artifacts/tiktoken-cache .venv-py313/bin/python -m pytest tests/unit -q
# integration 使用隔离 Eval 数据库，另行启用
JOBPILOT_TEST_POSTGRES=1 .venv-py313/bin/python -m pytest tests/integration
```

集成测试只使用 canonical `jobpilot_eval` 中随机独立 schema，deterministic embedding 与 model stubs，不触及产品库。测试应覆盖 unknown/unmapped scoring、缺失/额外 Judge check、非法 enum、语义失败聚合、provider/schema 分类、stream reasoning 隔离、A6 改稿边界、A9 精确 source、Memory target event/冲突/跨会话、真实 fault trigger/recovery 和产品数据库断言。执行计数与 DB smoke 实际结果以本轮实施报告为准。

此前 A1 的超时、截断、抽取因果误读和引用格式错误保留于 历史简化报告（历史资料未包含在公开快照中），不能证明新 direct Judge 的语义质量或成本。跨任务离线回归与实现变化见 本轮实施记录（历史资料未包含在公开快照中）。真实语义校准和 full Eval/variance 尚待用户执行，不冻结 baseline。先选不同任务类型的代表案例，固定版本观察正确/错误和少量边界样本；不在每次模型生成后修改代码。下面保留正常运行入口供人工执行。

```bash
.venv-py313/bin/python -m app.evals.runner \
  --case A3 --case A4 --case A6 --case A7 --case A8 \
  --case M4 --case M6 --case F1 --case F2 \
  --embedding-profile bge-m3-school-v1
.venv-py313/bin/python -m app.evals.review '<representative-run-directory>'
.venv-py313/bin/python -m app.evals.runner --suite agent_tasks --suite memory --suite runtime_failure --embedding-profile bge-m3-school-v1
.venv-py313/bin/python -m app.evals.runner --embedding-profile bge-m3-school-v1
.venv-py313/bin/python -m app.evals.runner --include-variance --embedding-profile bge-m3-school-v1
.venv-py313/bin/python -m app.evals.review '<full-with-variance-run-directory>'
# 仅在该完整 artifact 人工审核已通过并明确批准 freeze 后执行；目标须尚不存在
test ! -e ../jobpilot_eval_v0.1.1/baselines/baseline_v1 && cp -R '<approved-full-with-variance-run-directory>' ../jobpilot_eval_v0.1.1/baselines/baseline_v1
```

检查 review CLI 的输出与人工审核记录后，才进入 baseline freeze。`run_status=COMPLETED` 只证明没有 BLOCKED/ERROR，不代表全部 PASS，也不代表人工接受 baseline。

## 运行 artifact、比较与执行状态

`runs/<timestamp>_<git>_<run-id>/` 保存 run identity/comparison contract、compiled plans、case results、bounded Judge responses/telemetry、summary/report。所有新结果以 `result.outcome` 为唯一状态，没有旧 pass/status compatibility branch。比较使用明确文件 allow-list、合同/实现/配置/corpus/profile 摘要，不扫描 .DS_Store 等非语义文件。

- SCORED：Retrieval 正常计算四项指标，支持不足仍是 SCORED，质量由 metrics 表达。
- PASS：其他 suite 完整正常执行机会中所有 required checks 满足。
- FAIL：正常执行完成，但任务质量、Gold、预算、grounding 或来源要求失败。
- BLOCKED：主模型、embedding 或 Judge 的外部 provider interruption；不计算能力通过率。
- 真实工具内部的 provider 异常通过 eval-only `provider_error_code` 保留并分类为 BLOCKED；不改变公开 ToolError 合同。注入故障、缓存结果和工具自身预算到期不冒充外部 provider interruption。
- ERROR：dataset/schema/fixture/parser/migration/evaluator/evidence capture 等 harness 自身错误。

`scorable_case_count = PASS + FAIL + SCORED`，`pass_rate = PASS / (PASS + FAIL)`，Retrieval 不进入 PASS rate 分母；BLOCKED/ERROR 不进入可评分分母。summary 单列 SCORED 数与 answerable/unanswerable 数；有 ERROR 的 run_status 为 ERROR，有 BLOCKED 为 PARTIAL，否则 COMPLETED。context compactor 的调用/token/latency 单列，同时计入 provider 总量；主 Agent evidence 不使用只有 compactor 看见的记录。recovered_error_count 与 lineage 只使用实际 tool execution ledger，不由最终 COMPLETED 推断所有历史错误恢复。没有价格配置时不声称真实货币成本；正式模型能力、Judge 耗时和 variance 仍待真实验收。

2026-10-17 Context Lifecycle V1：普通 pressure eviction 不调用模型，因此新运行的 compactor calls/tokens/latency 应为 0；保留原指标口径与历史记录，不为此次实现修改 case、Gold、fixtures、threshold、Judge、Retrieval、failure classifier 或现有比较器。`read_working_context` 的内存 backing store 不依赖 trace capture；缺失时的 MODEL/TOOL 重建是 best-effort，失败必须输出 `WORKING_CONTEXT_NOT_RECOVERABLE`，不能用 capture 缺失证明内容从未存在。主 Agent 的实际工具可见结果仍遵守现有 evidence capture 合同，Working Task State 仍是派生数据。

本次不引入 `NONFATAL_ERROR` 或 Compactor recovery/retry 分类；终止 provider 异常保留已有 error code。Context lifecycle 的 reference/candidate 实验须保持案例、Gold、主模型/Judge、corpus、embedding profile 与其配置一致，另存实现与 lifecycle 设置的 manifest，逐 case 对照现有 artifact/trace；现有 embedding A/B 比较器要求不同 profile，不能直接用于此次 context 变更。真实 A1/Suite 尚未执行；执行前仍需 canonical Eval DB readiness、fixture/corpus/identity 预检与模型出站条件，离线/预检通过不证明真实能力或 latency 改善。

2026-10-17 当前执行边界：已实际尝试 canonical Eval DB 的只读 readiness smoke，受控重试返回 `connection_refused`；数据库 readiness 不可用，未启动真实 A1/Suite，也未调用真实 provider 或执行 migration。该失败来自实际连接结果，不依据配置推断服务状态；离线 preflight 不替代此 gate。

Embedding A/B 使用两个 explicit profile 分别运行同一合同，然后 `app.evals.comparison --baseline <qwen-run> --candidate <bge-run> --output <comparison-directory>` 只读取 artifact。这里的 `--baseline` 是 A/B 比较参考 run，不决定正式 `baseline_v1` 的 profile。comparison v4 记录并核对 retrieval_contract_v1 及三份 YAML 的 hash，拒绝旧 retrieval artifacts。合同、case set、corpus、query/检索限制、lexical trace 必须一致；BLOCKED/ERROR 为 inconclusive。旧 Qwen/BGE 结果是旧 evaluator 下的受控实验快照，不是正式 baseline，也不能直接代表新 pass_policy 下结果。

## 保留的历史证据（已被当前合同取代）

2026-10-09 supersession：以下 2026-10-06 记录保留当时结论和测试计数。其日期 Eval DB、一次性脚本、旧 any-hit PASS 和旧 Judge 路径已被本页当前合同取代。人工实验报告归档到 `docs/experiments/embedding_phase_b/`；旧 run 移到 `archive/prebaseline_before_contract_refactor/`，不迁移结果 schema，不进入 baseline。

2026-10-06 Phase A 的结构迁移测试记录：完整自动测试 75 passed，含 13 项隔离 PostgreSQL/pgvector 集成测试；10 个既有 corpus 向量查询迁移前后结果一致。该验证没有请求真实 query embedding、运行完整 Eval 或冻结新 baseline；仍需 Phase B 授权后按固定 profile 运行检索基线。

## 2026-10-06 Phase B 执行证据

Phase A 的 75-passed 记录保留为历史证据。identity v2 / Phase B 当前完整测试 109 passed（91 unit、18 隔离 PostgreSQL integration）。旧 isolated Eval 的同 SHA256 Corpus UUID 与当前 source 不一致；本次新建 `jobpilot_eval_phaseb_20261006`，复制源 UUID/SourceRef/向量，不清空历史 Eval。复现实验入口为 backend `PYTHONPATH=. python scripts/phase_b_eval.py --profile <name> --suite retrieval`；脚本仅给本次进程设置 isolated target 和直连网络规则，不改产品 ACTIVE。

Qwen 与 BGE 各执行现有 9 个 retrieval cases，代码/settings/Corpus/UUID/Gold/Prompt/预算和逐 case lexical trace fingerprint 通过严格对比。两边各 6 PASS、3 FAIL，均无 ERROR/BLOCKED；BGE R5 FAIL→PASS、R7 PASS→FAIL，R8 增加 g4，R9 两边 FAIL。R7 的 Qwen PASS 来自 g1；BGE 把同一组 g1 证据从 Qwen vector 第 1/4 位重排到第 7/8/10/11/14 位，因而失去 top-5 命中；两者都未在 top-30 命中 R7 g2。因该受控 R7 回归及 MRR 下降，A1/A4/A8/M4 Agent smoke 的前置门禁未通过，未调用 Agent/Judge。没有运行正式 full baseline/variance 或冻结新 baseline，没有调整阈值/Gold/检索参数。逐 case top-5、排名、Gold、新增 FP/FN 见 Phase B Retrieval Diff（历史资料未包含在公开快照中）。三路（lexical/vector/fused）top-20 的只读故障归因、Gold 位置和同形 SQL 的 `EXPLAIN` 记录在 Phase B Failure Attribution（历史资料未包含在公开快照中）；该诊断不调用 provider、不写数据库，也不改变产品配置或 planner。BGE 技术 validation 不代表检索质量通过或 activation approval。


> 2026-10-28 审计补充：上文 Phase B 的 R7 PASS→FAIL 是修正前旧 Gold 的历史结果。修正 R7 后 Qwen/BGE 双方均 PASS，详见 当前审计（历史资料未包含在公开快照中） 及其中原始对照；上文指向 final 的描述不适用于修正后的对照。10 月 26 日 retrieval 子集已评分，full run ERROR，未冻结正式 full-suite baseline；当前数据库连接被拒绝，active profile 未核实。
