# Embedding Profile 合约（2026-10-06 Phase A）

本合约取代单一全局 embedding model、实体内 `vector(1536)` 的旧设计。当前应用行为以 [技术总览](../docs/JOBPILOT_TECHNICAL_OVERVIEW.md) 为准。

## 数据与身份

    2026-10-06 identity v2 修订：Profile ID 表示不可混用的 embedding space / encoding contract。新 Profile 以版本化、排序后的 semantic payload 生成 UUID；不再对整个运行配置 `asdict` 求 hash。数据库 trigger 保护 ID、名称、编码语义和版本兼容快照；同名注册不同语义被拒绝，同语义的 runtime metadata 更新保留 ID、已有向量、validation 和 active 指针。不同 name 指向相同语义会给出明确 alias 冲突，要求复用已有 Profile。

| 字段 | v2 identity | 原因 / 变更要求 |
| --- | --- | --- |
| provider、model、dimension | 是 | 编码协议、模型/权重标识与返回坐标维度 |
| send_dimensions | 是 | 是否显式请求维度属于服务请求编码合约；未经验证不能认定与省略参数等价 |
| normalize、query_prefix、document_prefix | 是 | 客户端归一化及 query/document 预处理改变生成表示 |
| space_revision | 是 | 显式记录部署权重、服务端 tokenizer/预处理等语义版本；模型 alias 不足以区分部署时必须填写 |
| api_key_env | 否 | 只指定 credential 查找位置；环境变量改名不会产生新空间；实际 key 值永不保存 |
| base_url | 否 | 只作为无凭据的 transport endpoint；换地址必须验证其仍提供同一 space_revision，换模型部署语义需新 revision/Profile |
| batch_limit | 否 | 请求分批策略，不能改变同一条接受输入的向量 |
| max_input_tokens | 否 | 本地 admission 上限，只拒绝超长输入；不能静默截断。若服务端上限变化伴随截断/预处理变化，应改变 space_revision |
| tokenizer_encoding | 否 | 当前客户端仅计数并决定拒绝/接受，发送原文本；真正用于服务编码的 tokenizer 变更属于 space_revision |
| name | 否 | 人工查找与展示名称，数据库中仍保持不可改名 |

`0006_embedding_identity_v2` 为已存在的 v1 Profile 保存不可变 canonical `legacy_identity_json`、identity_version=1；保留历史 UUID、向量外键、索引名称、状态和 active 指针。v1 ID 始终由冻结旧 payload 校验，允许更新其 runtime metadata 而不重算 ID；与新 v2 注册比较时使用相同 semantic fingerprint。新 Profile 使用 identity_version=2、无 legacy 快照。`0005_embedding_profiles` 的 bootstrap payload/旧 UUID 算法已固定，不随运行时 dataclass 新字段漂移。0006 downgrade 仅允许未修改 runtime metadata 的 grandfathered v1 Profile，存在 v2 Profile 时明确拒绝。

此设计不自动证明两个 URL、credential 后端或模型 alias 返回等价向量。运营者必须用真实接口验证与部署 provenance 维持 space_revision 的语义；返回不同权重、服务端 tokenizer、池化或预处理的部署必须产生新 space_revision/Profile。API key 值只读环境或本地 `.env`，不存入 Profile。

可变生命周期字段为 status、validation_passed_at、validation_note、created_at。状态包括 REGISTERED、BACKFILLING、READY、INACTIVE、ACTIVE、FAILED。READY 表示 backfill 完整且索引可用；activate 仍要求显式 validation 记录。最多一个 ACTIVE status，并由 `active_embedding_profile(singleton_id=1, profile_id)` 保存唯一在线指针。

业务 `knowledge_chunk` / `memory_item` 无向量列。独立表为：

| 表 | 复合主键 | 向量 |
| --- | --- | --- |
| knowledge_chunk_embedding | chunk_id, profile_id | 非空通用 vector |
| memory_item_embedding | memory_item_id, profile_id | 非空通用 vector |

每行另存 created_at。业务实体删除级联删除其向量；Profile 被向量/active 指针引用时不可删除。数据库 trigger 校验实际 vector_dims 与 Profile dimension 一致。不同模型即使维度相同也使用独立 profile_id。

每个 Profile 的两个 ANN 索引分别为 `HNSW((embedding::vector(dimension)) vector_cosine_ops) WHERE profile_id='<uuid>'`。查询显式限定同一 UUID，再使用相同 dimension 的表达式排序，UUID 作为 SQL 常量供部分索引匹配。当前采用 pgvector 全精度 HNSW，接受 1..2000 维；更高维度需另行实现索引策略，禁止截断或补零。依据：[pgvector 官方说明](https://github.com/pgvector/pgvector#can-i-store-vectors-with-different-dimensions-in-the-same-column)。

## 在线解析与写入

产品操作的 `EmbeddingProfileService` 在每次操作解析数据库 active 指针；没有通过 EMBEDDING_MODEL/DIM 直接覆盖在线行为。读/写持 singleton 共享事务锁，activate 排他锁，确保写入、coverage 检查和切换不会交错。Client 绑定的 profile 必须与本次统一解析结果一致。隔离 Eval 的显式 `EvalScope.embedding_profile_id` 直接解析指定 Profile，不依赖 active 指针；仅接受 READY、INACTIVE 或 ACTIVE，REGISTERED 等未准备状态明确报错，不 fallback。

Knowledge 保留 parser、Chunker v2、retrieval_text、chunk/section/document ID 和 SourceRef。仅对 active Profile 编码并写一行 `(chunk_id, profile_id)`。Memory ADD/SUPERSEDE 同理；MERGE 保持实体 ID、原证据与业务语义，删除已失效的各 profile 正文向量后只生成当前 Profile。ARCHIVE 不生成向量。历史 profile 不实时维护，切回前补齐缺失。

Query 使用相同 Profile 的 query_prefix；Knowledge/Memory 使用 document_prefix。客户端先按显式 tokenizer_encoding 校验整个输入序列 max_input_tokens，再按 batch_limit 调用。超过能力必须报告 incompatibility，不能 truncate 或 rechunk。本地 tokenizer 规则不是对 provider 未公开 tokenizer 的保证，后续接口验证必须确认兼容性。send_dimensions=false 时请求无 dimensions 字段；normalize=true 才归一化。响应 index、数量、维度、有限性和非零 norm 均检查。OpenAI SDK 保留两次瞬时失败重试。

## 显式操作

在 backend 激活 Python 环境后：

```bash
python -m app.embedding register --file profile.json
python -m app.embedding coverage --profile target-profile-v1
python -m app.embedding compatibility --profile target-profile-v1
python -m app.embedding backfill --profile target-profile-v1
python -m app.embedding index --profile target-profile-v1
# 使用同一 profile 完成 retrieval smoke / Eval，再登记通过证据
python -m app.embedding validate --profile target-profile-v1 --retrieval-evidence 'passing smoke/eval artifact reference'
python -m app.embedding activate --profile target-profile-v1
```

等价短入口：`python -m app.embedding.backfill --profile ...`、`python -m app.embedding.activate --profile ...`。其中 index 不生成向量；完整 backfill 已自动建立目标索引，手动 index 适用于重建/检查。任何操作都不隐式切换或给其他 profile 双写。

注册文件为公开配置，不能含 API key 值；示例：

```json
{
  "name": "target-profile-v1",
  "provider": "openai-compatible",
  "base_url": "https://embedding.example/v1",
  "model": "target-embedding-model",
  "dimension": 1024,
  "max_input_tokens": 8192,
  "batch_limit": 20,
  "send_dimensions": false,
  "normalize": false,
  "query_prefix": "",
  "document_prefix": "",
  "tokenizer_encoding": "o200k_base",
  "api_key_env": "TARGET_EMBEDDING_API_KEY",
  "space_revision": "operator-verified-deployment-v1"
}
```

这些示例能力必须由运营者验证后填写，不表示已探测任何新 provider。READY/INCOMPLETE 为 coverage/backfill 报告状态；CLI 不完整/失败退出码为 2。

Coverage 的 Knowledge 分母是 READY 文档下的 chunks；Memory 是与当前检索业务 scope 一致的 ACTIVE items，产品操作排除 eval_run_id 非空的数据，Eval 仅统计当前 case 且未撤回的 memory。每类返回 total、existing、missing。空集完整。

`compatibility` 扫描所有 eligible Knowledge 与 Memory（含目标已有向量），输出 compatible count、too-long count、max observed tokens/characters 以及每个超限 UUID、tokens、limit、excess；不输出正文。Backfill 在 client 构造、状态修改与任何 API request 前强制执行此全局门禁；有超限立即停止，不截断、不 rechunk。输入按 document_prefix + 原文计数。`hf-json:<sha256>` 用本地 hash 校验的通用 JSON tokenizer，目录为项目 `artifacts/tokenizers` 或 `EMBEDDING_TOKENIZER_DIR`，禁用 truncation/padding、包含 special tokens；其与真实部署的 usage/输入边界必须由非敏感 probe 核对。

Backfill 默认只处理缺失复合主键，显式按目标 Profile 编码，使用原 retrieval_text/Memory content。逐批事务、锁定当前实体、成功提交、失败回滚该批；再次执行只补缺失。Knowledge/Memory 分别输出 total、existing、pending、completed、failed 及最终 coverage；embedding_batches 统计逻辑 SDK 调用，http_attempts 统计实际 HTTP embeddings 请求（含 retry），失败日志不打印输入正文或 provider 异常正文。新增向量清除旧 validation 记录，需重新验证。Backfill 完成不 activate。

Activate 在一个事务内检查：Profile 存在且 immutable identity/config 有效、eligible Knowledge/Memory 均 100%、两个实际 pg_catalog 索引 ready/valid 且 predicate/dimension/operator 正确、已记录通过的 retrieval/Eval 验证、目标生命周期可切换和凭据可用。失败保留当前 pointer；不会生成向量或调用 embedding API。`validate` 是显式 operator attestation + coverage/index 校验，并检查目标全部已存向量实际维度、finite 和非零 norm，不冒充自动完整 Eval。

## 无损迁移与 Eval

`0005_embedding_profiles` 对旧业务表加锁，检查现有维度/eligible coverage，按当前真实旧配置创建 `qwen-current-v1`，原样复制已有向量与时间，并逐行比对向量文本值。初始化该 Profile 为 ACTIVE，已有 baseline 以注明原因的 migration grandfather validation 保留；删除旧列和旧全局 HNSW 索引，建立 profile 索引。无需 provider 调用；旧 EMBEDDING_BASE_URL/MODEL/DIM 仅初始化现有空间。旧 client 固有规则（batch 20、send_dimensions=true、normalize=false、无 prefix）保留。Downgrade 仅允许原 Profile 已恢复 active 时回到历史 1536 schema。

Eval 通过 `--embedding-profile <name|UUID>` 指定目标，省略则在 run 开始固定 source active。Run/case/summary/Judge/report 与 retrieval trace 记录 profile ID/name/model/dimension。Eval 数据库唯一为可重建的 `jobpilot_eval`；setup/migration 清除共享 migration 在空库创建的 bootstrap active 指针，不将产品 ACTIVE 状态复制为 Eval ACTIVE。

2026-10-09首次正式baseline操作默认：统一显式传`--embedding-profile bge-m3-school-v1`，从corpus setup/backfill/check到A1、suite、full与variance都绑定同一BGE profile，表征当前产品配置。Qwen保留为历史/对照profile；此流程不省略profile、不执行activate，Eval binding独立于产品active pointer。通用CLI解析、历史迁移和评分合同保持不变。

Corpus 以 manifest 中逐文件 SHA256 校验的本地 frozen snapshot 为事实来源。与产品 READY 文档 SHA 和 source role 完全一致时，保留其业务 UUID/SourceRef、chunks 与指定 Profile 的既有向量；新 snapshot 在 canonical Eval 本地 parse/chunk，生成稳定 UUID。setup 不调用 provider，不改产品文档或向量。Profile 在 frozen corpus 向量未覆盖完整时保持 REGISTERED，完整时为 READY；真正 run 需要完整覆盖，不完整为 ERROR，不 fallback。EvalScope 的显式 Profile binding 可以在没有 active 指针的 canonical Eval 中工作，不更改产品 active 指针。补齐缺失向量需另外授权的 Eval-only backfill。

## Phase A 边界

本轮只完成现有 Qwen 的结构迁移、接口与隔离/模拟测试。没有注册真实 BGE Profile、没有调用 BGE API、没有真实模型 backfill、没有完整 Eval、没有改变 active embedding model。后续新 provider probe、目标 backfill、检索基线验证和切换属于用户确认后的 Phase B。

## 本轮执行证据

完整测试 **75 passed（62 unit、13 PostgreSQL integration）**，均未调用真实 embedding provider。真实数据库已迁移到 `0005_embedding_profiles`：Knowledge **3566/3566**、Memory **2/2**，missing=0；全部向量及业务实体/source 校验和未变，10/10 个既有向量查询的 ANN 有序结果未变。当前唯一 Profile 为 `qwen-current-v1`（模型 `qwen3.7-text-embedding`、1536 维、ACTIVE）。完整真实 Eval 与目标 provider probe 仍未执行。

## 2026-10-06 Phase B supersession

上述 Phase A 执行结果为历史证据。Phase B 授权范围更新为非敏感真实 provider probe → 目标注册 → 全局兼容性扫描 → 显式 missing-only backfill → validate/index EXPLAIN → 同条件 retrieval A/B → 条件满足后的 A1/A4/A8/M4 Agent smoke。Phase B 不 activate；切换仅属后续人工决定的 Phase C。当前证据见 Phase B 报告（历史资料未包含在公开快照中）。


> 2026-10-09 supersession: Phase B 为旧 contract 下 prebaseline 实验；日期 Eval DB 已移除。当前 Eval 仅使用 canonical jobpilot_eval，并通过显式 profile binding 进行 A/B。BGE 配置稳定路径为 backend/config/embedding_profiles/bge-m3-school-v1.json。本轮开始时产品 BGE 已 ACTIVE，Qwen 为 INACTIVE；本轮没有 activation，也没有修改产品数据库。当前 frozen JobPilot snapshot 没有可复用向量，两个 canonical Eval Profile 为 REGISTERED，active 指针为空；真实模型 Eval 尚未执行，正式 baseline 尚未冻结。清理与验证证据见 Eval DB 清理报告（历史资料未包含在公开快照中）。
