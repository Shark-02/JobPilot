# Jev 重排序离线实验记录

日期：2026-10-27。状态：**已完成一次真实第三方 JevModel A/B。MRR 小幅提高但只来自一个查询，Hit@3 和证据完整率不变；出现反例误排且串行延迟较高，暂不建议接入生产。**结果仅适用于本次合成小样本，不能证明生产效果。

## 官方资料

- [模型与价格](https://docs.typesafe.ai/models)：`jev-1.14.0`、`jev-1.13.0`；核查时 `jev-latest`、`jev-preview` 均指向 1.14.0。输入 $0.042／百万 token，输出免费；默认 64k／请求、32k／单问题。英文优于中文，中文效果必须实测。账户实际可用模型需用认证后的 `GET /v1/models` 确认。
- [API 与认证](https://api.typesafe.ai/redoc)：`POST https://api.typesafe.ai/v1/systemone`，`Authorization: Bearer` API Key；请求含 `model`、`state` 和 `questions`。官方直连脚手架固定 `jev-1.14.0`，但未进行真实直连。本次真实结果来自 JevModel，站方声称固定 `typesafe/jev-1.13`，见下文。逐个 query–candidate 用 [Score](https://docs.typesafe.ai/primitives/score) 获取 0–2 的期望相关度，再按分数降序重排，相同分数保留 RRF 顺序。confidence 不是相关度或正确率。[官方 RAG 示例](https://docs.typesafe.ai/cookbooks/classifying_rag_passages)也是逐候选判断。
- [官方控制台](https://console.typesafe.ai/login)需要登录，继续时提示同意客户协议及隐私政策；是否需要计费方式及该账户的模型权限未验证。

## 数据和方法

原有 retrieval Golden Set 只有 R1–R9，共九条且含真实资料。这里按其任务类型独立编写九条**合成派生场景**，加一条基于 memory M4 任务类型的合成场景；共十条、25 条虚构文档，内容和标注均非原 Gold。S09 不可回答，不纳入可答指标均值。不得将原始资料、原查询或真实候选替换进外发数据。脚本固定合成数据 SHA-256，读取候选时逐条验证查询和文本；任何数据改动须重新安全审查并更新固定哈希。

通过现有 `MemoryRetriever.search`，在本地 `jobpilot_eval` 的随机隔离 schema 内对每条合成查询运行 `pg_trgm` 词法检索 + `pgvector` 向量检索，使用现有 RRF k=60，取前 5。该 schema 已清理，没有修改生产主链路或上传资料。向量使用本地确定性字符 n-gram，而非生产 embedding，因此此基线只验证隔离实验链路，不能代表生产召回或生产效果。

指标为本实验的文档级口径，分母为 9 条可回答场景：MRR 是首个标注 gold 文档的倒数排名；Hit@3 为前 3 命中任意 gold；Evidence Success@k 要求所有标注 gold 均出现在前 k。当前项目正式 retrieval contract 的 MRR 是首个完整证据单元的模块排名，使用来源 span 和 proof；该合同没有本实验的 Hit@3/ESR，数值不可直接比较。没有 gold 的 S09 单独保留供人工复核，不当作零分。

| 方法 | MRR | Hit@3 | Evidence Success@3 | Evidence Success@5 |
| --- | ---: | ---: | ---: | ---: |
| 合成 RRF 基线 | 0.667 | 0.778 | 0.667 | 0.667 |
| JevModel 重排 | 0.722 | 0.778 | 0.667 | 0.667 |

MRR 提升 **0.056**，全部来自 S04：首个 gold 从第 2 升至第 1，单条倒数排名从 0.5 升至 1；其他 8 条可答案例不变。S01、S03、S07 的固定 Top 5 未覆盖全部 gold，重排无法补回缺失证据；本样本 6/9 已在 Top 3 获得全部证据，Evidence Success@3 没有提升空间。固定 Top 5 集合时 ESR@5 必然不变。样本量仅 9 条可答案例，不能证明生产效果。

### 典型案例：仅含虚构资料

- **S04 判断较好**：问题要求区分唯一执行者与重复通知。RRF 把不充分方案 n02 放第 1；Jev 将两条正确证据 d07（1.81/2）和 d08（1.76/2）排到前两位，n02（1.03/2）降至第 3。这里对反例的判断并非完全排除，只是相对排序改善。
- **S10 判断较好**：关于最近一次模拟面谈后的复习计划，时间不符的干扰项 n05 从第 2 降至第 5、两条 gold 从第 1/3 升至第 1/2，虽然所报指标没有变化。
- **S03 误排**：要求如何防止限流重试超配额，n01 明确是无法约束请求总数的错误方案，却以 0.95/2 排在正确的共享计数证据 d06（0.77/2）之前。另一条 gold d05 不在 Top 5；这属于候选召回缺失和候选相关性判断两个不同问题。
- **S07 误排与召回缺失**：查询要求有前后测量的性能改动，n03 明确表示方案还未测量，却以 1.12/2 升至第 1；两条含测量的 gold 都不在 Top 5，重排无法补回。应人工复核相关度及其对误导答案的影响，不能仅凭一个分数认定生成结果。
- **S09 不可回答**：所有候选均不是所需邮箱证据，最高分为 0.28/2。该实验没有预设拒答阈值或执行回答生成，因此这不是已验证的拒答成功。

### 真实延迟与费用

第三方结果文件包含 50 次成功请求和 3 次先前 403 失败。以下延迟仅统计 50 次成功请求：中位数 **4.843 秒**、平均 **5.206 秒**、P95 **7.156 秒**、最大 **7.893 秒**；本实验逐候选串行，50 次请求累计 **260.31 秒**，平均约 **26.03 秒／查询**，不含本地 RRF 和其他额外开销。这是该次 JevModel/OpenRouter 链路的实测值，不是官方直连 Jev 模型的纯推理时间，也不是并发后的延迟。

成功请求合计 **26,552 输入 token**。首个成功响应报告注册赠送额度 100,000、余额 99,470，最后一个成功响应余额 73,448、`hasPurchased=false`；与全部成功用量相符，表明响应所示额度消耗来自注册赠送余额。结果文件没有美元账单，因此实际美元费用**未核验**，不把官方直连价格当作本站实际收费。作为未来付费成本参考，按站方 [价格页](https://jevmodel.app/pricing)最小包 $4.90／800 万 token 折算，本次用量约 **$0.0163**；这是按包单价的估算，不是本次账单，也不含整包购买带来的预付金额。

响应中的 `usage.cost` 数值合计 **0.001115184**；站方公开 API 文档未明确该字段是用户美元账单，因此仅保存原值，不将其视为本次实付金额。

结论：当前证据不足以说 Jev 对 JobPilot 显著更好；尤其候选集已缺证据、中文反例有误排，而串行延迟较高。暂不集成主链路。后续若继续，应在隐私允许的独立数据集上扩大样本、提高候选召回、设计更有区分度的完整证据前 3 测试，并对并发或多问题请求另外实测；同时保留原始 RRF 作为基线并逐例审查反例。

## 运行

从项目根目录执行：

```sh
# 重新生成候选时，先在当前 shell 设置 JOBPILOT_JEV_EVAL_DATABASE_URL，
# 仅允许本机 jobpilot_eval，且不得与已配置的产品库相同。
backend/.venv-py313/bin/python experiments/jev_ab/rrf_synthetic.py

# 只计算基线，不发网络请求。
backend/.venv-py313/bin/python experiments/jev_ab/jev_ab.py

# 有官方密钥后，在终端安全设置 TYPESAFE_API_KEY，再运行：
backend/.venv-py313/bin/python experiments/jev_ab/jev_ab.py --live
```

`--live` 最多顺序发送 50 次请求，只发送固定合成 query 与 candidate；首次错误即停止，不输出不完整 A/B 均值。记录每次实际请求的耗时、状态、模型回答和 token 用量，按官方单价计算**估算输入费用**；实际账单、失败请求计费和其他费用需另在账户核验。输出 `result-*.json` 含逐候选结果以及供人工复核的提升、退化和 gold／干扰项排序案例；这些自动筛选不是经人工确认的正确率。未设置密钥时拒绝请求。请勿将密钥写进项目文件或聊天。

验证：数据和候选结构校验通过；无密钥 `--live` 退出且未发请求；脚本编译通过；现有 retrieval contract 与数据集单元测试 `69 passed`。所有实验文件为未跟踪文件，不改变原有评测集、阈值、生产代码或配置。

> 2026-11-08 补充：上述“未跟踪”描述仅反映实验时的工作区状态；是否纳入版本管理以当前仓库为准。原验证计数为当时结果，不代表当前整个测试套件通过。

## JevModel 第三方渠道（2026-10-27）

[JevModel](https://jevmodel.app/) 自称为独立服务，通过 OpenRouter 调用 `typesafe/jev-1.13`，不是 TypeSafe 官方渠道；本次真实响应没有提供可独立验证模型版本的字段，模型身份按站方声明记录。它与上面的 TypeSafe 直连 1.14 脚手架是不同模型版本和调用路径，不能把第三方网络延迟、计费和结果当作官方直连 1.14 的测量。只允许上传本实验经固定哈希校验的虚构数据。

[API 页面](https://jevmodel.app/jev-api) 与 [价格页面](https://jevmodel.app/pricing) 宣称邮箱验证后有一次性 100,000 免费 token；但 [较早的 API 指南](https://jevmodel.app/guides/api) 又写每日 5 次免费请求。该信息存在冲突，须以实际账户额度为准。本次响应显示注册赠送余额足以完成 50 次成功调用。该渠道须使用 JevModel 自己的密钥，端点和响应格式也与官方不同；不将 TypeSafe 官方输入价格用作该渠道账单。

独立适配器 `jevmodel_ab.py`：每个候选各发一次请求，最多 50 次；中断后可用 `--live --resume jevmodel-result-….json` 恢复，已成功的请求不会重复发送。恢复时核验固定合成数据、候选顺序及渠道身份。该站 [隐私政策](https://jevmodel.app/privacy-policy) 称登录后的运行内容由本站保存并经 OpenRouter 路由，因此只发送合成样本。本次逐候选调用和响应保存在 `jevmodel-result-20261027T070559462342Z.json`；费用须以账户账单为准。

在项目根目录的 zsh 中，取得 JevModel 密钥后可运行：

```sh
read -rs 'JEVMODEL_API_KEY?请输入 JevModel API Key（不回显）：'
export JEVMODEL_API_KEY
backend/.venv-py313/bin/python experiments/jev_ab/jevmodel_ab.py --live
unset JEVMODEL_API_KEY
```

无密钥时命令会退出，未发送请求；适配器编译及无密钥检查已通过。

如请求失败，脚本显示 HTTP 状态及错误是否符合本站 JSON 格式，不保存或显示原始错误响应，以免泄漏密钥。403 的原因需结合响应格式与账户验证状态排查，不能仅凭状态确定。

2026-10-27 排查：不带密钥的空请求使用 Python 默认 User-Agent 时返回 Cloudflare 403 / 1010；显式标识 `JobPilot-JevExperiment/0.1` 后返回本站 JSON 格式的 401。脚本现使用该应用标识，并已用真实密钥完成 50 次成功调用。参考 [Cloudflare 1010 文档](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-1xxx-errors/error-1010/)。
