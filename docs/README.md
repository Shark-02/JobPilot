# JobPilot Documentation

本目录中的 current-state 文档必须与同一迭代的代码、migration、配置、测试和运行边界同步更新。维护规则见仓库根目录的 [AGENTS.md](../AGENTS.md)；历史 iteration 报告只记录当时事实，不替代当前实现。

- [JobPilot Technical Overview](JOBPILOT_TECHNICAL_OVERVIEW.md)：以当前代码、migration、配置和测试为事实源的 canonical 技术档案；涵盖架构、运行链路、Agent、Memory、Knowledge/RAG、数据模型、Observability、前端、API、测试、运行与限制。
- [Evaluation Guide](EVALUATION.md)：真实离线评测的隔离、准备、运行、审阅与 baseline 冻结流程。
- [Context Lifecycle V1 合同](../SPEC/04_CONTEXT_LIFECYCLE_V2_DESIGN.md)：2026-10-17 按设计审查收敛的当前合同；内存 backing store、deterministic eviction、精确 read、add-only Working Task State 与 token/error 边界，无新增 Archive 表或 migration，文件名沿用原 v2 设计稿。
- Iteration 001（历史资料未包含在公开快照中）：历史实现与验收记录，仅作背景，不替代当前代码事实。

- Eval Contract Refactor（历史资料未包含在公开快照中）：正式 baseline 前收敛实施与验证。
- Eval DB Cleanup（历史资料未包含在公开快照中）：canonical DB inventory、安全与重建证据。
- Prebaseline Artifact Archive（历史资料未包含在公开快照中）：旧 runs 和人工实验报告归档库存。
- Historical Embedding Phase B（历史资料未包含在公开快照中）：旧 contract 的 Qwen/BGE 实验快照，不是 baseline。
