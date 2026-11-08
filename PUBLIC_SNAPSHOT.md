# 公开快照与私有材料

本仓库为独立公开快照，不包含原工作区的 Git 历史。历史评测使用个人资料、面试记录及项目档案，公开版不含这些资料，也不包含多数未经公开审查的历史记录。

要在受信任的电脑恢复私有评测输入，请从原工作区私下复制以下相对路径，保持目录结构：

- `jobpilot_eval_v0.1.1/cases/`、`fixtures/`、`corpora/`
- `jobpilot_eval_v0.1.1/retrieval_cases.yaml`、`retrieval_evidence.yaml`、`retrieval_protocol.yaml`
- `jobpilot_eval_v0.1.1/baselines/prebaseline_20261026/`（历史校准材料，非正式 baseline）
- `backend/tests/unit/test_eval_observations.py`、`backend/tests/unit/test_context_lifecycle_v1.py`

实际历史运行还依赖隔离数据库、既有向量、服务商和私有运行记录；复制文件不能保证重现同一结果。不要把这些资料、密钥或数据库备份提交到公开仓库。独立的合成实验及结果见 `experiments/jev_ab/README.md`。

## 公开快照的离线单元测试

在 `backend/` 已安装开发依赖且 tiktoken 词表已缓存后，可运行下列不依赖私有数据的测试子集：

```sh
uv run pytest tests/unit -q \
  --ignore=tests/unit/test_parallel_judge.py \
  --ignore=tests/unit/test_retrieval_dataset_oracles.py \
  --ignore=tests/unit/test_eval_comparison.py \
  --ignore=tests/unit/test_eval_harness.py \
  --ignore=tests/unit/test_eval_runner_identity.py \
  --ignore=tests/unit/test_eval_typed_contract.py
```

2026-11-08 在本机公开快照中运行结果为 447 passed；原工作区私人输入齐全时完整单元测试为 683 passed。新电脑首次运行可能需要下载 tiktoken 词表。测试结果不能替代真实模型或数据库集成验收。
