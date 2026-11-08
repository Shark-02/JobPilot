"""Historical Memory binding regressions using recorded observations only."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from app.evals.compiler import CompiledCheck, ExecutableEvalPlan, SemanticCheck
from app.config import Settings
from app.evals.executor import EvaluationContext, execute_plan
from app.evals.judge import JudgeSchemaError, StructuredJudge
from app.evals.judge_tasks import JudgeTaskPlanner


def source(ref, content):
    return {"source_ref": ref, "content": content,
            "id": "bound_source_" + sha256((ref + content).encode()).hexdigest()[:20]}


OLD = source("message://seed", "原计划做后端研发")
A = source("message://a", "Step A 决定先投后端研发")
B = source("message://b", "Step B 改为只投产品经理")


def memory(content, sources):
    return {"id": "same-memory", "kind": "CAREER_GOAL", "content": content,
            "attributes": {}, "status": "ACTIVE",
            "source_refs": [s["source_ref"] for s in sources],
            "bound_sources": deepcopy(sources)}


def event(turn, before, after, sources, *, phase="target"):
    return {"turn_id": turn, "phase": phase, "success": True, "operation": "MERGE",
            "target_memory_id": "same-memory", "result_memory_id": "same-memory",
            "before": [deepcopy(before)], "after": [deepcopy(after)],
            "sources": [{k: v for k, v in s.items() if k != "id"} for s in sources]}


def plan(scope, step="a"):
    check = SemanticCheck(id="fact", gold_path="gold.semantics", statement="recorded target fact",
                          support_mode="bound_memory_source", presence="memory",
                          observation_scope=scope, target_step_id=step)
    policy = CompiledCheck(id="mutation", gold_path="gold.mutation", evaluator_kind="memory_mutation",
                          expected={"target_step_id": step, "target_topic_ids": []})
    return ExecutableEvalPlan(case_id="opaque", suite="memory_cases", task_type="test",
                              checks=[policy], semantic_checks=[check], extractions=[])


def payload(scope):
    before = memory("原计划做后端研发", [OLD])
    after_a = memory("先投后端研发", [OLD, A])
    after_b = memory("只投产品经理", [OLD, A, B])
    recalled = {k: deepcopy(v) for k, v in after_a.items() if k not in {"bound_sources", "status"}}
    return {
        "task": [{"id": "earlier"}, {"id": "a"}, {"id": "b"}], "source_policy": {}, "final_answer": "产品经理",
        "final_answers_by_step": {"a": "先投后端研发", "b": "产品经理"},
        "answer_spans": [{"id": "step_a_1", "text": "先投后端研发"}, {"id": "answer_1", "text": "产品经理"}],
        "step_turn_ids": {"a": "turn-a", "b": "turn-b"}, "step_source_refs": {},
        "fixture_source_refs": {}, "evidence": [], "memory_before": [before], "memories": [after_b],
        "memory_events": [event("seed", before, before, [OLD], phase="seed"),
                          event("turn-a" if scope == "target_mutation" else "earlier", before, after_a, [A]),
                          event("turn-b", after_a, after_b, [B])],
        "tool_events": [{"turn_id": "turn-a", "tool_name": "search_memory", "execution_kind": "real",
                         "result": {"ok": True, "agent_payload": {"memories": [recalled]}}}],
    }


def task_data(task):
    return json.loads(task.messages[-1]["content"])


def result(evidence_id):
    return json.dumps({"items": [{"id": "c1", "result": "PASS", "answer_span_id": "step_a_1",
                                   "evidence_ids": [evidence_id], "reason": "recorded source"}]})


@pytest.mark.parametrize("scope", ["target_mutation", "memory_recall"])
def test_same_id_later_mutation_does_not_replace_observed_content(scope):
    data = payload(scope)
    if scope == "memory_recall":
        data["step_turn_ids"] = {"earlier": "earlier", **data["step_turn_ids"]}
    observed = task_data(JudgeTaskPlanner().plan(plan(scope), data)[0])
    assert observed["memories"][0]["content"] == "先投后端研发"


@pytest.mark.parametrize("scope", ["target_mutation", "memory_recall"])
def test_same_id_sources_keep_legal_history_and_reject_later_step(scope):
    data = payload(scope)
    if scope == "memory_recall":
        data["step_turn_ids"] = {"earlier": "earlier", **data["step_turn_ids"]}
    task = JudgeTaskPlanner().plan(plan(scope), data)[0]
    refs = {s["source_ref"] for m in task_data(task)["memories"] for s in m["bound_sources"]}
    assert refs == {OLD["source_ref"], A["source_ref"]}
    aliases = {s["source_ref"]: s["id"] for memory in task_data(task)["memories"] for s in memory["bound_sources"]}
    for legitimate in (OLD, A):
        validated = task.validate(result(aliases[legitimate["source_ref"]]))["checks"][0]
        assert validated["result"] == "PASS" and validated["evidence_ids"] == [legitimate["id"]]
    with pytest.raises(JudgeSchemaError, match="evidence"):
        task.validate(result(B["id"]))
    assert B["source_ref"] not in aliases


@pytest.mark.parametrize("scope", ["target_mutation", "memory_recall"])
def test_missing_historical_observation_is_an_execution_error(scope):
    data = payload(scope)
    if scope == "target_mutation":
        data["memory_events"][1].pop("after")
    else:
        data["tool_events"][0]["result"]["agent_payload"]["memories"] = [{"id": "same-memory"}]
    with pytest.raises(JudgeSchemaError, match="[Hh]istorical|[Ss]napshot|[Oo]bserved"):
        JudgeTaskPlanner().plan(plan(scope), data)


def test_recall_cannot_resolve_source_only_recorded_after_recall_in_same_turn():
    data = payload("memory_recall")
    data["memory_events"][1]["turn_id"] = "turn-a"
    # The content/refs really returned by recall are retained, but the separate
    # ledgers provide no proof whether same-turn source annotation preceded it.
    with pytest.raises(JudgeSchemaError, match="[Hh]istorical|[Oo]rder|[Ss]ource"):
        JudgeTaskPlanner().plan(plan("memory_recall"), data)


def test_recall_after_later_update_still_uses_baseline_source_at_recall_time():
    data = payload("memory_recall")
    data["tool_events"][0]["result"]["agent_payload"]["memories"] = [
        {k: v for k, v in data["memory_before"][0].items() if k not in {"bound_sources", "status"}}]
    data["memory_events"][1]["turn_id"] = "turn-a"
    observed = task_data(JudgeTaskPlanner().plan(plan("memory_recall"), data)[0])
    assert observed["memories"][0]["content"] == "原计划做后端研发"
    assert {s["source_ref"] for s in observed["memories"][0]["bound_sources"]} == {OLD["source_ref"]}


def test_recall_same_ref_changed_raw_content_has_no_inferred_order():
    data = payload("memory_recall")
    data["tool_events"][0]["result"]["agent_payload"]["memories"] = [
        {k: v for k, v in data["memory_before"][0].items() if k not in {"bound_sources", "status"}}]
    data["memory_events"][1]["turn_id"] = "turn-a"
    data["memory_events"][1]["sources"] = [{"source_ref": OLD["source_ref"], "content": "later changed raw content"}]
    with pytest.raises(JudgeSchemaError, match="order is ambiguous"):
        JudgeTaskPlanner().plan(plan("memory_recall"), data)


def test_step_turn_dictionary_order_does_not_establish_history():
    data = payload("memory_recall")
    data["step_turn_ids"] = {"b": "turn-b", "a": "turn-a", "earlier": "earlier"}
    observed = task_data(JudgeTaskPlanner().plan(plan("memory_recall"), data)[0])
    assert {s["source_ref"] for s in observed["memories"][0]["bound_sources"]} == {OLD["source_ref"], A["source_ref"]}


def test_unbound_historical_source_is_not_admitted_by_same_memory_id():
    data = payload("target_mutation")
    data["memory_events"][1]["after"][0]["source_refs"] = [A["source_ref"]]
    observed = task_data(JudgeTaskPlanner().plan(plan("target_mutation"), data)[0])
    assert [s["source_ref"] for s in observed["memories"][0]["bound_sources"]] == [A["source_ref"]]


def test_mutation_preserves_earlier_step_sources_even_without_baseline_content():
    data = payload("target_mutation")
    data["memory_before"] = []
    earlier_source = source("message://earlier", "Earlier committed fact")
    before_a = memory("earlier state", [OLD, earlier_source])
    data["memory_events"].insert(1, event("earlier", data["memory_events"][0]["after"][0], before_a, [earlier_source]))
    data["memory_events"][2]["before"] = [before_a]
    data["memory_events"][2]["after"][0]["source_refs"].append(earlier_source["source_ref"])
    observed = task_data(JudgeTaskPlanner().plan(plan("target_mutation"), data)[0])
    assert {s["source_ref"] for s in observed["memories"][0]["bound_sources"]} == {
        OLD["source_ref"], A["source_ref"], earlier_source["source_ref"]}


def test_latest_committed_snapshot_within_target_turn_is_used():
    data = payload("target_mutation")
    data["memory_events"][2]["turn_id"] = "turn-a"
    observed = task_data(JudgeTaskPlanner().plan(plan("target_mutation"), data)[0])
    assert observed["memories"][0]["content"] == "只投产品经理"
    assert {s["source_ref"] for s in observed["memories"][0]["bound_sources"]} == {OLD["source_ref"], A["source_ref"], B["source_ref"]}


def test_successful_recall_missing_its_snapshot_is_not_an_empty_search():
    data = payload("memory_recall")
    data["tool_events"][0]["result"]["agent_payload"] = {}
    with pytest.raises(JudgeSchemaError, match="snapshot is unavailable"):
        JudgeTaskPlanner().plan(plan("memory_recall"), data)


def test_recall_preserves_distinct_observed_versions_of_same_id():
    data = payload("memory_recall")
    data["step_turn_ids"]["earlier"] = "earlier"
    another = deepcopy(data["tool_events"][0])
    another["result"]["agent_payload"]["memories"] = [
        {k: v for k, v in data["memory_before"][0].items() if k not in {"bound_sources", "status"}}]
    data["tool_events"].append(another)
    observed = task_data(JudgeTaskPlanner().plan(plan("memory_recall"), data)[0])
    assert [m["content"] for m in observed["memories"]] == ["先投后端研发", "原计划做后端研发"]
    assert all("status" not in m for m in observed["memories"])


@pytest.mark.parametrize("scope", ["target_mutation", "memory_recall"])
async def test_missing_history_cannot_return_a_successful_judge_or_case(monkeypatch, scope):
    import app.evals.judge as module

    def client(**kwargs):
        pytest.fail("An incomplete Memory observation must fail before any provider request")

    monkeypatch.setattr(module, "AsyncOpenAI", client)
    monkeypatch.setattr(module, "TokenCounter", lambda _: SimpleNamespace(count=lambda text: len(text) // 4))
    judge = StructuredJudge(Settings(_env_file=None, eval_judge_base_url="https://synthetic.invalid/v1",
                                     eval_judge_api_key="synthetic", eval_judge_model="synthetic"))
    data = payload(scope)
    if scope == "target_mutation":
        data["memory_events"][1].pop("after")
    else:
        data["tool_events"][0]["result"]["agent_payload"]["memories"] = [{"id": "same-memory"}]
    response = await judge.judge(plan=plan(scope), payload=data)
    assert response["execution_error"]["code"] == "JUDGE_SCHEMA_ERROR"
    assert "checks" not in response
    assert execute_plan(plan(scope), EvaluationContext(), response)["outcome"] == "ERROR"


@pytest.mark.parametrize("scope", ["target_mutation", "memory_recall"])
def test_historical_task_view_excludes_later_authored_step_instructions(scope):
    data = payload(scope)
    data["task"] = [{"id": "earlier", "prompt": "Earlier task"},
                    {"id": "a", "prompt": "Current task"},
                    {"id": "b", "prompt": "Later request contains a changed fact"}]
    if scope == "memory_recall":
        data["step_turn_ids"]["earlier"] = "earlier"
    task = JudgeTaskPlanner().plan(plan(scope), data)[0]
    assert task_data(task)["task"] == [{"id": "a", "prompt": "Current task"}]
    assert "Later request contains a changed fact" not in task.messages[-1]["content"]


async def test_full_judge_keeps_valid_earlier_source_after_later_reference_removal(monkeypatch):
    import app.evals.judge as module

    class Stream:
        def __init__(self, body):
            self.body = body

        async def __aiter__(self):
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content=json.dumps(self.body), reasoning_content=None), finish_reason="stop")])

        async def close(self):
            pass

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            submitted = json.loads(kwargs["messages"][-1]["content"])
            bound = next(source for source in submitted["memories"][0]["bound_sources"]
                         if source["source_ref"] == A["source_ref"])
            return Stream({"items": [{"id": "c1", "result": "PASS", "reason": "Supported target snapshot",
                                       "evidence_ids": [bound["id"]]}]})

        async def close(self):
            pass

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module, "TokenCounter", lambda _: SimpleNamespace(count=lambda text: len(text) // 4))
    judge = StructuredJudge(Settings(_env_file=None, eval_judge_base_url="https://synthetic.invalid/v1",
                                     eval_judge_api_key="synthetic", eval_judge_model="synthetic"))
    data = payload("target_mutation")
    data["memory_before"] = []
    data["memories"][0]["bound_sources"] = [B]
    data["memories"][0]["source_refs"] = [B["source_ref"]]
    response = await judge.judge(plan=plan("target_mutation"), payload=data)
    assert "execution_error" not in response
    assert response["checks"][0]["evidence_ids"] == [A["id"]]
    assert response["checks"][0]["answer_span_id"] is None
