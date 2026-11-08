"""Offline contract regressions for the process-local parallel Judge.

Only synthetic observations and the existing frozen typed cases are used. These
tests do not create an Agent, connect to a database, or call a real provider.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.evals.compiler import CompiledCheck, ExecutableEvalPlan, ExtractionSpec, SemanticCheck, compile_case
from app.evals.datasets import load_dataset
from app.config import Settings
from app.evals.executor import EvaluationContext, execute_plan
from app.evals.judge import JudgeSchemaError, StructuredJudge, validate_response
from app.evals.judge_tasks import JudgeTaskPlanner


def observation(plan=None):
    steps = {check.target_step_id for check in (plan.semantic_checks if plan else []) if check.target_step_id}
    return {
        "task": "synthetic task", "source_policy": {"fact_time_policy": "current_snapshot"},
        "final_answer": "observed answer", "answer_spans": [
            {"id": "answer_1", "text": "observed answer"},
            *[{"id": f"step_{step}_1", "text": "observed answer"} for step in sorted(steps)],
        ],
        "evidence": [
            {"id": "raw", "content": "observed source", "visible_to_agent": True,
             "authoritative_for_grounding": True},
            {"id": "derived", "content": "assistant summary", "visible_to_agent": True,
             "authoritative_for_grounding": False},
            {"id": "hidden", "content": "unseen corpus fact", "visible_to_agent": False,
             "authoritative_for_grounding": True},
        ],
        "memories": [], "memory_before": [], "memory_events": [], "tool_events": [],
        "step_turn_ids": {step: f"turn:{step}" for step in steps},
        "final_answers_by_step": {step: "observed answer" for step in steps},
        "fixture_source_refs": {}, "step_source_refs": {},
    }


def request(checks=(), extractions=(), deterministic=()):
    return ExecutableEvalPlan(case_id="opaque", suite="agent_tasks", task_type="test",
                              checks=list(deterministic), semantic_checks=list(checks),
                              extractions=list(extractions))


def semantic_result(identifier="c1", *, result="PASS", span="answer_1", evidence=("e1",)):
    return {"items": [{"id": identifier, "result": result, "answer_span_id": span,
                        "evidence_ids": list(evidence), "reason": "atomic synthetic result"}]}


def task_data(task):
    return json.loads(task.messages[-1]["content"])


def task_contract(task):
    return json.loads(task.messages[0]["content"].split("\n", 1)[1])


@pytest.mark.parametrize("case_id", sorted(load_dataset().cases))
def test_frozen_a_r_m_f_plans_preserve_all_checks_and_nonmaterial_extractions(case_id):
    plan = compile_case(load_dataset().cases[case_id])
    tasks = JudgeTaskPlanner().plan(plan, observation(plan))
    semantic_tasks = [task for task in tasks if task.kind == "semantic"]
    assigned = [check for task in semantic_tasks
                for check in task_contract(task)["checks"]]
    expected = [check.model_dump(mode="json", exclude={"id", "gold_path"}) for check in plan.semantic_checks]
    ordering = lambda check: json.dumps(check, sort_keys=True)
    assert sorted([{k: v for k, v in check.items() if k != "id"} for check in assigned], key=ordering) == sorted(expected, key=ordering)
    assert [spec for task in tasks for spec in task_contract(task).get("observations", [])] == [
        spec.model_dump(mode="json") for spec in plan.extractions]
    assert all(spec.kind != "material_claims" for spec in plan.extractions)
    assert [task.id for task in semantic_tasks] == [
        "semantic:answer_grounding:shard:001" if check.id == "answer_grounding" else f"semantic:{check.id}"
        for check in plan.semantic_checks
    ]
    assert all(len(task_contract(task)["checks"]) == 1 for task in semantic_tasks)
    assert len(tasks) == len({task.id for task in tasks})
    # Case identity must not determine tasks, schemas, or submitted evidence.
    renamed = plan.model_copy(update={"case_id": "another-opaque-identity"})
    other = JudgeTaskPlanner().plan(renamed, observation(plan))
    assert [(task.id, task.messages) for task in tasks] == [(task.id, task.messages) for task in other]


def test_a1_non_grounding_semantic_requests_remain_byte_identical_for_cache_reuse():
    compiled = compile_case(load_dataset().cases["A1"])
    payload = observation(compiled)
    planner = JudgeTaskPlanner()
    sharded = planner.plan(compiled, payload)
    without_grounding = compiled.model_copy(update={
        "semantic_checks": [check for check in compiled.semantic_checks if check.id != "answer_grounding"],
    })
    baseline = planner.plan(without_grounding, payload)
    actual = [task for task in sharded if task.kind == "semantic"
              and not task.id.startswith("semantic:answer_grounding:shard:")]

    assert len(actual) == 10
    assert [(task.id, task.kind, task.stage, task.schema_version, task.messages) for task in actual] == [
        (task.id, task.kind, task.stage, task.schema_version, task.messages) for task in baseline
    ]


def test_semantic_task_has_only_assigned_check_and_minimal_schema():
    checks = [SemanticCheck(id="fact", gold_path="gold.facts", statement="one observed fact"),
              SemanticCheck(id="policy", gold_path="gold.prohibitions", statement="one absence",
                            support_mode="answer_only", presence="absence")]
    tasks = JudgeTaskPlanner().plan(request(checks), observation())
    for task, check in zip(tasks, checks, strict=True):
        contract = task_contract(task)
        assert contract["checks"] == [{"id": "c1", **check.model_dump(mode="json", exclude={"id", "gold_path"})}]
        assert set(contract["response_schema"]["properties"]) == {"items"}
        assert "material_claims" not in task.messages[0]["content"]
        assert task.validate(json.dumps(semantic_result(evidence=() if check.support_mode == "answer_only" else ("e1",))))["checks"][0]["id"] == check.id
    assert [item["id"] for item in task_data(tasks[0])["evidence"]] == ["e1"]
    assert task_data(tasks[1])["evidence"] == []


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "span", "evidence", "enum", "envelope"])
def test_semantic_task_rejects_invalid_or_out_of_scope_result(change):
    task = JudgeTaskPlanner().plan(request([SemanticCheck(id="fact", gold_path="gold.fact", statement="fact")]), observation())[0]
    body = semantic_result()
    if change == "missing":
        body["items"] = []
    elif change == "duplicate":
        body["items"].append(deepcopy(body["items"][0]))
    elif change == "unknown":
        body["items"][0]["id"] = "other-check"
    elif change == "span":
        body["items"][0]["answer_span_id"] = "missing-span"
    elif change == "evidence":
        body["items"][0]["evidence_ids"] = ["derived"]
    elif change == "enum":
        body["items"][0]["result"] = "MAYBE"
    else:
        body["extractions"] = []
    with pytest.raises(ValueError):
        task.validate(json.dumps(body))


def test_atomic_fail_and_unknown_survive_semantic_task_validation():
    task = JudgeTaskPlanner().plan(request([SemanticCheck(id="fact", gold_path="gold.fact", statement="fact")]), observation())[0]
    for status in ("FAIL", "UNKNOWN"):
        result = task.validate(json.dumps(semantic_result(result=status)))
        assert result["checks"][0]["result"] == status
        assert "outcome" not in result


def test_nonmaterial_extraction_validates_its_own_kind_and_exact_id():
    spec = ExtractionSpec(id="topics", kind="topic_set", allowed_ids=["topic"], instructions="extract actual topics")
    task = JudgeTaskPlanner().plan(request(extractions=[spec]), observation())[0]
    assert "MaterialClaim" not in json.dumps(task_contract(task)["response_schema"])
    body = {"id": "topics", "kind": "topic_set", "items": ["topic"], "answer_span_id": "answer_1"}
    assert task.validate(json.dumps({"items": [], "observations": [body]}))["extractions"] == [body]
    for field, value in [("id", "unexpected"), ("kind", "project_selection"),
                         ("items", ["invented"]), ("answer_span_id", "missing")]:
        with pytest.raises(ValueError):
            task.validate(json.dumps({"items": [], "observations": [{**body, field: value}]}))


def memory_request(scope):
    check = SemanticCheck(id="memory_fact", gold_path="gold.semantics", statement="target memory fact",
                          support_mode="bound_memory_source", presence="memory", observation_scope=scope,
                          target_step_id="decision")
    policy = CompiledCheck(id="mutation", gold_path="gold.mutation", evaluator_kind="memory_mutation",
                          expected={"target_step_id": "decision", "target_topic_ids": ["target-topic"]})
    return request([check], deterministic=[policy])


def memory_observation(scope):
    payload = observation(memory_request(scope))
    payload["memories"] = [
        {"id": "target", "kind": "CAREER_GOAL", "status": "ACTIVE", "attributes": {},
         "source_refs": ["message://target"], "content": "target memory fact", "bound_sources": [
            {"id": "bound-target", "content": "target user's fact", "source_ref": "message://target"}]},
        {"id": "unrelated", "kind": "CAREER_GOAL", "status": "ACTIVE", "attributes": {},
         "source_refs": ["message://other"], "content": "unrelated memory fact", "bound_sources": [
            {"id": "bound-unrelated", "content": "other user's fact", "source_ref": "message://other"}]},
    ]
    payload["memory_before"] = deepcopy(payload["memories"])
    payload["memory_events"] = [
        {"turn_id": "turn:decision", "phase": "target", "success": True,
         "target_memory_id": "target", "result_memory_id": "target", "sources": []},
        {"turn_id": "turn:decision", "phase": "target", "success": True,
         "target_memory_id": "unrelated", "result_memory_id": "unrelated", "sources": []},
        {"turn_id": "seed-turn", "phase": "seed", "success": True,
         "target_memory_id": "unrelated", "result_memory_id": "unrelated", "sources": []},
    ]
    for event in payload["memory_events"]:
        event["after"] = [deepcopy(m) for m in payload["memories"] if m["id"] == event["result_memory_id"]]
    payload["tool_events"] = [
        {"turn_id": "turn:decision", "tool_name": "search_memory", "execution_kind": "real",
         "result": {"ok": True, "agent_payload": {"memories": [
             {key: deepcopy(value) for key, value in payload["memories"][0].items()
              if key not in {"status", "bound_sources"}}]}}},
        {"turn_id": "another-turn", "tool_name": "search_memory", "execution_kind": "real",
         "result": {"ok": True, "agent_payload": {"memories": [{"id": "unrelated"}]}}},
        {"turn_id": "turn:decision", "tool_name": "search_memory", "execution_kind": "cache",
         "result": {"ok": True, "agent_payload": {"memories": [{"id": "unrelated"}]}}},
    ]
    return payload


@pytest.mark.parametrize("scope", ["target_mutation", "memory_recall"])
def test_memory_semantics_cannot_borrow_another_target_or_general_career_evidence(scope):
    extractions = [{"id": "memory_topics", "kind": "memory_topics", "items": [
        {"memory_id": "target", "topic_id": "target-topic", "state": "present"},
        {"memory_id": "unrelated", "topic_id": "other", "state": "present"}]}]
    task = JudgeTaskPlanner().plan(memory_request(scope), memory_observation(scope))[0]
    data = task_data(task)
    assert [memory["id"] for memory in data["memories"]] == (["target", "unrelated"] if scope == "target_mutation" else ["target"])
    assert data["evidence"] == []
    assert {span["id"] for span in data["answer_spans"]} <= {"answer_1", "step_decision_1"}
    accepted = semantic_result(span="step_decision_1", evidence=("b1",))
    assert task.validate(json.dumps(accepted))["checks"][0]["result"] == "PASS"
    for evidence in ("bound-unrelated", "raw", "hidden"):
        rejected = semantic_result(span="step_decision_1", evidence=(evidence,))
        with pytest.raises(ValueError):
            task.validate(json.dumps(rejected))


def test_memory_topic_extraction_preserves_conflicting_and_other_memories_but_removes_bound_sources():
    spec = ExtractionSpec(id="memory_topics", kind="memory_topics", allowed_ids=["target-topic"], instructions="map every memory")
    data = memory_observation("target_mutation")
    task = JudgeTaskPlanner().plan(request(extractions=[spec]), data)[0]
    assert {memory["id"] for memory in task_data(task)["memories"]} == {"target", "unrelated"}
    assert all("bound_sources" not in memory for memory in task_data(task)["memories"])
    body = {"id": "memory_topics", "kind": "memory_topics", "items": [
        {"memory_id": "target", "topic_id": "target-topic", "state": "present"},
        {"memory_id": "unrelated", "topic_id": "other", "state": "present"}]}
    assert task.validate(json.dumps({"items": [], "observations": [body]}))["extractions"] == [body]
    body["items"].pop()
    with pytest.raises(ValueError):
        task.validate(json.dumps({"items": [], "observations": [body]}))


@pytest.mark.parametrize("scope", ["target_mutation", "memory_recall"])
def test_memory_target_answer_cannot_borrow_a_step_with_a_shared_id_prefix(scope):
    payload = memory_observation(scope)
    payload["final_answers_by_step"]["decision_extra"] = "unrelated answer"
    payload["answer_spans"].append({"id": "step_decision_extra_1", "text": "unrelated answer"})
    task = JudgeTaskPlanner().plan(memory_request(scope), payload)[0]
    assert "step_decision_extra_1" not in {s["id"] for s in task_data(task)["answer_spans"]}
    with pytest.raises(ValueError):
        task.validate(json.dumps(semantic_result(span="step_decision_extra_1", evidence=("b1",))))


def test_assembled_contract_rejects_missing_duplicate_and_unknown_extraction_ids():
    specs = [ExtractionSpec(id=name, kind="topic_set", instructions="observed topics") for name in ("first", "second")]
    plan = request(extractions=specs)
    items = [{"id": spec.id, "kind": spec.kind, "items": [], "answer_span_id": None} for spec in specs]
    for values in (items[:1], [items[0], items[0]], [items[0], {**items[1], "id": "unknown"}]):
        with pytest.raises(JudgeSchemaError):
            validate_response(json.dumps({"checks": [], "extractions": values}), plan=plan, payload=observation())


class SyntheticStream:
    def __init__(self, body, *, delay=0, error=None, finish="stop", on_finished=None):
        self.body = body
        self.delay = delay
        self.error = error
        self.finish = finish
        self.on_finished = on_finished

    async def __aiter__(self):
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        if self.on_finished:
            self.on_finished()
        yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(content=json.dumps(self.body), reasoning_content=None),
            finish_reason=self.finish)])

    async def close(self):
        pass


def synthetic_judge(monkeypatch, respond):
    import app.evals.judge as module

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            assert kwargs["max_tokens"] == 6144
            return respond(kwargs["messages"])

        async def close(self):
            pass

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    # No tokenizer network/cache dependency belongs in a Mock Provider test.
    monkeypatch.setattr(module, "TokenCounter", lambda _: SimpleNamespace(count=lambda text: len(text) // 4))
    return StructuredJudge(Settings(_env_file=None, eval_judge_base_url="https://synthetic.example/v1",
                                    eval_judge_api_key="synthetic", eval_judge_model="synthetic",
                                    eval_judge_concurrency=2, eval_judge_transport_retries=0))


def semantic_execution_plan(statuses):
    definitions = [SemanticCheck(id=name, gold_path=f"gold.{name}", statement=name,
                                 support_mode="answer_only") for name in statuses]
    deterministic = [CompiledCheck(id=check.id, gold_path=check.gold_path, evaluator_kind="semantic",
                                   expected="PASS") for check in definitions]
    return request(definitions, deterministic=deterministic)


def answer_grounding_execution_plan():
    definition = SemanticCheck(id="answer_grounding", gold_path="source_policy", statement="all answer facts are grounded",
                               support_mode="authoritative_visible", presence="conditional")
    return request([definition], deterministic=[CompiledCheck(id="answer_grounding", gold_path="source_policy",
                                                               evaluator_kind="semantic", expected="PASS")])


def answer_grounding_observation():
    payload = observation()
    spans = [{"id": f"answer_{index}", "text": f"observed answer {index}"} for index in range(1, 11)]
    payload.update(final_answer="\n".join(item["text"] for item in spans),
                   final_answers_by_step={}, answer_spans=spans)
    return payload


def answer_grounding_body(messages, *, result="PASS"):
    submitted = json.loads(messages[-1]["content"])
    return semantic_result(result=result, span=submitted["answer_spans"][0]["id"], evidence=())


@pytest.mark.parametrize("mode, expected_result, expected_error, expected_outcome", [
    ("pass", "PASS", None, "PASS"),
    ("fail", "FAIL", None, "FAIL"),
    ("provider", None, "PROVIDER_HTTP_503", "BLOCKED"),
    ("fail_with_provider", "FAIL", None, "FAIL"),
    ("schema_with_fail", None, "JUDGE_SCHEMA_ERROR", "ERROR"),
])
async def test_answer_grounding_shards_use_all_aggregation_and_preserve_interruption_rules(
        monkeypatch, mode, expected_result, expected_error, expected_outcome):
    class ProviderUnavailable(Exception):
        status_code = 503

    def respond(messages):
        first_span = json.loads(messages[-1]["content"])["answer_spans"][0]["id"]
        if mode == "provider" and first_span == "answer_5":
            return SyntheticStream({}, error=ProviderUnavailable())
        if mode == "fail_with_provider":
            if first_span == "answer_1":
                return SyntheticStream(answer_grounding_body(messages, result="FAIL"))
            if first_span == "answer_5":
                return SyntheticStream({}, error=ProviderUnavailable())
        if mode == "schema_with_fail":
            if first_span == "answer_1":
                return SyntheticStream(answer_grounding_body(messages, result="FAIL"))
            if first_span == "answer_5":
                return SyntheticStream(semantic_result("unexpected", span=first_span, evidence=()))
        return SyntheticStream(answer_grounding_body(messages, result="FAIL" if mode == "fail" and first_span == "answer_5" else "PASS"))

    plan = answer_grounding_execution_plan()
    result = await synthetic_judge(monkeypatch, respond).judge(plan=plan, payload=answer_grounding_observation())

    assert len(result["subtasks"]) == 3
    if expected_error:
        assert result["execution_error"]["code"] == expected_error
        assert result["execution_error"]["task_id"] == "semantic:answer_grounding:shard:002"
        assert "checks" not in result
    else:
        assert "execution_error" not in result
        assert result["checks"] == [{
            "id": "answer_grounding", "result": expected_result,
            "answer_span_id": None, "evidence_ids": [],
            "reason": result["checks"][0]["reason"],
        }]
        assert "ALL shards" in result["checks"][0]["reason"]
    assert execute_plan(plan, EvaluationContext(), result)["outcome"] == expected_outcome


async def test_actual_execution_combines_individual_semantics_and_observation_without_dependency(monkeypatch):
    statuses = {"slow": "PASS", "fast": "UNKNOWN"}
    plan = semantic_execution_plan(statuses)
    plan.extractions = [ExtractionSpec(id="topics", kind="topic_set", allowed_ids=["topic"], instructions="actual topics")]
    completed = []

    def respond(messages):
        contract = json.loads(messages[0]["content"].split("\n", 1)[1])
        items = [semantic_result(check["id"], result=statuses[check["statement"]], evidence=())["items"][0]
                 for check in reversed(contract["checks"])]
        body = {"items": items}
        if "observations" in contract:
            body["observations"] = [{"id": "topics", "kind": "topic_set", "items": ["topic"],
                                     "answer_span_id": "answer_1"}]
        return SyntheticStream(body, on_finished=lambda: completed.append("direct"))

    result = await synthetic_judge(monkeypatch, respond).judge(plan=plan, payload=observation())
    assert "execution_error" not in result
    assert completed == ["direct", "direct"]
    assert [check["id"] for check in result["checks"]] == ["slow", "fast"]
    assert [check["result"] for check in result["checks"]] == ["PASS", "UNKNOWN"]
    assert result["summary"]["total_subtasks"] == 2
    assert result["telemetry"]["provider_output_tokens"] is None
    assert execute_plan(plan, EvaluationContext(), result)["outcome"] == "FAIL"


@pytest.mark.parametrize("status", ["FAIL", "UNKNOWN"])
async def test_parallel_atomic_semantic_failure_cannot_be_overridden_by_sibling_pass(monkeypatch, status):
    statuses = {"passing": "PASS", "failing": status}
    plan = semantic_execution_plan(statuses)

    def respond(messages):
        checks = json.loads(messages[0]["content"].split("\n", 1)[1])["checks"]
        return SyntheticStream({"items": [semantic_result(check["id"], result=statuses[check["statement"]],
                                                          evidence=())["items"][0] for check in checks]})

    result = await synthetic_judge(monkeypatch, respond).judge(plan=plan, payload=observation())
    assert "execution_error" not in result
    assert result["checks"][1]["result"] == status
    assert execute_plan(plan, EvaluationContext(), result)["outcome"] == "FAIL"


@pytest.mark.parametrize("failure,expected_code,expected_outcome", [
    ("schema", "JUDGE_SCHEMA_ERROR", "ERROR"),
    ("length", "JUDGE_OUTPUT_TRUNCATED", "ERROR"),
    ("timeout", "JUDGE_TIMEOUT", "BLOCKED"),
    ("provider", "PROVIDER_HTTP_503", "BLOCKED"),
])
async def test_partial_subtask_failure_has_no_public_judgement_or_false_pass(monkeypatch, failure, expected_code, expected_outcome):
    class ProviderUnavailable(Exception):
        status_code = 503

    # Every criterion is an independent task, so one failure cannot be hidden
    # behind successful sibling checks.
    plan = semantic_execution_plan({**{f"passing_{i}": "PASS" for i in range(4)}, "failing": "PASS"})
    plan.semantic_checks[-1] = plan.semantic_checks[-1].model_copy(update={"support_mode": "technical_rubric"})

    def respond(messages):
        checks = json.loads(messages[0]["content"].split("\n", 1)[1])["checks"]
        if checks[0]["statement"].startswith("passing_"):
            return SyntheticStream({"items": [semantic_result(check["id"], evidence=())["items"][0] for check in checks]})
        name = checks[0]["id"]
        if failure == "schema":
            return SyntheticStream(semantic_result("unknown", evidence=()))
        if failure == "length":
            return SyntheticStream(semantic_result(name, evidence=()), finish="length")
        return SyntheticStream({}, error=asyncio.TimeoutError() if failure == "timeout" else ProviderUnavailable())

    result = await synthetic_judge(monkeypatch, respond).judge(plan=plan, payload=observation())
    assert result["execution_error"]["code"] == expected_code
    assert result["execution_error"]["task_id"] == "semantic:failing"
    assert "checks" not in result and "extractions" not in result
    assert len(result["subtasks"]) == 5
    assert result["summary"]["failures_by_stage"]["semantic"] == 1
    assert execute_plan(plan, EvaluationContext(), result)["outcome"] == expected_outcome
