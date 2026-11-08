"""Compact semantic wire requests retain exact frozen scope and provenance."""
from copy import deepcopy
import json

import pytest

from app.evals.compiler import CompiledCheck, ExecutableEvalPlan, ExtractionSpec, SemanticCheck
from app.evals.executor import EvaluationContext, execute_plan
from app.evals.judge import JudgeSchemaError, restore_observation_text
from app.evals.judge_tasks import JudgeTaskPlanner, PROMPT_VERSION


RAW_ID = "evidence:083675f1-eae3-4b29-b871-67e123be19a4"
BOUND_A_ID = "source:94510fb1-2494-4c7a-869d-74edc9e038e5"
BOUND_B_ID = "source:22e0695f-6a01-4ef4-9396-9de8338f697f"


def check(identifier, **kwargs):
    return SemanticCheck(id=identifier, gold_path="gold.facts[0]", statement=f"Fact {identifier}", **kwargs)


def plan(checks=(), *, extractions=(), deterministic=()):
    return ExecutableEvalPlan(case_id="opaque", suite="agent_tasks", task_type="test",
                              checks=list(deterministic), semantic_checks=list(checks),
                              extractions=list(extractions))


def payload():
    return {
        "task": "Read every source and report the observed fact.",
        "source_policy": {"fact_time_policy": "current_snapshot"},
        "final_answer": "Observed answer.", "final_answers_by_step": {"last": "Observed answer."},
        "answer_spans": [{"id": "answer_1", "text": "Observed answer."}],
        "fixture_source_refs": {}, "step_turn_ids": {}, "step_source_refs": {},
        "evidence": [
            {"id": RAW_ID, "content": "Full raw excerpt. " * 1000, "source_ref": "document://source",
             "visible_to_agent": True, "authoritative_for_grounding": True,
             "evidence_origin": "career_document", "offset_start": 3, "offset_end": 18003},
            {"id": "hidden", "content": "Unobserved fact.", "visible_to_agent": False,
             "authoritative_for_grounding": True},
            {"id": "summary", "content": "Derived summary.", "visible_to_agent": True,
             "authoritative_for_grounding": False},
        ],
        "memories": [], "memory_before": [], "memory_events": [], "tool_events": [],
    }


def contract(task):
    return json.loads(task.messages[0]["content"].split("\n", 1)[1])


def data(task):
    return json.loads(task.messages[-1]["content"])


def result(task, *, evidence=("e1",), span="answer_1"):
    return {"items": [{"id": item["id"], "result": "PASS", "answer_span_id": span,
                       "evidence_ids": list(evidence), "reason": "Observed support."}
                      for item in contract(task)["checks"]]}


def test_same_scope_checks_get_individual_tasks_with_matching_contracts_and_ids():
    checks = [check("first"), check("second")]
    original_plan = plan(checks)
    original_payload = payload()
    before = deepcopy((original_plan.model_dump(), original_payload))
    tasks = JudgeTaskPlanner().plan(original_plan, original_payload)
    assert [task.id for task in tasks] == ["semantic:first", "semantic:second"]
    assert [len(contract(task)["checks"]) for task in tasks] == [1, 1]
    assert all(task.schema_version == PROMPT_VERSION for task in tasks)
    for task, own_check in zip(tasks, checks, strict=True):
        submitted = contract(task)
        assert submitted["checks"] == [{
            **own_check.model_dump(mode="json", exclude={"id", "gold_path"}), "id": "c1",
        }]
        assert set(submitted["response_schema"]["properties"]) == {"items"}
        restored = task.validate(json.dumps(result(task)))["checks"]
        assert restored == [{
            "id": own_check.id, "result": "PASS", "reason": "Observed support.",
            "answer_span_id": "answer_1", "evidence_ids": [RAW_ID],
        }]
        assert data(task)["source_policy"] == original_payload["source_policy"]
        assert data(task)["evidence"] == [{"id": "e1", "content": original_payload["evidence"][0]["content"]}]
    assert (original_plan.model_dump(), original_payload) == before


def test_answer_grounding_uses_deterministic_bounded_shards_without_widening_scope():
    grounding = check("answer_grounding", presence="conditional")
    ordinary = check("ordinary", support_mode="answer_only")
    original_plan = plan([grounding, ordinary])
    original_payload = payload()
    spans = [{"id": f"answer_{index}", "text": f"Observed line {index}."} for index in range(1, 11)]
    original_payload.update(final_answer="\n".join(item["text"] for item in spans),
                            final_answers_by_step={}, answer_spans=spans)
    before = deepcopy((original_plan.model_dump(), original_payload))

    tasks = JudgeTaskPlanner().plan(original_plan, original_payload)
    grounding_tasks, ordinary_task = tasks[:3], tasks[3]
    assert [task.id for task in grounding_tasks] == [
        "semantic:answer_grounding:shard:001",
        "semantic:answer_grounding:shard:002",
        "semantic:answer_grounding:shard:003",
    ]
    assert [task.id for task in tasks] == [* [task.id for task in grounding_tasks], "semantic:ordinary"]
    assert [[span["id"] for span in data(task)["answer_spans"]] for task in grounding_tasks] == [
        ["answer_1", "answer_2", "answer_3", "answer_4"],
        ["answer_5", "answer_6", "answer_7", "answer_8"],
        ["answer_9", "answer_10"],
    ]
    assert [span for task in grounding_tasks for span in data(task)["answer_spans"]] == spans

    expected_criterion = {**grounding.model_dump(mode="json", exclude={"id", "gold_path"}), "id": "c1"}
    expected_evidence = [{"id": "e1", "content": original_payload["evidence"][0]["content"]}]
    for task in grounding_tasks:
        assert contract(task)["checks"] == [expected_criterion]
        assert set(contract(task)["response_schema"]["properties"]) == {"items"}
        assert data(task)["source_policy"] == original_payload["source_policy"]
        assert data(task)["evidence"] == expected_evidence
        own_span = data(task)["answer_spans"][0]["id"]
        assert task.validate(json.dumps(result(task, span=own_span)))["checks"] == [{
            "id": "answer_grounding", "result": "PASS", "reason": "Observed support.",
            "answer_span_id": own_span, "evidence_ids": [RAW_ID],
        }]
        foreign_span = next(span["id"] for span in spans if span["id"] != own_span
                            and span not in data(task)["answer_spans"])
        with pytest.raises(JudgeSchemaError, match="span"):
            task.validate(json.dumps(result(task, span=foreign_span)))

    assert ordinary_task.id == "semantic:ordinary"
    assert contract(ordinary_task)["checks"] == [{
        **ordinary.model_dump(mode="json", exclude={"id", "gold_path"}), "id": "c1",
    }]
    assert data(ordinary_task)["answer_spans"] == spans
    assert (original_plan.model_dump(), original_payload) == before


def test_answer_grounding_shards_keep_existing_extraction_on_full_answer_view():
    grounding = check("answer_grounding", presence="conditional")
    ordinary = check("ordinary", support_mode="answer_only")
    spec = ExtractionSpec(id="bullets", kind="bullets", instructions="Select all delivered bullets.")
    original_payload = payload()
    spans = [{"id": f"answer_{index}", "text": f"Observed line {index}."} for index in range(1, 11)]
    original_payload.update(final_answer="\n".join(item["text"] for item in spans),
                            final_answers_by_step={}, answer_spans=spans)
    tasks = JudgeTaskPlanner().plan(plan([grounding, ordinary], extractions=[spec]), original_payload)
    shards = tasks[:3]
    observation_task = tasks[-1]

    assert [task.id for task in tasks] == [
        "semantic:answer_grounding:shard:001",
        "semantic:answer_grounding:shard:002",
        "semantic:answer_grounding:shard:003",
        "semantic:ordinary",
        "observation:bullets",
    ]
    assert all("observations" not in contract(task) for task in [*shards, tasks[3]])
    assert contract(observation_task)["checks"] == []
    assert contract(observation_task)["observations"] == [spec.model_dump(mode="json")]
    assert data(observation_task)["answer_spans"] == spans
    assert data(observation_task)["source_policy"] == original_payload["source_policy"]
    assert data(observation_task)["evidence"] == [{
        "id": "e1", "content": original_payload["evidence"][0]["content"],
    }]
    extraction = {"id": "bullets", "kind": "bullets", "items": [
        {"id": None, "answer_span_id": "answer_10"},
    ]}
    assert observation_task.validate(json.dumps({"items": [], "observations": [extraction]}))["extractions"] == [extraction]
    with pytest.raises(JudgeSchemaError, match="span"):
        shards[0].validate(json.dumps(result(shards[0], span="answer_10")))


@pytest.mark.parametrize("scope_change", [
    {"support_mode": "answer_only"},
    {"support_mode": "technical_rubric"},
    {"observation_scope": "memory_recall"},
    {"target_step_id": "another-step"},
])
def test_different_declared_scope_or_dependencies_never_share_a_request(scope_change):
    tasks = JudgeTaskPlanner().plan(plan([check("first"), check("second", **scope_change)]), payload())
    assert len(tasks) == 2
    assert [len(contract(task)["checks"]) for task in tasks] == [1, 1]


def test_semantic_view_keeps_full_visible_evidence_and_only_one_answer_representation():
    observation = payload()
    task = JudgeTaskPlanner().plan(plan([check("fact")]), observation)[0]
    submitted = data(task)
    assert "final_answer" not in submitted
    assert "final_answers_by_step" not in submitted
    assert submitted["answer_spans"] == observation["answer_spans"]
    assert submitted["evidence"] == [{"id": "e1", "content": observation["evidence"][0]["content"]}]
    assert RAW_ID not in task.messages[-1]["content"]
    assert set(contract(task)["response_schema"]["properties"]) == {"items"}


def test_projection_keeps_full_provenance_and_original_spans_for_validation(monkeypatch):
    import app.evals.judge_tasks as judge_tasks

    observation = payload()
    observation["answer_spans"].append({"id": "step_last_1", "text": "Observed answer."})
    before = deepcopy(observation)
    captured = {}
    original_validate = judge_tasks.validate_response

    def capture(raw, *, plan, payload):
        captured.update(deepcopy(payload))
        return original_validate(raw, plan=plan, payload=payload)

    monkeypatch.setattr(judge_tasks, "validate_response", capture)
    task = JudgeTaskPlanner().plan(plan([check("fact")]), observation)[0]
    assert data(task)["answer_spans"] == [{"id": "step_last_1", "text": "Observed answer."}]
    restored = task.validate(json.dumps(result(task, span="step_last_1")))["checks"]
    assert restored[0]["evidence_ids"] == [RAW_ID]
    assert captured["evidence"] == [before["evidence"][0]]
    assert captured["answer_spans"] == before["answer_spans"]
    assert observation == before


def test_fixture_dependency_keeps_raw_source_identity_without_other_source_metadata():
    observation = payload()
    observation["fixture_source_refs"] = {"supplement": "document://source"}
    observation["evidence"].append({
        "id": "another-observed-source", "content": "Another entire raw excerpt.",
        "source_ref": "document://another", "visible_to_agent": True,
        "authoritative_for_grounding": True, "offset_start": 7, "turn_id": "observed-turn",
    })
    before = deepcopy(observation)
    task = JudgeTaskPlanner().plan(plan([check("fact", required_fixture_id="supplement")]), observation)[0]
    submitted = data(task)
    assert submitted["fixture_source_refs"] == {"supplement": "document://source"}
    assert submitted["evidence"] == [
        {"id": "e1", "content": observation["evidence"][0]["content"], "source_ref": "document://source"},
        {"id": "e2", "content": "Another entire raw excerpt.", "source_ref": "document://another"},
    ]
    assert task.validate(json.dumps(result(task)))["checks"][0]["evidence_ids"] == [RAW_ID]
    assert observation == before


@pytest.mark.parametrize("fact_time_policy", ["current_snapshot", "preserve_event_time"])
def test_historical_conflicting_evidence_preserves_observation_time_only_when_required(fact_time_policy):
    observation = payload()
    observation["source_policy"]["fact_time_policy"] = fact_time_policy
    observation["evidence"] = [
        {"id": RAW_ID, "content": "Historical observed state: backend role.",
         "source_ref": "document://same-source", "turn_id": "turn-earlier", "phase": "seed",
         "visible_to_agent": True, "authoritative_for_grounding": True},
        {"id": "later-state", "content": "Later observed state: product role.",
         "source_ref": "document://same-source", "turn_id": "turn-later", "phase": "target",
         "visible_to_agent": True, "authoritative_for_grounding": True},
        {"id": "undated-state", "content": "A source without an observed turn.",
         "visible_to_agent": True, "authoritative_for_grounding": True},
    ]
    before = deepcopy(observation)
    task = JudgeTaskPlanner().plan(plan([check("historical-role")]), observation)[0]
    sources = data(task)["evidence"]
    assert [source["content"] for source in sources] == [item["content"] for item in before["evidence"]]
    assert [source["id"] for source in sources] == ["e1", "e2", "e3"]
    if fact_time_policy == "preserve_event_time":
        assert [(source["turn_id"], source["phase"]) for source in sources[:2]] == [
            ("turn-earlier", "seed"), ("turn-later", "target")]
        assert set(sources[2]) == {"id", "content"}
    else:
        assert all(set(source) == {"id", "content"} for source in sources)
    assert task.validate(json.dumps(result(task)))["checks"][0]["evidence_ids"] == [RAW_ID]
    assert observation == before


@pytest.mark.parametrize("authored_order", [True, False])
def test_equal_text_in_different_steps_is_preserved_and_final_alias_requires_authored_order(authored_order):
    observation = payload()
    observation.update({
        "task": [{"id": "early"}, {"id": "last"}] if authored_order else "Read and report.",
        # Deliberately reverse dictionary order: only authored steps define time.
        "final_answers_by_step": {"last": "Observed answer.", "early": "Observed answer."},
        "answer_spans": [{"id": "answer_1", "text": "Observed answer."},
                         {"id": "step_early_1", "text": "Observed answer."},
                         {"id": "step_last_1", "text": "Observed answer."}],
    })
    before = deepcopy(observation)
    task = JudgeTaskPlanner().plan(plan([check("fact")]), observation)[0]
    submitted = data(task)["answer_spans"]
    expected_ids = ["step_early_1", "step_last_1"] if authored_order else [
        "answer_1", "step_early_1", "step_last_1"]
    assert [span["id"] for span in submitted] == expected_ids
    assert observation == before


def test_incomplete_last_step_observation_does_not_remove_any_final_alias():
    observation = payload()
    observation.update({
        "final_answer": "First line.\nSecond line.",
        "final_answers_by_step": {"last": "First line.\nSecond line."},
        "answer_spans": [{"id": "answer_1", "text": "First line."},
                         {"id": "answer_2", "text": "Second line."},
                         {"id": "step_last_1", "text": "First line."}],
    })
    task = JudgeTaskPlanner().plan(plan([check("fact")]), observation)[0]
    assert data(task)["answer_spans"] == observation["answer_spans"]


@pytest.mark.parametrize("extracted_span", ["answer_1", "step_last_1"])
def test_item_scoped_semantics_and_observation_share_request(extracted_span):
    observation = payload()
    observation["answer_spans"].append({"id": "step_last_1", "text": "Observed answer."})
    extraction = {"id": "revised_bullets", "kind": "bullets", "items": [{
        "id": "target", "answer_span_id": extracted_span,
    }]}
    definition = check("target-fact", support_mode="answer_only",
                       required_extraction_id="revised_bullets", required_item_id="target")
    original_plan = plan([definition], extractions=[ExtractionSpec(
        id="revised_bullets", kind="bullets", allowed_ids=["target"], instructions="Select actual bullet spans.")],
        deterministic=[CompiledCheck(id="target-fact", gold_path="gold.required_target_facts[0]",
                                     evaluator_kind="semantic", expected={})])
    tasks = JudgeTaskPlanner().plan(original_plan, observation)
    assert len(tasks) == 1
    body = {**result(tasks[0], evidence=(), span=extracted_span), "observations": [extraction]}
    response = tasks[0].validate(json.dumps(body))
    response["extractions"] = restore_observation_text(response["extractions"], observation)
    evaluated = execute_plan(original_plan, EvaluationContext(data={"final_answers": ["Observed answer."]}), response)
    assert evaluated["outcome"] == "PASS"


@pytest.mark.parametrize("change", ["missing", "duplicate", "forged_check", "original_check", "forged_evidence",
                                    "original_evidence", "duplicate_evidence", "span", "old_envelope"])
def test_wire_rejects_omissions_forged_ids_and_old_long_id_protocol(change):
    task = JudgeTaskPlanner().plan(plan([check("original-check")]), payload())[0]
    body = result(task)
    if change == "missing":
        body["items"] = []
    elif change == "duplicate":
        body["items"].append(deepcopy(body["items"][0]))
    elif change == "forged_check":
        body["items"][0]["id"] = "c2"
    elif change == "original_check":
        body["items"][0]["id"] = "original-check"
    elif change == "forged_evidence":
        body["items"][0]["evidence_ids"] = ["e2"]
    elif change == "original_evidence":
        body["items"][0]["evidence_ids"] = [RAW_ID]
    elif change == "duplicate_evidence":
        body["items"][0]["evidence_ids"] = ["e1", "e1"]
    elif change == "span":
        body["items"][0]["answer_span_id"] = "missing"
    else:
        body = {"checks": body["items"]}
    with pytest.raises(ValueError):
        task.validate(json.dumps(body))


def test_semantic_task_order_follows_check_order_and_each_validator_restores_its_own_id():
    tasks = JudgeTaskPlanner().plan(plan([check("first"), check("second")]), payload())
    assert [task.id for task in tasks] == ["semantic:first", "semantic:second"]
    assert [task.validate(json.dumps(result(task)))["checks"][0]["id"] for task in tasks] == [
        "first", "second"]


def memory_payload():
    observation = payload()
    sources = [
        {"id": BOUND_A_ID, "source_ref": "message://alpha", "content": "Alpha observed source."},
        {"id": BOUND_B_ID, "source_ref": "message://beta", "content": "Beta observed source."},
    ]
    memories = [{"id": "same-memory", "kind": "CAREER_GOAL", "content": f"{step} saved fact",
                 "attributes": {}, "status": "ACTIVE", "source_refs": [source["source_ref"]],
                 "bound_sources": [source]}
                for step, source in zip(("alpha", "beta"), sources, strict=True)]
    observation.update({
        "task": [{"id": "alpha"}, {"id": "beta"}],
        "final_answer": "Beta answer.", "final_answers_by_step": {"alpha": "Alpha answer.", "beta": "Beta answer."},
        "answer_spans": [{"id": "step_alpha_1", "text": "Alpha answer."},
                         {"id": "step_beta_1", "text": "Beta answer."},
                         {"id": "answer_1", "text": "Beta answer."}],
        "fixture_source_refs": {"alpha-fixture": "message://alpha", "beta-fixture": "message://beta"},
        "step_turn_ids": {"alpha": "turn-alpha", "beta": "turn-beta"},
        "step_source_refs": {"alpha": "message://alpha", "beta": "message://beta"},
        "memories": [memories[1]], "memory_before": [],
        "memory_events": [
            {"turn_id": f"turn-{step}", "phase": "target", "success": True,
             "target_memory_id": "same-memory", "result_memory_id": "same-memory",
             "after": [memory], "sources": memory["bound_sources"]}
            for step, memory in zip(("alpha", "beta"), memories, strict=True)
        ],
    })
    return observation


def test_bound_memory_tasks_do_not_mix_checks_target_steps_or_later_versions():
    checks = [check(f"{step}-fact-{index}", support_mode="bound_memory_source", presence="memory",
                    observation_scope="target_mutation", target_step_id=step, required_fixture_id=f"{step}-fixture")
              for step in ("alpha", "beta") for index in range(2)]
    policies = [CompiledCheck(id=f"mutation-{step}", gold_path="gold.mutation", evaluator_kind="memory_mutation",
                              expected={"target_step_id": step, "target_topic_ids": []})
                for step in ("alpha", "beta")]
    observation = memory_payload()
    before = deepcopy(observation)
    tasks = JudgeTaskPlanner().plan(plan(checks, deterministic=policies), observation)
    assert [task.id for task in tasks] == [
        "semantic:alpha-fact-0", "semantic:alpha-fact-1",
        "semantic:beta-fact-0", "semantic:beta-fact-1",
    ]
    for task in tasks:
        check_id = task.id.removeprefix("semantic:")
        step = check_id.split("-", 1)[0]
        original_source = BOUND_A_ID if step == "alpha" else BOUND_B_ID
        assert len(contract(task)["checks"]) == 1
        submitted = data(task)
        assert submitted["source_policy"] == observation["source_policy"]
        assert set(submitted["step_turn_ids"]) == {step}
        assert set(submitted["step_source_refs"]) == {step}
        assert submitted["memories"][0]["content"] == f"{step} saved fact"
        assert submitted["memories"][0]["status"] == "ACTIVE"
        assert submitted["memories"][0]["source_refs"] == [f"message://{step}"]
        assert submitted["answer_spans"] == [{"id": f"step_{step}_1", "text": f"{step.title()} answer."}]
        assert submitted["memories"][0]["bound_sources"] == [{
            **observation["memory_events"][0 if step == "alpha" else 1]["sources"][0], "id": "b1",
        }]
        assert not {"memory_before", "memory_events", "tool_events", "final_answer", "final_answers_by_step"} & set(submitted)
        restored = task.validate(json.dumps(result(task, evidence=("b1",), span=f"step_{step}_1")))["checks"]
        assert [item["id"] for item in restored] == [check_id]
        assert all(item["evidence_ids"] == [original_source] for item in restored)
        with pytest.raises(JudgeSchemaError, match="span"):
            task.validate(json.dumps(result(task, evidence=("b1",), span=f"step_{'beta' if step == 'alpha' else 'alpha'}_1")))
    assert observation == before


def test_observation_only_uses_original_spans_and_its_minimal_schema():
    spec = ExtractionSpec(id="topics", kind="topic_set", allowed_ids=["observed-topic"], instructions="Identify observed topics.")
    observation = payload()
    task = JudgeTaskPlanner().plan(plan(extractions=[spec]), observation)[0]
    assert "final_answer" not in data(task)
    assert contract(task)["observations"] == [spec.model_dump(mode="json")]
    body = {"id": "topics", "kind": "topic_set", "items": ["observed-topic"], "answer_span_id": "answer_1"}
    assert task.validate(json.dumps({"items": [], "observations": [body]}))["extractions"] == [body]


def test_fixture_and_item_dependencies_keep_individual_semantic_tasks():
    checks = [check("first"), check("fixture", required_fixture_id="supplement"),
              check("item", required_extraction_id="bullets", required_item_id="target")]
    tasks = JudgeTaskPlanner().plan(plan(checks), payload())
    assert [task.id for task in tasks] == ["semantic:first", "semantic:fixture", "semantic:item"]
    assert [contract(task)["checks"][0]["required_fixture_id"] for task in tasks] == [
        None, "supplement", None]
    assert all(len(contract(task)["checks"]) == 1 for task in tasks)
    assert "source_ref" not in data(tasks[0])["evidence"][0]
    assert data(tasks[1])["evidence"][0]["source_ref"] == "document://source"
    assert "source_ref" not in data(tasks[2])["evidence"][0]


def test_ordinary_semantics_accepts_minimal_result_without_copying_spans_or_evidence():
    task = JudgeTaskPlanner().plan(plan([check("fact")]), payload())[0]
    response = task.validate(json.dumps({"items": [{"id": "c1", "result": "PASS", "reason": "Faithful answer."}]}))
    assert response["checks"] == [{"id": "fact", "result": "PASS", "reason": "Faithful answer.",
                                    "answer_span_id": None, "evidence_ids": []}]
