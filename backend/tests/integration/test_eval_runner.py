"""Real runtime/runner wiring with synthetic model and Judge, no provider calls."""
import pytest

from app.agent.model import ModelCompleted, ModelResponse
from app.config import Settings
from app.evals.datasets import load_dataset
from app.evals.runner import EvalRunner, RunOptions


@pytest.mark.parametrize("case_id", ["A8", "A9"])
async def test_runner_visible_fixtures_are_unique_and_annotations_never_reach_agent(
        profile_database_factory, monkeypatch, case_id):
    import app.evals.runner as runner_module
    import app.agent.loop as loop_module
    import app.agent.trace as trace_module

    database = profile_database_factory()
    requests = []

    class Model:
        def __init__(self, settings):
            pass

        async def stream(self, request):
            requests.append(request)
            yield ModelCompleted(response=ModelResponse(
                assistant_message={"role": "assistant", "content": "Synthetic incomplete answer"},
                final_text="Synthetic incomplete answer", finish_reason="stop"))

    class Judge:
        async def judge(self, *, plan, payload):
            # This intentionally incomplete answer must fail its semantic Gold.
            checks = [{"id": check.id, "result": "FAIL", "answer_span_id": "answer_1",
                       "evidence_ids": [], "reason": "Synthetic answer omits required facts"}
                      for check in plan.semantic_checks]
            extractions = [{"id": spec.id, "kind": spec.kind, "items": [],
                            **({"answer_span_id": None} if spec.kind in {"project_selection", "topic_set"} else {})}
                           for spec in plan.extractions]
            return {"checks": checks, "extractions": extractions}

    monkeypatch.setattr(runner_module, "session_factory", database.sessions)
    monkeypatch.setattr(trace_module, "session_factory", database.sessions)
    monkeypatch.setattr(loop_module, "OpenAICompatibleModelClient", Model)
    settings = Settings(_env_file=None, llm_model="synthetic")
    dataset = load_dataset()
    runner = EvalRunner(settings, dataset, RunOptions(case_ids={case_id}), "unused-read-only-source")
    runner.judge = Judge()
    row, judged = await runner.run_case(dataset.cases[case_id])
    assert row["result"]["outcome"] == "FAIL", row["result"]
    assert requests and judged is not None
    visible = [message["content"] for request in requests for message in request.messages
               if message["role"] == "user"]
    fixture = dataset.cases[case_id].setup.user_fixtures[0]
    fixture_text = dataset.fixture(fixture.path)
    assert sum(fixture_text in content for content in visible) == 1
    assert "evaluator_annotation" not in "\n".join(visible)
    assert "Eval Gold" not in "\n".join(visible)
    assert row["observable"]["evidence"]["capture_complete"] is True
    ref = row["observable"]["fixture_source_refs"][fixture.id]
    raw = [item for item in row["observable"]["evidence"]["items"] if item["source_ref"] == ref]
    assert len(raw) == 1 and raw[0]["visible_to_agent"] and raw[0]["authoritative_for_grounding"]
