from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.embedding import cli
from app.embedding.service import BackfillReport, CoverageReport, EntityCoverage


@pytest.mark.parametrize(
    ("existing", "failed", "status", "exit_code"),
    [(1, 0, "READY", 0), (0, 1, "INCOMPLETE", 2), (1, 1, "INCOMPLETE", 2)],
)
async def test_backfill_cli_reports_status_and_exit_code_without_activating(
    monkeypatch, capsys, existing, failed, status, exit_code,
) -> None:
    profile = SimpleNamespace(id=uuid4(), name="synthetic-space-v1")
    coverage = CoverageReport(profile.id, profile.name, EntityCoverage(1, existing), EntityCoverage(0, 0))
    report = BackfillReport(total=1, existing=0, pending=1, completed=existing, failed=failed, coverage=coverage)
    calls = []

    class FakeService:
        async def get(self, name):
            calls.append(("get", name))
            return profile

        async def backfill(self, selected):
            assert selected is profile
            calls.append(("backfill", selected.name))
            return report

    @asynccontextmanager
    async def fake_sessions():
        yield object()

    monkeypatch.setattr(cli, "session_factory", fake_sessions)
    monkeypatch.setattr(cli, "get_settings", lambda: object())
    monkeypatch.setattr(cli, "EmbeddingProfileService", lambda *args: FakeService())

    assert report.status == status
    assert await cli.run(SimpleNamespace(operation="backfill", profile=profile.name)) == exit_code
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == status
    assert output["completed"] == existing
    assert output["failed"] == failed
    assert calls == [("get", profile.name), ("backfill", profile.name)]
