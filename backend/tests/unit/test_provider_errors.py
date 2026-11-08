from __future__ import annotations

import httpcore
import httpx
import pytest

from app.infra.provider_errors import normalize_provider_error


@pytest.mark.parametrize("error", [
    httpx.RemoteProtocolError("peer disconnected"),
    httpx.ReadError("connection reset"),
    httpx.ConnectError("connection refused"),
    httpcore.RemoteProtocolError("peer disconnected"),
    httpcore.ReadError("connection reset"),
    httpcore.ConnectError("connection refused"),
])
def test_http_transport_interruptions_use_provider_connection_code(error):
    assert normalize_provider_error(error, timeout_code="JUDGE_TIMEOUT") == "PROVIDER_CONNECTION_ERROR"


@pytest.mark.parametrize("error", [
    httpx.ReadTimeout("read deadline exceeded"),
    httpx.ConnectTimeout("connect deadline exceeded"),
    httpcore.ReadTimeout("read deadline exceeded"),
    httpcore.ConnectTimeout("connect deadline exceeded"),
])
def test_http_transport_timeouts_keep_the_callers_timeout_namespace(error):
    assert normalize_provider_error(error, timeout_code="JUDGE_TIMEOUT") == "JUDGE_TIMEOUT"


@pytest.mark.parametrize("error", [
    httpx.LocalProtocolError("invalid request"),
    httpx.UnsupportedProtocol("unsupported scheme"),
    httpcore.LocalProtocolError("invalid request"),
    ValueError("invalid JSON"),
])
def test_local_contract_failures_are_not_transient_provider_interruptions(error):
    assert normalize_provider_error(error, timeout_code="JUDGE_TIMEOUT") is None
