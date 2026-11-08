from __future__ import annotations

import asyncio
import re
from enum import Enum

import httpcore
import httpx
from openai import APIConnectionError, APITimeoutError


class ProviderFailureKind(str, Enum):
    TIMEOUT = "TIMEOUT"
    BILLING = "BILLING"
    RATE_LIMITED = "RATE_LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


_TRANSIENT_HTTP_STATUSES = frozenset({402, 429, 500, 502, 503, 504})


def normalize_provider_error(error: Exception, *, timeout_code: str) -> str | None:
    """Return a durable code for errors caused by the model provider.

    Callers retain their own timeout namespace (for example ``MODEL_TIMEOUT``
    or ``JUDGE_TIMEOUT``), while connection and HTTP failures share one
    normalization path.  ``None`` deliberately means that the error has no
    evidence of being a provider interruption.
    """

    if isinstance(error, (asyncio.TimeoutError, APITimeoutError,
                          httpx.TimeoutException, httpcore.TimeoutException)):
        return timeout_code
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return f"PROVIDER_HTTP_{status_code}"
    # Streaming errors can escape the SDK after request creation, without its
    # APIConnectionError wrapper. Only remote/network failures are transient;
    # local protocol errors (for example an invalid request) stay evaluator errors.
    if isinstance(error, (APIConnectionError, OSError, httpx.NetworkError,
                          httpx.RemoteProtocolError, httpcore.NetworkError,
                          httpcore.RemoteProtocolError)):
        return "PROVIDER_CONNECTION_ERROR"
    return None


def classify_provider_error_code(error_code: str | None) -> ProviderFailureKind | None:
    """Classify only provider interruptions that invalidate an eval attempt."""

    if error_code in {"MODEL_TIMEOUT", "JUDGE_TIMEOUT"}:
        return ProviderFailureKind.TIMEOUT
    if error_code == "PROVIDER_CONNECTION_ERROR":
        return ProviderFailureKind.UNAVAILABLE
    match = re.fullmatch(r"PROVIDER_HTTP_(\d{3})", error_code or "")
    if not match:
        return None
    status = int(match.group(1))
    if status == 402:
        return ProviderFailureKind.BILLING
    if status == 429:
        return ProviderFailureKind.RATE_LIMITED
    if status in _TRANSIENT_HTTP_STATUSES:
        return ProviderFailureKind.UNAVAILABLE
    return None
