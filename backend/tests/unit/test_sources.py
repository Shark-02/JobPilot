import pytest

from app.knowledge.sources import SourceRef


def test_source_ref_parses_all_supported_kinds() -> None:
    value = "00000000-0000-0000-0000-000000000001"
    document = "00000000-0000-0000-0000-000000000002"
    message = SourceRef(f"message://{value}")
    section = SourceRef(f"document://{document}/section/{value}")
    chunk = SourceRef(f"document://{document}/chunk/{value}")
    assert message.parts()[0] == "message"
    assert section.parts()[2] == "section"
    assert chunk.parts()[2] == "chunk"


@pytest.mark.parametrize("value", ["message://not-a-uuid", "document://x/chunk/y", "unknown://x"])
def test_source_ref_rejects_invalid_uri(value: str) -> None:
    with pytest.raises(ValueError):
        SourceRef(value)
