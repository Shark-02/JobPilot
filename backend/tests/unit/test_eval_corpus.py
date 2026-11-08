import hashlib
from types import SimpleNamespace

import pytest

from app.evals.corpus import frozen_snapshot_texts
from app.evals.datasets import DatasetError
from app.evals.database import EvalDatabaseSafetyError


def dataset(root, **item):
    return SimpleNamespace(root=root, manifest=SimpleNamespace(model_dump=lambda **kwargs: {"frozen_documents": [item]}))


def test_frozen_snapshot_reads_exact_bytes_and_requires_hash(tmp_path):
    path = tmp_path / "raw.md"
    raw = b"# Frozen\r\nExact bytes\r\n"
    path.write_bytes(raw)
    data = dataset(tmp_path, key="example", snapshot_path="raw.md", sha256=hashlib.sha256(raw).hexdigest())
    assert frozen_snapshot_texts(data) == {"example": raw.decode()}
    path.write_bytes(raw + b"changed")
    with pytest.raises(DatasetError, match="SHA256 mismatch"):
        frozen_snapshot_texts(data)


@pytest.mark.parametrize("relative", [None, "missing.md", "../outside.md"])
def test_frozen_snapshot_missing_or_outside_package_is_error(tmp_path, relative):
    with pytest.raises(DatasetError):
        frozen_snapshot_texts(dataset(tmp_path, key="example", snapshot_path=relative, sha256="a" * 64))


async def test_direct_local_snapshot_writer_refuses_product_before_query_or_mutation():
    from app.evals.corpus import setup_local_document
    from sqlalchemy.engine import make_url
    session = SimpleNamespace(bind=SimpleNamespace(url=make_url("postgresql://localhost/jobpilot")))
    with pytest.raises(EvalDatabaseSafetyError, match="Product database"):
        await setup_local_document(session, {}, "", "corpus_v1")
