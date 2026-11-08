from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import UUID


@dataclass(frozen=True)
class IncompatibleEntity:
    entity_id: UUID
    tokens: int
    limit: int

    def as_dict(self) -> dict:
        return {"entity_id": str(self.entity_id), "tokens": self.tokens,
                "limit": self.limit, "excess": self.tokens - self.limit}


@dataclass(frozen=True)
class EntityCompatibility:
    total: int
    compatible: int
    max_observed_tokens: int
    max_observed_characters: int
    incompatible: tuple[IncompatibleEntity, ...]

    def as_dict(self) -> dict:
        return {"total": self.total, "compatible": self.compatible, "too_long": len(self.incompatible),
                "max_observed_tokens": self.max_observed_tokens,
                "max_observed_characters": self.max_observed_characters,
                "incompatible": [item.as_dict() for item in self.incompatible]}


@dataclass(frozen=True)
class CompatibilityReport:
    profile_id: UUID
    profile_name: str
    tokenizer_encoding: str
    max_input_tokens: int
    knowledge: EntityCompatibility
    memory: EntityCompatibility

    @property
    def compatible(self) -> bool:
        return not self.knowledge.incompatible and not self.memory.incompatible

    def as_dict(self) -> dict:
        return {"profile_id": str(self.profile_id), "profile_name": self.profile_name,
                "tokenizer_encoding": self.tokenizer_encoding, "max_input_tokens": self.max_input_tokens,
                "knowledge": self.knowledge.as_dict(), "memory": self.memory.as_dict(),
                "compatible": self.compatible}


class CompatibilityError(ValueError):
    def __init__(self, report: CompatibilityReport) -> None:
        self.report = report
        super().__init__(f"Embedding corpus incompatible with profile {report.profile_name}; "
                         f"knowledge_too_long={len(report.knowledge.incompatible)} "
                         f"memory_too_long={len(report.memory.incompatible)}; no provider request performed")


@dataclass(frozen=True)
class EntityBackfillReport:
    total: int
    existing: int
    pending: int
    completed: int
    failed: int

    def as_dict(self) -> dict:
        return asdict(self)
