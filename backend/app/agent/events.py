from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class AgentEvent:
    type: str
    data: dict

    def sse(self) -> str:
        return f"event: {self.type}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"

    def json(self) -> str:
        return json.dumps({"type": self.type, **self.data}, ensure_ascii=False)


class EventBroker:
    """Per-process replay plus fan-out for the AG-UI-compatible delta stream."""

    def __init__(self) -> None:
        self._history: dict[UUID, list[AgentEvent]] = defaultdict(list)
        self._listeners: dict[UUID, list[asyncio.Queue[AgentEvent | None]]] = defaultdict(list)
        self._finished: set[UUID] = set()

    async def publish(self, turn_id: UUID, event_type: str, **data: object) -> None:
        event = AgentEvent(event_type, {"turn_id": str(turn_id), **data})
        self._history[turn_id].append(event)
        for queue in list(self._listeners[turn_id]):
            await queue.put(event)

    async def finish(self, turn_id: UUID) -> None:
        self._finished.add(turn_id)
        for queue in list(self._listeners[turn_id]):
            await queue.put(None)

    async def subscribe(self, turn_id: UUID) -> AsyncIterator[AgentEvent]:
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        for event in list(self._history[turn_id]):
            yield event
        if turn_id in self._finished:
            return
        self._listeners[turn_id].append(queue)
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            if queue in self._listeners[turn_id]:
                self._listeners[turn_id].remove(queue)


event_broker = EventBroker()


class CancellationRegistry:
    def __init__(self) -> None:
        self._events: dict[UUID, asyncio.Event] = {}

    def event(self, turn_id: UUID) -> asyncio.Event:
        return self._events.setdefault(turn_id, asyncio.Event())

    def cancel(self, turn_id: UUID) -> None:
        self.event(turn_id).set()

    def remove(self, turn_id: UUID) -> None:
        self._events.pop(turn_id, None)


cancellation_registry = CancellationRegistry()
