from __future__ import annotations

import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.agent.model import ModelCompleted, ModelEvent, ModelRequest, ModelResponse, TextDelta, ToolCall
from app.config import Settings


class ConfigurationError(RuntimeError):
    pass


class OpenAICompatibleModelClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not (settings.llm_base_url and settings.llm_api_key and settings.llm_model):
            self._client = None
        else:
            self._client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    def _require_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise ConfigurationError("LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL must be configured before running an agent turn")
        return self._client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response: ModelResponse | None = None
        async for event in self.stream(request):
            if isinstance(event, ModelCompleted):
                response = event.response
        if response is None:
            raise RuntimeError("Provider ended without a completion")
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        client = self._require_client()
        stream = await client.chat.completions.create(
            model=request.model,
            messages=request.messages,  # type: ignore[arg-type]
            tools=request.tools,  # type: ignore[arg-type]
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            reasoning_effort="low",
            stream=True,
        )
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage_input: int | None = None
        usage_output: int | None = None
        async for chunk in stream:
            if chunk.usage:
                usage_input = chunk.usage.prompt_tokens
                usage_output = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            if delta.content:
                text.append(delta.content)
                yield TextDelta(text=delta.content)
            for tool_delta in delta.tool_calls or []:
                row = calls.setdefault(tool_delta.index, {"id": "", "name": "", "arguments": ""})
                if tool_delta.id:
                    row["id"] = tool_delta.id
                if tool_delta.function and tool_delta.function.name:
                    row["name"] += tool_delta.function.name
                if tool_delta.function and tool_delta.function.arguments:
                    row["arguments"] += tool_delta.function.arguments
        tool_calls: list[ToolCall] = []
        assistant_calls: list[dict] = []
        for row in calls.values():
            try:
                arguments = json.loads(row["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"__raw_arguments__": row["arguments"]}
            tool_calls.append(ToolCall(id=row["id"], name=row["name"], arguments=arguments))
            assistant_calls.append({"id": row["id"], "type": "function", "function": {"name": row["name"], "arguments": row["arguments"]}})
        assistant_message: dict = {"role": "assistant", "content": "".join(text) or None}
        if assistant_calls:
            assistant_message["tool_calls"] = assistant_calls
        yield ModelCompleted(response=ModelResponse(
            assistant_message=assistant_message, final_text="".join(text), tool_calls=tool_calls,
            finish_reason=finish_reason, input_tokens=usage_input, output_tokens=usage_output,
        ))
