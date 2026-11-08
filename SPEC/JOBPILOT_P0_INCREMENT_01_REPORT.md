# JobPilot P0 Increment 01 Report

## Scope

This increment completes the P0 chat convergence foundation and Knowledge deletion. It deliberately does not claim the separate Observation Persistence and Observability Dashboard phases as complete.

## Phase 0 audit

- Frontend: React 19 with Vite.
- Existing chat path: `Chat.tsx` manually created an EventSource, accumulated `streaming`, held a separate tool-activity array, and reloaded conversation history at completion.
- Message ownership before this change: persisted history was rendered from `App`, while streamed assistant text was rendered from a second local buffer. This is the concrete duplicate-message cause: final history and the stream were independently rendered.
- Markdown before this change: plain React text rendering, so Markdown syntax was literal.
- Knowledge deletion before this change: the backend DELETE endpoint existed but returned only `204`; the UI offered only Archive.
- FK audit: `knowledge_document -> knowledge_section -> knowledge_chunk` uses `ON DELETE CASCADE`. `memory_source` references document sections/chunks with `ON DELETE CASCADE`; orphan memory cleanup remains in the same deletion transaction.
- Existing traces: `agent_turn`, `model_call_trace`, and `tool_call_trace` only retain summary data. Retrieval is implemented in `backend/app/knowledge/retrieval.py` as lexical + semantic candidates, RRF fusion, then section/module assembly.

## Implemented convergence

- Added the official `@assistant-ui/react-ag-ui` adapter and aligned the assistant-ui React core version with it.
- Replaced the hand-written EventSource parser, stream string, pending state, and final history reload path with one `useAgUiRuntime` owner.
- Added a standard AG-UI POST streaming endpoint. The server emits AG-UI event payloads and assigns a UUID before streaming a model response; the final persisted assistant message uses that same UUID.
- The chat transcript now uses assistant-ui primitives. User messages are optimistic runtime messages; the adapter owns stream assembly, message IDs, cancellation state, and transcript hydration.
- Assistant text is rendered by `react-markdown` with GFM. Raw HTML is not enabled.
- Added an explicit destructive-confirmation delete action for READY, ARCHIVED, and FAILED document cards.
- DELETE now reports the deleted document and section/chunk counts after the transaction commits.

## Verification

- `frontend`: `npm run build` passed.
- Backend Python AST syntax validation passed.
- `git diff --check` passed.

## Observability completion

- Migration `0002_agent_observability` adds the complete payload store and records `started_at` / `finished_at` on the existing Model and Tool trace tables. Existing rows use their historical creation timestamp as the only available backfill value; new rows carry real boundary timestamps.
- `TraceRecorder` is the one observation boundary for Model and Tool execution. It persists actual outbound model messages/tool schemas and the returned assistant/tool payload without storing credentials or hidden reasoning.
- `search_knowledge` records its query filters, lexical and vector candidate ranks/scores or distances, RRF score/rank, module assembly, and the exact modules returned to the Agent. This debug capture is derived after the normal calculation and does not affect ordering.
- Read-only APIs: `GET /api/observability/turns`, `/turns/{turn_id}`, `/turns/{turn_id}/observations`, and `/observations/{observation_id}`.
- The renamed Observability view provides runs, an ordered timeline, expandable Input/Output/Metadata/Sources/Raw payloads, and a read-only play/pause/step replay. It never invokes the Agent or any tool.
