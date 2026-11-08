# JobPilot Contribution Rules

## Documentation is part of every change

Every iteration that changes application code, schema, configuration, runtime
semantics, evaluation behavior, tests, API contracts, or operator workflow
must update the relevant project Markdown in the same iteration. Do not leave
the implemented behavior to drift from its documentation.

- Treat `docs/JOBPILOT_TECHNICAL_OVERVIEW.md` as the canonical current-state
  document; update it for implementation, configuration, migration,
  observability, validation, and limitation changes.
- Update the root `README.md` when local setup, runbook, public capability, or
  developer verification guidance changes. Update `docs/EVALUATION.md` when
  evaluation execution, isolation, metrics, or review semantics change.
- Update a `SPEC/` document when its stated contract has changed. Preserve
  historical reports as historical evidence; append a dated supersession note
  instead of rewriting their past results.
- Before handoff, search the tracked Markdown for superseded identifiers,
  removed behavior, stale defaults, and changed test counts. Documentation
  claims must be grounded in the current code, migrations, and executed
  verification rather than planned behavior.

## Scope discipline

Keep evaluation cases, thresholds, and fixtures unchanged unless the requested
work explicitly includes them. Preserve unrelated user changes and never put
secrets, private source content, or database URLs into Markdown.
