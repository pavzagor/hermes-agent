# Stage cron model proposals; never run live jobs

This is a default-profile, read-only audit. Changes here are tooling only. Parent Astra owns architecture, independent review, activation, and production verification.

Guardrails: never alter prompts, schedules, delivery, safety, reasoning pins, or completeness budgets; leave no-agent and disabled jobs unchanged; preserve explicit model/provider/endpoint pins; exact supported model/provider pairs come from the live authorized Sol route. Public provider tests use no tools or user data. No live cron run is a test.

Implementation order: add validator/checker behavior tests, observe red, implement minimal pure validator and exact-output checks, collect scoped usage/history aggregates, run bounded real provider pairs, run all affected tests, commit. Missing latency/projection fields remain unknown, not model-based guesses.
