# Acceptance Criteria Patterns

## Pattern A: Given/When/Then
- Given <context/state>
- When <action>
- Then <expected outcome>
- And <additional observable outcome>

## Pattern B: API Contract
- Request:
- Response:
- Status codes:
- Error body schema:
- Retries/timeouts:
- Idempotency rules:

## Pattern C: Performance / Reliability
- p95 latency <= X ms for <operation> at <load>
- Error rate <= Y% over Z minutes
- RPO/RTO targets (if relevant)

## Pattern D: Security
- AuthN method:
- AuthZ rules:
- Audit log events emitted:
- Data retention policy applied:
