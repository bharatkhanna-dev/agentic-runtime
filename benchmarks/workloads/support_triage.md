# Support Triage Workload

This workload is the headline demo for the first paper cycle.

## Why this workload matters

It exercises runtime behaviors that are easy to validate and important for production:

- classification and routing steps
- typed tool calls
- escalation logic
- approval checkpoints
- bounded retries

## Early benchmark ideas

- classify severity correctly
- route to the correct queue
- require approval for sensitive actions
- avoid invalid tool order
- stop when policy denies execution