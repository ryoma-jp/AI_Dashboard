# Task Queue Format

This folder is the task queue used by the project manager and specialist agents.

## Files
- `current.md`: Active sequential queue.
- `archive/`: Optional storage for completed task snapshots.

## Task Entry Schema
Use this YAML-like block for each task item:

```yaml
- id: T-001
  title: Add dataset filter API
  owner: backend-developer
  status: todo
  inputs:
    - endpoint spec
    - serializer update
  done_criteria:
    - endpoint returns filtered results
    - test added or updated
  handoff_to: frontend-developer
```

## Rules
- Keep tasks small and independently verifiable.
- Maintain strict sequential execution: only one `in_progress` task.
- Update `owner` and `handoff_to` when transferring work.
- Mark blocked tasks with clear blocker notes.
