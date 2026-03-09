# Task Queue Format

This folder is the task queue used by the project manager and specialist agents.

## Files
- `current.md`: Source of truth for active queue operation (active and recent tasks only).
- `archive/`: Completed task history snapshots rotated out from `current.md`.

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

## Archive Operation
- Naming convention for archive files: `archive/YYYY-MM.md`.
- Archive trigger:
  - completed tasks in `current.md` reach 5 or more since last archive, or
  - `current.md` exceeds about 200 lines.
- Migration steps:
  1. Move eligible `status: done` tasks from `current.md` to the month archive file.
  2. Keep task fields unchanged to preserve traceability.
  3. Record moved id range and archive destination in `current.md` metadata.
  4. Ensure `current.md` remains compact and focused on active work.
