# Task Archive Rules

This directory stores completed task history moved out of `.github/tasks/current.md`.

## Naming convention
- Monthly archive file: `YYYY-MM.md` (example: `2026-03.md`)
- Add archived tasks to the month file matching the archive execution date.

## Update rules
- `current.md` remains the source of truth for active queue operation.
- `archive/*.md` stores completed task history and must not contain `todo` or `in_progress` tasks.
- Archived entries keep original fields (`id`, `title`, `owner`, `status`, `inputs`, `done_criteria`, `handoff_to`) to preserve traceability.

## Rotation trigger
Run archive migration when either condition is met:
- completed tasks in `current.md` reach 5 or more since last archive, or
- `current.md` grows beyond roughly 200 lines.

## Migration procedure
1. Select eligible `status: done` tasks from `current.md`.
2. Append them to the target `archive/YYYY-MM.md` file.
3. Remove the moved tasks from `current.md`.
4. Record moved id range and destination in `current.md` metadata (`recent_archive`).
