# Active Task Queue

metadata:
  project: AI_Dashboard
  mode: sequential
  updated_at: 2026-03-09
  source_of_truth:
    active_queue: .github/tasks/current.md
    completed_history: .github/tasks/archive/*.md
  recent_archive:
    moved_task_ids: T-101..T-118
    destination: .github/tasks/archive/2026-03.md
    moved_at: 2026-03-09

tasks:
  - id: T-119
    title: Redesign task file operation to prevent current.md growth
    owner: project-manager
    status: done
    inputs:
      - user request for archive-based operation
      - existing .github/tasks/current.md with many done entries
    done_criteria:
      - operation model is defined with current and archive responsibilities
      - naming/update/rotation trigger rules are documented
      - queue remains sequentially operable
    handoff_to: documentation

  - id: T-120
    title: Implement archive structure and migrate completed tasks
    owner: documentation
    status: done
    inputs:
      - migrated done tasks (T-101 to T-118)
      - task archive operation model from T-119
    done_criteria:
      - .github/tasks/archive directory and archive rule document are created
      - completed task history is moved to archive markdown without field loss
      - current.md no longer contains the migrated tasks
    handoff_to: documentation

  - id: T-121
    title: Align project-wide task management guideline with new operation
    owner: documentation
    status: done
    inputs:
      - .github/copilot-instructions.md Task Management Rules
      - updated task operation files under .github/tasks/
    done_criteria:
      - guideline text has no contradiction with archive operation
      - source-of-truth statement remains explicit for active queue
      - migration trigger/traceability expectation is documented
    handoff_to: infra-qa

  - id: T-122
    title: Validate archive migration references and queue consistency
    owner: infra-qa
    status: done
    inputs:
      - .github/tasks/current.md
      - .github/tasks/archive/2026-03.md
      - .github/tasks/README.md
    done_criteria:
      - moved task IDs are traceable from current to archive
      - only one or zero in_progress tasks exist in current queue
      - archive destination files referenced in rules actually exist
    handoff_to: project-manager
