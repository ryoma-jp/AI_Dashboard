---
name: task-management
description: 'Manage sequential engineering tasks in .github/tasks/current.md. Use for decomposition, assignment, status transitions, and handoff control in this project.'
argument-hint: 'Describe the request, current queue state, and desired next owner'
user-invocable: true
---

# Task Management Skill

## When to Use
- A new user request must be broken into executable tasks.
- A task status needs to move across `todo`, `in_progress`, `blocked`, `done`.
- Ownership must be transferred between customer, project manager, and specialists.
- The project manager needs to enforce single-thread sequential execution.

## Procedure
1. Read `.github/tasks/current.md` and identify the active queue state.
2. Convert the incoming request into small tasks with clear `done_criteria`.
3. Ensure only one task is `in_progress`.
4. Assign each task to one owner from:
   - `customer`
   - `project-manager`
   - `backend-developer`
   - `frontend-developer`
   - `ml-engineer`
   - `infra-qa`
   - `documentation`
   - `performance`
5. Set `handoff_to` for the next owner after completion.
6. Write concise blocker notes when status is `blocked`.
7. Keep tasks in strict sequence and avoid parallel execution unless explicitly requested.

## Output Contract
Return a concise report with:
- queue changes
- active task id
- next owner
- blockers (if any)

## Guardrails
- Do not edit files outside `.github/tasks/` while using this skill.
- Do not mark a task `done` without matching `done_criteria`.
- Do not keep multiple `in_progress` tasks.
