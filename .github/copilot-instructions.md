# AI Dashboard Project Guidelines

## Objective
- This workspace uses a multi-agent workflow with a customer proxy agent and a specialist implementation team.
- Always execute user requests by decomposing requirements into sequential tasks first.
- Prefer stable, incremental edits over broad refactors.

## Architecture Awareness
- Backend: Django + DRF in `django_project/app/` and `django_project/project/`.
- Frontend: Django templates in `django_project/templates/` and assets in `django_project/static/`.
- ML: Training and inference in `django_project/app/machine_learning/`.
- Infra: Docker Compose and nginx configuration in root and `nginx/`.

## Multi-Agent Orchestration Rules
- The customer agent is the user-facing proxy and should delegate implementation only through the project manager agent.
- The project manager agent must split requirements into sequential, testable tasks.
- Specialist agents must work only in their owned scope and return concise completion reports.
- If a task crosses multiple scopes, execute in this order unless explicitly changed: backend -> frontend -> ml -> infra/qa -> documentation -> performance.

## Task Management Rules
- Source of truth is `.github/tasks/current.md`.
- Every task entry must include: `id`, `title`, `owner`, `status`, `inputs`, `done_criteria`.
- Allowed status values: `todo`, `in_progress`, `blocked`, `done`.
- Only one task may be `in_progress` at a time for sequential execution.

## Coding Conventions
- Follow existing project style in each file; avoid unrelated formatting changes.
- Keep comments minimal and useful.
- Preserve public APIs unless the task explicitly requests API changes.

## Validation
- Run the smallest relevant checks for touched scope.
- For local test execution, prefer Docker-based commands and environment.
- Report what was validated and what remains unverified.

## Safety
- Never delete unrelated files or revert unrelated user changes.
- If requirements are ambiguous, propose assumptions in the task breakdown first.
