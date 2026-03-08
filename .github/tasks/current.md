# Active Task Queue

metadata:
  project: AI_Dashboard
  mode: sequential
  updated_at: 2026-03-09

tasks:
  - id: T-001
    title: Intake request and decompose into executable tasks
    owner: customer
    status: done
    inputs:
      - user requirement
    done_criteria:
      - requirement is split into sequential implementation tasks
      - each task has owner and completion criteria
    handoff_to: project-manager

  - id: T-002
    title: Plan README update tasks for Copilot agent architecture
    owner: project-manager
    status: done
    inputs:
      - user requirement
      - existing README structure
      - task queue rules
    done_criteria:
      - tasks are sequential and testable
      - exactly one active owner at each transition
      - documentation task includes concrete acceptance points
    handoff_to: documentation

  - id: T-003
    title: Add Japanese section to README for Copilot multi-agent configuration
    owner: documentation
    status: done
    inputs:
      - architecture points to include
      - existing README tone and section flow
    done_criteria:
      - new section is added without breaking existing content
      - required architecture items are explicitly documented in Japanese
      - key configuration files are referenced
    handoff_to: project-manager

  - id: T-004
    title: Review documentation update against acceptance criteria
    owner: project-manager
    status: done
    inputs:
      - updated README content
      - task done criteria
    done_criteria:
      - all required items are present
      - placement is natural in README flow
      - report is returned to caller
    handoff_to: customer
