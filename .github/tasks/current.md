# Active Task Queue

metadata:
  project: AI_Dashboard
  mode: sequential
  updated_at: 2026-03-09

tasks:
  - id: T-101
    title: Reorganize design information from current implementation
    owner: project-manager
    status: done
    inputs:
      - user requirement
      - current repository structure and key implementation files
    done_criteria:
      - architecture and implementation points are extracted by technical area
      - documentation information architecture is defined for docs directory
    handoff_to: documentation

  - id: T-102
    title: Create docs directory and add technical markdown documents
    owner: documentation
    status: done
    inputs:
      - task T-101 output
      - existing README and project structure
    done_criteria:
      - docs directory exists at repository root
      - multiple markdown files are added and grouped by technology
      - document links and cross references are valid
    handoff_to: documentation

  - id: T-103
    title: Simplify README to overview and docs navigation
    owner: documentation
    status: done
    inputs:
      - existing README
      - new docs information architecture
    done_criteria:
      - README contains concise project overview
      - README links users to detailed docs under docs/
      - no unrelated large scale refactoring is introduced
    handoff_to: infra-qa

  - id: T-104
    title: Validate links and file existence for documentation changes
    owner: infra-qa
    status: done
    inputs:
      - updated README and docs files
    done_criteria:
      - markdown links in README and docs resolve correctly
      - expected docs files exist in repository
      - verification result and residual risks are reported
    handoff_to: project-manager

  - id: T-105
    title: Integrate results and report completion
    owner: project-manager
    status: done
    inputs:
      - outputs from T-101 to T-104
    done_criteria:
      - requested deliverables are compiled for user
      - queue statuses are updated consistently
      - final report includes verification summary and risks
    handoff_to: customer

  - id: T-106
    title: Register additional README cleanup request in task queue
    owner: project-manager
    status: done
    inputs:
      - user additional request (README should contain overview only)
      - current task queue state
    done_criteria:
      - additional request is recorded as sequential task entries
      - each entry contains id/title/owner/status/inputs/done_criteria
      - task order is explicit and traceable
    handoff_to: documentation

  - id: T-107
    title: Remove legacy detailed block from README and keep minimal structure
    owner: documentation
    status: done
    inputs:
      - current README content including legacy HTML comment block
      - requirement: overview + docs guidance + minimal quickstart only
    done_criteria:
      - legacy detailed content is removed from README including HTML comments
      - README keeps concise overview section
      - README keeps docs navigation and minimal quickstart section
    handoff_to: infra-qa

  - id: T-108
    title: Re-validate docs link consistency after README cleanup
    owner: infra-qa
    status: done
    inputs:
      - updated README links
      - docs directory file set
    done_criteria:
      - every README docs link target exists in repository
      - no docs files are modified by this cleanup
      - verification results are recorded in completion report
    handoff_to: project-manager

  - id: T-109
    title: Integrate additional-fix results and provide report
    owner: project-manager
    status: done
    inputs:
      - outputs from T-106 to T-108
      - user requirement to fully overwrite README with minimal sections only
    done_criteria:
      - changed files are listed
      - verification evidence is included
      - residual risks are stated if any
    handoff_to: customer

  - id: T-110
    title: Register final README truncation fix request in queue
    owner: project-manager
    status: done
    inputs:
      - user final requirement (keep only concise top structure and remove all trailing legacy content)
      - current README state with duplicated old detailed section
    done_criteria:
      - final fix request is added as sequential task entries
      - each new entry has id/title/owner/status/inputs/done_criteria
      - queue remains sequential with a single active task at a time during execution
    handoff_to: documentation

  - id: T-111
    title: Truncate README after supplement references and remove legacy block
    owner: documentation
    status: done
    inputs:
      - README current content
      - requirement to keep only title/overview/docs links/quickstart/supplement
    done_criteria:
      - README keeps only the concise top structure
      - all lines after the two supplement bullets are removed
      - old sections such as PostgreSQL setup are absent
    handoff_to: infra-qa

  - id: T-112
    title: Verify README head section and legacy section removal
    owner: infra-qa
    status: done
    inputs:
      - updated README content
      - check scope: first 120 lines
    done_criteria:
      - first 120 lines are reviewed
      - no legacy section headers remain (e.g., PostgreSQLを設定する)
      - verification result is ready for final integration report
    handoff_to: project-manager

  - id: T-113
    title: Integrate final README fix completion report
    owner: project-manager
    status: done
    inputs:
      - outputs from T-110 to T-112
      - user instruction not to stop in blocked state
    done_criteria:
      - queue update, active/completed/blocker status are reported
      - README verification evidence is included
      - final response confirms completion without blocked status
    handoff_to: customer

  - id: T-114
    title: Register README detail-migration request and define sequential work items
    owner: project-manager
    status: done
    inputs:
      - user request: move remaining README detailed text into docs and keep README minimal
      - current README and docs structure
    done_criteria:
      - new tasks are added with id/title/owner/status/inputs/done_criteria
      - task sequence is explicit and traceable
      - one-task-at-a-time execution policy is preserved
    handoff_to: documentation

  - id: T-115
    title: Split README detailed content into maintainable docs documents
    owner: documentation
    status: done
    inputs:
      - existing README detailed sections (references and operational notes)
      - existing docs technical documents
    done_criteria:
      - README keeps only overview/docs navigation/minimal quickstart
      - detailed content is migrated to newly added docs markdown files
      - migrated docs are structured by purpose for future maintenance
    handoff_to: documentation

  - id: T-116
    title: Update docs index and organize cross-reference relationships
    owner: documentation
    status: done
    inputs:
      - docs/README.md current table of contents
      - newly added docs files
    done_criteria:
      - docs/README.md includes links to new docs
      - overlap with existing architecture/backend/frontend/ml/infrastructure docs is minimized
      - role boundaries between core docs and supplemental docs are documented
    handoff_to: infra-qa

  - id: T-117
    title: Verify README-docs link consistency and migration completeness
    owner: infra-qa
    status: done
    inputs:
      - updated README.md
      - updated docs/README.md and newly created docs files
    done_criteria:
      - README and docs links resolve to existing files
      - migrated destination files exist
      - README has no residual detailed reference blocks
    handoff_to: project-manager

  - id: T-118
    title: Integrate migration results and report deliverables
    owner: project-manager
    status: done
    inputs:
      - outputs from T-114 to T-117
      - user deliverable format requirements
    done_criteria:
      - changed files are listed
      - new docs and their roles are reported
      - verification results and residual risks are reported
    handoff_to: customer
