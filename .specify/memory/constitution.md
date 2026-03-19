<!--
Sync Impact Report
- Version change: N/A -> 1.0.0
- Modified principles:
	- Principle slot 1 -> I. Clarity Over Cleverness
	- Principle slot 2 -> II. Small, Focused Units
	- Principle slot 3 -> III. Behavior-First Testing
	- Principle slot 4 -> IV. Minimal Surface Area
	- Principle slot 5 -> V. Refactor Before Extend
- Added sections:
	- Engineering Standards
	- Development Workflow & Review
- Removed sections:
	- None
- Templates requiring updates:
	- ✅ updated: .specify/templates/plan-template.md
	- ✅ updated: .specify/templates/spec-template.md
	- ✅ updated: .specify/templates/tasks-template.md
	- ⚠ pending: .specify/templates/commands/*.md (directory not present)
- Deferred follow-up TODOs:
	- None.
-->

# Recruitment AI Assistant Constitution

## Core Principles

### I. Clarity Over Cleverness
All production code MUST optimize for readability over novelty. Code MUST use explicit,
domain-meaningful names and straightforward control flow so a new contributor can understand
intent within one reading. Dense abstractions, trick patterns, or implicit behavior are
prohibited unless a written justification demonstrates a measurable benefit.

Rationale: Clear code reduces onboarding cost, review risk, and defect escape rates.

### II. Small, Focused Units
Modules, classes, and functions MUST have a single clear responsibility. New code MUST prefer
small composable units over monolithic blocks, and each unit SHOULD remain easy to test in
isolation. Pull requests that combine unrelated concerns MUST be split before merge.

Rationale: KISS is enforced by constraining scope and preventing hidden coupling.

### III. Behavior-First Testing
Every behavior change MUST include automated tests that describe expected outcomes and failure
conditions. Tests MUST be understandable specifications, not implementation snapshots.
Complex logic without tests is non-compliant and cannot be merged.

Rationale: Readable, behavior-focused tests preserve correctness while enabling safe refactoring.

### IV. Minimal Surface Area
Public APIs, configuration knobs, and dependencies MUST be kept minimal. New dependencies MUST
be justified with a brief alternatives review, and wrappers/adapters SHOULD shield the codebase
from dependency churn. Backward-incompatible API changes MUST be explicitly versioned and
documented.

Rationale: Smaller surfaces are easier to reason about, maintain, and evolve safely.

### V. Refactor Before Extend
When code is hard to understand, teams MUST simplify existing structure before adding features.
Feature delivery MUST include targeted cleanup where readability or duplication would otherwise
increase. New abstractions MUST be introduced only after repeated concrete use cases are present.

Rationale: Continuous simplification prevents complexity debt from compounding.

## Engineering Standards

- Code MUST pass formatter, linter, and test checks in CI before merge.
- Functions SHOULD remain short and intention-revealing; if flow requires extensive comments to
	explain, the implementation MUST be restructured.
- Architectural decision records (ADRs) MUST be added for major structural choices and for any
	approved exception to these principles.
- Logging and error messages MUST be actionable and plain-language to aid rapid diagnosis.

## Development Workflow & Review

- Planning artifacts MUST include a constitution compliance check before implementation begins.
- Pull requests MUST document: scope, tests, readability impact, dependency impact, and any
	deliberate complexity trade-offs.
- Reviewers MUST block merges that violate clarity, single-responsibility, or unnecessary
	abstraction constraints.
- Post-merge retrospectives SHOULD capture recurring complexity patterns and the simplifications
	applied.

## Governance

This constitution supersedes conflicting local conventions for design, implementation, and
review.

Amendment procedure:
1. Propose changes in a pull request with rationale, migration impact, and version bump type.
2. Obtain approval from at least one maintainer and one active contributor.
3. Update dependent templates and guidance artifacts in the same change set.

Versioning policy:
- MAJOR: Removes or materially redefines a principle or governance requirement.
- MINOR: Adds a new principle/section or significantly expands mandatory guidance.
- PATCH: Clarifications, wording improvements, and non-semantic refinements.

Compliance review expectations:
- Every plan, specification, and task list MUST reference and satisfy constitution gates.
- Every pull request review MUST explicitly verify constitution compliance.
- Exceptions MUST include a documented expiry or follow-up remediation task.

**Version**: 1.0.0 | **Ratified**: 2026-03-17 | **Last Amended**: 2026-03-17
