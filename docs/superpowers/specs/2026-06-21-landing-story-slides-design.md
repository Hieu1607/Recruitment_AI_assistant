# Landing Story Slides Design

Date: 2026-06-21
Status: Approved for implementation
Scope: Replace the three landing page showcase images with lightweight JavaScript-driven story slides that explain the product flow more clearly

## Goal

Turn the three large landing page image areas into self-running product stories so a first-time visitor understands:

- resumes move through an AI-assisted intake flow
- scoring is an interactive, explainable ranking workflow
- the assistant can answer recruiter questions against the candidate pool

The result should feel simpler and cleaner than the current screenshots while still looking like product UI rather than decorative illustration.

## User Decisions Captured

- The largest showcase at the top uses a timeline-style process story.
- The two feature showcases below use UI walkthrough stories.
- Motion should use gentle zoom in and zoom out effects.
- The behavior should stay simple enough for users to understand the message quickly.

## Existing Context

The current landing page lives in [frontend/src/routes/landing.tsx](C:/Users/Admin/Desktop/Recruitment_AI_assistant/frontend/src/routes/landing.tsx). It currently renders:

- one large browser-frame screenshot near the top
- one square screenshot for `Scoring Engine`
- one square screenshot for `AI Assistant`

There is existing Playwright coverage in [frontend/tests/e2e/landing-gallery.spec.ts](C:/Users/Admin/Desktop/Recruitment_AI_assistant/frontend/tests/e2e/landing-gallery.spec.ts), but it validates the old static image implementation and must be updated.

## Experience Principles

- Show one core idea per step.
- Use UI state changes, not dense text blocks, to explain value.
- Keep the motion subtle and directional.
- Make each scene readable even if the user only glances at it for a second.
- Keep the overall visual language aligned with the current landing page tone.

## Story Structure

### Hero Showcase

This becomes a three-step timeline loop inside the existing browser frame:

1. `Upload CVs`
2. `AI parses profiles`
3. `Ranked shortlist`

The active step should be visible through:

- a simple progress rail or step pills
- a focused content panel for the current state
- one zoomed highlight region that eases in and returns softly

### Scoring Showcase

This becomes a UI walkthrough loop with three emphasis states:

1. weighting the evaluation dimensions
2. scores updating across candidates
3. rationale or explanation panel surfacing the decision

The message is that recruiters can tune scoring and immediately understand the outcome.

### Assistant Showcase

This becomes a UI walkthrough loop with three emphasis states:

1. a recruiter asks a natural-language question
2. filters or match criteria become active
3. the best candidates and answer summary appear

The message is that the assistant queries recruiting data, not a generic chatbot.

## Motion Language

Use restrained DOM-based transitions:

- opacity transitions between scene states
- slight translate for step changes
- a gentle scale transform for the focus area
- no large sweeps, flashing, or exaggerated parallax

Reduced-motion users should see coherent static states without looping zoom behavior.

## Implementation Shape

Keep the work inside the landing route unless a helper is clearly necessary. The feature is small enough that it should not introduce a large new subsystem.

Recommended structure:

- shared story data arrays for the three sections
- a small React timing hook or inline `useEffect` interval logic
- presentational helpers for repeated story chrome if that keeps `landing.tsx` readable
- scoped CSS classes in the route for scene styling and zoom behavior

Do not add a third-party carousel or animation dependency.

## Accessibility And Performance

- The slides are explanatory but non-critical, so failure should degrade to a readable static first or final state.
- Motion should pause cleanly when JavaScript is unavailable or reduced motion is requested.
- The layout must remain readable on desktop and mobile.
- The story text should exist in the DOM so Playwright and assistive tools can observe it.

## Testing Strategy

Update the existing landing Playwright test to verify:

- the three story areas render
- expected story labels or steps are visible
- legacy screenshot alt-text is gone
- placeholder bracketed screenshot copy does not return

Typecheck and build should still pass after the route changes.

## Non-Goals

- adding manual carousel controls
- introducing real backend data into the landing stories
- redesigning the landing page copy outside the three showcase areas
- adding a global animation framework

## Recommendation

Implement one lightweight React-based autoplay story system in the landing route, then apply it to:

- a timeline-style hero process story
- a scoring walkthrough card
- an assistant walkthrough card

This keeps the change focused, easier to test, and visually cleaner than continuing to crop screenshots.
