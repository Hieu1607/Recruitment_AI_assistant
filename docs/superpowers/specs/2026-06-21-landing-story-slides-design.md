# Landing Storyboard Redesign

Date: 2026-06-21
Status: Drafted for user review
Scope: Replace the landing page showcase slideshow treatment with a storyboard-style section that reflects the real EasyHR product flow more directly

## Goal

Redesign the landing page showcase area so it reads less like a rotating demo and more like product evidence. The section should explain how the app actually works through static UI snapshots, with one dominant narrative block and two supporting product views.

The key outcome is that a first-time visitor can scan the page and understand:

- how a recruiter moves from creating a job to a shortlist
- that the product has deeper operational screens beyond the hero flow
- that the landing visuals are based on product workflow states, not decorative placeholders

## User Decisions Captured

- Keep the main focus on the first showcase only.
- The first showcase should tell a single workflow using 3 to 4 static snapshots.
- The chosen primary flow is:
  - `Tạo job`
  - `Upload CV`
  - `AI parse CV`
  - `Shortlist`
- The second and third showcases should become static UI snapshots rather than animated walkthroughs.
- Remove progress rails, autoplay behavior, and play or pause controls from the experience.
- Color usage can be more expressive than the current muted landing demo, as long as the result still feels intentional and product-focused.

## Existing Context

The current implementation lives in [frontend/src/routes/landing.tsx](C:/Users/Admin/Desktop/Recruitment_AI_assistant/frontend/src/routes/landing.tsx).

Today the route contains:

- `OverviewStory`, which presents a browser-framed three-step autoplay sequence
- `ShowcaseWalkthrough`, which is reused for the scoring and assistant sections and also uses autoplay state
- route-local hooks for reduced motion and timed step rotation

The existing design emphasizes motion and step switching. The redesign should shift emphasis toward layout, hierarchy, and believable UI composition.

## Experience Principles

- Lead with one strong visual story instead of three equally active demos.
- Show product workflow states, not abstract marketing cards.
- Keep each snapshot understandable at a glance.
- Use color to separate stages and create hierarchy, not to simulate interaction.
- Treat the second and third showcases as supporting proof, not competing narratives.

## Information Architecture

### Primary Showcase

The first showcase becomes a large storyboard block inside the existing browser-style shell or a refined equivalent shell.

It should contain:

- a short heading and supporting line that frames the workflow
- four static product snapshots arranged as a sequential narrative
- compact labels or captions that anchor each snapshot to one stage

The four frames represent:

1. `Tạo job`
2. `Upload CV`
3. `AI parse CV`
4. `Shortlist`

These frames should feel like the same workflow progressing through real screens or distinct states of the same workspace. They should not behave like tabs, slides, or a carousel.

### Supporting Showcase Two

The scoring section becomes one static snapshot. Its purpose is to show that candidate evaluation and ranking logic exist as a deeper operational screen.

This card should prioritize:

- readable score hierarchy
- visible weighting or evaluation structure
- clear product chrome over explanatory motion

### Supporting Showcase Three

The assistant section becomes one static snapshot. Its purpose is to show a conversational or query-driven recruiter surface without turning it into another mini-story.

This card should prioritize:

- visible prompt or query area
- returned candidate matches or answer summary
- restrained supporting annotations, if any

## Layout Recommendation

Use a `Storyboard trung tam` composition:

- one large hero product card for the four-step narrative
- two smaller secondary product cards below or beside later sections

The primary card should dominate through both size and contrast. The other two should keep the same visual system but with less ornament and fewer explanatory elements.

## Visual Direction

The current beige-and-forest palette can be expanded. The redesign should move toward a more intentional product editorial look, for example:

- a soft neutral page base
- one stronger accent family for the main storyboard
- one or two supporting accent tones for badges, states, and highlighted tables

Recommended visual moves:

- browser or app shell framing that feels cleaner and more premium
- layered cards, tables, badges, and queue states that resemble app surfaces
- subtle gradients or light atmospheric backgrounds behind the snapshot area
- serif headline usage can stay, but the product snapshots themselves should feel sharper and more system-like

Avoid:

- progress bars as narrative devices
- carousel dots, play buttons, pause buttons, or fake transport controls
- heavy animation dependence
- oversimplified wireframes that no longer feel like the EasyHR product

Implementation reminder:

- if any showcase frame is built from placeholder or reconstructed UI imagery during implementation, replace it with real EasyHR interface captures before final sign-off when better product-faithful screens are available

## Component Shape

Keep the work local to the landing route unless extraction meaningfully improves readability.

Expected shape:

- remove or simplify autoplay hooks now that the main showcase is static
- replace `OverviewStory` with a storyboard-focused component
- replace the animated `ShowcaseWalkthrough` usage with static snapshot compositions for scoring and assistant
- keep repeated shell or card chrome in small local helpers if it improves route clarity

The new structure should favor declarative snapshot data and layout blocks over timing logic.

## Accessibility And Responsive Behavior

- The section must remain understandable without animation.
- Snapshot captions and labels should exist as real text in the DOM.
- Mobile should stack the four primary frames in a readable order without requiring hover or micro-interaction.
- Secondary snapshots should remain legible when collapsed into a single column.

## Testing Impact

Update landing page tests so they validate the new static storyboard content rather than autoplay behavior.

Tests should confirm:

- the main storyboard area renders the four workflow stages
- the scoring and assistant showcase areas still render with the expected product labels
- removed slide-era UI affordances do not appear

## Non-Goals

- redesigning the full landing page information architecture
- wiring the landing page to real backend data
- creating a fully interactive product tour
- adding a new animation framework
- preserving the existing autoplay slideshow concept

## Recommendation

Implement a storyboard-led landing showcase where:

- the first section tells the core recruiting flow through four static snapshots
- the second and third sections act as static supporting product proof
- motion is reduced to decorative polish only, if used at all

This keeps the message closer to the real app and aligns the landing page with the user's request to foreground believable interface states over slideshow behavior.

## Motion Addendum

The user later approved adding light motion after the storyboard redesign was complete.

The motion scope is:

- page-load reveal for the hero copy and storyboard shell
- scroll reveal for major showcase blocks
- micro-motion inside key product states only

The motion must not reintroduce any slideshow, autoplay story switching, or rotating content states.

Recommended motion language:

- fade-up and slight rise on first render
- intersection-based reveal for sections as they enter the viewport
- very soft pulse, shimmer, or breathing emphasis for meaningful UI states such as:
  - `Uploading`
  - top shortlist candidate
  - `Live ranking`
  - `Match`

Accessibility constraints:

- respect `prefers-reduced-motion`
- motion should remain subtle enough that the page still reads as a product landing page, not an animated demo
- no essential information may depend on animation timing
