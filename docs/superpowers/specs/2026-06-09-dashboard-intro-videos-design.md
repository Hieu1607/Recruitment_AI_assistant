# Dashboard Intro Videos Design

Date: 2026-06-09
Status: Proposed
Scope: Replace the current dashboard placeholder previews with three HTML/CSS/JS motion cards that explain product value at first glance for recruiter users

## Goal

Turn the three large placeholder panels on the dashboard into self-running visual stories that make a first-time recruiter immediately understand:

- this product centralizes recruiting workflow in one workspace
- AI scoring reduces manual CV review
- AI chat lets recruiters query candidate data directly

The experience should feel calm, premium, and product-led rather than loud or ad-like. The reference tone is a lightweight product teaser: minimal text, gentle zooms, clean transitions, and clear value moments.

## User Decisions Captured

- Each panel is its own multi-scene micro-video.
- The audience is recruiter users, not founders or hiring managers.
- The storytelling style is low narration and high visual clarity.
- Motion should feel simple, elegant, and premium, with controlled zoom in and zoom out.

## Existing Context

The current recruiter dashboard already has a first-run state in [frontend/src/routes/dashboard.tsx](C:/Users/Admin/Desktop/Recruitment_AI_assistant/frontend/src/routes/dashboard.tsx). That route currently shows:

- a workspace-ready hero
- a recommended flow panel
- an onboarding checklist

The visual placeholders referenced by the user represent three product areas:

- workspace overview
- scoring
- AI chat

This design assumes those three large panels either already exist in the dashboard layout or will replace the current placeholder area in the first-run dashboard section.

## Experience Principles

The three cards should behave like product vignettes, not marketing trailers.

- Show one idea at a time.
- Reveal value through state changes in the UI, not explanatory paragraphs.
- Use motion to direct the eye, not to entertain by itself.
- End each loop on a clear outcome frame that communicates benefit in under a second.
- Make the loop seam invisible or close to invisible.

The cards should feel cohesive as a set:

- same motion language
- same pacing family
- same visual polish
- different stories and focal areas

## Narrative Structure

Each card uses the same high-level loop model:

1. establish a clean full-frame product view
2. zoom or focus into the operational area
3. show 2 to 4 meaningful state transitions
4. land on an outcome state
5. hold briefly
6. reset softly into the next loop

Target loop length:

- 12 to 16 seconds per card

Target information density:

- enough change to explain the feature
- not so much change that the recruiter has to chase motion

## Card 1: Workspace Overview

### Purpose

Show that the platform is the main operating surface for recruiting activity rather than a disconnected set of tools.

### Story

The recruiter enters a clean workspace, then sees the system organize the flow from incoming materials to actionable next steps.

### Scene Breakdown

1. Wide app-shell view with sidebar and main content visible.
2. Gentle zoom toward the main workspace canvas.
3. Resume-related elements appear or populate.
4. A job description or setup marker becomes active.
5. Status chips or counters update to show that the system is ready for scoring and follow-up.
6. Final hold on a tidy, coordinated workspace state.

### User Understanding To Convey

- this is where the recruiter operates
- data and actions are connected
- the system reduces context switching

## Card 2: Scoring

### Purpose

Show that scoring turns a noisy candidate list into a ranked, decision-ready shortlist.

### Story

The recruiter starts from too many candidates, then the system applies fit logic and surfaces stronger matches.

### Scene Breakdown

1. Candidate list appears dense or unordered.
2. Camera focus shifts into the scoring or fit-analysis area.
3. Progress and evaluation states animate in.
4. Candidate rows reorder, score bars grow, or match labels resolve.
5. Top candidates become visually emphasized.
6. Final hold on a crisp shortlist or ranked outcome.

### User Understanding To Convey

- the recruiter does not need to inspect every CV manually before prioritizing
- the system translates job-fit signals into an ordered review queue
- scoring creates clarity, not just another report

## Card 3: AI Chat

### Purpose

Show that recruiters can query the candidate pool conversationally and get structured answers from real recruiting context.

### Story

The recruiter asks a short natural-language question, then the system answers with grounded candidate results and follow-up paths.

### Scene Breakdown

1. Clean chat view with minimal idle state.
2. A short recruiter query types in naturally.
3. Focus moves toward the response region as the answer starts to build.
4. Candidate cards, evidence snippets, or filters materialize with the response.
5. One or two follow-up suggestions appear subtly.
6. Final hold on a resolved answer that feels immediately useful.

### User Understanding To Convey

- this is not a generic chatbot
- the AI is operating on recruiting data
- the recruiter can ask instead of digging manually through screens

## Visual Language

The reference style is premium SaaS teaser motion.

- restrained zoom in and zoom out
- soft easing, not spring-heavy motion
- layered fades, mask reveals, focus shifts, and card elevation
- clean whitespace and deliberate pauses
- minimal on-screen copy

Avoid:

- aggressive glow effects
- cinematic camera sweeps
- fake terminal theatrics
- particle-heavy backgrounds
- fast-cut transitions that reduce comprehension

## Copy Strategy

Copy should be optional and sparse.

- Use at most one short headline or label per card if needed.
- Avoid explanatory paragraphs inside the animation.
- Prefer UI state labels, chips, score tags, and query text as the source of meaning.

If extra reinforcement is needed, place it outside the animation card in the static dashboard layout rather than inside the moving scenes.

## Implementation Shape

These should be implemented as real HTML/CSS/JS motion compositions, not embedded MP4 or GIF assets.

Recommended structure:

- one reusable `IntroMotionCard` container
- one scene-definition module per card
- shared timeline utilities for sequencing and loop reset
- DOM-based mock or semi-real UI fragments for motion targets

Animation responsibilities:

- CSS handles transform, opacity, blur, mask, and elevation changes
- JavaScript coordinates scene timing, class/state changes, and loop reset
- layout remains responsive and does not depend on fixed video dimensions

## Component Boundaries

Keep the implementation split into clear units:

- dashboard section component that renders the three cards
- shared animation shell component
- `workspace`, `scoring`, and `chat` scene components
- shared timing constants and reduced-motion helpers

Do not place all scene logic inline in `dashboard.tsx`.

## Data And Content Strategy

The intro cards should not depend on live API data for their primary narrative.

Preferred approach:

- use deterministic mock scene data based on realistic recruiter workflows
- keep content localized or localization-ready
- optionally borrow visual tokens from real product UI so the scene feels authentic

This avoids empty or inconsistent motion when a new workspace has no data yet.

## Performance And Accessibility

The cards should be decorative-explanatory, not required for task completion. They still need guardrails.

Required behavior:

- autoplay only when visible in viewport
- pause or idle when offscreen
- no heavy JS layout thrashing
- support `prefers-reduced-motion`
- reduced-motion mode falls back to subtle fades or static poster states

The cards must not delay dashboard interactivity or create obvious scroll jank on modest hardware.

## Error Handling

Because the animations are DOM-driven, failure should degrade gracefully.

- if motion setup fails, render a static final-state composition
- if JavaScript timing does not initialize, the card should remain visually coherent
- if reduced motion is requested, no looping zoom behavior should run

This prevents the intro area from becoming a blank panel.

## Testing Strategy

Implementation should be verified at three levels:

- component rendering and reduced-motion behavior
- scene-state sequencing for each card
- in-browser visual verification for loop smoothness and layout on desktop and mobile

Verification focus:

- cards remain readable at first glance
- motion does not cause layout shift outside the card
- loop restart is visually smooth
- viewport gating works
- static fallback remains acceptable

## Non-Goals

- producing offline rendered marketing videos
- adding voiceover, captions, or long-form tutorial copy
- turning the dashboard intro into a carousel or manual slideshow
- wiring the cards to real recruiter data on first load
- redesigning unrelated dashboard metrics or activity modules as part of this work

## Recommendation

Implement three recruiter-focused intro motion cards on the dashboard as HTML/CSS/JS scene compositions with:

- shared premium motion language
- separate storylines for workspace, scoring, and chat
- deterministic mock content
- graceful reduced-motion and static fallback behavior

Build the `Scoring` card first to lock the motion system and narrative density, then apply the same language to `Workspace Overview` and `AI Chat`.
