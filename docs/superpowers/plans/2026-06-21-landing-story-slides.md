# Landing Story Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the landing page's three static showcase images with autoplay story slides that explain overview, scoring, and assistant workflows.

**Architecture:** Keep the feature local to `frontend/src/routes/landing.tsx` with small shared data structures and lightweight React timing logic. Update the existing Playwright landing test first so implementation is driven by observable story content instead of screenshot markup.

**Tech Stack:** React 18, TypeScript, React Router 7, Tailwind utilities, inline route-scoped CSS classes, Playwright, Vite typecheck/build.

---

## File Structure

### Files to modify

- `frontend/src/routes/landing.tsx`
  - Replace the image-based showcases with timeline and walkthrough slide compositions plus autoplay state logic.
- `frontend/tests/e2e/landing-gallery.spec.ts`
  - Assert story-slide content and remove assumptions about screenshot `img` elements.

### Files to create

- `docs/superpowers/specs/2026-06-21-landing-story-slides-design.md`
  - Approved design reference for the landing slide behavior.
- `docs/superpowers/plans/2026-06-21-landing-story-slides.md`
  - This implementation plan.

## Task 1: Replace Screenshot-Based Test Expectations

**Files:**
- Modify: `frontend/tests/e2e/landing-gallery.spec.ts`
- Test: `frontend/tests/e2e/landing-gallery.spec.ts`

- [ ] **Step 1: Write a failing Playwright test that expects story content instead of images**

```ts
import { expect, test } from "@playwright/test";

test("landing page renders autoplay story slides for overview, scoring, and assistant", async ({
  page,
  baseURL,
}) => {
  await page.goto(`${baseURL}/`);

  await expect(page.getByTestId("landing-story-overview")).toBeVisible();
  await expect(page.getByTestId("landing-story-overview")).toContainText("Upload CVs");
  await expect(page.getByTestId("landing-story-scoring")).toContainText("Adjust score weights");
  await expect(page.getByTestId("landing-story-assistant")).toContainText("Ask the candidate pool");

  await expect(page.getByAltText("EasyHR dashboard overview")).toHaveCount(0);
  await expect(page.getByAltText("EasyHR scoring interface")).toHaveCount(0);
  await expect(page.getByAltText("EasyHR AI assistant interface")).toHaveCount(0);
  await expect(page.getByText("[Scoring Interface Screenshot]")).toHaveCount(0);
  await expect(page.getByText("[AI Chat Interface Screenshot]")).toHaveCount(0);
});
```

- [ ] **Step 2: Run the landing spec to verify it fails before implementation**

Run:

```bash
cd frontend
npx playwright test tests/e2e/landing-gallery.spec.ts
```

Expected:

```text
FAIL tests/e2e/landing-gallery.spec.ts
  Locator: getByTestId('landing-story-overview')
```

## Task 2: Implement Autoplay Story Slides In The Landing Route

**Files:**
- Modify: `frontend/src/routes/landing.tsx`
- Test: `frontend/tests/e2e/landing-gallery.spec.ts`

- [ ] **Step 1: Add shared story definitions and autoplay timing state**

```tsx
const overviewSteps = [
  { id: "upload", title: "Upload CVs", detail: "Drop resumes into one hiring workspace." },
  { id: "parse", title: "AI parses profiles", detail: "Extract skills, years, and role signals fast." },
  { id: "shortlist", title: "Ranked shortlist", detail: "Surface the strongest matches in minutes." },
] as const;

const scoringSteps = [
  { id: "weights", title: "Adjust score weights", detail: "Balance technical, domain, and communication fit." },
  { id: "scores", title: "See rankings update", detail: "Scores move instantly as priorities change." },
  { id: "reasons", title: "Read the rationale", detail: "Explain why top candidates stand out." },
] as const;
```

```tsx
function useAutoplaySteps(length: number, delayMs: number) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setIndex((value) => (value + 1) % length);
    }, delayMs);
    return () => window.clearTimeout(timeoutId);
  }, [delayMs, index, length]);

  return index;
}
```

- [ ] **Step 2: Replace the hero screenshot with a timeline-style story slide**

```tsx
<div data-testid="landing-story-overview" className="landing-story landing-story--hero">
  <div className="landing-story__steps">
    {overviewSteps.map((step, index) => (
      <div key={step.id} data-active={index === overviewIndex}>
        <span>{index + 1}</span>
        <div>
          <p>{step.title}</p>
          <p>{step.detail}</p>
        </div>
      </div>
    ))}
  </div>
</div>
```

- [ ] **Step 3: Replace the scoring and assistant screenshots with walkthrough cards**

```tsx
<div data-testid="landing-story-scoring" className="landing-walkthrough">
  <h3>Adjust score weights</h3>
  <p>Balance technical, domain, and communication fit.</p>
</div>

<div data-testid="landing-story-assistant" className="landing-walkthrough">
  <h3>Ask the candidate pool</h3>
  <p>Query your shortlist in natural language and get matched candidates fast.</p>
</div>
```

- [ ] **Step 4: Add zoom/focus transitions and reduced-motion guard**

```tsx
const prefersReducedMotion = useMemo(() => {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}, []);
```

```tsx
const overviewIndex = prefersReducedMotion ? overviewSteps.length - 1 : useAutoplaySteps(overviewSteps.length, 2600);
```

```tsx
className={cn("landing-story__focus", active && "landing-story__focus--active")}
```

- [ ] **Step 5: Re-run the landing spec and verify it passes**

Run:

```bash
cd frontend
npx playwright test tests/e2e/landing-gallery.spec.ts
```

Expected:

```text
PASS tests/e2e/landing-gallery.spec.ts
  ✓ landing page renders autoplay story slides for overview, scoring, and assistant
```

## Task 3: Final Verification And Commit

**Files:**
- Modify: `frontend/src/routes/landing.tsx`
- Modify: `frontend/tests/e2e/landing-gallery.spec.ts`
- Create: `docs/superpowers/specs/2026-06-21-landing-story-slides-design.md`
- Create: `docs/superpowers/plans/2026-06-21-landing-story-slides.md`

- [ ] **Step 1: Run typecheck and build after the landing route changes**

Run:

```bash
cd frontend
npm run typecheck
npm run build
```

Expected:

```text
typecheck: PASS
build: PASS
```

- [ ] **Step 2: Inspect the final diff to keep the scope on landing-page story slides**

Run:

```bash
git diff -- frontend/src/routes/landing.tsx frontend/tests/e2e/landing-gallery.spec.ts docs/superpowers/specs/2026-06-21-landing-story-slides-design.md docs/superpowers/plans/2026-06-21-landing-story-slides.md
```

Expected:

```text
Only the landing route, landing E2E test, and the new spec/plan files are changed for this feature.
```

- [ ] **Step 3: Commit the finished feature**

```bash
git add frontend/src/routes/landing.tsx frontend/tests/e2e/landing-gallery.spec.ts docs/superpowers/specs/2026-06-21-landing-story-slides-design.md docs/superpowers/plans/2026-06-21-landing-story-slides.md
git commit -m "feat: turn landing screenshots into story slides"
```

## Self-Review

### Spec coverage

- hero timeline story: covered by Task 2
- scoring and assistant walkthrough stories: covered by Task 2
- light motion and reduced-motion fallback: covered by Task 2
- landing test migration away from screenshots: covered by Task 1
- verification and scoped commit: covered by Task 3

### Placeholder scan

- no `TODO`, `TBD`, or implied follow-up work remains
- each task includes concrete file paths and commands
- testing commands are explicit

### Type consistency

- `landing-story-*` test ids are defined once in the plan and reused consistently
- autoplay state is introduced before any slide markup depends on it
- the same route file owns both timing logic and story presentation
