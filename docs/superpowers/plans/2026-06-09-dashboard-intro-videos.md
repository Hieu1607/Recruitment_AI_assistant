# Dashboard Intro Videos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dashboard's first-run placeholder area with three recruiter-focused HTML/CSS/JS intro motion cards for `Workspace`, `Scoring`, and `AI Chat`.

**Architecture:** Build a small intro-motion subsystem under `frontend/src/components/dashboard/intro-motion/` with a shared shell, deterministic scene data, and a lightweight DOM-driven timeline hook. Keep the dashboard route responsible only for layout and empty-state gating; keep animation sequencing, scene markup, and fallback behavior in focused files that can be tested and iterated independently.

**Tech Stack:** React 18, TypeScript, React Router 7, Tailwind v4 utilities plus global CSS keyframes, Playwright E2E, Vite build/typecheck, ESLint.

---

## File Structure

### Files to create

- `frontend/src/components/dashboard/DashboardIntroGallery.tsx`
  - First-run dashboard section that lays out the three cards and optional static labels.
- `frontend/src/components/dashboard/intro-motion/IntroMotionCard.tsx`
  - Shared container, viewport observer, reduced-motion handling, and scene-state data attributes.
- `frontend/src/components/dashboard/intro-motion/useIntroMotionTimeline.ts`
  - Lightweight timing hook that advances scene index and loop state without layout thrashing.
- `frontend/src/components/dashboard/intro-motion/useReducedMotion.ts`
  - Media-query hook for `prefers-reduced-motion`.
- `frontend/src/components/dashboard/intro-motion/scenes.ts`
  - Shared types, deterministic scene definitions, timing constants, and mock content.
- `frontend/src/components/dashboard/intro-motion/WorkspaceIntroScene.tsx`
  - Workspace overview micro-story markup.
- `frontend/src/components/dashboard/intro-motion/ScoringIntroScene.tsx`
  - Scoring micro-story markup.
- `frontend/src/components/dashboard/intro-motion/ChatIntroScene.tsx`
  - AI chat micro-story markup.
- `frontend/src/components/dashboard/intro-motion/index.ts`
  - Re-exports for the dashboard route.
- `frontend/tests/e2e/dashboard-intro-videos.spec.ts`
  - End-to-end coverage for first-run rendering, reduced-motion fallback, and loop-state observability.

### Files to modify

- `frontend/src/routes/dashboard.tsx`
  - Replace the current first-run placeholder area with the intro gallery while preserving CTA and checklist behavior.
- `frontend/src/styles/globals.css`
  - Add scoped `.intro-motion-*` classes, keyframes, and reduced-motion fallback styles.
- `frontend/tests/e2e/helpers.ts`
  - Export an empty-workspace seeding helper so Playwright can hit the dashboard first-run state without uploaded resumes or active downstream activity.

### Files to verify during implementation

- `frontend/package.json`
  - Confirm existing commands are still sufficient: `lint`, `typecheck`, `build`, and `test:e2e`.
- `frontend/playwright.config.ts`
  - Reuse current Playwright setup; do not add a second test runner for this feature.

## Task 1: Add First-Run Dashboard Test Coverage

**Files:**
- Modify: `frontend/tests/e2e/helpers.ts`
- Create: `frontend/tests/e2e/dashboard-intro-videos.spec.ts`
- Test: `frontend/tests/e2e/dashboard-intro-videos.spec.ts`

- [ ] **Step 1: Write the failing Playwright test for the new first-run intro gallery**

```ts
import { expect, test } from "@playwright/test";

import { authenticatePage, seedEmptyWorkspace } from "./helpers";

test("first-run dashboard shows three intro motion cards for an empty workspace", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedEmptyWorkspace(request, "Intro Motion Empty State");
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/`);

  await expect(page.getByText("Workspace Ready")).toBeVisible();
  await expect(page.getByTestId("dashboard-intro-gallery")).toBeVisible();
  await expect(page.getByTestId("intro-card-workspace")).toBeVisible();
  await expect(page.getByTestId("intro-card-scoring")).toBeVisible();
  await expect(page.getByTestId("intro-card-chat")).toBeVisible();
});

test("dashboard intro cards fall back to non-looping reduced-motion state", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedEmptyWorkspace(request, "Intro Motion Reduced");
  await authenticatePage(page, setup);
  await page.emulateMedia({ reducedMotion: "reduce" });

  await page.goto(`${baseURL}/`);

  const scoringCard = page.getByTestId("intro-card-scoring");
  await expect(scoringCard).toHaveAttribute("data-motion-mode", "reduced");
  await expect(scoringCard).toHaveAttribute("data-scene-state", "final");
});
```

- [ ] **Step 2: Run the new spec to verify it fails before implementation**

Run:

```bash
cd frontend
npx playwright test tests/e2e/dashboard-intro-videos.spec.ts
```

Expected:

```text
FAIL tests/e2e/dashboard-intro-videos.spec.ts
  Error: expect(locator).toBeVisible()
  Locator: getByTestId('dashboard-intro-gallery')
```

- [ ] **Step 3: Export an empty-workspace helper instead of forcing candidate uploads**

```ts
async function createAccountAndJob(request: APIRequestContext, title: string): Promise<AuthSetup> {
  // existing implementation stays the same
}

export async function seedEmptyWorkspace(
  request: APIRequestContext,
  title: string,
): Promise<AuthSetup> {
  return createAccountAndJob(request, title);
}

export async function seedWorkspace(
  request: APIRequestContext,
  title: string,
  candidates: Array<{ fullName: string; email: string; lines: string[] }>,
): Promise<AuthSetup> {
  const setup = await createAccountAndJob(request, title);
  // existing upload + polling logic stays here
  return setup;
}
```

- [ ] **Step 4: Re-run the failing spec to confirm the helper works and the failure is now about missing UI**

Run:

```bash
cd frontend
npx playwright test tests/e2e/dashboard-intro-videos.spec.ts --grep "first-run dashboard"
```

Expected:

```text
FAIL tests/e2e/dashboard-intro-videos.spec.ts
  Expected intro gallery test ids are missing, but navigation reaches the dashboard without seed helper errors.
```

- [ ] **Step 5: Commit the test scaffold**

```bash
git add frontend/tests/e2e/helpers.ts frontend/tests/e2e/dashboard-intro-videos.spec.ts
git commit -m "test: add dashboard intro gallery coverage"
```

## Task 2: Build Shared Intro Motion Infrastructure

**Files:**
- Create: `frontend/src/components/dashboard/intro-motion/useReducedMotion.ts`
- Create: `frontend/src/components/dashboard/intro-motion/useIntroMotionTimeline.ts`
- Create: `frontend/src/components/dashboard/intro-motion/scenes.ts`
- Create: `frontend/src/components/dashboard/intro-motion/IntroMotionCard.tsx`
- Create: `frontend/src/components/dashboard/intro-motion/index.ts`
- Modify: `frontend/src/styles/globals.css`
- Test: `frontend/tests/e2e/dashboard-intro-videos.spec.ts`

- [ ] **Step 1: Define the shared types, timelines, and mock scene content**

```ts
export type IntroCardKind = "workspace" | "scoring" | "chat";

export type IntroSceneDefinition = {
  id: string;
  enterMs: number;
  holdMs: number;
};

export type IntroCardDefinition = {
  kind: IntroCardKind;
  label: string;
  headline: string;
  scenes: IntroSceneDefinition[];
};

export const INTRO_CARD_DEFINITIONS: Record<IntroCardKind, IntroCardDefinition> = {
  workspace: {
    kind: "workspace",
    label: "Workspace overview",
    headline: "One recruiting workspace",
    scenes: [
      { id: "frame", enterMs: 1200, holdMs: 1200 },
      { id: "resumes", enterMs: 1600, holdMs: 1200 },
      { id: "job-ready", enterMs: 1600, holdMs: 1200 },
      { id: "handoff", enterMs: 1600, holdMs: 1600 },
    ],
  },
  scoring: {
    kind: "scoring",
    label: "Scoring",
    headline: "Rank candidates by fit",
    scenes: [
      { id: "inbox", enterMs: 1200, holdMs: 1000 },
      { id: "analysis", enterMs: 1800, holdMs: 1000 },
      { id: "reorder", enterMs: 1800, holdMs: 1200 },
      { id: "shortlist", enterMs: 1400, holdMs: 1800 },
    ],
  },
  chat: {
    kind: "chat",
    label: "AI chat",
    headline: "Ask the candidate pool",
    scenes: [
      { id: "idle", enterMs: 1000, holdMs: 1200 },
      { id: "query", enterMs: 1600, holdMs: 1000 },
      { id: "answer", enterMs: 1800, holdMs: 1200 },
      { id: "followups", enterMs: 1400, holdMs: 1800 },
    ],
  },
};
```

- [ ] **Step 2: Implement the reduced-motion and timeline hooks**

```ts
export function useReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setPrefersReducedMotion(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  return prefersReducedMotion;
}

export function useIntroMotionTimeline(
  scenes: IntroSceneDefinition[],
  options: { active: boolean; reducedMotion: boolean },
) {
  const { active, reducedMotion } = options;
  const [sceneIndex, setSceneIndex] = useState(reducedMotion ? scenes.length - 1 : 0);
  const [loopTick, setLoopTick] = useState(0);

  useEffect(() => {
    if (!active || reducedMotion || scenes.length === 0) return;

    let cancelled = false;
    let timeoutId: number | undefined;

    const runScene = (index: number) => {
      if (cancelled) return;
      setSceneIndex(index);
      const scene = scenes[index];
      timeoutId = window.setTimeout(() => {
        if (index === scenes.length - 1) {
          setLoopTick((value) => value + 1);
          runScene(0);
          return;
        }
        runScene(index + 1);
      }, scene.enterMs + scene.holdMs);
    };

    runScene(0);

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [active, reducedMotion, scenes]);

  return {
    sceneIndex,
    sceneId: scenes[sceneIndex]?.id ?? "final",
    motionMode: reducedMotion ? "reduced" : active ? "live" : "idle",
    loopTick,
  };
}
```

- [ ] **Step 3: Implement the shared card shell with viewport gating and test ids**

```tsx
type IntroMotionCardProps = {
  definition: IntroCardDefinition;
  testId: string;
  children: (state: {
    sceneId: string;
    motionMode: "idle" | "live" | "reduced";
    loopTick: number;
  }) => React.ReactNode;
};

export function IntroMotionCard({ definition, testId, children }: IntroMotionCardProps) {
  const reducedMotion = useReducedMotion();
  const [inView, setInView] = useState(false);
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = hostRef.current;
    if (!node || reducedMotion) return;
    const observer = new IntersectionObserver(([entry]) => {
      setInView(entry.isIntersecting);
    }, { threshold: 0.35 });
    observer.observe(node);
    return () => observer.disconnect();
  }, [reducedMotion]);

  const timeline = useIntroMotionTimeline(definition.scenes, {
    active: inView,
    reducedMotion,
  });

  return (
    <div
      ref={hostRef}
      data-testid={testId}
      data-card-kind={definition.kind}
      data-motion-mode={timeline.motionMode}
      data-scene-state={reducedMotion ? "final" : timeline.sceneId}
      className="intro-motion-card"
    >
      <div className="intro-motion-card__chrome">
        <div className="intro-motion-card__label">{definition.label}</div>
        {children(timeline)}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Add scoped motion classes and reduced-motion CSS**

```css
.intro-motion-card {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--hairline-strong);
  border-radius: 28px;
  background:
    radial-gradient(circle at top left, rgba(31, 58, 46, 0.08), transparent 32%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(250, 250, 247, 0.92));
  box-shadow: var(--shadow-lg);
}

.intro-motion-card__chrome {
  min-height: 420px;
  padding: 24px;
}

.intro-motion-zoom {
  transition:
    transform 1400ms var(--ease-out),
    opacity 400ms var(--ease-out),
    filter 600ms var(--ease-out);
  will-change: transform, opacity, filter;
}

[data-motion-mode="reduced"] .intro-motion-zoom {
  transition: none;
  transform: none !important;
  opacity: 1 !important;
  filter: none !important;
}
```

- [ ] **Step 5: Run lint and the focused Playwright spec to verify the shared shell compiles but the scenes are still intentionally incomplete**

Run:

```bash
cd frontend
npm run lint
npx playwright test tests/e2e/dashboard-intro-videos.spec.ts
```

Expected:

```text
lint: PASS
playwright: FAIL because the dashboard route still does not render the intro gallery into the first-run layout.
```

## Task 3: Implement and Integrate the Scoring Card First

**Files:**
- Create: `frontend/src/components/dashboard/intro-motion/ScoringIntroScene.tsx`
- Create: `frontend/src/components/dashboard/DashboardIntroGallery.tsx`
- Modify: `frontend/src/components/dashboard/intro-motion/index.ts`
- Modify: `frontend/src/routes/dashboard.tsx`
- Test: `frontend/tests/e2e/dashboard-intro-videos.spec.ts`

- [ ] **Step 1: Write a more specific failing assertion for the scoring card outcome**

```ts
test("scoring intro card ends on a ranked shortlist frame", async ({ page, request, baseURL }) => {
  const setup = await seedEmptyWorkspace(request, "Intro Motion Scoring");
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/`);

  const scoringCard = page.getByTestId("intro-card-scoring");
  await expect(scoringCard.getByText("Rank candidates by fit")).toBeVisible();
  await expect(scoringCard.getByText("Top matches")).toBeVisible({ timeout: 20_000 });
  await expect(scoringCard.getByText("92")).toBeVisible();
});
```

- [ ] **Step 2: Run the focused test and confirm the missing scoring scene fails**

Run:

```bash
cd frontend
npx playwright test tests/e2e/dashboard-intro-videos.spec.ts --grep "ranked shortlist"
```

Expected:

```text
FAIL tests/e2e/dashboard-intro-videos.spec.ts
  Locator: getByTestId('intro-card-scoring')
  Missing text: "Top matches"
```

- [ ] **Step 3: Implement the scoring scene and the gallery shell**

```tsx
export function ScoringIntroScene({
  sceneId,
}: {
  sceneId: string;
}) {
  const rows = [
    { name: "Avery Chen", score: 92, active: sceneId === "shortlist" },
    { name: "Jordan Lee", score: 88, active: sceneId === "shortlist" || sceneId === "reorder" },
    { name: "Priya Raman", score: 84, active: sceneId === "shortlist" },
  ];

  return (
    <div className="intro-motion-scene intro-motion-zoom" data-scene={sceneId}>
      <div className="intro-motion-header">
        <p className="intro-motion-kicker">Scoring</p>
        <h3 className="intro-motion-title">Rank candidates by fit</h3>
      </div>
      <div className="intro-motion-panel">
        <div className="intro-motion-panel__meta">
          <span>Match analysis</span>
          <span>{sceneId === "analysis" ? "Running…" : "Top matches"}</span>
        </div>
        {rows.map((row, index) => (
          <div
            key={row.name}
            className="intro-motion-score-row"
            data-rank={index + 1}
            data-active={row.active}
          >
            <span>{row.name}</span>
            <span>{row.score}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function DashboardIntroGallery() {
  return (
    <section data-testid="dashboard-intro-gallery" className="grid gap-6 xl:grid-cols-3">
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.workspace} testId="intro-card-workspace">
        {(timeline) => <div data-scene={timeline.sceneId} />}
      </IntroMotionCard>
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.scoring} testId="intro-card-scoring">
        {(timeline) => <ScoringIntroScene sceneId={timeline.sceneId} />}
      </IntroMotionCard>
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.chat} testId="intro-card-chat">
        {(timeline) => <div data-scene={timeline.sceneId} />}
      </IntroMotionCard>
    </section>
  );
}
```

- [ ] **Step 4: Render the new gallery inside the first-run dashboard state without removing CTA or checklist**

```tsx
import { DashboardIntroGallery } from "@/components/dashboard/intro-motion";

function FirstRunDashboardState({
  jobTitle,
  onUpload,
  onAddJobDescription,
}: {
  jobTitle: string;
  onUpload: () => void;
  onAddJobDescription: () => void;
}) {
  return (
    <div className="mx-auto max-w-7xl">
      <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6 sm:p-8">
        <div className="max-w-2xl">
          {/* existing title + copy + CTAs stay here */}
        </div>
        <div className="mt-8">
          <DashboardIntroGallery />
        </div>
      </div>
      <div className="mt-8">
        <OnboardingChecklist
          hasCandidates={false}
          hasJDs={false}
          hasScored={false}
          hasChatted={false}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Re-run the focused spec and make sure the scoring card test passes**

Run:

```bash
cd frontend
npx playwright test tests/e2e/dashboard-intro-videos.spec.ts --grep "ranked shortlist"
```

Expected:

```text
PASS tests/e2e/dashboard-intro-videos.spec.ts
  ✓ scoring intro card ends on a ranked shortlist frame
```

## Task 4: Implement Workspace and Chat Motion Stories

**Files:**
- Create: `frontend/src/components/dashboard/intro-motion/WorkspaceIntroScene.tsx`
- Create: `frontend/src/components/dashboard/intro-motion/ChatIntroScene.tsx`
- Modify: `frontend/src/components/dashboard/DashboardIntroGallery.tsx`
- Modify: `frontend/src/styles/globals.css`
- Test: `frontend/tests/e2e/dashboard-intro-videos.spec.ts`

- [ ] **Step 1: Extend the Playwright spec with assertions that each card communicates its intended value**

```ts
test("workspace and chat cards communicate setup flow and recruiter query value", async ({
  page,
  request,
  baseURL,
}) => {
  const setup = await seedEmptyWorkspace(request, "Intro Motion Full Set");
  await authenticatePage(page, setup);

  await page.goto(`${baseURL}/`);

  await expect(page.getByTestId("intro-card-workspace").getByText("One recruiting workspace")).toBeVisible();
  await expect(page.getByTestId("intro-card-workspace").getByText("Scoring ready")).toBeVisible({ timeout: 20_000 });

  await expect(page.getByTestId("intro-card-chat").getByText("Ask the candidate pool")).toBeVisible();
  await expect(page.getByTestId("intro-card-chat").getByText("Python and FastAPI")).toBeVisible({ timeout: 20_000 });
});
```

- [ ] **Step 2: Run the extended spec and confirm the two unfinished cards fail**

Run:

```bash
cd frontend
npx playwright test tests/e2e/dashboard-intro-videos.spec.ts --grep "workspace and chat cards"
```

Expected:

```text
FAIL tests/e2e/dashboard-intro-videos.spec.ts
  Workspace card missing "Scoring ready"
  Chat card missing "Python and FastAPI"
```

- [ ] **Step 3: Implement the workspace intro scene**

```tsx
export function WorkspaceIntroScene({ sceneId }: { sceneId: string }) {
  return (
    <div className="intro-motion-scene intro-motion-zoom" data-scene={sceneId}>
      <div className="intro-motion-browser">
        <aside className="intro-motion-browser__sidebar">
          <span className="is-active">Candidates</span>
          <span>Scoring</span>
          <span>Chat</span>
        </aside>
        <main className="intro-motion-browser__canvas">
          <div className="intro-motion-chip-row">
            <span className={sceneId !== "frame" ? "is-visible" : ""}>3 resumes parsed</span>
            <span className={sceneId === "job-ready" || sceneId === "handoff" ? "is-visible" : ""}>
              JD attached
            </span>
            <span className={sceneId === "handoff" ? "is-visible" : ""}>Scoring ready</span>
          </div>
        </main>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Implement the chat intro scene and wire both scenes into the gallery**

```tsx
export function ChatIntroScene({ sceneId }: { sceneId: string }) {
  return (
    <div className="intro-motion-scene intro-motion-zoom" data-scene={sceneId}>
      <div className="intro-motion-header">
        <p className="intro-motion-kicker">AI Chat</p>
        <h3 className="intro-motion-title">Ask the candidate pool</h3>
      </div>
      <div className="intro-motion-chat">
        <div className="intro-motion-chat__prompt">
          Top backend candidates with Python and FastAPI?
        </div>
        <div className={sceneId === "answer" || sceneId === "followups" ? "intro-motion-chat__answer is-visible" : "intro-motion-chat__answer"}>
          <p>Avery Chen, Jordan Lee, and Priya Raman match the strongest backend signals.</p>
        </div>
      </div>
    </div>
  );
}

export function DashboardIntroGallery() {
  return (
    <section data-testid="dashboard-intro-gallery" className="grid gap-6 xl:grid-cols-3">
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.workspace} testId="intro-card-workspace">
        {(timeline) => <WorkspaceIntroScene sceneId={timeline.sceneId} />}
      </IntroMotionCard>
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.scoring} testId="intro-card-scoring">
        {(timeline) => <ScoringIntroScene sceneId={timeline.sceneId} />}
      </IntroMotionCard>
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.chat} testId="intro-card-chat">
        {(timeline) => <ChatIntroScene sceneId={timeline.sceneId} />}
      </IntroMotionCard>
    </section>
  );
}
```

- [ ] **Step 5: Re-run the full dashboard intro spec and verify all three cards now pass**

Run:

```bash
cd frontend
npx playwright test tests/e2e/dashboard-intro-videos.spec.ts
```

Expected:

```text
PASS tests/e2e/dashboard-intro-videos.spec.ts
  ✓ first-run dashboard shows three intro motion cards for an empty workspace
  ✓ dashboard intro cards fall back to non-looping reduced-motion state
  ✓ scoring intro card ends on a ranked shortlist frame
  ✓ workspace and chat cards communicate setup flow and recruiter query value
```

## Task 5: Polish Motion, Responsive Layout, and Final Verification

**Files:**
- Modify: `frontend/src/components/dashboard/DashboardIntroGallery.tsx`
- Modify: `frontend/src/routes/dashboard.tsx`
- Modify: `frontend/src/styles/globals.css`
- Test: `frontend/tests/e2e/dashboard-intro-videos.spec.ts`
- Test: `frontend/tests/e2e/workspace-smoke.spec.ts`

- [ ] **Step 1: Refine the gallery layout so the cards stay elegant on desktop and stack cleanly on smaller screens**

```tsx
export function DashboardIntroGallery() {
  return (
    <section
      data-testid="dashboard-intro-gallery"
      className="grid gap-5 md:gap-6 xl:grid-cols-[1.1fr_1fr_1.1fr]"
    >
      {/* existing three IntroMotionCard instances */}
    </section>
  );
}
```

```css
@media (max-width: 1279px) {
  .intro-motion-card__chrome {
    min-height: 360px;
    padding: 20px;
  }
}

@media (max-width: 767px) {
  .intro-motion-card__chrome {
    min-height: 300px;
    padding: 18px;
  }
}
```

- [ ] **Step 2: Tighten the first-run copy so it complements the videos instead of duplicating them**

```tsx
<p className="mt-4 max-w-2xl text-sm leading-7 text-fg-muted sm:text-base">
  Upload resumes, define the role, and let the workspace carry recruiters from setup to ranking
  to fast AI-assisted review.
</p>
```

- [ ] **Step 3: Run full frontend verification including lint, typecheck, build, and the relevant Playwright coverage**

Run:

```bash
cd frontend
npm run lint
npm run typecheck
npm run build
npx playwright test tests/e2e/dashboard-intro-videos.spec.ts tests/e2e/workspace-smoke.spec.ts
```

Expected:

```text
lint: PASS
typecheck: PASS
build: PASS
playwright: PASS
```

- [ ] **Step 4: Inspect the final diff and verify the feature stays scoped to dashboard intro motion**

Run:

```bash
git diff --stat
git diff -- frontend/src/routes/dashboard.tsx frontend/src/components/dashboard frontend/src/styles/globals.css frontend/tests/e2e/dashboard-intro-videos.spec.ts frontend/tests/e2e/helpers.ts
```

Expected:

```text
Only dashboard intro files, shared motion helpers, CSS, and the new Playwright coverage are changed for this feature.
```

- [ ] **Step 5: Commit the finished feature**

```bash
git add frontend/src/routes/dashboard.tsx frontend/src/styles/globals.css frontend/src/components/dashboard frontend/tests/e2e/dashboard-intro-videos.spec.ts frontend/tests/e2e/helpers.ts
git commit -m "feat: add dashboard intro motion cards"
```

## Self-Review

### Spec coverage

- three independent recruiter-focused cards: covered by Tasks 3 and 4
- shared premium motion language and restrained zoom behavior: covered by Tasks 2 and 5
- deterministic mock content instead of live API dependence: covered by Task 2
- reduced motion, viewport gating, and graceful fallback: covered by Tasks 1 and 2
- dashboard integration without rewriting the rest of the page: covered by Tasks 3 and 5

### Placeholder scan

- no `TODO`, `TBD`, or "similar to above" shortcuts remain
- all file paths, commands, and code-entry points are explicit
- every test step includes the exact command to run

### Type consistency

- `IntroCardKind`, `IntroSceneDefinition`, and `IntroMotionCard` props are introduced once in Task 2 and reused consistently in later tasks
- `seedEmptyWorkspace` is introduced in Task 1 and reused consistently in all Playwright steps
- `data-motion-mode` and `data-scene-state` names are defined in Task 2 and reused consistently in the tests
