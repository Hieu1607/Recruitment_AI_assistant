# Landing Storyboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the landing page slideshow-style showcases with one storyboard-led hero flow and two supporting static product snapshots, then verify the final result in the browser with screenshots.

**Architecture:** Keep the work local to `frontend/src/routes/landing.tsx`. Remove or neutralize autoplay-oriented presentation for the landing showcases, introduce static snapshot data for the hero storyboard and the two supporting product cards, then layer in subtle motion hooks for load, reveal, and micro-interactions without bringing back rotating story states.

**Tech Stack:** React 18, TypeScript, React Router 7, Tailwind utilities, Playwright, Vite build pipeline.

---

## File Structure

### Files to modify

- `frontend/src/routes/landing.tsx`
  - Replace `OverviewStory` and animated walkthrough emphasis with static storyboard UI and supporting static product cards.
- `frontend/tests/e2e/landing-gallery.spec.ts`
  - Shift assertions from autoplay story copy to static storyboard stages and static supporting snapshots.
- `docs/superpowers/specs/2026-06-21-landing-story-slides-design.md`
  - Approved design reference for the redesign.
- `docs/superpowers/plans/2026-06-21-landing-story-slides.md`
  - This execution plan.

## Task 1: Update The Landing E2E Spec First

**Files:**
- Modify: `frontend/tests/e2e/landing-gallery.spec.ts`
- Test: `frontend/tests/e2e/landing-gallery.spec.ts`

- [ ] **Step 1: Replace the existing landing assertions with the new static storyboard expectations**

```ts
import { expect, test } from "@playwright/test";

test("landing page renders the storyboard hero and static supporting product snapshots", async ({
  page,
  baseURL,
}) => {
  await page.goto(`${baseURL}/`);

  const overview = page.getByTestId("landing-story-overview");
  const scoring = page.getByTestId("landing-story-scoring");
  const assistant = page.getByTestId("landing-story-assistant");

  await expect(overview).toBeVisible();
  await expect(overview).toContainText("Tạo job");
  await expect(overview).toContainText("Upload CV");
  await expect(overview).toContainText("AI parse CV");
  await expect(overview).toContainText("Shortlist");

  await expect(scoring).toBeVisible();
  await expect(scoring).toContainText("Scoring Engine");
  await expect(scoring).toContainText("Candidate scorecard");

  await expect(assistant).toBeVisible();
  await expect(assistant).toContainText("AI Assistant");
  await expect(assistant).toContainText("Talent search");

  await expect(overview).not.toContainText("Upload CVs");
  await expect(page.getByText("Adjust score weights")).toHaveCount(0);
  await expect(page.getByText("Ask the candidate pool")).toHaveCount(0);
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
  Expected locator to contain text "Tạo job"
```

## Task 2: Rebuild The Hero Showcase As A Static Storyboard

**Files:**
- Modify: `frontend/src/routes/landing.tsx`
- Test: `frontend/tests/e2e/landing-gallery.spec.ts`

- [ ] **Step 1: Remove story timing and autoplay dependencies that only exist for the landing showcases**

```tsx
import { Link } from "react-router";
```

```tsx
type StoryboardFrame = {
  id: string;
  title: string;
  detail: string;
  kicker: string;
  accentClass: string;
};
```

- [ ] **Step 2: Define four static storyboard frames for the hero showcase**

```tsx
const storyboardFrames: StoryboardFrame[] = [
  {
    id: "job",
    title: "Tạo job",
    detail: "Role brief, hiring targets, and must-have signals are defined in one workspace.",
    kicker: "01",
    accentClass: "from-sky-100 via-white to-cyan-50",
  },
  {
    id: "upload",
    title: "Upload CV",
    detail: "Recruiters drop a batch of resumes and watch the queue organize itself instantly.",
    kicker: "02",
    accentClass: "from-amber-100 via-white to-orange-50",
  },
  {
    id: "parse",
    title: "AI parse CV",
    detail: "Skills, years, seniority, and role evidence are extracted into structured profile cards.",
    kicker: "03",
    accentClass: "from-violet-100 via-white to-fuchsia-50",
  },
  {
    id: "shortlist",
    title: "Shortlist",
    detail: "Top candidates surface with scores, rationale, and recruiter-ready next actions.",
    kicker: "04",
    accentClass: "from-emerald-100 via-white to-teal-50",
  },
];
```

- [ ] **Step 3: Replace `OverviewStory` with a static storyboard composition**

```tsx
function OverviewStory() {
  return (
    <BrowserFrame>
      <div
        data-testid="landing-story-overview"
        className="grid gap-6 bg-[radial-gradient(circle_at_top_left,_rgba(29,78,216,0.12),_transparent_32%),linear-gradient(180deg,_#fffdf8,_#f5efe4)] p-5 md:p-8"
      >
        <div className="max-w-2xl">
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-accent-700">
            Product storyboard
          </p>
          <h3 className="mt-3 font-serif text-3xl text-forest-900 md:text-4xl">
            From job setup to a recruiter-ready shortlist.
          </h3>
          <p className="mt-3 text-sm leading-6 text-forest-600 md:text-base">
            Four static snapshots show the actual workflow instead of a slideshow demo.
          </p>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          {storyboardFrames.map((frame) => (
            <StoryboardSnapshot key={frame.id} frame={frame} />
          ))}
        </div>
      </div>
    </BrowserFrame>
  );
}
```

- [ ] **Step 4: Add a dedicated snapshot renderer that draws believable app states per frame**

```tsx
function StoryboardSnapshot({ frame }: { frame: StoryboardFrame }) {
  return (
    <article className="rounded-[28px] border border-sand-200 bg-white/90 p-4 shadow-lg shadow-forest-900/10">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-forest-400">{frame.kicker}</p>
          <h4 className="mt-2 font-serif text-2xl text-forest-900">{frame.title}</h4>
        </div>
        <span className="rounded-full bg-forest-900 px-3 py-1 text-xs font-medium text-white">
          Snapshot
        </span>
      </div>
      <p className="mt-3 text-sm leading-6 text-forest-600">{frame.detail}</p>
      <div className={`mt-4 rounded-[24px] bg-gradient-to-br ${frame.accentClass} p-4`}>
        {/* frame-specific UI blocks live here */}
      </div>
    </article>
  );
}
```

- [ ] **Step 5: Render frame-specific UI content for `Tạo job`, `Upload CV`, `AI parse CV`, and `Shortlist`**

```tsx
function StoryboardPanel({ frameId }: { frameId: StoryboardFrame["id"] }) {
  if (frameId === "job") {
    return (
      <div className="grid gap-3">
        <div className="rounded-2xl bg-white px-4 py-3">
          <p className="text-xs uppercase tracking-[0.16em] text-forest-400">Role setup</p>
          <p className="mt-2 text-sm font-medium text-forest-900">Senior Frontend Engineer</p>
          <p className="mt-1 text-sm text-forest-600">React, TypeScript, design systems, remote-first hiring.</p>
        </div>
      </div>
    );
  }

  if (frameId === "upload") {
    return (
      <div className="grid gap-3">
        {["FrontendLead.pdf", "StaffReact.pdf", "ProductUI.pdf"].map((fileName) => (
          <div key={fileName} className="flex items-center justify-between rounded-2xl bg-white px-4 py-3">
            <span className="text-sm font-medium text-forest-900">{fileName}</span>
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-800">Queued</span>
          </div>
        ))}
      </div>
    );
  }

  if (frameId === "parse") {
    return (
      <div className="grid gap-3 md:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-[24px] bg-white p-4">
          <p className="text-sm font-medium text-forest-900">Avery Chen</p>
          <div className="mt-4 flex flex-wrap gap-2">
            {["React", "TypeScript", "7 years", "Staff+"].map((chip) => (
              <span key={chip} className="rounded-full bg-sand-50 px-3 py-1 text-xs text-forest-700">{chip}</span>
            ))}
          </div>
        </div>
        <div className="space-y-3">
          {[
            ["Signals extracted", "18"],
            ["Seniority confidence", "High"],
            ["Role match", "92%"],
          ].map(([label, value]) => (
            <div key={label} className="rounded-2xl bg-white px-4 py-3">
              <p className="text-xs uppercase tracking-[0.18em] text-forest-400">{label}</p>
              <p className="mt-2 font-serif text-2xl text-forest-900">{value}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      {[
        ["Avery Chen", "94", "Strong architecture depth and hiring loop ownership"],
        ["Noah Martins", "91", "Reliable React platform and mentoring experience"],
        ["Mia Tran", "88", "Product-minded frontend delivery for growth teams"],
      ].map(([name, score, reason], index) => (
        <div
          key={name}
          className={`rounded-[24px] px-4 py-4 ${index === 0 ? "bg-forest-900 text-white" : "bg-white text-forest-900"}`}
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="font-medium">{name}</p>
              <p className={`mt-1 text-sm ${index === 0 ? "text-sand-100" : "text-forest-600"}`}>{reason}</p>
            </div>
            <span className={`rounded-2xl px-3 py-2 text-lg font-semibold ${index === 0 ? "bg-white text-forest-900" : "bg-sand-50 text-forest-900"}`}>{score}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
```

## Task 3: Turn The Two Secondary Sections Into Static Product Proof

**Files:**
- Modify: `frontend/src/routes/landing.tsx`
- Test: `frontend/tests/e2e/landing-gallery.spec.ts`

- [ ] **Step 1: Replace `ShowcaseWalkthrough` with a static shell component**

```tsx
type SupportingSnapshotProps = {
  eyebrow: string;
  title: string;
  description: string;
  testId: string;
  variant: "assistant" | "scoring";
};
```

```tsx
function SupportingSnapshot({
  eyebrow,
  title,
  description,
  testId,
  variant,
}: SupportingSnapshotProps) {
  return (
    <div
      data-testid={testId}
      className="relative aspect-square overflow-hidden rounded-3xl border border-sand-300 bg-[radial-gradient(circle_at_top_right,_rgba(27,55,39,0.08),_transparent_36%),linear-gradient(180deg,_#fffcf4,_#f4efe2)] p-4 md:p-5"
    >
      <div className="rounded-[28px] border border-white/70 bg-white/92 p-4 shadow-xl shadow-forest-900/10">
        <p className="text-xs uppercase tracking-[0.2em] text-forest-400">{eyebrow}</p>
        <h3 className="mt-2 font-serif text-2xl text-forest-900">{title}</h3>
        <p className="mt-3 text-sm leading-6 text-forest-600">{description}</p>
      </div>
      <div className="mt-4 rounded-[26px] border border-sand-200 bg-white/88 p-4 shadow-lg shadow-forest-900/5">
        {variant === "scoring" ? <ScoringSnapshotPanel /> : <AssistantSnapshotPanel />}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add a static scoring panel that reads as a scorecard screen**

```tsx
function ScoringSnapshotPanel() {
  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between rounded-2xl bg-sand-50 px-4 py-3">
        <div>
          <p className="text-sm font-medium text-forest-900">Candidate scorecard</p>
          <p className="text-xs uppercase tracking-[0.16em] text-forest-400">Technical · Communication · Domain</p>
        </div>
        <span className="rounded-full bg-accent-50 px-3 py-1 text-xs font-medium text-accent-700">Live ranking</span>
      </div>
      {[
        ["Avery Chen", "92", "Technical fit 40%"],
        ["Jordan Lee", "88", "Communication 30%"],
        ["Priya Raman", "84", "Domain fit 30%"],
      ].map(([name, score, note]) => (
        <div key={name} className="flex items-center justify-between rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5">
          <div>
            <p className="text-sm font-medium text-forest-900">{name}</p>
            <p className="text-xs text-forest-500">{note}</p>
          </div>
          <span className="rounded-xl bg-forest-900 px-3 py-2 text-sm font-semibold text-white">{score}</span>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Add a static assistant panel that reads as a talent search screen**

```tsx
function AssistantSnapshotPanel() {
  return (
    <div className="grid gap-3">
      <div className="rounded-[20px] bg-sand-50 p-4">
        <p className="text-xs uppercase tracking-[0.16em] text-forest-400">Talent search</p>
        <p className="mt-2 text-sm text-forest-700">Who has React platform depth, 5+ years, and experience mentoring teams?</p>
      </div>
      {[
        ["Avery Chen", "Staff Frontend Engineer"],
        ["Noah Martins", "Senior React Engineer"],
      ].map(([name, role]) => (
        <div key={name} className="flex items-center justify-between rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5">
          <div>
            <p className="text-sm font-medium text-forest-900">{name}</p>
            <p className="text-xs text-forest-500">{role}</p>
          </div>
          <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">Match</span>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Swap the landing route call sites to the new static component**

```tsx
<SupportingSnapshot
  eyebrow="Scoring Engine"
  title="Turn evaluation rules into a readable scorecard."
  description="A static product snapshot shows how recruiters compare candidates and understand rank changes."
  testId="landing-story-scoring"
  variant="scoring"
/>
```

```tsx
<SupportingSnapshot
  eyebrow="AI Assistant"
  title="Search the talent pool in natural language."
  description="The assistant surface should read like a real recruiter query, not another animated stepper."
  testId="landing-story-assistant"
  variant="assistant"
/>
```

## Task 4: Verify The New Storyboard Behavior

**Files:**
- Modify: `frontend/src/routes/landing.tsx`
- Modify: `frontend/tests/e2e/landing-gallery.spec.ts`
- Test: `frontend/tests/e2e/landing-gallery.spec.ts`

- [ ] **Step 1: Run the landing spec again and verify it passes**

Run:

```bash
cd frontend
npx playwright test tests/e2e/landing-gallery.spec.ts
```

Expected:

```text
PASS tests/e2e/landing-gallery.spec.ts
  ✓ landing page renders the storyboard hero and static supporting product snapshots
```

- [ ] **Step 2: Run typecheck**

Run:

```bash
cd frontend
npm run typecheck
```

Expected:

```text
> easyhr-frontend@0.0.1 typecheck
> tsc -b
```

- [ ] **Step 3: Run production build**

Run:

```bash
cd frontend
npm run build
```

Expected:

```text
> easyhr-frontend@0.0.1 build
> tsc -b && vite build
```

## Task 5: Add Light Motion Without Reintroducing Slides

**Files:**
- Modify: `frontend/tests/e2e/landing-gallery.spec.ts`
- Modify: `frontend/src/routes/landing.tsx`

- [ ] **Step 1: Extend the landing spec so motion hooks are required**

```ts
  await expect(page.getByTestId("landing-story-overview")).toHaveAttribute("data-motion", "storyboard-shell");
  await expect(page.getByTestId("landing-story-scoring")).toHaveAttribute("data-motion", "reveal");
  await expect(page.getByTestId("landing-story-assistant")).toHaveAttribute("data-motion", "reveal");
  await expect(page.getByTestId("landing-uploading-badge")).toHaveAttribute("data-motion", "pulse");
  await expect(page.getByTestId("landing-shortlist-top")).toHaveAttribute("data-motion", "hero-highlight");
```

- [ ] **Step 2: Run the landing spec and verify it fails before motion implementation**

Run:

```bash
cd frontend
npx playwright test tests/e2e/landing-gallery.spec.ts
```

Expected:

```text
FAIL tests/e2e/landing-gallery.spec.ts
  Expected locator to have attribute data-motion
```

- [ ] **Step 3: Add route-local motion hooks and reduced-motion handling**

```tsx
import { useEffect, useState } from "react";
```

```tsx
function usePrefersReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setPrefersReducedMotion(mediaQuery.matches);
    update();
    mediaQuery.addEventListener("change", update);
    return () => mediaQuery.removeEventListener("change", update);
  }, []);
  return prefersReducedMotion;
}
```

- [ ] **Step 4: Add scroll-reveal state and animate key landing surfaces**

```tsx
<section
  data-motion={prefersReducedMotion ? "static" : "storyboard-shell"}
  className="..."
>
```

```tsx
<span data-testid="landing-uploading-badge" data-motion={prefersReducedMotion ? "static" : "pulse"}>
  Uploading
</span>
```

```tsx
<div
  data-testid="landing-shortlist-top"
  data-motion={prefersReducedMotion ? "static" : "hero-highlight"}
  className="..."
>
```

- [ ] **Step 5: Add lightweight keyframes and transition classes directly in the route**

Use:

- fade-up for hero load
- reveal-on-scroll for storyboard and supporting cards
- pulse or shimmer for `Uploading`
- slow breathing glow for shortlist winner
- small badge pulse for `Live ranking` and `Match`

- [ ] **Step 6: Re-run the landing spec and ensure it passes**

Run:

```bash
cd frontend
npx playwright test tests/e2e/landing-gallery.spec.ts
```

Expected:

```text
PASS tests/e2e/landing-gallery.spec.ts
```

## Task 6: Browser Review And Visual Evidence

**Files:**
- Modify: `frontend/src/routes/landing.tsx` if browser review reveals visual defects
- Evidence: screenshot artifacts saved under the workspace

- [ ] **Step 1: Start the frontend locally**

Run:

```bash
cd frontend
npm run dev
```

Expected:

```text
VITE v5.x ready
  Local: http://localhost:5173/
```

- [ ] **Step 2: Open the landing page in the browser and inspect the hero storyboard**

Check:

- four storyboard frames are visible and readable
- visual hierarchy clearly favors the hero storyboard
- no progress rail or play or pause controls appear
- the four snapshots read in the chosen flow order

- [ ] **Step 3: Inspect the scoring and assistant static snapshots**

Check:

- both cards still look like product UI, not placeholders
- typography and spacing remain coherent with the landing page
- the supporting cards do not overpower the hero storyboard

- [ ] **Step 4: Capture verification screenshots**

Save:

- one full landing screenshot
- one focused screenshot of the hero storyboard
- one focused screenshot of the scoring card
- one focused screenshot of the assistant card

- [ ] **Step 5: If browser review finds issues, adjust the route and re-run the relevant verification commands before finishing**

## Self-Review

### Spec coverage

- primary storyboard with four static workflow states: covered by Task 2
- secondary scoring and assistant static snapshots: covered by Task 3
- removal of slideshow-era controls and behavior: covered by Tasks 1 and 2
- light load, reveal, and micro-motion without autoplay: covered by Task 5
- browser inspection and screenshot evidence: covered by Task 6

### Placeholder scan

- no `TODO`, `TBD`, or vague implementation steps remain
- each task names exact files and commands
- the browser verification deliverables are explicit

### Type consistency

- `landing-story-overview`, `landing-story-scoring`, and `landing-story-assistant` are the stable test ids used across plan tasks
- hero storyboard data is defined once and consumed by dedicated snapshot components
- the secondary cards use `SupportingSnapshot`, `ScoringSnapshotPanel`, and `AssistantSnapshotPanel` consistently
