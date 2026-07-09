import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const dashboardSource = readFileSync(new URL("../src/routes/dashboard.tsx", import.meta.url), "utf8");
const localizerSource = readFileSync(new URL("../src/components/UiLocalizer.tsx", import.meta.url), "utf8");

const onboardingStepsBlock = dashboardSource.match(/const ONBOARDING_STEPS = \[(?<steps>[\s\S]*?)\n\];/);

assert.ok(onboardingStepsBlock?.groups?.steps, "Could not locate ONBOARDING_STEPS in dashboard.tsx.");

const stepCount = (onboardingStepsBlock.groups.steps.match(/\bid:\s*"/g) ?? []).length;

assert.equal(stepCount, 3, `Expected dashboard onboarding checklist to contain 3 steps, found ${stepCount}.`);
assert.ok(
  !dashboardSource.includes('label: "Run AI scoring"'),
  "Deprecated 'Run AI scoring' checklist item is still present in dashboard.tsx.",
);
assert.ok(
  !dashboardSource.includes('"Run scoring after both are available."'),
  "Recommended dashboard flow still references removed scoring logic.",
);
assert.ok(
  !localizerSource.includes('["Run AI scoring", "Chạy chấm điểm AI"]'),
  "UiLocalizer still contains the removed 'Run AI scoring' copy.",
);
assert.ok(
  !localizerSource.includes('["0 of 4 complete", "Hoàn thành 0/4 bước"]'),
  "UiLocalizer still contains stale 4-step onboarding progress copy.",
);

console.log("Dashboard onboarding checklist regression passed.");
