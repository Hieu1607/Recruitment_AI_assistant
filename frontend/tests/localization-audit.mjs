import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const localizerPath = path.join(root, "src", "components", "UiLocalizer.tsx");
const localizerSource = fs.readFileSync(localizerPath, "utf8");

function collectArrayEntries(name) {
  const match = localizerSource.match(
    new RegExp(
      `const\\s+${name}[^=]*=\\s*(?:new Map<string, string>\\()?\\s*\\[([\\s\\S]*?)\\]\\s*\\)?;`,
      "m",
    ),
  );
  if (!match) {
    throw new Error(`Could not find ${name} in UiLocalizer.tsx`);
  }

  const entries = [];
  const pairRegex = /\[\s*"([\s\S]*?)"\s*,\s*"([\s\S]*?)"\s*\]/g;
  let pair;
  while ((pair = pairRegex.exec(match[1]))) {
    entries.push([pair[1], pair[2]]);
  }
  return entries;
}

const exactTranslations = new Map(collectArrayEntries("exactTranslations"));
const inlineTranslations = new Map(collectArrayEntries("inlineTranslations"));

const requiredExactStrings = [
  "Start your free trial",
  "Book a demo",
  "Resume Parsing",
  "Smart Scoring",
  "Candidate Pools",
  "AI Interview Prep",
  "Stop guessing.",
  "Start measuring.",
  "VP of Talent, Nexus",
  "© 2030 RecruitAI Platform Inc. All rights reserved.",
  "Workspace job description",
  "This page follows the selected workspace. Manage the current job description from the active job context instead of treating it as a standalone resource.",
  "Back to workspace",
  "This editor follows the currently selected workspace.",
  "Manage parsed resumes and candidate profiles",
  "Search by name or file…",
  "New Conversation",
  "No conversations yet",
  "Ask anything about your candidates",
  "Who has 5+ years of Python experience?",
  "Find candidates with machine learning skills",
  "Show candidates available in New York",
  "Who has a Master's degree in Computer Science?",
  "Rename collection",
  "Interview Templates",
  "Interview templates",
  "New template",
  "Before you begin",
  "Start interview",
  "Page not found.",
  "The page you're looking for moved, never existed, or you mistyped the URL.",
  "Back to dashboard",
  "Open workspace",
  "No job description yet",
  "Current JD",
  "No outreach messages",
  "No interview invitations",
  "Send interview invitation",
  "Candidate profile unavailable",
  "Overview",
  "Resume PDF",
  "Outreach",
  "Interviews",
  "Open PDF",
  "Uploaded",
  "Processing",
  "Select all",
  "Select row",
  "Rows per page",
  "Previous page",
  "Next page",
  "First-time setup",
  "Set up your first job.",
  "Choose a hiring workspace.",
  "Create the workspace context first. You can refine the full job description after you enter the dashboard.",
  "Every resume, JD, score, and chat session is scoped to one job.",
  "Unable to load workspaces",
  "The jobs list did not load. Retry once the API is available.",
  "Retry",
  "Step 1 of 4",
  "Name the role you are hiring for.",
  "This creates the workspace boundary for resumes, job description authoring, scoring runs, and AI recruiter chat.",
  "Job title",
  "Senior Backend Engineer",
  "Keep it specific. This title will anchor the rest of the hiring workflow.",
  "Responsibilities, required skills, or a short hiring brief.",
  "Core responsibilities",
  "Required skills",
  "Hiring priorities",
  "You can edit both fields later.",
  "Create workspace",
  "What this unlocks",
  "Candidate scope",
  "Uploaded resumes stay tied to the right role and pipeline.",
  "JD authoring",
  "You can expand this into a full job description when ready.",
  "AI recruiter chat",
  "Chat answers stay grounded in the selected job workspace.",
  "Recommended next",
  "Upload resumes to generate candidate profiles.",
  "Add the full job description to define fit.",
  "Run scoring once both are ready.",
  "Use AI chat to interrogate the pipeline.",
  "The dashboard will guide the rest of the setup after this step.",
  "You can switch jobs anytime from the top bar.",
  "Search jobs...",
  "Search jobs",
];

const forbiddenInlineSources = ["question", "turn"];

const missingExactStrings = requiredExactStrings.filter((entry) => !exactTranslations.has(entry));
const forbiddenInlineEntries = forbiddenInlineSources.filter((entry) => inlineTranslations.has(entry));

if (missingExactStrings.length > 0 || forbiddenInlineEntries.length > 0) {
  const problems = [];
  if (missingExactStrings.length > 0) {
    problems.push(
      `Missing exact translations:\n- ${missingExactStrings.join("\n- ")}`,
    );
  }
  if (forbiddenInlineEntries.length > 0) {
    problems.push(
      `Unsafe inline translations still present:\n- ${forbiddenInlineEntries.join("\n- ")}`,
    );
  }
  throw new Error(problems.join("\n\n"));
}

console.log(
  `Localization audit passed with ${requiredExactStrings.length} required exact translations present.`,
);
