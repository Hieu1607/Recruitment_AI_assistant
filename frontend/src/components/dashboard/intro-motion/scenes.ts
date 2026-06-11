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
