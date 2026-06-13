import { api, queryClient, type ScoreResponse } from "@/api";
import { create } from "zustand";

const STORAGE_KEY = "recruit_ai_scoring_runs_v1";
const INTERNAL_SCORING_BATCH_SIZE = 3;

export type ScoringRunStatus = "idle" | "running" | "completed" | "failed";

export interface StartScoringRunInput {
  scoreThreshold: number;
  sectionWeights: Record<string, number>;
  candidateProfileIds?: string[];
  hiddenTextSnapshot?: string;
}

export interface StoredScoringRun {
  status: ScoringRunStatus;
  latestResult: ScoreResponse | null;
  lastError: string | null;
  startedAt: string | null;
  completedAt: string | null;
  hiddenTextSnapshot: string;
}

interface ScoringStoreState {
  runs: Record<string, StoredScoringRun>;
  startRun: (jobId: string, input: StartScoringRunInput) => Promise<ScoreResponse>;
  clearError: (jobId: string) => void;
}

const inflightRuns = new Map<string, Promise<ScoreResponse>>();

function createDefaultRun(): StoredScoringRun {
  return {
    status: "idle",
    latestResult: null,
    lastError: null,
    startedAt: null,
    completedAt: null,
    hiddenTextSnapshot: "",
  };
}

function readStoredRuns(): Record<string, StoredScoringRun> {
  if (typeof window === "undefined") return {};

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, Partial<StoredScoringRun>>;
    return Object.fromEntries(
      Object.entries(parsed).map(([jobId, run]) => [
        jobId,
        {
          ...createDefaultRun(),
          ...run,
          status:
            run.status === "running" ||
            run.status === "completed" ||
            run.status === "failed"
              ? run.status
              : "idle",
        },
      ]),
    );
  } catch {
    return {};
  }
}

function writeStoredRuns(runs: Record<string, StoredScoringRun>) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
}

function updateRun(
  runs: Record<string, StoredScoringRun>,
  jobId: string,
  patch: Partial<StoredScoringRun>,
): Record<string, StoredScoringRun> {
  return {
    ...runs,
    [jobId]: {
      ...(runs[jobId] ?? createDefaultRun()),
      ...patch,
    },
  };
}

export const useScoringStore = create<ScoringStoreState>((set) => ({
  runs: readStoredRuns(),

  clearError: (jobId) =>
    set((state) => {
      const runs = updateRun(state.runs, jobId, { lastError: null });
      writeStoredRuns(runs);
      return { runs };
    }),

  startRun: async (jobId, input) => {
    const existing = inflightRuns.get(jobId);
    if (existing) return existing;

    set((state) => {
      const runs = updateRun(state.runs, jobId, {
        status: "running",
        lastError: null,
        startedAt: new Date().toISOString(),
        completedAt: null,
        hiddenTextSnapshot:
          input.hiddenTextSnapshot ?? state.runs[jobId]?.hiddenTextSnapshot ?? "",
      });
      writeStoredRuns(runs);
      return { runs };
    });

    const request = api.jobs
      .score(jobId, {
        score_threshold: input.scoreThreshold,
        batch_size: INTERNAL_SCORING_BATCH_SIZE,
        section_weights: input.sectionWeights,
        candidate_profile_ids: input.candidateProfileIds,
      })
      .then((result) => {
        set((state) => {
          const runs = updateRun(state.runs, jobId, {
            status: "completed",
            latestResult: result,
            lastError: null,
            completedAt: new Date().toISOString(),
            hiddenTextSnapshot: input.hiddenTextSnapshot ?? state.runs[jobId]?.hiddenTextSnapshot ?? "",
          });
          writeStoredRuns(runs);
          return { runs };
        });

        queryClient.invalidateQueries({ queryKey: ["jobs", jobId, "setup-status"] });
        return result;
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : "Scoring failed";
        set((state) => {
          const runs = updateRun(state.runs, jobId, {
            status: "failed",
            lastError: message,
            completedAt: new Date().toISOString(),
            hiddenTextSnapshot: input.hiddenTextSnapshot ?? state.runs[jobId]?.hiddenTextSnapshot ?? "",
          });
          writeStoredRuns(runs);
          return { runs };
        });
        throw error;
      })
      .finally(() => {
        inflightRuns.delete(jobId);
      });

    inflightRuns.set(jobId, request);
    return request;
  },
}));
