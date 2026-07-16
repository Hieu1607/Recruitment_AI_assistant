import {
  api,
  type CandidateEvaluationResponse,
} from "@/api";
import {
  SECTION_LABELS,
  completedEvaluations,
  formatCriterionLabel,
} from "@/components/scoring-utils";
import { CriteriaRadar } from "@/components/scoring-visuals";
import { Avatar, Badge, Button, EmptyState, Pagination, Skeleton } from "@/components/ui";
import { useSelectedJobId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { toast } from "sonner";

const DEFAULT_THRESHOLD = 50;
const DEFAULT_PAGE_SIZE = 10;
const SECTION_ORDER = ["skills", "experience", "education", "projects", "summary", "languages", "achievements", "certifications", "publications", "other"];
type SortOrder = "natural" | "score_desc" | "score_asc";

function normalizeSectionWeights(sectionWeights: Record<string, number>) {
  const cleaned = Object.entries(sectionWeights).reduce<Record<string, number>>((acc, [key, value]) => {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) {
      acc[key] = numeric;
    }
    return acc;
  }, {});
  const total = Object.values(cleaned).reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    return null;
  }
  return Object.fromEntries(
    Object.entries(cleaned).map(([key, value]) => [key, value / total]),
  );
}

function positiveWeight(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function recalculateEvaluation(
  evaluation: CandidateEvaluationResponse,
  sectionWeights: Record<string, number>,
  scoreThreshold: number,
): CandidateEvaluationResponse {
  const normalizedWeights = normalizeSectionWeights(sectionWeights);
  const componentScores = evaluation.componentScores;
  const sectionCounts = componentScores.reduce<Record<string, number>>((acc, component) => {
    const section = component.section ?? "";
    if (!section) return acc;
    acc[section] = (acc[section] ?? 0) + 1;
    return acc;
  }, {});
  const rawWeightTotals = componentScores.reduce<Record<string, number>>((acc, component) => {
    const section = component.section ?? "";
    const weight = positiveWeight(component.weight);
    if (!section || weight === null) return acc;
    acc[section] = (acc[section] ?? 0) + weight;
    return acc;
  }, {});
  const totalRawWeight = componentScores.reduce((sum, component) => sum + (positiveWeight(component.weight) ?? 0), 0);

  let totalScore = 0;
  const recalculatedScores = componentScores.map((component) => {
    const section = component.section ?? "";
    const rawWeight = positiveWeight(component.weight);
    let effectiveWeight = 0;

    if (normalizedWeights) {
      const sectionWeight = normalizedWeights[section] ?? 0;
      const sectionTotal = rawWeightTotals[section] ?? 0;
      if (rawWeight !== null && sectionTotal > 0) {
        effectiveWeight = sectionWeight * (rawWeight / sectionTotal);
      } else {
        const sectionCount = sectionCounts[section] ?? 0;
        effectiveWeight = sectionWeight > 0 && sectionCount > 0 ? sectionWeight / sectionCount : 0;
      }
    } else if (rawWeight !== null && totalRawWeight > 0) {
      effectiveWeight = rawWeight / totalRawWeight;
    }

    const weightedScore = Number((component.scorePercent * effectiveWeight).toFixed(2));
    totalScore += weightedScore;
    return {
      ...component,
      effectiveWeight,
      weightedScore,
    };
  });

  totalScore = Number(totalScore.toFixed(2));
  return {
    ...evaluation,
    totalScore,
    passedThreshold: totalScore >= scoreThreshold,
    componentScores: recalculatedScores,
  };
}

function scoreVariant(status: CandidateEvaluationResponse["status"]) {
  if (status === "completed") return "success";
  if (status === "failed") return "danger";
  if (status === "outdated") return "warning";
  return "neutral";
}

function candidateDisplayLabel(evaluation: CandidateEvaluationResponse) {
  return (
    evaluation.candidateDisplayName?.trim()
    || evaluation.candidateName?.trim()
    || evaluation.resumeFileName?.trim()
    || evaluation.candidate_profile_id
  );
}

export default function ScoringSetupRoute() {
  const queryClient = useQueryClient();
  const selectedJobId = useSelectedJobId();
  const [draftWeights, setDraftWeights] = useState<Record<string, number>>({});
  const [thresholdDraft, setThresholdDraft] = useState(DEFAULT_THRESHOLD);
  const [prefsDirty, setPrefsDirty] = useState(false);
  const [expandedCandidateId, setExpandedCandidateId] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<SortOrder>("natural");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);

  const { data: evaluations, isLoading } = useQuery({
    queryKey: ["jobs", selectedJobId, "evaluations"],
    queryFn: () => selectedJobId ? api.jobs.evaluations.list(selectedJobId) : Promise.resolve(null),
    enabled: !!selectedJobId,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data && (data.pending_count > 0 || data.running_count > 0) ? 3000 : false;
    },
  });

  useEffect(() => {
    setPage(1);
    setExpandedCandidateId(null);
  }, [selectedJobId]);

  useEffect(() => {
    if (!evaluations || prefsDirty) return;
    setDraftWeights(evaluations.section_weights);
    setThresholdDraft(evaluations.score_threshold);
  }, [evaluations, prefsDirty]);

  const savePreferences = useMutation({
    mutationFn: (body: { section_weights: Record<string, number>; score_threshold: number }) =>
      api.jobs.scoringPreferences.update(selectedJobId!, body),
    onSuccess: () => {
      setPrefsDirty(false);
      void queryClient.invalidateQueries({ queryKey: ["jobs", selectedJobId, "evaluations"] });
      toast.success("Scoring preferences saved");
    },
    onError: () => {
      toast.error("Failed to save scoring preferences");
    },
  });

  const scoreAgain = useMutation({
    mutationFn: () => api.jobs.evaluations.scoreAgain(selectedJobId!),
    onSuccess: () => {
      toast.success("Scoring queued");
      void queryClient.invalidateQueries({ queryKey: ["jobs", selectedJobId, "evaluations"] });
      void queryClient.invalidateQueries({ queryKey: ["jobs", selectedJobId, "setup-status"] });
    },
    onError: () => {
      toast.error("Failed to queue scoring");
    },
  });

  const orderedWeightEntries = useMemo(() => {
    const keys = new Set<string>([
      ...SECTION_ORDER,
      ...Object.keys(draftWeights),
      ...(evaluations ? evaluations.items.flatMap((item) => item.componentScores.map((component) => component.section ?? "")) : []),
    ]);
    return [...keys]
      .filter(Boolean)
      .map((key) => ({
        key,
        label: SECTION_LABELS[key] ?? key.replace(/\b\w/g, (char) => char.toUpperCase()),
        value: draftWeights[key] ?? 0,
      }));
  }, [draftWeights, evaluations]);

  const previewItems = useMemo(() => {
    if (!evaluations) return [];
    if (!prefsDirty && !evaluations.scoring_preferences_applied) {
      return evaluations.items;
    }
    return evaluations.items.map((item) => recalculateEvaluation(item, draftWeights, thresholdDraft));
  }, [draftWeights, evaluations, prefsDirty, thresholdDraft]);

  const displayedItems = useMemo(() => {
    if (sortOrder === "natural") {
      return previewItems;
    }
    return previewItems
      .map((item, index) => ({ item, index }))
      .sort((left, right) => {
        const leftScore = left.item.status === "completed" || left.item.status === "outdated" ? left.item.totalScore : null;
        const rightScore = right.item.status === "completed" || right.item.status === "outdated" ? right.item.totalScore : null;

        if (leftScore === null && rightScore === null) return left.index - right.index;
        if (leftScore === null) return 1;
        if (rightScore === null) return -1;
        if (leftScore === rightScore) return left.index - right.index;

        return sortOrder === "score_desc" ? rightScore - leftScore : leftScore - rightScore;
      })
      .map(({ item }) => item);
  }, [previewItems, sortOrder]);

  const totalPages = Math.max(1, Math.ceil(displayedItems.length / pageSize));
  const paginatedItems = useMemo(
    () => displayedItems.slice((page - 1) * pageSize, page * pageSize),
    [displayedItems, page, pageSize],
  );

  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages);
      setExpandedCandidateId(null);
    }
  }, [page, totalPages]);

  const scoreSummary = useMemo(() => {
    const scored = previewItems.filter((item) => item.status === "completed" || item.status === "outdated");
    const average = scored.length > 0
      ? Number((scored.reduce((sum, item) => sum + item.totalScore, 0) / scored.length).toFixed(2))
      : 0;
    const highest = scored.length > 0 ? Math.max(...scored.map((item) => item.totalScore)) : 0;
    const passed = scored.filter((item) => item.passedThreshold).length;
    const passRate = scored.length > 0 ? (passed / scored.length) * 100 : 0;
    return { average, highest, passRate };
  }, [previewItems]);

  if (!selectedJobId) {
    return (
      <div className="px-8 py-8 min-h-full">
        <EmptyState
          heading="No workspace selected"
          body="Select a job workspace first to review scoring results."
        />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="px-8 py-8 space-y-6">
        <Skeleton className="h-10 w-72" />
        <div className="grid gap-4 md:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-28 rounded-[var(--radius-lg)]" />
          ))}
        </div>
        <Skeleton className="h-[420px] rounded-[var(--radius-lg)]" />
      </div>
    );
  }

  if (!evaluations || evaluations.total_candidates === 0) {
    return (
      <div className="px-8 py-8 min-h-full">
        <EmptyState
          heading="No evaluations yet"
          body="Upload and parse candidates first, then queue scoring for the current job description."
          action={{
            label: scoreAgain.isPending ? "Queueing…" : "Score again",
            onClick: () => void scoreAgain.mutate(),
          }}
        />
      </div>
    );
  }

  const showScoreAgain = evaluations.outdated_count > 0 || evaluations.failed_count > 0 || evaluations.total_candidates === 0;

  return (
    <div className="px-8 py-8 min-h-full space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl">
          <h1 className="font-display text-[2rem] font-medium text-fg">Scoring dashboard</h1>
          <p className="mt-2 text-sm leading-6 text-fg-muted">
            Review evaluation status, adjust job-level weights, and requeue outdated results without reopening the scoring wizard.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="secondary"
            icon={<RefreshCw size={14} strokeWidth={1.75} />}
            loading={scoreAgain.isPending}
            onClick={() => void scoreAgain.mutate()}
          >
            Score again
          </Button>
          <Button
            variant="primary"
            icon={<Save size={14} strokeWidth={1.75} />}
            loading={savePreferences.isPending}
            disabled={!prefsDirty}
            onClick={() => void savePreferences.mutate({ section_weights: draftWeights, score_threshold: thresholdDraft })}
          >
            Save weights
          </Button>
        </div>
      </div>

      {showScoreAgain && (
        <div className="rounded-[var(--radius-lg)] border border-[rgba(184,68,46,0.24)] bg-[rgba(184,68,46,0.06)] px-4 py-3 text-sm text-fg">
          {evaluations.outdated_count > 0
            ? `${evaluations.outdated_count} evaluation${evaluations.outdated_count === 1 ? "" : "s"} are outdated because the JD scoring input changed.`
            : "Some evaluations are missing or failed. Queue scoring again to refresh them."}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-5">
        <SummaryCard label="Candidates" value={String(evaluations.total_candidates)} />
        <SummaryCard label="Completed" value={String(evaluations.completed_count)} />
        <SummaryCard label="Average score" value={scoreSummary.average.toFixed(1)} />
        <SummaryCard label="Highest score" value={scoreSummary.highest.toFixed(1)} />
        <SummaryCard label="Pass rate" value={`${scoreSummary.passRate.toFixed(0)}%`} />
      </div>

      <ScoringInsights items={previewItems} />

      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
        <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="font-display text-xl font-medium text-fg">Job weights</h2>
              <p className="mt-1 text-sm text-fg-muted">
                Changing weights recalculates displayed totals from stored criterion percentages.
              </p>
            </div>
            {prefsDirty && (
              <Badge variant="warning" size="sm">
                Unsaved
              </Badge>
            )}
          </div>

          <div className="mt-5 space-y-4">
            {orderedWeightEntries.map((entry) => (
              <label key={entry.key} className="block space-y-2">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-fg">{entry.label}</span>
                  <span className="font-mono text-fg-muted">{Number(entry.value || 0).toFixed(0)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={entry.value}
                  onChange={(event) => {
                    const nextValue = Number(event.target.value);
                    setDraftWeights((current) => ({ ...current, [entry.key]: nextValue }));
                    setPrefsDirty(true);
                  }}
                  className="w-full accent-accent"
                />
              </label>
            ))}
          </div>

          <div className="mt-5 border-t border-[color:var(--hairline)] pt-4">
            <label className="block space-y-2">
              <span className="text-sm text-fg">Score threshold</span>
              <input
                type="number"
                min={0}
                max={100}
                step={1}
                value={thresholdDraft}
                onChange={(event) => {
                  setThresholdDraft(Number(event.target.value));
                  setPrefsDirty(true);
                }}
                className="w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2 text-sm text-fg"
              />
            </label>
          </div>
        </section>

        <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated overflow-hidden">
          <div className="flex items-center justify-between gap-4 border-b border-[color:var(--hairline)] px-5 py-3">
            <p className="text-sm text-fg-muted">Review candidate results and expand a row for criterion-level detail.</p>
            <label className="flex items-center gap-3 text-sm text-fg">
              <span>Sort candidates</span>
              <select
                aria-label="Sort candidates"
                value={sortOrder}
                onChange={(event) => {
                  setSortOrder(event.target.value as SortOrder);
                  setPage(1);
                  setExpandedCandidateId(null);
                }}
                className="rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2 text-sm text-fg"
              >
                <option value="natural">Natural</option>
                <option value="score_desc">Highest score</option>
                <option value="score_asc">Lowest score</option>
              </select>
            </label>
          </div>
          <div className="grid grid-cols-[minmax(220px,1.5fr)_110px_120px_100px] gap-4 border-b border-[color:var(--hairline)] px-5 py-3 text-xs font-semibold uppercase tracking-wide text-fg-muted">
            <span>Candidate</span>
            <span>Score</span>
            <span>Status</span>
            <span>Passed</span>
          </div>

          <div className="divide-y divide-[color:var(--hairline)]">
            {paginatedItems.map((item) => {
              const isExpanded = expandedCandidateId === item.candidate_profile_id;
              return (
                <div key={item.id}>
                  <button
                    type="button"
                    onClick={() => setExpandedCandidateId(isExpanded ? null : item.candidate_profile_id)}
                    className="grid w-full grid-cols-[minmax(220px,1.5fr)_110px_120px_100px] gap-4 px-5 py-4 text-left transition-colors hover:bg-[color:var(--hairline)]"
                  >
                    <span className="flex items-center gap-3">
                      <Avatar name={candidateDisplayLabel(item)} size="sm" />
                      <span className="min-w-0">
                        <span data-testid="scoring-candidate-name" className="block truncate text-sm font-medium text-fg">
                          {candidateDisplayLabel(item)}
                        </span>
                        <span className="block truncate text-xs text-fg-muted">{item.resumeFileName ?? item.candidate_profile_id}</span>
                      </span>
                    </span>
                    <span className="font-display text-2xl font-medium text-fg tabular-nums">
                      {item.status === "completed" || item.status === "outdated" ? item.totalScore.toFixed(1) : "—"}
                    </span>
                    <span>
                      <Badge variant={scoreVariant(item.status)} size="sm">
                        {item.status}
                      </Badge>
                    </span>
                    <span>
                      {item.status === "completed" || item.status === "outdated" ? (
                        <Badge variant={item.passedThreshold ? "success" : "danger"} size="sm">
                          {item.passedThreshold ? "Passed" : "Failed"}
                        </Badge>
                      ) : (
                        <span className="text-sm text-fg-muted">—</span>
                      )}
                    </span>
                  </button>

                  {isExpanded && (
                    <div className="bg-bg px-5 pb-5">
                      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
                        <div className="space-y-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-wide text-fg-muted">Rationale</p>
                            <p className="mt-2 text-sm leading-6 text-fg">{item.rationale || "No rationale available yet."}</p>
                          </div>

                          <div className="space-y-3">
                            {item.componentScores.length === 0 ? (
                              <p className="text-sm text-fg-muted">No component scores available yet.</p>
                            ) : (
                              item.componentScores.map((component) => (
                                <div
                                  key={`${item.id}-${component.criterionKey}`}
                                  className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg p-3"
                                >
                                  <div className="flex flex-wrap items-start justify-between gap-3">
                                    <div className="min-w-0">
                                      <p className="text-sm font-medium text-fg">{formatCriterionLabel(component)}</p>
                                      <div className="mt-2 flex flex-wrap items-center gap-2">
                                        <Badge
                                          variant={component.evaluationMode === "measurable" ? "success" : "neutral"}
                                          size="sm"
                                        >
                                          {component.evaluationMode === "measurable" ? "Rule-based" : "Semantic"}
                                        </Badge>
                                        {component.section && (
                                          <span className="text-xs text-fg-muted">{SECTION_LABELS[component.section] ?? component.section}</span>
                                        )}
                                      </div>
                                    </div>
                                    <div className="text-right">
                                      <p className="font-mono text-lg text-fg tabular-nums">{component.scorePercent.toFixed(1)}%</p>
                                      <p className="text-xs text-fg-muted">
                                        {(component.effectiveWeight ?? 0).toFixed(2)} × {component.weightedScore.toFixed(1)}
                                      </p>
                                    </div>
                                  </div>
                                  {component.evidenceSummary && (
                                    <p className="mt-3 text-sm leading-6 text-fg-muted">{component.evidenceSummary}</p>
                                  )}
                                </div>
                              ))
                            )}
                          </div>
                        </div>

                        <div className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
                          <p className="text-xs font-semibold uppercase tracking-wide text-fg-muted">Evaluation state</p>
                          <dl className="mt-4 space-y-3 text-sm">
                            <div className="flex items-center justify-between gap-4">
                              <dt className="text-fg-muted">Status</dt>
                              <dd><Badge variant={scoreVariant(item.status)} size="sm">{item.status}</Badge></dd>
                            </div>
                            <div className="flex items-center justify-between gap-4">
                              <dt className="text-fg-muted">Threshold</dt>
                              <dd className="font-mono text-fg">{thresholdDraft.toFixed(1)}</dd>
                            </div>
                            <div className="flex items-center justify-between gap-4">
                              <dt className="text-fg-muted">Passed</dt>
                              <dd className={cn("font-medium", item.passedThreshold ? "text-success" : "text-danger")}>
                                {item.passedThreshold ? "Yes" : "No"}
                              </dd>
                            </div>
                            <div className="flex items-center justify-between gap-4">
                              <dt className="text-fg-muted">Components</dt>
                              <dd className="font-mono text-fg">{item.componentScores.length}</dd>
                            </div>
                          </dl>
                          <div className="mt-5 border-t border-[color:var(--hairline)] pt-5">
                            <p className="text-xs font-semibold uppercase tracking-wide text-fg-muted">Criteria radar</p>
                            <div className="mt-3">
                              <CriteriaRadar evaluation={item} jobEvaluations={previewItems} size={300} />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <div className="border-t border-[color:var(--hairline)] px-4">
            <Pagination
              total={displayedItems.length}
              page={page}
              pageSize={pageSize}
              onPageChange={(nextPage) => {
                setPage(nextPage);
                setExpandedCandidateId(null);
              }}
              onPageSizeChange={(nextPageSize) => {
                setPageSize(nextPageSize);
                setPage(1);
                setExpandedCandidateId(null);
              }}
            />
          </div>
        </section>
      </div>
    </div>
  );
}

function ScoringInsights({ items }: { items: CandidateEvaluationResponse[] }) {
  const scored = completedEvaluations(items);
  const distribution = [
    { range: "0–49", candidates: scored.filter((item) => item.totalScore < 50).length },
    { range: "50–69", candidates: scored.filter((item) => item.totalScore >= 50 && item.totalScore < 70).length },
    { range: "70–79", candidates: scored.filter((item) => item.totalScore >= 70 && item.totalScore < 80).length },
    { range: "80–100", candidates: scored.filter((item) => item.totalScore >= 80).length },
  ];
  const criteria = new Map<string, { label: string; total: number; count: number }>();
  scored.forEach((item) => {
    item.componentScores.forEach((component) => {
      const current = criteria.get(component.criterionKey) ?? {
        label: formatCriterionLabel(component),
        total: 0,
        count: 0,
      };
      current.total += component.scorePercent;
      current.count += 1;
      criteria.set(component.criterionKey, current);
    });
  });
  const criteriaAverages = [...criteria.entries()].map(([key, value]) => ({
    key,
    label: value.label,
    score: Number((value.total / value.count).toFixed(1)),
  }));

  if (scored.length === 0) {
    return (
      <EmptyState
        heading="Scoring insights unavailable"
        body="Charts will appear after at least one candidate has a completed evaluation."
      />
    );
  }

  const tooltipStyle = {
    background: "var(--bg-elevated)",
    border: "1px solid var(--hairline-strong)",
    borderRadius: "var(--radius-sm)",
    color: "var(--fg)",
    fontSize: 12,
  };

  return (
    <section className="space-y-4">
      <div>
        <h2 className="font-display text-xl font-medium text-fg">Scoring insights</h2>
        <p className="mt-1 text-sm text-fg-muted">A job-level view of candidate score distribution and criteria strength.</p>
      </div>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
          <h3 className="text-sm font-medium text-fg">Score distribution</h3>
          <p className="mt-1 text-xs text-fg-muted">Completed candidates by score range</p>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distribution} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
                <CartesianGrid vertical={false} stroke="var(--hairline)" />
                <XAxis dataKey="range" tick={{ fill: "var(--fg-muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis allowDecimals={false} tick={{ fill: "var(--fg-muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
                <RechartsTooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--hairline)" }} />
                <Bar dataKey="candidates" name="Candidates" fill="var(--accent)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
          <h3 className="text-sm font-medium text-fg">Average by criteria</h3>
          <p className="mt-1 text-xs text-fg-muted">Average percentage across completed candidates</p>
          <div className="mt-4 max-h-[360px] overflow-y-auto">
            <div style={{ height: Math.max(240, criteriaAverages.length * 44) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={criteriaAverages} layout="vertical" margin={{ top: 4, right: 28, bottom: 4, left: 8 }}>
                  <CartesianGrid horizontal={false} stroke="var(--hairline)" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: "var(--fg-muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis
                    type="category"
                    dataKey="label"
                    width={150}
                    tick={{ fill: "var(--fg-muted)", fontSize: 11 }}
                    tickFormatter={(value: string) => value.length > 24 ? `${value.slice(0, 21)}…` : value}
                    axisLine={false}
                    tickLine={false}
                  />
                  <RechartsTooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--hairline)" }} formatter={(value) => [`${Number(value).toFixed(1)}%`, "Average"]} />
                  <Bar dataKey="score" name="Average" fill="var(--success)" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
      <p className="text-xs font-semibold uppercase tracking-wide text-fg-muted">{label}</p>
      <p className="mt-3 font-display text-3xl font-medium text-fg tabular-nums">{value}</p>
    </div>
  );
}
