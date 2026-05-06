import { api, type JobDescriptionResponse, type ScoreResponse } from "@/api";
import {
    Avatar,
    Badge,
    Button,
    ScoreDonut,
    ScoreRadar,
    Skeleton,
    type ScoreSegment,
} from "@/components/ui";
import { useSelectedJobId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
    BarChart2,
    Check,
    ChevronDown,
    ChevronUp,
    ClipboardCopy,
    Plus,
    RefreshCw,
    Search,
    X,
} from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { toast } from "sonner";

// ── constants ────────────────────────────────────────────────────────────────

const PROCESSING_MESSAGES = [
  "Reading job description…",
  "Evaluating candidate profiles…",
  "Applying section weights…",
  "Calibrating match scores…",
  "Generating rationales…",
  "Finalising results…",
];

const SEG_COLORS = ["#1F3A2E", "#2A5A78", "#5A3A7E", "#7A3A3A", "#3A5A3A", "#5A4A2A"];

const EXTRA_SECTIONS = [
  { key: "languages", label: "Languages" },
  { key: "achievements", label: "Achievements" },
  { key: "certifications", label: "Certifications" },
  { key: "publications", label: "Publications" },
  { key: "other", label: "Other" },
];

// ── types ────────────────────────────────────────────────────────────────────

type Step = 1 | 2 | 3;
type WeightSection = { key: string; label: string; value: number };

const DEFAULT_SECTIONS: WeightSection[] = [
  { key: "skills", label: "Skills", value: 25 },
  { key: "experience", label: "Experience", value: 25 },
  { key: "education", label: "Education", value: 20 },
  { key: "projects", label: "Projects", value: 20 },
  { key: "summary", label: "Summary", value: 10 },
];

const DEFAULT_KEYS = new Set(DEFAULT_SECTIONS.map((s) => s.key));

// ── helpers ──────────────────────────────────────────────────────────────────

function truncateId(id: string) {
  return `#${id.slice(0, 4)}…${id.slice(-4)}`;
}

function fileToName(f: string) {
  return (
    f
      .replace(/\.pdf$/i, "")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .trim() || f
  );
}

function scoreColor(n: number) {
  if (n >= 80) return "var(--success)";
  if (n >= 60) return "var(--warning)";
  return "var(--danger)";
}

// ── main component ───────────────────────────────────────────────────────────

export default function ScoringSetupRoute() {
  const [searchParams] = useSearchParams();
  const selectedJobId = useSelectedJobId();

  // ── step state ────────────────────────────────────────────────────────────
  const [step, setStep] = useState<Step>(1);

  // ── step-1 state ──────────────────────────────────────────────────────────
  const [selectedJdId, setSelectedJdId] = useState(searchParams.get("jd") ?? "");
  const [candidateMode, setCandidateMode] = useState<"all" | "specific">("all");
  const [candSearch, setCandSearch] = useState("");
  const [selectedCandIds, setSelectedCandIds] = useState<Set<string>>(new Set());
  const [sections, setSections] = useState<WeightSection[]>(DEFAULT_SECTIONS);
  const [threshold, setThreshold] = useState(50);
  const [batchSize, setBatchSize] = useState(10);
  const [addSectionOpen, setAddSectionOpen] = useState(false);

  // ── step-2 state ──────────────────────────────────────────────────────────
  const [msgIdx, setMsgIdx] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  // ── step-3 state ──────────────────────────────────────────────────────────
  const [scoreResult, setScoreResult] = useState<ScoreResponse | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [selIds, setSelIds] = useState<Set<string>>(new Set());
  const [resultSort, setResultSort] = useState<{
    key: "totalScore" | "passed";
    dir: "asc" | "desc";
  }>({ key: "totalScore", dir: "desc" });
  const [copiedId, setCopiedId] = useState(false);

  // ── data ──────────────────────────────────────────────────────────────────

  const { data: jdData, isLoading: jdsLoading } = useQuery({
    queryKey: ["jobDescriptions"],
    queryFn: async () => {
      if (!selectedJobId) return { items: [], total: 0 };
      try {
        const jd = await api.jobs.jobDescription.get(selectedJobId);
        return { items: [jd], total: 1 };
      } catch {
        return { items: [], total: 0 };
      }
    },
  });

  const { data: resumeData, isLoading: resumesLoading } = useQuery({
    queryKey: ["resumes", 200],
    queryFn: () => (selectedJobId ? api.jobs.resumes.list(selectedJobId, { limit: 200 }) : Promise.resolve({ items: [], total: 0 })),
    enabled: candidateMode === "specific",
  });

  const jds: JobDescriptionResponse[] = jdData?.items ?? [];
  const resumes = resumeData?.items ?? [];
  const selectedJd = jds.find((j) => j.id === selectedJdId);

  useEffect(() => {
    if (!selectedJdId && jds.length > 0) {
      setSelectedJdId(jds[0].id);
    }
  }, [jds, selectedJdId]);

  // ── processing timers ─────────────────────────────────────────────────────

  useEffect(() => {
    if (step !== 2) return;
    setMsgIdx(0);
    setElapsed(0);
    const t1 = setInterval(
      () => setMsgIdx((i) => (i + 1) % PROCESSING_MESSAGES.length),
      2500
    );
    const t2 = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => {
      clearInterval(t1);
      clearInterval(t2);
    };
  }, [step]);

  // ── mutation ──────────────────────────────────────────────────────────────

  const scoreMutation = useMutation({
    mutationFn: () => {
      const sw: Record<string, number> = {};
      sections.forEach((s) => {
        sw[s.key] = s.value;
      });
      return api.jobs.score(selectedJobId!, {
        score_threshold: threshold,
        batch_size: batchSize,
        section_weights: sw,
        candidate_profile_ids:
          candidateMode === "specific" && selectedCandIds.size > 0
            ? [...selectedCandIds]
            : undefined,
      });
    },
    onSuccess: (data) => {
      setScoreResult(data);
      setStep(3);
      localStorage.setItem("recruiter_onboarding_scored", "true");
    },
    onError: () => {
      toast.error("Scoring failed — please try again");
      setStep(1);
    },
  });

  // ── weight helpers ────────────────────────────────────────────────────────

  const totalWeight = sections.reduce((sum, s) => sum + s.value, 0);

  const donutSegments: ScoreSegment[] = sections.map((s, i) => ({
    label: s.label,
    value: totalWeight > 0 ? Math.round((s.value / totalWeight) * 100) : 0,
    color: SEG_COLORS[i % SEG_COLORS.length],
  }));

  const availableToAdd = EXTRA_SECTIONS.filter(
    (es) => !sections.find((s) => s.key === es.key)
  );

  // ── candidate helpers ─────────────────────────────────────────────────────

  const filteredResumes = useMemo(() => {
    const q = candSearch.toLowerCase().trim();
    if (!q) return resumes;
    return resumes.filter(
      (r) =>
        fileToName(r.original_file_name).toLowerCase().includes(q) ||
        r.original_file_name.toLowerCase().includes(q)
    );
  }, [resumes, candSearch]);

  // ── result helpers ────────────────────────────────────────────────────────

  const sortedScores = useMemo(() => {
    if (!scoreResult) return [];
    return [...scoreResult.scores].sort((a, b) => {
      const av =
        resultSort.key === "totalScore" ? a.totalScore : a.passedThreshold ? 1 : 0;
      const bv =
        resultSort.key === "totalScore" ? b.totalScore : b.passedThreshold ? 1 : 0;
      return resultSort.dir === "desc" ? bv - av : av - bv;
    });
  }, [scoreResult, resultSort]);

  const avgScore =
    scoreResult && scoreResult.scores.length > 0
      ? Math.round(
          scoreResult.scores.reduce((s, c) => s + c.totalScore, 0) /
            scoreResult.scores.length
        )
      : 0;

  const highScore =
    scoreResult && scoreResult.scores.length > 0
      ? Math.max(...scoreResult.scores.map((c) => c.totalScore))
      : 0;

  function toggleResultSort(key: typeof resultSort.key) {
    setResultSort((prev) =>
      prev.key === key
        ? { ...prev, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "desc" }
    );
  }

  function toggleExpand(id: string) {
    setExpandedIds((prev) => {
      const n = new Set(prev);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  }

  function toggleSel(id: string) {
    setSelIds((prev) => {
      const n = new Set(prev);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  }

  function copyRunId() {
    if (!scoreResult) return;
    navigator.clipboard.writeText(scoreResult.match_run_id).then(() => {
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
    });
  }

  function startScoring() {
    if (!selectedJobId || !selectedJdId) {
      toast.error("Please select a job description");
      return;
    }
    setStep(2);
    scoreMutation.mutate();
  }

  function resetToSetup() {
    setStep(1);
    setScoreResult(null);
    setExpandedIds(new Set());
    setSelIds(new Set());
  }

  const estSeconds = (() => {
    const n =
      candidateMode === "all" ? (resumeData?.total ?? 0) : selectedCandIds.size;
    return n > 0 ? Math.max(15, Math.ceil(n / batchSize) * 15) : null;
  })();

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="px-8 py-8 min-h-full">

      {/* Stepper */}
      <div className="flex items-center gap-0 mb-10">
        {[
          { n: 1, label: "Setup" },
          { n: 2, label: "Processing" },
          { n: 3, label: "Results" },
        ].map(({ n, label }, i) => (
          <div key={n} className="flex items-center">
            {i > 0 && (
              <div
                className={cn(
                  "h-px w-14 mx-3",
                  step > i ? "bg-accent" : "bg-[color:var(--hairline-strong)]"
                )}
              />
            )}
            <div className="flex items-center gap-2">
              <div
                className={cn(
                  "h-7 w-7 rounded-full flex items-center justify-center text-xs font-medium font-sans transition-colors",
                  step >= n
                    ? "bg-accent text-accent-fg"
                    : "bg-[color:var(--hairline)] text-fg-muted"
                )}
              >
                {step > n ? <Check size={12} strokeWidth={2.5} /> : n}
              </div>
              <span
                className={cn(
                  "text-sm font-sans",
                  step === n ? "text-fg font-medium" : "text-fg-muted"
                )}
              >
                {label}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* ── STEP 1: Setup ── */}
      {step === 1 && (
        <div className="grid grid-cols-5 gap-10">

          {/* Left: JD + Candidates */}
          <div className="col-span-3 space-y-8">

            {/* JD Selector */}
            <div>
              <h2 className="font-display text-xl font-medium text-fg mb-4">
                Job Description
              </h2>
              {jdsLoading ? (
                <Skeleton className="h-10 w-full" />
              ) : (
                <div className="space-y-3">
                  <select
                    value={selectedJdId}
                    onChange={(e) => setSelectedJdId(e.target.value)}
                    className={cn(
                      "w-full h-10 px-3 text-sm font-sans rounded-[var(--radius-md)]",
                      "border border-[color:var(--hairline-strong)] bg-bg text-fg",
                      "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
                    )}
                  >
                    <option value="">Select a job description…</option>
                    {jds.map((j) => (
                      <option key={j.id} value={j.id}>
                        {j.title ?? "Untitled position"}
                      </option>
                    ))}
                  </select>
                  {selectedJd && (
                    <div
                      className={cn(
                        "p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)]",
                        "bg-bg-elevated text-sm text-fg-muted font-sans leading-relaxed line-clamp-4"
                      )}
                    >
                      {selectedJd.jd_text}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Candidate Selector */}
            <div>
              <h2 className="font-display text-xl font-medium text-fg mb-4">
                Candidates
              </h2>
              <div className="flex items-center gap-5 mb-4">
                {(["all", "specific"] as const).map((mode) => (
                  <label key={mode} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="candidateMode"
                      checked={candidateMode === mode}
                      onChange={() => setCandidateMode(mode)}
                      className="accent-accent"
                    />
                    <span className="text-sm font-sans text-fg">
                      {mode === "all" ? "All candidates" : "Specific candidates"}
                    </span>
                  </label>
                ))}
              </div>

              {candidateMode === "specific" && (
                <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] overflow-hidden">
                  <div className="relative border-b border-[color:var(--hairline)]">
                    <Search
                      size={13}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-muted pointer-events-none"
                    />
                    <input
                      type="text"
                      placeholder="Search candidates…"
                      value={candSearch}
                      onChange={(e) => setCandSearch(e.target.value)}
                      className="w-full h-9 pl-8 pr-3 text-sm font-sans bg-bg-elevated text-fg placeholder:text-fg-subtle outline-none"
                    />
                  </div>
                  <div className="max-h-52 overflow-y-auto">
                    {resumesLoading ? (
                      Array.from({ length: 4 }).map((_, i) => (
                        <div
                          key={i}
                          className="px-3 py-2.5 flex items-center gap-3 hairline-b"
                        >
                          <Skeleton className="h-4 w-4 shrink-0" />
                          <Skeleton className="h-4 flex-1" />
                        </div>
                      ))
                    ) : filteredResumes.length === 0 ? (
                      <p className="px-3 py-4 text-sm text-fg-muted text-center font-sans">
                        No candidates found
                      </p>
                    ) : (
                      filteredResumes.map((r) => (
                        <label
                          key={r.id}
                          className="flex items-center gap-3 px-3 py-2.5 hairline-b cursor-pointer hover:bg-[color:var(--hairline)] transition-colors"
                        >
                          <input
                            type="checkbox"
                            checked={selectedCandIds.has(r.id)}
                            onChange={() =>
                              setSelectedCandIds((prev) => {
                                const n = new Set(prev);
                                n.has(r.id) ? n.delete(r.id) : n.add(r.id);
                                return n;
                              })
                            }
                            className="h-4 w-4 rounded-[var(--radius-sm)] accent-accent shrink-0"
                          />
                          <span className="text-sm font-sans text-fg">
                            {fileToName(r.original_file_name)}
                          </span>
                        </label>
                      ))
                    )}
                  </div>
                  {selectedCandIds.size > 0 && (
                    <div className="px-3 py-2 border-t border-[color:var(--hairline)] bg-bg text-xs text-fg-muted font-sans">
                      {selectedCandIds.size} selected
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right: Weights + Config */}
          <div className="col-span-2 space-y-8">

            {/* Section Weights */}
            <div>
              <h2 className="font-display text-xl font-medium text-fg mb-4">
                Section Weights
              </h2>
              <div className="flex justify-center mb-5">
                <ScoreDonut score={100} segments={donutSegments} size={152} />
              </div>
              <div className="space-y-3">
                {sections.map((s, i) => (
                  <div key={s.key} className="flex items-center gap-2.5">
                    <div
                      className="h-2 w-2 rounded-full shrink-0"
                      style={{ backgroundColor: SEG_COLORS[i % SEG_COLORS.length] }}
                    />
                    <span className="text-xs font-sans text-fg-muted w-[72px] shrink-0">
                      {s.label}
                    </span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={s.value}
                      onChange={(e) =>
                        setSections((prev) =>
                          prev.map((sec) =>
                            sec.key === s.key ? { ...sec, value: +e.target.value } : sec
                          )
                        )
                      }
                      className="flex-1 accent-accent h-1"
                    />
                    <input
                      type="number"
                      min={0}
                      max={100}
                      value={s.value}
                      onChange={(e) =>
                        setSections((prev) =>
                          prev.map((sec) =>
                            sec.key === s.key
                              ? { ...sec, value: Math.min(100, Math.max(0, +e.target.value)) }
                              : sec
                          )
                        )
                      }
                      className={cn(
                        "w-11 h-7 px-1 text-xs font-mono text-center tabular-nums",
                        "rounded-[var(--radius-sm)] border border-[color:var(--hairline-strong)]",
                        "bg-bg text-fg outline-none",
                        "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
                      )}
                    />
                    {!DEFAULT_KEYS.has(s.key) && (
                      <button
                        type="button"
                        onClick={() =>
                          setSections((prev) => prev.filter((sec) => sec.key !== s.key))
                        }
                        className="text-fg-muted hover:text-danger transition-colors"
                        aria-label={`Remove ${s.label}`}
                      >
                        <X size={12} strokeWidth={2} />
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {availableToAdd.length > 0 && (
                <div className="relative mt-3">
                  <button
                    type="button"
                    onClick={() => setAddSectionOpen((v) => !v)}
                    className="inline-flex items-center gap-1.5 text-xs text-fg-muted hover:text-fg transition-colors font-sans"
                  >
                    <Plus size={11} strokeWidth={2} />
                    Add section
                  </button>
                  {addSectionOpen && (
                    <div
                      className={cn(
                        "absolute left-0 top-full mt-1 z-10 w-44 py-1",
                        "rounded-[var(--radius-md)] bg-bg-elevated",
                        "border border-[color:var(--hairline)] shadow-[var(--shadow-md)]"
                      )}
                    >
                      {availableToAdd.map((es) => (
                        <button
                          key={es.key}
                          type="button"
                          onClick={() => {
                            setSections((prev) => [
                              ...prev,
                              { key: es.key, label: es.label, value: 10 },
                            ]);
                            setAddSectionOpen(false);
                          }}
                          className="w-full px-3 py-2 text-left text-sm font-sans text-fg hover:bg-[color:var(--hairline)] transition-colors"
                        >
                          {es.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Config */}
            <div>
              <h2 className="font-display text-xl font-medium text-fg mb-4">
                Configuration
              </h2>
              <div className="space-y-5">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-sans font-medium text-fg-muted">
                      Pass threshold
                    </label>
                    <span className="font-mono text-sm tabular-nums text-fg">
                      {threshold}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={threshold}
                    onChange={(e) => setThreshold(+e.target.value)}
                    className="w-full accent-accent h-1"
                  />
                  <div className="flex justify-between text-[10px] text-fg-subtle mt-1 font-sans">
                    <span>0 — score all</span>
                    <span>100 — perfect only</span>
                  </div>
                </div>

                <div>
                  <label className="text-xs font-sans font-medium text-fg-muted block mb-1.5">
                    Batch size
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={batchSize}
                    onChange={(e) =>
                      setBatchSize(Math.min(50, Math.max(1, +e.target.value)))
                    }
                    className={cn(
                      "w-24 h-9 px-3 text-sm font-mono tabular-nums rounded-[var(--radius-md)]",
                      "border border-[color:var(--hairline-strong)] bg-bg text-fg outline-none",
                      "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
                    )}
                  />
                  <p className="text-[11px] text-fg-subtle font-sans mt-1">
                    candidates per LLM batch (1–50)
                  </p>
                </div>

                {estSeconds !== null && (
                  <p className="text-xs text-fg-muted font-sans">
                    Estimated time:{" "}
                    <span className="text-fg font-medium tabular-nums">
                      {estSeconds < 60 ? `~${estSeconds}s` : `~${Math.round(estSeconds / 60)}m`}
                    </span>
                    {" "}for{" "}
                    <span className="text-fg font-medium tabular-nums">
                      {candidateMode === "all" ? resumeData?.total ?? "all" : selectedCandIds.size}
                    </span>{" "}
                    candidates
                  </p>
                )}
              </div>
            </div>

            <Button
              variant="primary"
              size="lg"
              className="w-full justify-center"
              icon={<BarChart2 size={16} strokeWidth={2} />}
              disabled={!selectedJdId}
              onClick={startScoring}
            >
              Start scoring
            </Button>
          </div>
        </div>
      )}

      {/* ── STEP 2: Processing ── */}
      {step === 2 && (
        <div className="flex flex-col items-center justify-center min-h-[420px] gap-7">
          <div className="relative w-28 h-28">
            <div className="absolute inset-0 rounded-full bg-accent/5 animate-ping [animation-duration:2000ms]" />
            <div className="absolute inset-4 rounded-full bg-accent/10 animate-pulse" />
            <div className="absolute inset-0 flex items-center justify-center">
              <BarChart2 size={36} strokeWidth={1.25} className="text-accent" />
            </div>
          </div>

          <div className="text-center max-w-sm">
            <h2 className="font-display text-2xl font-medium text-fg mb-2">
              Scoring candidates
            </h2>
            <p className="text-sm text-fg-muted font-sans min-h-[20px] transition-all duration-500">
              {PROCESSING_MESSAGES[msgIdx]}
            </p>
          </div>

          <div className="w-72 h-1 rounded-full bg-[color:var(--hairline)] overflow-hidden">
            <div
              className="h-full w-1/3 rounded-full bg-accent"
              style={{ animation: "indeterminate 1.5s ease-in-out infinite" }}
            />
          </div>

          <p className="font-mono text-xs text-fg-subtle tabular-nums">
            {elapsed}s elapsed
          </p>

          <p className="text-xs text-fg-muted font-sans text-center max-w-xs">
            Do not navigate away — the LLM is actively evaluating candidates.
            This may take several minutes.
          </p>

          <Button variant="secondary" size="sm" disabled className="opacity-40 cursor-not-allowed">
            Cancel
          </Button>
        </div>
      )}

      {/* ── STEP 3: Results ── */}
      {step === 3 && scoreResult && (
        <div className="space-y-6">

          {/* Summary strip */}
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: "Total candidates", value: scoreResult.total_candidates },
              { label: "Passed threshold", value: scoreResult.total_passed_candidates },
              { label: "Average score", value: avgScore },
              { label: "Highest score", value: highScore },
            ].map(({ label, value }) => (
              <div
                key={label}
                className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5"
              >
                <p className="text-xs text-fg-muted font-sans mb-1">{label}</p>
                <p className="font-display text-4xl font-medium text-fg tabular-nums">
                  {value}
                </p>
              </div>
            ))}
          </div>

          {/* Run info bar */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-sm text-fg-muted font-sans">Match run:</span>
              <button
                type="button"
                onClick={copyRunId}
                className="inline-flex items-center gap-1.5 font-mono text-sm text-fg hover:text-accent transition-colors"
                title="Copy match run ID"
              >
                {truncateId(scoreResult.match_run_id)}
                {copiedId ? (
                  <Check size={13} strokeWidth={2.5} className="text-success" />
                ) : (
                  <ClipboardCopy size={13} strokeWidth={1.75} />
                )}
              </button>
              {selectedJd && (
                <span className="text-sm text-fg-muted font-sans">
                  vs{" "}
                  <span className="text-fg font-medium">
                    {selectedJd.title ?? "Untitled position"}
                  </span>
                </span>
              )}
            </div>
            <Button
              variant="ghost"
              size="sm"
              icon={<RefreshCw size={13} strokeWidth={1.75} />}
              onClick={resetToSetup}
            >
              Score again
            </Button>
          </div>

          {/* Results table */}
          <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] overflow-hidden">
            <table className="w-full border-collapse font-sans text-sm">
              <thead className="sticky top-0 z-10 bg-bg">
                <tr className="hairline-b">
                  <th className="w-10 px-4 py-2.5">
                    <input
                      type="checkbox"
                      aria-label="Select all"
                      checked={
                        scoreResult.scores.length > 0 &&
                        selIds.size === scoreResult.scores.length
                      }
                      onChange={(e) =>
                        setSelIds(
                          e.target.checked
                            ? new Set(scoreResult.scores.map((s) => s.candidateId))
                            : new Set()
                        )
                      }
                      className="h-4 w-4 rounded-[var(--radius-sm)] accent-accent cursor-pointer"
                    />
                  </th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wide text-fg-subtle w-10">
                    #
                  </th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wide text-fg-subtle">
                    Candidate
                  </th>
                  <th
                    className="px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wide text-fg-subtle cursor-pointer hover:text-fg transition-colors select-none"
                    onClick={() => toggleResultSort("totalScore")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Score
                      {resultSort.key === "totalScore" &&
                        (resultSort.dir === "asc" ? (
                          <ChevronUp size={12} strokeWidth={2} />
                        ) : (
                          <ChevronDown size={12} strokeWidth={2} />
                        ))}
                    </span>
                  </th>
                  <th
                    className="px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wide text-fg-subtle cursor-pointer hover:text-fg transition-colors select-none"
                    onClick={() => toggleResultSort("passed")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Result
                      {resultSort.key === "passed" &&
                        (resultSort.dir === "asc" ? (
                          <ChevronUp size={12} strokeWidth={2} />
                        ) : (
                          <ChevronDown size={12} strokeWidth={2} />
                        ))}
                    </span>
                  </th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wide text-fg-subtle">
                    Breakdown
                  </th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wide text-fg-subtle">
                    Rationale
                  </th>
                  <th className="w-10 px-4 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {sortedScores.map((score, idx) => (
                  <Fragment key={score.candidateId}>
                    <tr
                      className={cn(
                        "hairline-b transition-colors hover:bg-[color:var(--hairline)]",
                        selIds.has(score.candidateId) && "bg-[rgba(31,58,46,0.04)]"
                      )}
                    >
                      <td className="w-10 px-4 py-3">
                        <input
                          type="checkbox"
                          checked={selIds.has(score.candidateId)}
                          onChange={() => toggleSel(score.candidateId)}
                          className="h-4 w-4 rounded-[var(--radius-sm)] accent-accent cursor-pointer"
                        />
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-fg-muted tabular-nums">
                        {idx + 1}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2.5">
                          <Avatar name={score.candidateId} size="sm" />
                          <span className="font-mono text-xs text-fg tabular-nums">
                            {truncateId(score.candidateId)}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className="font-display text-2xl font-medium tabular-nums"
                          style={{ color: scoreColor(score.totalScore) }}
                        >
                          {score.totalScore}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <Badge
                          variant={score.passedThreshold ? "success" : "neutral"}
                          size="sm"
                          dot
                        >
                          {score.passedThreshold ? "Passed" : "Failed"}
                        </Badge>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-end gap-0.5 h-6">
                          {score.componentScores.map((cs, ci) => (
                            <div
                              key={cs.criterionKey}
                              title={`${cs.criterionKey}: ${cs.score}`}
                              className="w-3 rounded-sm"
                              style={{
                                height: `${Math.max(4, (cs.score / 100) * 24)}px`,
                                backgroundColor: SEG_COLORS[ci % SEG_COLORS.length],
                                opacity: 0.85,
                              }}
                            />
                          ))}
                        </div>
                      </td>
                      <td className="px-4 py-3 max-w-[240px]">
                        <p className="text-xs text-fg-muted line-clamp-2 leading-relaxed">
                          {score.rationale}
                        </p>
                      </td>
                      <td className="px-4 py-3">
                        <button
                          type="button"
                          onClick={() => toggleExpand(score.candidateId)}
                          className="inline-flex items-center justify-center h-6 w-6 rounded-[var(--radius-sm)] text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors"
                        >
                          {expandedIds.has(score.candidateId) ? (
                            <ChevronUp size={13} strokeWidth={2} />
                          ) : (
                            <ChevronDown size={13} strokeWidth={2} />
                          )}
                        </button>
                      </td>
                    </tr>

                    {expandedIds.has(score.candidateId) && (
                      <tr className="bg-bg-elevated">
                        <td colSpan={8} className="px-8 py-6">
                          <div className="grid grid-cols-2 gap-8">
                            <div className="space-y-5">
                              <div>
                                <p className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-2 font-sans">
                                  Full Rationale
                                </p>
                                <p className="text-sm text-fg font-sans leading-relaxed">
                                  {score.rationale}
                                </p>
                              </div>
                              <div>
                                <p className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-2 font-sans">
                                  Component Scores
                                </p>
                                <table className="w-full text-xs font-sans border-collapse">
                                  <thead>
                                    <tr className="hairline-b">
                                      <th className="py-1.5 pr-3 text-left text-fg-subtle font-medium">
                                        Criterion
                                      </th>
                                      <th className="py-1.5 pr-3 text-right text-fg-subtle font-medium tabular-nums">
                                        Wt
                                      </th>
                                      <th className="py-1.5 pr-3 text-right text-fg-subtle font-medium tabular-nums">
                                        Score
                                      </th>
                                      <th className="py-1.5 pr-3 text-right text-fg-subtle font-medium tabular-nums">
                                        Weighted
                                      </th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {score.componentScores.map((cs) => (
                                      <tr key={cs.criterionKey} className="hairline-b">
                                        <td className="py-2 pr-3 text-fg font-medium capitalize">
                                          {cs.criterionKey}
                                        </td>
                                        <td className="py-2 pr-3 text-right text-fg-muted tabular-nums">
                                          {cs.weight}
                                        </td>
                                        <td className="py-2 pr-3 text-right text-fg tabular-nums">
                                          {cs.score}
                                        </td>
                                        <td className="py-2 pr-3 text-right text-fg tabular-nums">
                                          {cs.weightedScore}
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                                {score.componentScores.some((cs) => cs.evidenceSummary) && (
                                  <div className="mt-3 space-y-1.5">
                                    {score.componentScores
                                      .filter((cs) => cs.evidenceSummary)
                                      .map((cs) => (
                                        <p
                                          key={cs.criterionKey}
                                          className="text-[11px] text-fg-muted italic font-sans"
                                        >
                                          <span className="not-italic font-medium text-fg-subtle capitalize">
                                            {cs.criterionKey}:
                                          </span>{" "}
                                          {cs.evidenceSummary}
                                        </p>
                                      ))}
                                  </div>
                                )}
                              </div>
                            </div>
                            <div>
                              <p className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-2 font-sans">
                                Score Radar
                              </p>
                              <ScoreRadar
                                data={score.componentScores.map((cs) => ({
                                  subject: cs.criterionKey,
                                  value: cs.score,
                                  fullMark: 100,
                                }))}
                                size={280}
                              />
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Bulk action bar */}
      {step === 3 && selIds.size > 0 && (
        <div
          className={cn(
            "fixed bottom-6 left-1/2 -translate-x-1/2 z-30",
            "flex items-center gap-3 px-5 py-3 rounded-[var(--radius-lg)]",
            "bg-fg text-bg shadow-[var(--shadow-lg)]",
          )}
        >
          <span className="font-sans text-sm font-medium tabular-nums">
            {selIds.size} selected
          </span>
          <div className="w-px h-4 bg-current opacity-20" />
          <button
            type="button"
            className="text-sm font-sans font-medium text-bg/80 hover:text-bg transition-colors"
          >
            Add {selIds.size} to shortlist
          </button>
          <button
            type="button"
            className="text-sm font-sans font-medium text-bg/80 hover:text-bg transition-colors"
          >
            Export
          </button>
          <button
            type="button"
            className="text-sm font-sans font-medium text-bg/80 hover:text-bg transition-colors"
          >
            Draft outreach
          </button>
          <div className="w-px h-4 bg-current opacity-20" />
          <button
            type="button"
            onClick={() => setSelIds(new Set())}
            className="text-sm font-sans font-medium text-bg/40 hover:text-bg transition-colors"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
