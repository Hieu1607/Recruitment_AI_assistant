import { api, type JobResponse } from "@/api";
import { Button, EmptyState, FilterChip, Skeleton } from "@/components/ui";
import { useAuthStore } from "@/lib/auth";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, BriefcaseBusiness, FileUp, MessageSquare, Plus, Search, Sparkles, Users } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { toast } from "sonner";
import { routes } from "@/routes";
import { cn } from "@/lib/cn";
import { CurrentWorkspaceBadge, JobStatusBadge } from "./job-ui";
import {
  fieldClasses,
  formatAbsoluteDate,
  formatRelativeDate,
  jobMatchesFilter,
  panelClasses,
  type JobFilter,
} from "./job-utils";

interface JobWorkspaceGateProps {
  jobs: JobResponse[];
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
}

export function JobWorkspaceGate({
  jobs,
  isLoading,
  error,
  onRetry,
}: JobWorkspaceGateProps) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const setSelectedJobId = useAuthStore((state) => state.setSelectedJobId);
  const [jobTitle, setJobTitle] = useState("");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<JobFilter>("all");

  const createJob = useMutation({
    mutationFn: async () => {
      const title = jobTitle.trim();
      const job = await api.jobs.create({ title });
      return { job };
    },
    onSuccess: ({ job }) => {
      setSelectedJobId(job.id);
      setJobTitle("");
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["jobs", job.id, "job-description", "summary"] });
      qc.invalidateQueries({ queryKey: ["dashboard-jds", job.id] });
      qc.invalidateQueries({ queryKey: ["jobs", job.id, "setup-status"] });
      toast.success("Workspace created");
      navigate(routes.dashboard);
    },
    onError: (mutationError) => {
      toast.error(mutationError instanceof Error ? mutationError.message : "Unable to create job");
    },
  });
  const canCreateJob = jobTitle.trim().length > 0 && !createJob.isPending;

  const filteredJobs = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return jobs.filter((job) => {
      if (!jobMatchesFilter(job, filter)) return false;
      if (!needle) return true;
      return job.title.toLowerCase().includes(needle);
    });
  }, [filter, jobs, search]);

  const openWorkspace = (job: JobResponse) => {
    setSelectedJobId(job.id);
    navigate(routes.dashboard);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-bg">
      <div className="min-h-full px-6 py-8 sm:px-8 lg:px-12">
        <div className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-6xl flex-col">
          <div className="flex items-start justify-between gap-6">
            <div>
              <p className="font-display text-2xl text-fg">EasyHR</p>
              <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.24em] text-fg-subtle">
                Editorial Intelligence
              </p>
            </div>
            {jobs.length > 0 && (
              <p className="max-w-xs text-right text-xs text-fg-subtle">
                You can switch jobs anytime from the top bar.
              </p>
            )}
          </div>

          <div className="flex flex-1 items-center justify-center py-10">
            <div className="w-full max-w-5xl">
              <div className="mx-auto max-w-2xl text-center">
                {jobs.length === 0 && (
                  <div className="inline-flex items-center gap-2 rounded-full border border-[rgba(31,58,46,0.12)] bg-[rgba(31,58,46,0.05)] px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em] text-fg-muted">
                    <Sparkles size={12} strokeWidth={1.75} className="text-accent" />
                    First-time setup
                  </div>
                )}
                <h1 className="font-display text-4xl leading-tight text-fg sm:text-5xl">
                  {jobs.length === 0
                    ? "Set up your first job."
                    : "Choose a hiring workspace."}
                </h1>
                <p className="mt-4 text-base leading-7 text-fg-muted">
                  {jobs.length === 0
                    ? "Create the workspace boundary first. You will add the full job description from the dashboard next."
                    : "Every resume, JD, score, and chat session is scoped to one job."}
                </p>
              </div>

              {isLoading ? (
                <div className="mt-10 grid gap-4 md:grid-cols-2">
                  <div className={cn(panelClasses, "p-6")}>
                    <Skeleton className="h-6 w-44" />
                    <Skeleton className="mt-4 h-12 w-full" />
                    <Skeleton className="mt-6 h-10 w-28" />
                  </div>
                  <div className={cn(panelClasses, "p-6")}>
                    <Skeleton className="h-10 w-full" />
                    <div className="mt-6 space-y-3">
                      {Array.from({ length: 3 }).map((_, index) => (
                        <Skeleton key={index} className="h-28 w-full rounded-[var(--radius-lg)]" />
                      ))}
                    </div>
                  </div>
                </div>
              ) : error ? (
                <div className={cn(panelClasses, "mx-auto mt-10 max-w-xl p-8")}>
                  <EmptyState
                    icon={<Sparkles size={36} strokeWidth={1.5} />}
                    heading="Unable to load workspaces"
                    body="The jobs list did not load. Retry once the API is available."
                    action={{ label: "Retry", onClick: onRetry }}
                  />
                </div>
              ) : jobs.length === 0 ? (
                <div className="relative mx-auto mt-12 max-w-4xl overflow-hidden rounded-[28px] border border-[color:var(--hairline-strong)] bg-bg-elevated shadow-[var(--shadow-lg)]">
                  <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-[radial-gradient(circle_at_top_left,rgba(31,58,46,0.16),transparent_52%),radial-gradient(circle_at_top_right,rgba(31,58,46,0.09),transparent_42%)]" />
                  <div className="relative grid lg:grid-cols-[minmax(0,1fr)_300px]">
                    <form
                      className="space-y-6 p-6 sm:p-8 lg:p-10"
                      onSubmit={(event) => {
                        event.preventDefault();
                        if (canCreateJob) createJob.mutate();
                      }}
                    >
                      <div className="inline-flex items-center gap-2 rounded-full border border-[rgba(31,58,46,0.10)] bg-bg px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em] text-fg-muted shadow-[var(--shadow-sm)]">
                        <BriefcaseBusiness size={12} strokeWidth={1.75} className="text-accent" />
                        First-time setup
                      </div>

                      <div>
                        <h2 className="font-display text-3xl leading-tight text-fg sm:text-[2.5rem]">
                          Name the role you are hiring for.
                        </h2>
                        <p className="mt-3 max-w-2xl text-sm leading-7 text-fg-muted sm:text-base">
                          This creates the workspace boundary for resumes, job description authoring,
                          scoring runs, and AI recruiter chat.
                        </p>
                      </div>

                      <div>
                        <label className="text-sm font-medium text-fg-muted" htmlFor="first-job-title">
                          Job title
                        </label>
                        <input
                          id="first-job-title"
                          value={jobTitle}
                          onChange={(event) => setJobTitle(event.target.value)}
                          placeholder="Senior Backend Engineer"
                          className={cn(
                            fieldClasses,
                            "mt-2 h-14 bg-bg font-display text-2xl shadow-[var(--shadow-sm)] sm:text-[2rem]",
                          )}
                        />
                        <p className="mt-3 max-w-xl text-sm leading-6 text-fg-muted">
                          Keep it specific. This title will anchor the rest of the hiring workflow.
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center justify-between gap-4 border-t border-[color:var(--hairline)] pt-5">
                        <p className="text-xs uppercase tracking-[0.18em] text-fg-subtle">
                          The dashboard will guide the next setup steps.
                        </p>
                        <Button
                          type="submit"
                          icon={<Plus size={16} strokeWidth={1.75} />}
                          loading={createJob.isPending}
                          disabled={!canCreateJob}
                        >
                          Create workspace
                        </Button>
                      </div>
                    </form>

                    <div className="border-t border-white/10 bg-accent text-accent-fg lg:border-l lg:border-t-0">
                      <div className="flex h-full flex-col p-6 sm:p-8">
                        <div className="rounded-[22px] border border-white/15 bg-white/5 p-5">
                          <p className="text-[11px] uppercase tracking-[0.22em] text-[rgba(250,250,247,0.66)]">
                            What this unlocks
                          </p>
                          <div className="mt-4 space-y-4">
                            {[ 
                              {
                                icon: Users,
                                title: "Candidate scope",
                                body: "Uploaded resumes stay tied to the right role and pipeline.",
                              },
                              {
                                icon: MessageSquare,
                                title: "Workspace context",
                                body: "Scoring, AI chat, and candidate activity all stay grounded in the selected role.",
                              },
                              {
                                icon: FileUp,
                                title: "Guided next steps",
                                body: "After entering the dashboard, add the JD first, then start uploading resumes.",
                              },
                            ].map((item) => {
                              const Icon = item.icon;
                              return (
                                <div key={item.title} className="flex items-start gap-3">
                                  <div className="mt-0.5 inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/15 bg-white/10">
                                    <Icon size={15} strokeWidth={1.75} />
                                  </div>
                                  <div>
                                    <p className="text-sm font-medium text-accent-fg">{item.title}</p>
                                    <p className="mt-1 text-sm leading-6 text-[rgba(250,250,247,0.72)]">
                                      {item.body}
                                    </p>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>

                        <div className="mt-6">
                          <p className="text-[11px] uppercase tracking-[0.22em] text-[rgba(250,250,247,0.66)]">
                            Recommended next
                          </p>
                          <div className="mt-3 space-y-2">
                            {[
                              "Add the full job description from the dashboard.",
                              "Upload resumes once the JD is ready.",
                              "Run scoring after both data sources exist.",
                              "Use AI chat with the workspace context in place.",
                            ].map((step, index) => (
                              <div
                                key={step}
                                className="flex items-start gap-3 rounded-[var(--radius-md)] border border-white/10 bg-white/5 px-3 py-3"
                              >
                                <div className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white/10 text-xs font-medium">
                                  {index + 1}
                                </div>
                                <p className="text-sm leading-6 text-[rgba(250,250,247,0.78)]">
                                  {step}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="mt-auto pt-6">
                          <div className="flex items-center gap-2 rounded-[var(--radius-lg)] border border-white/10 bg-white/5 px-4 py-3">
                            <ArrowRight size={16} strokeWidth={1.75} className="shrink-0" />
                            <p className="text-sm leading-6 text-[rgba(250,250,247,0.78)]">
                              Create the workspace now, then complete the JD inside the dashboard.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-10 grid gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
                  <section className={cn(panelClasses, "p-5 sm:p-6")}>
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                      <label className="relative block flex-1">
                        <Search
                          size={15}
                          strokeWidth={1.75}
                          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-fg-subtle"
                        />
                        <input
                          value={search}
                          onChange={(event) => setSearch(event.target.value)}
                          placeholder="Search jobs..."
                          className={cn(fieldClasses, "pl-9")}
                          aria-label="Search jobs"
                        />
                      </label>
                      <div className="flex flex-wrap gap-2">
                        {(["all", "active", "archived"] as const).map((option) => (
                          <FilterChip
                            key={option}
                            selected={filter === option}
                            onClick={() => setFilter(option)}
                          >
                            {option === "all"
                              ? "All"
                              : option === "active"
                                ? "Active"
                                : "Archived"}
                          </FilterChip>
                        ))}
                      </div>
                    </div>

                    <div className="mt-5 grid gap-3 md:grid-cols-2">
                      {filteredJobs.map((job) => {
                        const isCurrent = job.id === selectedJobId;
                        return (
                          <div
                            key={job.id}
                            role="button"
                            tabIndex={0}
                            onClick={() => openWorkspace(job)}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                openWorkspace(job);
                              }
                            }}
                            className={cn(
                              "group rounded-[var(--radius-lg)] border p-5 text-left transition-all duration-[var(--duration-base)] ease-[var(--ease-out)] hover:-translate-y-0.5 hover:border-[color:var(--hairline-strong)] hover:shadow-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
                              isCurrent
                                ? "border-[color:var(--hairline-strong)] bg-[rgba(31,58,46,0.04)]"
                                : "border-[color:var(--hairline)] bg-bg",
                            )}
                            aria-label={`Open ${job.title}`}
                          >
                            <div className="flex items-start justify-between gap-4">
                              <div className="min-w-0">
                                <p className="font-display text-2xl leading-tight text-fg">{job.title}</p>
                                <div className="mt-3 flex flex-wrap items-center gap-2">
                                  <JobStatusBadge status={job.status} />
                                  {isCurrent && <CurrentWorkspaceBadge />}
                                </div>
                              </div>
                              <BriefcaseBusiness
                                size={18}
                                strokeWidth={1.5}
                                className="mt-1 shrink-0 text-fg-subtle"
                              />
                            </div>
                            <div className="mt-5 flex flex-wrap gap-5 text-xs text-fg-muted">
                              <span title={formatAbsoluteDate(job.updated_at)}>
                                Updated {formatRelativeDate(job.updated_at)}
                              </span>
                              <span title={formatAbsoluteDate(job.created_at)}>
                                Created {formatAbsoluteDate(job.created_at)}
                              </span>
                            </div>
                          </div>
                        );
                      })}

                      <button
                        type="button"
                        onClick={() => navigate(routes.jobsNew)}
                        className={cn(
                          panelClasses,
                          "flex min-h-60 flex-col items-start justify-between border-dashed p-5 text-left transition-colors hover:border-[color:var(--hairline-strong)] hover:bg-bg",
                        )}
                      >
                        <div className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-[color:var(--hairline)] bg-bg">
                          <Plus size={18} strokeWidth={1.75} />
                        </div>
                        <div>
                          <p className="font-display text-2xl text-fg">New job</p>
                          <p className="mt-2 text-sm leading-6 text-fg-muted">
                            Create a new workspace for another role.
                          </p>
                        </div>
                      </button>
                    </div>

                    {filteredJobs.length === 0 && (
                      <div className="mt-8">
                        <EmptyState
                          heading="No jobs match this view"
                          body="Try a broader search or switch the status filter."
                        />
                      </div>
                    )}
                  </section>

                  <aside className={cn(panelClasses, "p-6")}>
                    <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Workspace scope</p>
                    <p className="mt-4 text-sm leading-6 text-fg-muted">
                      The selected job defines which resumes, job description, scoring runs, and chat
                      context the rest of the product uses.
                    </p>
                    <div className="mt-8 rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-4">
                      <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Current selection</p>
                      <p className="mt-3 font-display text-2xl text-fg">
                        {jobs.find((job) => job.id === selectedJobId)?.title ?? "No workspace selected"}
                      </p>
                      <p className="mt-2 text-sm text-fg-muted">
                        Select a card to enter the dashboard in that job context.
                      </p>
                    </div>
                  </aside>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
