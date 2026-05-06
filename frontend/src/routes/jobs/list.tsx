import { ApiError, api, type JobResponse } from "@/api";
import { DeleteJobDialog } from "@/components/jobs/DeleteJobDialog";
import {
  CurrentWorkspaceBadge,
  JobStatusBadge,
} from "@/components/jobs/job-ui";
import {
  fieldClasses,
  formatAbsoluteDate,
  formatRelativeDate,
  isArchivedJob,
  jobMatchesFilter,
  panelClasses,
  type JobFilter,
} from "@/components/jobs/job-utils";
import { Button, EmptyState, FilterChip, Skeleton } from "@/components/ui";
import { useAuthStore } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowUpRight,
  BriefcaseBusiness,
  CircleEllipsis,
  FileSearch,
  PencilLine,
  Plus,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { toast } from "sonner";
import { routes } from "@/routes";

const EMPTY_JOBS: JobResponse[] = [];

function getNextSelectedJob(jobs: JobResponse[], removedJobId: string): string | null {
  const remaining = jobs.filter((job) => job.id !== removedJobId);
  const nextActive = remaining.find((job) => !isArchivedJob(job));
  return nextActive?.id ?? remaining[0]?.id ?? null;
}

function StatCard({
  label,
  value,
  hint,
  loading = false,
}: {
  label: string;
  value: string;
  hint: string;
  loading?: boolean;
}) {
  return (
    <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-4">
      <p className="text-xs uppercase tracking-[0.2em] text-fg-subtle">{label}</p>
      {loading ? (
        <Skeleton className="mt-4 h-8 w-24" />
      ) : (
        <p className="mt-4 font-display text-3xl leading-none text-fg">{value}</p>
      )}
      <p className="mt-2 text-sm text-fg-muted">{hint}</p>
    </div>
  );
}

function JobActionsMenu({
  job,
  onEdit,
  onToggleArchive,
  onDelete,
}: {
  job: JobResponse;
  onEdit: (job: JobResponse) => void;
  onToggleArchive: (job: JobResponse) => void;
  onDelete: (job: JobResponse) => void;
}) {
  const menuRef = useRef<HTMLDetailsElement>(null);
  const closeMenu = () => {
    if (menuRef.current) menuRef.current.open = false;
  };

  const handleClick = (callback: (job: JobResponse) => void) => (event: React.MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    closeMenu();
    callback(job);
  };

  return (
    <details ref={menuRef} className="relative" onClick={(event) => event.stopPropagation()}>
      <summary
        className="flex h-9 w-9 cursor-pointer list-none items-center justify-center rounded-[var(--radius-md)] text-fg-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-fg"
        aria-label={`Actions for ${job.title}`}
      >
        <CircleEllipsis size={16} strokeWidth={1.5} />
      </summary>
      <div className="absolute right-0 top-11 z-20 w-44 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-1 shadow-[var(--shadow-md)]">
        <button
          type="button"
          onClick={handleClick(onEdit)}
          className="flex w-full items-center gap-2 rounded-[var(--radius-sm)] px-3 py-2 text-sm text-fg-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-fg"
        >
          <PencilLine size={14} strokeWidth={1.5} />
          Rename
        </button>
        <button
          type="button"
          onClick={handleClick(onToggleArchive)}
          className="flex w-full items-center gap-2 rounded-[var(--radius-sm)] px-3 py-2 text-sm text-fg-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-fg"
        >
          <FileSearch size={14} strokeWidth={1.5} />
          {isArchivedJob(job) ? "Mark active" : "Archive"}
        </button>
        <button
          type="button"
          onClick={handleClick(onDelete)}
          className="flex w-full items-center gap-2 rounded-[var(--radius-sm)] px-3 py-2 text-sm text-danger transition-colors hover:bg-[rgba(184,68,46,0.10)]"
        >
          <Trash2 size={14} strokeWidth={1.5} />
          Delete
        </button>
      </div>
    </details>
  );
}

export default function JobsListRoute() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const setSelectedJobId = useAuthStore((state) => state.setSelectedJobId);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<JobFilter>("all");
  const [jobToDelete, setJobToDelete] = useState<JobResponse | null>(null);

  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.jobs.list(),
    staleTime: 60_000,
  });

  const jobs = jobsQuery.data?.items ?? EMPTY_JOBS;
  const selectedJob = jobs.find((job) => job.id === selectedJobId) ?? jobs[0] ?? null;

  const candidateCountQuery = useQuery({
    queryKey: ["jobs", selectedJob?.id, "candidates", "count"],
    queryFn: () => api.jobs.listCandidates(selectedJob!.id),
    enabled: !!selectedJob,
  });

  const jdQuery = useQuery({
    queryKey: ["jobs", selectedJob?.id, "job-description", "summary"],
    enabled: !!selectedJob,
    queryFn: async () => {
      try {
        return await api.jobs.jobDescription.get(selectedJob!.id);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
  });

  const updateJob = useMutation({
    mutationFn: ({ jobId, body }: { jobId: string; body: { title?: string; status?: string } }) =>
      api.jobs.update(jobId, body),
    onSuccess: (job, variables) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["jobs", variables.jobId] });
      toast.success(
        variables.body.status
          ? `Workspace marked ${variables.body.status}`
          : "Workspace updated",
      );
      if (selectedJobId === variables.jobId && variables.body.status) {
        setSelectedJobId(job.id);
      }
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Unable to update job");
    },
  });

  const deleteJob = useMutation({
    mutationFn: (job: JobResponse) => api.jobs.remove(job.id),
    onSuccess: (_, job) => {
      const nextJobId = getNextSelectedJob(jobs, job.id);
      if (selectedJobId === job.id) {
        setSelectedJobId(nextJobId);
      }
      setJobToDelete(null);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      toast.success("Workspace deleted");
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Unable to delete job");
    },
  });

  const filteredJobs = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return jobs.filter((job) => {
      if (!jobMatchesFilter(job, filter)) return false;
      if (!needle) return true;
      return job.title.toLowerCase().includes(needle);
    });
  }, [filter, jobs, search]);

  const openJob = (job: JobResponse) => {
    setSelectedJobId(job.id);
    navigate(routes.dashboard);
  };

  const makeCurrent = (job: JobResponse) => {
    setSelectedJobId(job.id);
    toast.success("Current workspace updated");
  };

  return (
    <>
      <div className="px-6 py-8 sm:px-8">
        <div className="flex flex-col gap-4 border-b border-[color:var(--hairline)] pb-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Jobs</p>
            <h1 className="mt-3 font-display text-4xl leading-tight text-fg">Jobs</h1>
            <p className="mt-3 text-sm leading-6 text-fg-muted sm:text-base">
              Manage hiring workspaces and choose the active context for your recruiting pipeline.
            </p>
          </div>
          <Button
            icon={<Plus size={16} strokeWidth={1.75} />}
            onClick={() => navigate(routes.jobsNew)}
          >
            Create job
          </Button>
        </div>

        <div className="mt-8 grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
          <section className={cn(panelClasses, "overflow-hidden")}>
            <div className="flex flex-col gap-4 border-b border-[color:var(--hairline)] p-5 sm:flex-row sm:items-center sm:justify-between">
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
                    {option === "all" ? "All" : option === "active" ? "Active" : "Archived"}
                  </FilterChip>
                ))}
              </div>
            </div>

            {jobsQuery.isLoading ? (
              <div className="space-y-3 p-5">
                {Array.from({ length: 5 }).map((_, index) => (
                  <Skeleton key={index} className="h-32 w-full rounded-[var(--radius-lg)]" />
                ))}
              </div>
            ) : jobsQuery.error ? (
              <div className="p-6">
                <EmptyState
                  heading="Unable to load jobs"
                  body="The workspace list could not be loaded. Retry after the API is available."
                  action={{
                    label: "Retry",
                    onClick: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
                  }}
                />
              </div>
            ) : filteredJobs.length === 0 ? (
              <div className="p-6">
                <EmptyState
                  heading="No workspaces match this view"
                  body="Try a broader search or switch the status filter."
                  action={{ label: "Create job", onClick: () => navigate(routes.jobsNew) }}
                />
              </div>
            ) : (
              <div className="divide-y divide-[color:var(--hairline)]">
                {filteredJobs.map((job) => {
                  const isCurrent = job.id === selectedJobId;
                  return (
                    <article
                      key={job.id}
                      role="button"
                      tabIndex={0}
                      onClick={() => makeCurrent(job)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          makeCurrent(job);
                        }
                      }}
                      className={cn(
                        "group grid gap-5 p-5 transition-colors hover:bg-[rgba(31,58,46,0.03)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent lg:grid-cols-[minmax(0,1fr)_auto]",
                        isCurrent && "bg-[rgba(31,58,46,0.04)]",
                      )}
                    >
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h2 className="font-display text-2xl leading-tight text-fg">{job.title}</h2>
                          <JobStatusBadge status={job.status} />
                          {isCurrent && <CurrentWorkspaceBadge />}
                        </div>
                        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-sm text-fg-muted">
                          <span title={formatAbsoluteDate(job.updated_at)}>
                            Updated {formatRelativeDate(job.updated_at)}
                          </span>
                          <span title={formatAbsoluteDate(job.created_at)}>
                            Created {formatAbsoluteDate(job.created_at)}
                          </span>
                          <span className="font-mono text-[0.8125rem] text-fg-subtle">
                            {job.id.slice(0, 8)}
                          </span>
                        </div>
                      </div>

                      <div
                        className="flex flex-wrap items-center gap-2"
                        onClick={(event) => event.stopPropagation()}
                      >
                        <Button
                          variant="secondary"
                          icon={<ArrowUpRight size={15} strokeWidth={1.5} />}
                          onClick={() => openJob(job)}
                        >
                          Open
                        </Button>
                        <Button
                          variant="ghost"
                          disabled={isCurrent}
                          onClick={() => makeCurrent(job)}
                        >
                          Set current
                        </Button>
                        <JobActionsMenu
                          job={job}
                          onEdit={(targetJob) => navigate(routes.jobEdit(targetJob.id))}
                          onToggleArchive={(targetJob) =>
                            updateJob.mutate({
                              jobId: targetJob.id,
                              body: { status: isArchivedJob(targetJob) ? "active" : "archived" },
                            })
                          }
                          onDelete={setJobToDelete}
                        />
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </section>

          <aside className="space-y-4">
            <section className={cn(panelClasses, "p-6")}>
              <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Workspace scope</p>
              <p className="mt-4 text-sm leading-6 text-fg-muted">
                Candidates, JD authoring, scoring runs, and AI chat all resolve through the current
                job. Treat this page as the workspace container, not the job description editor.
              </p>
            </section>

            <section className={cn(panelClasses, "p-6")}>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Selected job</p>
                  <p className="mt-3 font-display text-3xl leading-tight text-fg">
                    {selectedJob?.title ?? "No workspace selected"}
                  </p>
                </div>
                <BriefcaseBusiness size={18} strokeWidth={1.5} className="mt-1 text-fg-subtle" />
              </div>
              <div className="mt-5 flex flex-wrap gap-2">
                {selectedJob && <JobStatusBadge status={selectedJob.status} />}
                {selectedJob && <CurrentWorkspaceBadge />}
              </div>
              <div className="mt-6 grid gap-3">
                <StatCard
                  label="Candidates"
                  value={String(candidateCountQuery.data?.total ?? 0)}
                  hint="Visible in the selected workspace."
                  loading={candidateCountQuery.isLoading}
                />
                <StatCard
                  label="Job Description"
                  value={jdQuery.isLoading ? "—" : jdQuery.data ? "Ready" : "Missing"}
                  hint={
                    jdQuery.data
                      ? `Updated ${formatRelativeDate(jdQuery.data.created_at)}`
                      : "No active JD is attached yet."
                  }
                  loading={jdQuery.isLoading}
                />
              </div>
            </section>

            <section className={cn(panelClasses, "p-6")}>
              <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Quick actions</p>
              <div className="mt-4 flex flex-col gap-2">
                <Button
                  variant="secondary"
                  icon={<Plus size={15} strokeWidth={1.75} />}
                  onClick={() => navigate(routes.jobsNew)}
                >
                  Create job
                </Button>
                <Button
                  variant="ghost"
                  icon={<Upload size={15} strokeWidth={1.75} />}
                  disabled={!selectedJob}
                  onClick={() => navigate(routes.candidates)}
                >
                  Upload resumes
                </Button>
                <Button
                  variant="ghost"
                  icon={<PencilLine size={15} strokeWidth={1.75} />}
                  disabled={!selectedJob}
                  onClick={() => selectedJob && navigate(routes.jobEdit(selectedJob.id))}
                >
                  Edit selected job
                </Button>
                <Button
                  variant="ghost"
                  icon={<ArrowUpRight size={15} strokeWidth={1.75} />}
                  disabled={!selectedJob}
                  onClick={() => selectedJob && openJob(selectedJob)}
                >
                  Open job dashboard
                </Button>
              </div>
            </section>
          </aside>
        </div>
      </div>

      <DeleteJobDialog
        job={jobToDelete}
        open={jobToDelete !== null}
        loading={deleteJob.isPending}
        onClose={() => setJobToDelete(null)}
        onConfirm={() => jobToDelete && deleteJob.mutate(jobToDelete)}
      />
    </>
  );
}
