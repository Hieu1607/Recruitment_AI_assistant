import { ApiError, api, type JobResponse } from "@/api";
import { DeleteJobDialog } from "@/components/jobs/DeleteJobDialog";
import { PublicApplicationLinkCard } from "@/components/jobs/PublicApplicationLinkCard";
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
import {
  Badge,
  Button,
  EmptyState,
  FilterChip,
  Modal,
  ModalContent,
  ModalDescription,
  ModalFooter,
  ModalHeader,
  ModalTitle,
  Skeleton,
} from "@/components/ui";
import { useAuthStore } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowUpRight,
  CircleEllipsis,
  Copy,
  ExternalLink,
  FileSearch,
  PencilLine,
  Plus,
  QrCode,
  Search,
  Trash2,
} from "lucide-react";
import QRCode from "qrcode";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { toast } from "sonner";
import { routes } from "@/routes";

const EMPTY_JOBS: JobResponse[] = [];

async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function getNextSelectedJob(jobs: JobResponse[], removedJobId: string): string | null {
  const remaining = jobs.filter((job) => job.id !== removedJobId);
  const nextActive = remaining.find((job) => !isArchivedJob(job));
  return nextActive?.id ?? remaining[0]?.id ?? null;
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

function truncateText(text: string, max = 180) {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}...` : flat;
}

function JobApplicationQrModal({
  job,
  open,
  onOpenChange,
}: {
  job: JobResponse | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [qrDataUrl, setQrDataUrl] = useState("");
  const publicApplyUrl = job?.public_apply_url ?? "";

  useEffect(() => {
    let cancelled = false;
    if (!publicApplyUrl) {
      setQrDataUrl("");
      return;
    }

    QRCode.toDataURL(publicApplyUrl, {
      errorCorrectionLevel: "M",
      margin: 2,
      width: 260,
      color: {
        dark: "#1f3a2e",
        light: "#ffffff",
      },
    })
      .then((dataUrl) => {
        if (!cancelled) setQrDataUrl(dataUrl);
      })
      .catch(() => {
        if (!cancelled) setQrDataUrl("");
      });

    return () => {
      cancelled = true;
    };
  }, [publicApplyUrl]);

  return (
    <Modal open={open} onOpenChange={onOpenChange}>
      <ModalContent size="large" className="sm:max-w-[640px]">
        <ModalHeader>
          <ModalTitle>{job ? `Application link for ${job.title}` : "Application link"}</ModalTitle>
          <ModalDescription>This QR code and link are unique to this job.</ModalDescription>
        </ModalHeader>

        <div className="flex flex-col items-center gap-4">
          <div className="flex h-44 w-44 shrink-0 items-center justify-center rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-white p-3">
            {qrDataUrl ? (
              <img
                src={qrDataUrl}
                alt={job ? `QR code for ${job.title} resume upload` : "Job application QR code"}
                className="h-full w-full object-contain"
              />
            ) : (
              <QrCode size={48} strokeWidth={1.5} className="text-fg-subtle" />
            )}
          </div>
          <div className="w-full min-w-0 space-y-3">
            <div className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg px-3 py-2">
              <p className="truncate font-mono text-xs text-fg-muted" title={publicApplyUrl}>
                {publicApplyUrl || "No public URL available"}
              </p>
            </div>
            <p className="max-w-[56ch] text-sm leading-6 text-fg-muted sm:text-center">
              {job?.public_apply_enabled === false
                ? "This job's public application link is currently disabled."
                : "Candidates can open this link or scan the QR code to submit a PDF resume for this job."}
            </p>
          </div>
        </div>

        <ModalFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          <Button
            variant="secondary"
            icon={<Copy size={15} strokeWidth={1.75} />}
            disabled={!publicApplyUrl}
            onClick={() => {
              copyText(publicApplyUrl)
                .then(() => toast.success("Candidate upload link copied"))
                .catch(() => toast.error("Unable to copy link"));
            }}
          >
            Copy link
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
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
  const [qrJob, setQrJob] = useState<JobResponse | null>(null);

  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.jobs.list(),
    staleTime: 60_000,
  });

  const jobs = jobsQuery.data?.items ?? EMPTY_JOBS;
  const selectedJob = jobs.find((job) => job.id === selectedJobId) ?? jobs[0] ?? null;

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
            <h1 className="font-display text-4xl leading-tight text-fg">Jobs</h1>
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
                        onKeyDown={(event) => event.stopPropagation()}
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
                        <Button
                          variant="icon"
                          aria-label={`Copy application link for ${job.title}`}
                          title="Copy application link"
                          disabled={!job.public_apply_url}
                          icon={<Copy size={15} strokeWidth={1.75} />}
                          onClick={() => {
                            copyText(job.public_apply_url)
                              .then(() => toast.success("Candidate upload link copied"))
                              .catch(() => toast.error("Unable to copy link"));
                          }}
                        />
                        <a
                          href={job.public_apply_url || undefined}
                          target="_blank"
                          rel="noreferrer"
                          aria-label={`Open application link for ${job.title}`}
                          title="Open application link"
                          className={cn(
                            "inline-flex h-9 w-9 items-center justify-center rounded-[var(--radius-md)] text-fg-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
                            !job.public_apply_url && "pointer-events-none opacity-50",
                          )}
                        >
                          <ExternalLink size={15} strokeWidth={1.75} />
                        </a>
                        <Button
                          variant="icon"
                          aria-label={`Show QR code for ${job.title}`}
                          title="Show QR code"
                          disabled={!job.public_apply_url}
                          icon={<QrCode size={15} strokeWidth={1.75} />}
                          onClick={() => setQrJob(job)}
                        />
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
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Job description</p>
                  {jdQuery.isLoading ? (
                    <Skeleton className="mt-3 h-8 w-48" />
                  ) : (
                    <p className="mt-3 font-display text-3xl leading-tight text-fg">
                      {jdQuery.data ? (jdQuery.data.title ?? "Current job description") : "No job description yet"}
                    </p>
                  )}
                </div>
                {jdQuery.data && (
                  <Badge variant={jdQuery.data.is_active ? "success" : "neutral"} size="sm" dot>
                    {jdQuery.data.is_active ? "Active" : "Inactive"}
                  </Badge>
                )}
              </div>
              <p
                className="mt-3 text-sm text-fg-subtle"
                title={jdQuery.data ? formatAbsoluteDate(jdQuery.data.created_at) : undefined}
              >
                {jdQuery.data
                  ? `Updated ${formatRelativeDate(jdQuery.data.created_at)}`
                  : selectedJob
                    ? "No description update yet"
                    : "Select a job to manage its description."}
              </p>
              {jdQuery.isLoading ? (
                <div className="mt-5 space-y-3">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-11/12" />
                  <Skeleton className="h-4 w-4/5" />
                </div>
              ) : (
                <p className="mt-5 text-sm leading-6 text-fg-muted">
                  {jdQuery.data
                    ? truncateText(jdQuery.data.jd_text)
                    : "Create the job description here so scoring, interview prep, and AI workflows all follow the current job."}
                </p>
              )}
              <div className="mt-5 flex flex-wrap gap-2">
                <Button
                  variant="secondary"
                  icon={<PencilLine size={15} strokeWidth={1.75} />}
                  disabled={!selectedJob}
                  onClick={() => navigate(routes.jobDescriptionNew)}
                >
                  {jdQuery.data ? "Edit job description" : "Create job description"}
                </Button>
                <Button
                  variant="ghost"
                  icon={<ArrowUpRight size={15} strokeWidth={1.75} />}
                  disabled={!selectedJob || !jdQuery.data}
                  onClick={() => jdQuery.data && navigate(`${routes.scoring}?jd=${jdQuery.data.id}`)}
                >
                  Score this job
                </Button>
              </div>
            </section>

            <PublicApplicationLinkCard job={selectedJob} />
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
      <JobApplicationQrModal
        job={qrJob}
        open={qrJob !== null}
        onOpenChange={(open) => {
          if (!open) setQrJob(null);
        }}
      />
    </>
  );
}
