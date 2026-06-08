import { ApiError, api } from "@/api";
import { DeleteJobDialog } from "@/components/jobs/DeleteJobDialog";
import { JobDescriptionRichTextBody } from "@/components/jobs/JobDescriptionRichTextBody";
import { PublicApplicationLinkCard } from "@/components/jobs/PublicApplicationLinkCard";
import { JobStatusBadge } from "@/components/jobs/job-ui";
import { fieldClasses, formatAbsoluteDate, panelClasses } from "@/components/jobs/job-utils";
import { Button, Skeleton } from "@/components/ui";
import { useAuthStore } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BriefcaseBusiness, CalendarDays, ChevronLeft, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { toast } from "sonner";
import { routes } from "@/routes";
import { htmlToMarkdown } from "@/components/jobs/job-description-markdown";

const statusOptions = [
  { value: "active", label: "Active" },
  { value: "archived", label: "Archived" },
] as const;

export default function JobEditRoute() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { jobId } = useParams();
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const setSelectedJobId = useAuthStore((state) => state.setSelectedJobId);
  const isEditMode = Boolean(jobId);
  const [title, setTitle] = useState("");
  const [status, setStatus] = useState("active");
  const [candidateMessage, setCandidateMessage] = useState("");
  const [publicApplyEnabled, setPublicApplyEnabled] = useState(true);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const createJobDescriptionRef = useRef<HTMLDivElement>(null);

  const jobQuery = useQuery({
    queryKey: ["jobs", jobId],
    enabled: !!jobId,
    queryFn: () => api.jobs.get(jobId!),
  });

  useEffect(() => {
    if (!jobQuery.data) return;
    setTitle(jobQuery.data.title);
    setStatus(jobQuery.data.status);
    setCandidateMessage(jobQuery.data.candidate_message ?? "");
    setPublicApplyEnabled(jobQuery.data.public_apply_enabled);
  }, [jobQuery.data]);

  const saveJob = useMutation({
    mutationFn: async () => {
      if (isEditMode) {
        return {
          job: await api.jobs.update(jobId!, {
            title: title.trim(),
            status,
            candidate_message: candidateMessage.trim() || null,
            public_apply_enabled: publicApplyEnabled,
          }),
          descriptionSaved: false,
          descriptionSaveError: null as string | null,
        };
      }

      const job = await api.jobs.create({
        title: title.trim(),
        status,
        candidate_message: candidateMessage.trim() || null,
        public_apply_enabled: publicApplyEnabled,
      });
      const jobDescription = htmlToMarkdown(createJobDescriptionRef.current?.innerHTML ?? "");

      if (!jobDescription) {
        return { job, descriptionSaved: false, descriptionSaveError: null as string | null };
      }

      try {
        await api.jobs.jobDescription.upsert(job.id, {
          title: title.trim() || undefined,
          jd_text: jobDescription,
          is_active: true,
        });
        return { job, descriptionSaved: true, descriptionSaveError: null as string | null };
      } catch (saveError) {
        return {
          job,
          descriptionSaved: false,
          descriptionSaveError:
            saveError instanceof Error ? saveError.message : "Unable to save job description",
        };
      }
    },
    onSuccess: ({ job, descriptionSaved, descriptionSaveError }) => {
      setSelectedJobId(job.id);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["jobs", job.id] });
      qc.invalidateQueries({ queryKey: ["jobs", job.id, "job-description"] });
      qc.invalidateQueries({ queryKey: ["dashboard-jds", job.id] });
      qc.invalidateQueries({ queryKey: ["jobs", job.id, "setup-status"] });
      toast.success(isEditMode ? "Workspace updated" : "Workspace created");
      if (!isEditMode && descriptionSaveError && !descriptionSaved) {
        toast.error(`Workspace created, but the description was not saved. ${descriptionSaveError}`);
      }
      navigate(isEditMode ? routes.jobs : routes.dashboard);
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Unable to save job");
    },
  });

  const deleteJob = useMutation({
    mutationFn: () => api.jobs.remove(jobId!),
    onSuccess: () => {
      if (selectedJobId === jobId) setSelectedJobId(null);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      setDeleteOpen(false);
      toast.success("Workspace deleted");
      navigate(routes.jobs);
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Unable to delete job");
    },
  });

  const canSubmit = title.trim().length > 0 && !saveJob.isPending;
  const titleText = isEditMode ? "Edit job" : "New job";
  const subtitleText = isEditMode
    ? jobQuery.data
      ? `Created ${formatAbsoluteDate(jobQuery.data.created_at)}`
      : "Update the workspace title and status."
    : "Create a workspace for one role, candidate pool, JD, scoring history, and AI chat context.";

  const errorMessage = useMemo(() => {
    if (!jobQuery.error) return null;
    if (jobQuery.error instanceof ApiError && jobQuery.error.status === 404) {
      return "The requested job no longer exists.";
    }
    return jobQuery.error instanceof Error ? jobQuery.error.message : "Unable to load job";
  }, [jobQuery.error]);

  return (
    <>
      <div className="px-6 py-8 sm:px-8">
        <div className="flex flex-col gap-4 border-b border-[color:var(--hairline)] pb-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <Link
              to={routes.jobs}
              className="inline-flex items-center gap-2 text-sm text-fg-muted transition-colors hover:text-fg"
            >
              <ChevronLeft size={15} strokeWidth={1.75} />
              Back to Jobs
            </Link>
            <h1 className="mt-4 font-display text-4xl leading-tight text-fg">{titleText}</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-fg-muted sm:text-base">
              {subtitleText}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="ghost" onClick={() => navigate(routes.jobs)}>
              Cancel
            </Button>
            <Button loading={saveJob.isPending} disabled={!canSubmit} onClick={() => saveJob.mutate()}>
              {isEditMode ? "Save changes" : "Create job"}
            </Button>
          </div>
        </div>

        <div className="mt-8 grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
          <section className={cn(panelClasses, "p-6 sm:p-8")}>
            {isEditMode && jobQuery.isLoading ? (
              <div className="space-y-5">
                <Skeleton className="h-5 w-28" />
                <Skeleton className="h-14 w-full" />
                <Skeleton className="h-5 w-24" />
                <Skeleton className="h-12 w-full" />
              </div>
            ) : errorMessage ? (
              <div className="rounded-[var(--radius-lg)] border border-[rgba(184,68,46,0.24)] bg-[rgba(184,68,46,0.06)] p-5">
                <p className="font-medium text-danger">{errorMessage}</p>
                <p className="mt-2 text-sm text-fg-muted">
                  Return to the Jobs list and choose another workspace.
                </p>
              </div>
            ) : (
              <form
                className="space-y-6"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (canSubmit) saveJob.mutate();
                }}
              >
                <div>
                  <label htmlFor="job-title" className="text-sm font-medium text-fg-muted">
                    Job title
                  </label>
                  <p className="mt-2 text-sm text-fg-muted">
                    Enter the role name candidates are applying for.
                  </p>
                  <div className="mt-3 rounded-[var(--radius-lg)] border border-[color:var(--hairline-strong)] bg-bg-elevated px-5 py-4 shadow-[var(--shadow-sm)] transition-colors focus-within:border-accent focus-within:shadow-[var(--shadow-md)]">
                    <input
                      id="job-title"
                      value={title}
                      onChange={(event) => setTitle(event.target.value)}
                      placeholder="Senior Backend Engineer"
                      className="w-full border-none bg-transparent px-0 py-0 font-display text-4xl leading-tight text-fg outline-none placeholder:text-fg-subtle"
                    />
                  </div>
                </div>

                {!isEditMode && (
                  <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5 sm:p-6">
                    <div className="mb-6">
                      <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">
                        Step 2
                      </p>
                      <h2 className="mt-3 font-display text-3xl leading-tight text-fg">
                        Job description
                      </h2>
                      <p className="mt-3 max-w-3xl text-sm leading-6 text-fg-muted">
                        Write the full JD here while you create the workspace so scoring and AI workflows are ready from the start.
                      </p>
                    </div>
                    <JobDescriptionRichTextBody
                      editorRef={createJobDescriptionRef}
                      minHeightClassName="min-h-[360px]"
                    />
                  </div>
                )}

                <div className="max-w-sm">
                  <label htmlFor="job-status" className="text-sm font-medium text-fg-muted">
                    Status
                  </label>
                  <select
                    id="job-status"
                    value={status}
                    onChange={(event) => setStatus(event.target.value)}
                    className={cn(fieldClasses, "mt-2")}
                  >
                    {statusOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">
                        Candidate upload link
                      </p>
                      <p className="mt-3 text-sm leading-6 text-fg-muted">
                        This controls whether candidates can submit resumes through the public link
                        and QR code for this job.
                      </p>
                    </div>
                    <label className="inline-flex items-center gap-2 text-sm font-medium text-fg">
                      <input
                        type="checkbox"
                        checked={publicApplyEnabled}
                        onChange={(event) => setPublicApplyEnabled(event.target.checked)}
                        className="h-4 w-4 rounded border-[color:var(--hairline-strong)] accent-[color:var(--accent)]"
                      />
                      Enabled
                    </label>
                  </div>

                  <label htmlFor="candidate-message" className="mt-5 block text-sm font-medium text-fg-muted">
                    Candidate message
                  </label>
                  <textarea
                    id="candidate-message"
                    value={candidateMessage}
                    onChange={(event) => setCandidateMessage(event.target.value)}
                    placeholder="Add instructions candidates will see before uploading their PDF."
                    rows={5}
                    className={cn(fieldClasses, "mt-2 resize-y leading-6")}
                  />
                </div>

                <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
                  <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">
                    What belongs to a job?
                  </p>
                  <p className="mt-3 text-sm leading-6 text-fg-muted">
                    One JD, uploaded resumes, parsed candidate profiles, score runs, and chat context.
                  </p>
                </div>
              </form>
            )}
          </section>

          <aside className="space-y-4">
            <section className={cn(panelClasses, "p-6")}>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Workspace preview</p>
                  <p className="mt-3 font-display text-3xl leading-tight text-fg">
                    {title.trim() || "Untitled job"}
                  </p>
                </div>
                <BriefcaseBusiness size={18} strokeWidth={1.5} className="mt-1 text-fg-subtle" />
              </div>
              <div className="mt-4">
                <JobStatusBadge status={status} />
              </div>
              <p className="mt-4 text-sm leading-6 text-fg-muted">
                {isEditMode
                  ? "This workspace will define the data boundary for the rest of the recruiting pipeline."
                  : "This workspace and JD will be created together so candidates, scoring, and chat start from the same context."}
              </p>
            </section>

            {isEditMode && jobQuery.data && (
              <PublicApplicationLinkCard job={jobQuery.data} showRotate />
            )}

            {isEditMode && jobQuery.data && (
              <section className={cn(panelClasses, "p-6")}>
                <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Metadata</p>
                <dl className="mt-4 space-y-4 text-sm">
                  <div className="flex items-start justify-between gap-3">
                    <dt className="text-fg-muted">Created at</dt>
                    <dd className="text-right text-fg">{formatAbsoluteDate(jobQuery.data.created_at)}</dd>
                  </div>
                  <div className="flex items-start justify-between gap-3">
                    <dt className="text-fg-muted">Updated at</dt>
                    <dd className="text-right text-fg">{formatAbsoluteDate(jobQuery.data.updated_at)}</dd>
                  </div>
                  <div className="flex items-start justify-between gap-3">
                    <dt className="text-fg-muted">Owner</dt>
                    <dd className="font-mono text-[0.8125rem] text-fg">
                      {jobQuery.data.owner_user_id.slice(0, 8)}
                    </dd>
                  </div>
                </dl>
              </section>
            )}

            {isEditMode && jobQuery.data && (
              <section className="rounded-[var(--radius-lg)] border border-[rgba(184,68,46,0.24)] bg-[rgba(184,68,46,0.06)] p-6">
                <p className="text-xs uppercase tracking-[0.22em] text-danger">Danger zone</p>
                <p className="mt-3 text-sm leading-6 text-fg-muted">
                  Archive this workspace by setting its status, or delete it permanently.
                </p>
                <div className="mt-5 flex flex-col gap-2">
                  <Button
                    variant="ghost"
                    icon={<CalendarDays size={15} strokeWidth={1.75} />}
                    onClick={() => setStatus(status === "archived" ? "active" : "archived")}
                  >
                    {status === "archived" ? "Mark active" : "Archive job"}
                  </Button>
                  <Button
                    variant="danger"
                    icon={<Trash2 size={15} strokeWidth={1.75} />}
                    onClick={() => setDeleteOpen(true)}
                  >
                    Delete job
                  </Button>
                </div>
              </section>
            )}
          </aside>
        </div>
      </div>

      <DeleteJobDialog
        job={jobQuery.data ?? null}
        open={deleteOpen}
        loading={deleteJob.isPending}
        onClose={() => setDeleteOpen(false)}
        onConfirm={() => deleteJob.mutate()}
      />
    </>
  );
}
