import { ApiError, api } from "@/api";
import {
  Badge,
  Button,
  EmptyState,
  Modal,
  ModalContent,
  ModalDescription,
  ModalFooter,
  ModalHeader,
  ModalTitle,
  Skeleton,
} from "@/components/ui";
import { useSelectedJobId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, Save, ToggleLeft, ToggleRight, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { toast } from "sonner";
import { JobDescriptionRichTextBody } from "./JobDescriptionRichTextBody";
import { htmlToMarkdown } from "./job-description-markdown";

type SaveState = "idle" | "saving" | "saved" | "error";

export function WorkspaceJobDescriptionEditor() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const selectedJobId = useSelectedJobId();

  const titleRef = useRef<HTMLInputElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  const autosaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [savedAt, setSavedAt] = useState<Date | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [localIsActive, setLocalIsActive] = useState(true);
  const [hiddenText, setHiddenText] = useState("");
  const [, forceRender] = useState(0);

  useEffect(() => {
    if (saveState !== "saved") return;
    const timer = setInterval(() => forceRender((value) => value + 1), 5000);
    return () => clearInterval(timer);
  }, [saveState]);

  const { data: jd, isLoading } = useQuery({
    queryKey: ["jobs", selectedJobId, "job-description", "editor"],
    queryFn: async () => {
      try {
        return await api.jobs.jobDescription.get(selectedJobId!);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    enabled: !!selectedJobId,
  });

  const { data: evaluations } = useQuery({
    queryKey: ["jobs", selectedJobId, "evaluations"],
    queryFn: () => selectedJobId ? api.jobs.evaluations.list(selectedJobId) : Promise.resolve(null),
    enabled: !!selectedJobId,
  });

  useEffect(() => {
    if (!jd) return;
    if (titleRef.current && !titleRef.current.value) {
      titleRef.current.value = jd.title ?? "";
    }
    setLocalIsActive(jd.is_active);
    setHiddenText(jd.hidden_text ?? "");
  }, [jd]);

  const createMutation = useMutation({
    mutationFn: (body: { title: string; jd_text: string; hidden_text: string }) =>
      api.jobs.jobDescription.upsert(selectedJobId!, {
        title: body.title || undefined,
        jd_text: body.jd_text,
        hidden_text: body.hidden_text,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobDescriptions"] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "job-description"] });
      qc.invalidateQueries({ queryKey: ["dashboard-jds", selectedJobId] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "setup-status"] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "evaluations"] });
      setSaveState("saved");
      setSavedAt(new Date());
    },
    onError: () => {
      setSaveState("error");
      toast.error("Failed to save job description");
    },
  });

  const updateMutation = useMutation({
    mutationFn: (body: Partial<{ title: string; jd_text: string; hidden_text: string; is_active: boolean }>) =>
      api.jobs.jobDescription.patch(selectedJobId!, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobDescriptions"] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "job-description"] });
      qc.invalidateQueries({ queryKey: ["dashboard-jds", selectedJobId] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "setup-status"] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "evaluations"] });
      setSaveState("saved");
      setSavedAt(new Date());
    },
    onError: () => {
      setSaveState("error");
      toast.error("Failed to save job description");
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.jobs.jobDescription.patch(selectedJobId!, { is_active: false }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobDescriptions"] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "job-description"] });
      qc.invalidateQueries({ queryKey: ["dashboard-jds", selectedJobId] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "setup-status"] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "evaluations"] });
      setDeleteOpen(false);
      setLocalIsActive(false);
      toast.success("Job description deactivated");
    },
    onError: () => toast.error("Failed to delete job description"),
  });

  const scoreAgainMutation = useMutation({
    mutationFn: () => api.jobs.evaluations.scoreAgain(selectedJobId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "evaluations"] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "setup-status"] });
      toast.success("Scoring queued");
    },
    onError: () => toast.error("Failed to queue scoring"),
  });

  function getCurrentValues() {
    return {
      title: titleRef.current?.value.trim() ?? "",
      jd_text: htmlToMarkdown(editorRef.current?.innerHTML ?? ""),
      hidden_text: hiddenText,
    };
  }

  function saveNow(extra?: Partial<{ is_active: boolean }>) {
    const { title, jd_text, hidden_text } = getCurrentValues();
    if (!jd_text && !title) return;
    setSaveState("saving");
    if (!jd) {
      createMutation.mutate({ title, jd_text, hidden_text });
      return;
    }
    updateMutation.mutate({ title, jd_text, hidden_text, ...extra });
  }

  function scheduleAutosave() {
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    autosaveTimer.current = setTimeout(() => {
      if (selectedJobId) saveNow();
    }, 1500);
  }

  function cancelAndSaveNow() {
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    if (selectedJobId) saveNow();
  }

  useEffect(() => {
    return () => {
      if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    };
  }, []);

  function toggleActive() {
    const next = !localIsActive;
    setLocalIsActive(next);
    if (jd) {
      updateMutation.mutate({ is_active: next });
    }
  }

  function savedLabel(): string {
    if (saveState === "saving") return "Saving…";
    if (saveState === "error") return "Save failed";
    if (saveState === "saved" && savedAt) {
      const diff = Math.round((Date.now() - savedAt.getTime()) / 1000);
      if (diff < 5) return "Saved just now";
      if (diff < 60) return `Saved ${diff}s ago`;
      return `Saved ${Math.round(diff / 60)}m ago`;
    }
    return "";
  }

  if (!selectedJobId) {
    return (
      <div className="px-8 py-8 min-h-full">
        <EmptyState
          heading="No workspace selected"
          body="Select a job workspace first. Job description management now follows the active workspace."
          action={{ label: "Open jobs", onClick: () => navigate(routes.jobs) }}
        />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="px-8 py-8 min-h-full space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full rounded-[var(--radius-lg)]" />
        <Skeleton className="h-80 w-full rounded-[var(--radius-lg)]" />
      </div>
    );
  }

  const isSaving = createMutation.isPending || updateMutation.isPending;
  const scoringStatus =
    !evaluations || evaluations.total_candidates === 0
      ? "Not scored"
      : evaluations.outdated_count > 0
        ? "Outdated"
        : evaluations.running_count > 0 || evaluations.pending_count > 0
          ? "Scoring"
          : "Current";
  const showScoreAgain = scoringStatus === "Outdated" || scoringStatus === "Not scored";

  return (
    <div className="px-8 py-8 min-h-full">
      <div className="flex flex-col gap-4 border-b border-[color:var(--hairline)] pb-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl">
          <h1 className="font-display text-[2rem] font-medium leading-tight text-fg">
            Workspace job description
          </h1>
          <p className="mt-2 text-sm leading-6 text-fg-muted">
            This page follows the selected workspace. Manage the current job description from the active job context instead of treating it as a standalone resource.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {savedLabel() && (
            <span
              className={cn(
                "text-xs font-sans tabular-nums transition-colors",
                saveState === "error" ? "text-danger" : "text-fg-muted",
              )}
            >
              {savedLabel()}
            </span>
          )}
          <Button
            variant="ghost"
            icon={<ArrowUpRight size={15} strokeWidth={1.75} />}
            onClick={() => navigate(routes.jobs)}
          >
            Open workspace
          </Button>
          <Button
            variant="primary"
            icon={<Save size={13} strokeWidth={2} />}
            loading={isSaving}
            onClick={() => saveNow()}
          >
            {jd ? "Save" : "Create"}
          </Button>
        </div>
      </div>

      <div className="mt-8 grid gap-6 xl:grid-cols-[minmax(0,1fr)_260px]">
          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6 sm:p-8">
          <input
            ref={titleRef}
            type="text"
            placeholder="Untitled position"
            defaultValue={jd?.title ?? ""}
            className={cn(
              "w-full font-display text-[2.5rem] font-medium leading-tight",
              "text-fg placeholder:text-fg-subtle",
              "bg-transparent border-none outline-none mb-6",
            )}
            onBlur={scheduleAutosave}
          />
          <div className="h-px bg-[color:var(--hairline)] mb-6" />
            <JobDescriptionRichTextBody
              editorRef={editorRef}
              initialMarkdown={jd?.jd_text ?? ""}
              onInput={scheduleAutosave}
              onBlur={cancelAndSaveNow}
              minHeightClassName="min-h-[400px]"
            />
            <div className="mt-8 border-t border-[color:var(--hairline)] pt-6">
              <h2 className="font-display text-xl font-medium text-fg">Recruiter-only hidden information</h2>
              <p className="mt-2 text-sm leading-6 text-fg-muted">
                This content is stored on the job description and affects evaluation signatures, but is not shown to candidates.
              </p>
              <textarea
                aria-label="Recruiter-only hidden information"
                value={hiddenText}
                onChange={(event) => setHiddenText(event.target.value)}
                onBlur={cancelAndSaveNow}
                className="mt-4 min-h-28 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2 text-sm text-fg"
              />
            </div>
          </section>

        <aside className="space-y-4">
          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6">
            <p className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-3">
              Settings
            </p>
            <div className="flex items-center justify-between py-2">
              <span className="text-sm text-fg">Active</span>
              <button
                type="button"
                onClick={toggleActive}
                aria-pressed={localIsActive}
                title={localIsActive ? "Deactivate" : "Activate"}
                className="text-fg-muted hover:text-fg transition-colors"
              >
                {localIsActive ? (
                  <ToggleRight size={28} strokeWidth={1.5} className="text-accent" />
                ) : (
                  <ToggleLeft size={28} strokeWidth={1.5} />
                )}
              </button>
            </div>
            <p className="text-xs text-fg-muted">
              {localIsActive ? "Available for scoring" : "Not available for scoring"}
            </p>
            <div className="mt-2">
              <Badge variant={localIsActive ? "success" : "neutral"} size="sm" dot>
                {localIsActive ? "Active" : "Inactive"}
              </Badge>
            </div>
          </section>

          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6">
            <p className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-3">
              Scoring
            </p>
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm text-fg">Evaluation status</span>
              <Badge
                variant={
                  scoringStatus === "Current"
                    ? "success"
                    : scoringStatus === "Scoring"
                      ? "warning"
                      : scoringStatus === "Outdated"
                        ? "danger"
                        : "neutral"
                }
                size="sm"
              >
                {scoringStatus}
              </Badge>
            </div>
            <p className="mt-3 text-xs leading-6 text-fg-muted">
              {scoringStatus === "Current"
                ? "Current evaluations match the active JD signature."
                : scoringStatus === "Scoring"
                  ? "New candidate evaluations are still running."
                  : scoringStatus === "Outdated"
                    ? "The JD text or hidden information changed after the last completed evaluation."
                    : "No candidate evaluation has been saved for this job yet."}
            </p>
            {showScoreAgain && (
              <div className="mt-4">
                <Button
                  variant="secondary"
                  size="sm"
                  loading={scoreAgainMutation.isPending}
                  onClick={() => void scoreAgainMutation.mutate()}
                >
                  Score again
                </Button>
              </div>
            )}
          </section>

          {jd && (
            <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6">
              <p className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-2">
                Created
              </p>
              <p className="text-xs text-fg-muted" title={new Date(jd.created_at).toUTCString()}>
                {new Date(jd.created_at).toLocaleDateString(undefined, {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })}
              </p>
            </section>
          )}

          {jd && (
            <section className="rounded-[var(--radius-lg)] border border-[rgba(184,68,46,0.24)] bg-[rgba(184,68,46,0.06)] p-6">
              <p className="text-xs uppercase tracking-[0.22em] text-danger">Danger zone</p>
              <p className="mt-3 text-sm leading-6 text-fg-muted">
                Deactivate this job description if it should no longer be used for scoring.
              </p>
              <div className="mt-5">
                <Button
                  variant="danger"
                  size="sm"
                  className="w-full justify-center"
                  icon={<Trash2 size={13} strokeWidth={1.75} />}
                  onClick={() => setDeleteOpen(true)}
                >
                  Delete JD
                </Button>
              </div>
            </section>
          )}
        </aside>
      </div>

      <Modal open={deleteOpen} onOpenChange={setDeleteOpen}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Delete job description</ModalTitle>
            <ModalDescription>
              Are you sure you want to delete{" "}
              <span className="font-medium text-fg">
                {(jd?.title ?? titleRef.current?.value.trim()) || "Untitled position"}
              </span>
              ? This cannot be undone.
            </ModalDescription>
          </ModalHeader>
          <ModalFooter>
            <Button variant="ghost" onClick={() => setDeleteOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              loading={deleteMutation.isPending}
              onClick={() => deleteMutation.mutate()}
            >
              Delete
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}
