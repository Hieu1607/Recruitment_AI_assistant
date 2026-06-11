import {
  api,
  type CandidateProfileResponse,
  type DispatchCandidateResponse,
  type QuestionSetResponse,
} from "@/api";
import {
  Button,
  EmptyState,
  Modal,
  ModalContent,
  ModalDescription,
  ModalFooter,
  ModalHeader,
  ModalTitle,
} from "@/components/ui";
import { useUserId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Layers, UserRound, Users } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { toast } from "sonner";

type SourceMode = "individual" | "shortlist";

type PreviewCandidate = {
  id: string;
  fullName: string;
  email: string | null;
  subtitle: string | null;
};

type CreateSummary = {
  createdCount: number;
  skippedCount: number;
  failedCount: number;
};

function formatQuestionSetLabel(questionSet: QuestionSetResponse) {
  const createdAt = new Date(questionSet.created_at).toLocaleDateString();
  return questionSet.candidate_full_name
    ? `${questionSet.candidate_full_name} · ${createdAt}`
    : `${questionSet.job_description_title || "Question set"} · ${createdAt}`;
}

function toPreviewCandidate(candidate: CandidateProfileResponse | DispatchCandidateResponse): PreviewCandidate {
  if ("candidate_profile_id" in candidate) {
    return {
      id: candidate.candidate_profile_id,
      fullName: candidate.full_name || "Unknown candidate",
      email: candidate.email,
      subtitle: candidate.current_job_title,
    };
  }
  return {
    id: candidate.id,
    fullName: candidate.full_name || "Unknown candidate",
    email: candidate.email,
    subtitle: candidate.current_job_title,
  };
}

function SourceCard({
  active,
  icon,
  title,
  body,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  title: string;
  body: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-[var(--radius-lg)] border p-4 text-left transition-colors",
        active
          ? "border-accent bg-accent/5 shadow-[inset_0_0_0_1px_var(--accent)]"
          : "border-[color:var(--hairline)] bg-bg hover:border-[color:var(--hairline-strong)]",
      )}
    >
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "mt-0.5 flex h-9 w-9 items-center justify-center rounded-[var(--radius-md)]",
            active ? "bg-accent/10 text-accent" : "bg-[color:var(--hairline)] text-fg-muted",
          )}
        >
          {icon}
        </div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-fg">{title}</span>
            {active && <Check size={14} strokeWidth={2} className="text-accent" />}
          </div>
          <p className="text-sm text-fg-muted">{body}</p>
        </div>
      </div>
    </button>
  );
}

export function InterviewLinkComposerModal({
  open,
  onOpenChange,
  jobId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  jobId: string | null;
}) {
  const queryClient = useQueryClient();
  const userId = useUserId();
  const [sourceMode, setSourceMode] = useState<SourceMode>("individual");
  const [questionSetId, setQuestionSetId] = useState("");
  const [expiresInHours, setExpiresInHours] = useState("72");
  const [shortlistId, setShortlistId] = useState("");
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!open) return;
    setSourceMode("individual");
    setQuestionSetId("");
    setExpiresInHours("72");
    setShortlistId("");
    setSelectedCandidateIds(new Set());
  }, [open]);

  useEffect(() => {
    setSelectedCandidateIds(new Set());
    if (sourceMode === "individual") {
      setShortlistId("");
    }
  }, [sourceMode]);

  const { data: jobDescription, isLoading: jobDescriptionLoading } = useQuery({
    queryKey: ["job-description", jobId],
    queryFn: () => api.jobs.jobDescription.get(jobId!).catch(() => null),
    enabled: open && !!jobId,
  });

  const { data: questionSetsData, isLoading: questionSetsLoading } = useQuery({
    queryKey: ["interview-question-sets", jobDescription?.id],
    queryFn: () => api.interviewQuestions.list({ job_description_id: jobDescription!.id, limit: 200 }),
    enabled: open && !!jobDescription?.id,
  });

  const { data: candidatesData, isLoading: candidatesLoading } = useQuery({
    queryKey: ["job-candidates", jobId],
    queryFn: () => api.jobs.listCandidates(jobId!),
    enabled: open && sourceMode === "individual" && !!jobId,
  });

  const { data: shortlistsData, isLoading: shortlistsLoading } = useQuery({
    queryKey: ["shortlist-collections", userId],
    queryFn: () => api.shortlist.collections.list({ user_id: userId!, limit: 200 }),
    enabled: open && sourceMode === "shortlist" && !!userId,
  });

  const { data: shortlistSummary, isLoading: shortlistSummaryLoading } = useQuery({
    queryKey: ["shortlist-dispatch-summary", shortlistId],
    queryFn: () => api.shortlist.dispatch.summary(shortlistId),
    enabled: open && sourceMode === "shortlist" && !!shortlistId,
  });

  const shortlistMatchesJob = !shortlistSummary?.job || shortlistSummary.job.id === jobId;

  const previewCandidates = useMemo<PreviewCandidate[]>(() => {
    if (sourceMode === "shortlist") {
      return shortlistMatchesJob
        ? (shortlistSummary?.candidates ?? []).map(toPreviewCandidate)
        : [];
    }
    return (candidatesData?.items ?? []).map(toPreviewCandidate);
  }, [candidatesData?.items, shortlistMatchesJob, shortlistSummary?.candidates, sourceMode]);

  useEffect(() => {
    if (sourceMode !== "shortlist" || !shortlistSummary || !shortlistMatchesJob) return;
    setSelectedCandidateIds(new Set((shortlistSummary.candidates ?? []).map((candidate) => candidate.candidate_profile_id)));
  }, [shortlistMatchesJob, shortlistSummary, sourceMode]);

  useEffect(() => {
    if (sourceMode !== "individual") return;
    setSelectedCandidateIds((current) => {
      const validIds = new Set((candidatesData?.items ?? []).map((candidate) => candidate.id));
      const next = new Set([...current].filter((candidateId) => validIds.has(candidateId)));
      return next.size === current.size ? current : next;
    });
  }, [candidatesData?.items, sourceMode]);

  const selectedPreviewCandidates = previewCandidates.filter((candidate) => selectedCandidateIds.has(candidate.id));
  const allSelected = previewCandidates.length > 0 && selectedCandidateIds.size === previewCandidates.length;
  const questionSets = questionSetsData?.items ?? [];

  const createMutation = useMutation({
    mutationFn: async (): Promise<CreateSummary> => {
      const expires = expiresInHours ? Number(expiresInHours) : undefined;
      if (sourceMode === "shortlist") {
        const result = await api.shortlist.dispatch.createInterviewInvitations(shortlistId, {
          candidate_profile_ids: [...selectedCandidateIds],
          job_id: jobId!,
          interview_question_set_id: questionSetId,
          expires_in_hours: expires,
          send_email: false,
        });
        return {
          createdCount: result.created_count,
          skippedCount: result.skipped_count,
          failedCount: result.failed_count,
        };
      }

      const responses = await Promise.allSettled(
        [...selectedCandidateIds].map((candidateId) =>
          api.interviewInvitations.create({
            job_id: jobId!,
            candidate_profile_id: candidateId,
            interview_question_set_id: questionSetId,
            expires_in_hours: expires,
            send_email: false,
          }),
        ),
      );

      const createdCount = responses.filter((response) => response.status === "fulfilled").length;
      const failed = responses.filter((response) => response.status === "rejected");

      if (createdCount === 0 && failed.length > 0) {
        throw failed[0].reason;
      }

      return {
        createdCount,
        skippedCount: 0,
        failedCount: failed.length,
      };
    },
    onSuccess: async (summary) => {
      await queryClient.invalidateQueries({ queryKey: ["interview-invitations", jobId] });
      if (shortlistId) {
        await queryClient.invalidateQueries({ queryKey: ["shortlist-dispatch-summary", shortlistId] });
        await queryClient.invalidateQueries({ queryKey: ["collection-dispatch", shortlistId] });
      }
      const suffix =
        summary.skippedCount || summary.failedCount
          ? ` (${summary.skippedCount} skipped, ${summary.failedCount} failed)`
          : "";
      toast.success(`Created ${summary.createdCount} interview link${summary.createdCount === 1 ? "" : "s"}${suffix}`);
      onOpenChange(false);
    },
    onError: (error: Error) => toast.error(error.message || "Failed to create interview links"),
  });

  function toggleCandidate(candidateId: string) {
    setSelectedCandidateIds((current) => {
      const next = new Set(current);
      if (next.has(candidateId)) {
        next.delete(candidateId);
      } else {
        next.add(candidateId);
      }
      return next;
    });
  }

  function toggleAll() {
    setSelectedCandidateIds(() => {
      if (allSelected) return new Set();
      return new Set(previewCandidates.map((candidate) => candidate.id));
    });
  }

  const createDisabled =
    !jobId ||
    !questionSetId ||
    selectedCandidateIds.size === 0 ||
    (sourceMode === "shortlist" && (!shortlistId || !shortlistMatchesJob));

  return (
    <Modal open={open} onOpenChange={onOpenChange}>
      <ModalContent size="large">
        <ModalHeader>
          <ModalTitle>Create interview links</ModalTitle>
          <ModalDescription>
            Choose a source, pick an interview question set, then review who should receive a draft link. Links are created now and not sent yet.
          </ModalDescription>
        </ModalHeader>

        {!jobId ? (
          <EmptyState
            heading="Select a job first"
            body="Pick the active job in the top bar before creating interview links."
          />
        ) : jobDescriptionLoading ? (
          <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] px-4 py-6 text-sm text-fg-muted">
            Loading job context...
          </div>
        ) : !jobDescription ? (
          <EmptyState
            heading="No active job description"
            body="Interview question sets depend on the active job description for the selected job."
          />
        ) : (
          <div className="space-y-5">
            <section className="space-y-2">
              <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">Source</span>
              <div className="grid gap-3 md:grid-cols-2">
                <SourceCard
                  active={sourceMode === "individual"}
                  icon={<UserRound size={16} strokeWidth={1.9} />}
                  title="Individual"
                  body="Pick one or more candidates from the selected job."
                  onClick={() => setSourceMode("individual")}
                />
                <SourceCard
                  active={sourceMode === "shortlist"}
                  icon={<Layers size={16} strokeWidth={1.9} />}
                  title="Shortlist"
                  body="Load candidates from a shortlist, selected by default, then remove anyone you do not want."
                  onClick={() => setSourceMode("shortlist")}
                />
              </div>
            </section>

            <section className="space-y-4">
              <label className="space-y-1.5">
                <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">Interview question set</span>
                <select
                  aria-label="Interview question set"
                  value={questionSetId}
                  onChange={(event) => setQuestionSetId(event.target.value)}
                  className={cn(
                    "h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3",
                    "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                  )}
                >
                  <option value="">
                    {questionSetsLoading ? "Loading question sets..." : "Select question set..."}
                  </option>
                  {questionSets.map((questionSet) => (
                    <option key={questionSet.id} value={questionSet.id}>
                      {formatQuestionSetLabel(questionSet)}
                    </option>
                  ))}
                </select>
              </label>

              {sourceMode === "shortlist" && (
                <label className="space-y-1.5">
                  <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">Shortlist</span>
                  <select
                    aria-label="Shortlist"
                    value={shortlistId}
                    onChange={(event) => setShortlistId(event.target.value)}
                    className={cn(
                      "h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3",
                      "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                    )}
                  >
                    <option value="">
                      {shortlistsLoading ? "Loading shortlists..." : "Select shortlist..."}
                    </option>
                    {(shortlistsData?.items ?? []).map((collection) => (
                      <option key={collection.id} value={collection.id}>
                        {collection.name} ({collection.item_count})
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <label className="space-y-1.5">
                <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">Expires in hours</span>
                <input
                  aria-label="Expires in hours"
                  inputMode="numeric"
                  value={expiresInHours}
                  onChange={(event) => setExpiresInHours(event.target.value)}
                  className={cn(
                    "h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3",
                    "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                  )}
                />
              </label>
            </section>

            {questionSets.length === 0 && !questionSetsLoading ? (
              <EmptyState
                heading="No interview question sets"
                body="Generate at least one interview question set for this job before creating links here."
              />
            ) : sourceMode === "shortlist" && shortlistId && !shortlistMatchesJob ? (
              <div className="rounded-[var(--radius-md)] border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
                This shortlist belongs to a different job. Choose a shortlist for the active job.
              </div>
            ) : (
              <section className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">Review candidates</span>
                    <p className="mt-1 text-sm text-fg-muted">
                      {sourceMode === "shortlist"
                        ? "Candidates from the selected shortlist are preselected. Untick anyone you want to exclude."
                        : "Choose one or more candidates from the selected job."}
                    </p>
                  </div>
                  {previewCandidates.length > 0 && (
                    <Button variant="ghost" size="sm" onClick={toggleAll}>
                      {allSelected ? "Clear all" : "Select all"}
                    </Button>
                  )}
                </div>

                {(sourceMode === "individual" && candidatesLoading) ||
                (sourceMode === "shortlist" && shortlistSummaryLoading) ? (
                  <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] px-4 py-6 text-sm text-fg-muted">
                    Loading candidates...
                  </div>
                ) : previewCandidates.length === 0 ? (
                  <EmptyState
                    heading="No candidates to review"
                    body={
                      sourceMode === "shortlist"
                        ? "Select a shortlist for this job to load its candidates."
                        : "Candidates in the selected job will appear here."
                    }
                  />
                ) : (
                  <div className="max-h-[320px] overflow-y-auto rounded-[var(--radius-lg)] border border-[color:var(--hairline)]">
                    {previewCandidates.map((candidate) => {
                      const checked = selectedCandidateIds.has(candidate.id);
                      return (
                        <label
                          key={candidate.id}
                          className="flex cursor-pointer items-start gap-3 border-b border-[color:var(--hairline)] px-4 py-3 last:border-b-0 hover:bg-[color:var(--hairline)]/30"
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleCandidate(candidate.id)}
                            className="mt-1 h-4 w-4 shrink-0 accent-[color:var(--accent)]"
                          />
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <span className="truncate text-sm font-medium text-fg">{candidate.fullName}</span>
                              {!candidate.email && (
                                <span className="rounded-full bg-[color:var(--hairline)] px-2 py-0.5 text-[11px] text-fg-muted">
                                  no email
                                </span>
                              )}
                            </div>
                            <p className="mt-0.5 text-xs text-fg-muted">
                              {candidate.subtitle || "Candidate"}
                              {candidate.email ? ` · ${candidate.email}` : ""}
                            </p>
                          </div>
                        </label>
                      );
                    })}
                  </div>
                )}

                <div className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated px-3 py-2 text-sm text-fg-muted">
                  <span className="font-medium text-fg">{selectedPreviewCandidates.length}</span> candidate
                  {selectedPreviewCandidates.length === 1 ? "" : "s"} selected for link creation.
                </div>
              </section>
            )}
          </div>
        )}

        <ModalFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            loading={createMutation.isPending}
            disabled={createDisabled}
            icon={<Users size={14} strokeWidth={1.75} />}
            onClick={() => createMutation.mutate()}
          >
            Create links
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
