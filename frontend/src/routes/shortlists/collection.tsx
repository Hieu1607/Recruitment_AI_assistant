import { api, type BatchActionResponse, type DispatchCandidateResponse } from "@/api";
import { parseAxiosError } from "@/api/errors";
import {
    Avatar,
    Badge,
    Button,
    EmptyState,
    Modal,
    ModalContent,
    ModalDescription,
    ModalFooter,
    ModalHeader,
    ModalTitle,
    Pagination,
    Skeleton,
    Tooltip
} from "@/components/ui";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    ArrowLeft,
    Check,
    Layers,
    Mail,
    MessageSquare,
    Mic2,
    Pencil,
    Trash2,
    Users,
    X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { toast } from "sonner";

// ── constants ─────────────────────────────────────────────────────────────────

const PAGE_SIZE = 20;

// ── helpers ───────────────────────────────────────────────────────────────────

function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d}d ago`;
  return new Date(iso).toLocaleDateString();
}

function truncateId(id: string): string {
  return `#${id.slice(0, 6)}…${id.slice(-4)}`;
}

// ── InlineRename ───────────────────────────────────────────────────────────────

function InlineRename({
  value,
  onSave,
  onCancel,
  loading,
  conflict,
}: {
  value: string;
  onSave: (name: string) => void;
  onCancel: () => void;
  loading: boolean;
  conflict: boolean;
}) {
  const [name, setName] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); inputRef.current?.select(); }, []);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <input
          ref={inputRef}
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && name.trim() && name.trim() !== value) onSave(name.trim());
            if (e.key === "Escape") onCancel();
          }}
          className={cn(
            "h-10 px-3 text-xl font-display font-medium rounded-[var(--radius-md)] w-full max-w-sm",
            "border bg-bg-elevated text-fg outline-none",
            conflict
              ? "border-danger focus:ring-2 focus:ring-danger/30"
              : "border-[color:var(--hairline-strong)] focus:border-accent focus:ring-2 focus:ring-accent/20",
          )}
        />
        <button
          type="button"
          disabled={!name.trim() || name.trim() === value || loading}
          onClick={() => name.trim() && name.trim() !== value && onSave(name.trim())}
          className={cn(
            "h-8 w-8 rounded-[var(--radius-sm)] flex items-center justify-center transition-colors",
            name.trim() && name.trim() !== value
              ? "text-accent hover:bg-accent/10"
              : "text-fg-subtle cursor-not-allowed",
          )}
          aria-label="Save"
        >
          <Check size={15} strokeWidth={2} />
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="h-8 w-8 rounded-[var(--radius-sm)] flex items-center justify-center text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors"
          aria-label="Cancel"
        >
          <X size={15} strokeWidth={2} />
        </button>
      </div>
      {conflict && (
        <p className="text-xs font-sans text-danger ml-1">
          A collection with this name already exists.
        </p>
      )}
    </div>
  );
}

// ── MemberRow ──────────────────────────────────────────────────────────────────

function MemberRow({
  candidate,
  selected,
  onToggleSelected,
  onRemove,
  removing,
}: {
  candidate: DispatchCandidateResponse;
  selected: boolean;
  onToggleSelected: (candidateId: string) => void;
  onRemove: (candidateId: string) => void;
  removing: boolean;
}) {
  const displayName = candidate.full_name || `Candidate ${truncateId(candidate.candidate_profile_id)}`;
  const skills = candidate.skills_text
    ? candidate.skills_text.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean).slice(0, 4)
    : [];
  const outreachLabel = candidate.outreach?.status?.replace("_", " ") ?? "not started";
  const interviewLabel = candidate.interview?.status ?? "not invited";

  return (
    <div className="group flex items-center gap-4 px-5 py-3.5 hairline-b last:border-b-0 hover:bg-[color:var(--hairline)]/30 transition-colors">
      <input
        type="checkbox"
        checked={selected}
        onChange={() => onToggleSelected(candidate.candidate_profile_id)}
        aria-label={`Select ${displayName}`}
        className="h-4 w-4 shrink-0 accent-[color:var(--accent)]"
      />
      <Avatar name={displayName} size="md" />

      <div className="flex-1 min-w-0">
        <p className="text-sm font-sans font-medium text-fg truncate">
          {displayName}
        </p>
        {candidate.current_job_title && (
          <p className="text-xs font-sans text-fg-muted mt-0.5 truncate">
            {candidate.current_job_title}
          </p>
        )}
      </div>

      {/* Skill chips */}
      <div className="hidden xl:flex items-center gap-1.5 flex-1 min-w-0">
        {skills.map((s) => (
          <span
            key={s}
            className="px-2 py-0.5 text-[11px] font-sans text-fg-muted rounded-[var(--radius-sm)] border border-[color:var(--hairline)] bg-bg shrink-0"
          >
            {s}
          </span>
        ))}
      </div>

      <div className="hidden lg:flex w-32 justify-end">
        <Badge variant={candidate.contact_status === "ready" ? "success" : "warning"} size="sm" dot={false}>
          {candidate.contact_status === "ready" ? "email ok" : "missing email"}
        </Badge>
      </div>

      <div className="hidden xl:flex w-32 justify-end">
        <Badge
          variant={candidate.outreach?.status === "sent" ? "success" : candidate.outreach?.status === "failed" ? "danger" : "neutral"}
          size="sm"
          dot={false}
        >
          {outreachLabel}
        </Badge>
      </div>

      <div className="hidden xl:flex w-32 justify-end">
        <Badge
          variant={candidate.interview?.completed_at ? "success" : candidate.interview ? "warning" : "neutral"}
          size="sm"
          dot={false}
        >
          {interviewLabel}
        </Badge>
      </div>

      {/* Remove button */}
      <button
        type="button"
        disabled={removing}
        onClick={() => onRemove(candidate.candidate_profile_id)}
        aria-label="Remove from collection"
        className={cn(
          "opacity-0 group-hover:opacity-100 transition-opacity",
          "h-7 w-7 rounded-[var(--radius-sm)] flex items-center justify-center",
          "text-fg-muted hover:text-danger hover:bg-danger/10 transition-colors",
          removing && "opacity-50 cursor-not-allowed",
        )}
      >
        <Trash2 size={13} strokeWidth={1.75} />
      </button>
    </div>
  );
}

function selectedCandidateList(
  candidates: DispatchCandidateResponse[],
  selectedIds: Set<string>,
) {
  return candidates.filter((candidate) => selectedIds.has(candidate.candidate_profile_id));
}

function BatchResultSummary({ result }: { result: BatchActionResponse | null }) {
  if (!result) return null;
  return (
    <div className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated px-3 py-2 text-sm text-fg-muted">
      Created {result.created_count}, skipped {result.skipped_count}, failed {result.failed_count}.
    </div>
  );
}

function OutreachDraftModal({
  open,
  onOpenChange,
  collectionId,
  candidates,
  onComplete,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  collectionId: string;
  candidates: DispatchCandidateResponse[];
  onComplete: () => void;
}) {
  const [subjectTemplate, setSubjectTemplate] = useState("Following up on {{job_title}}");
  const [bodyTemplate, setBodyTemplate] = useState("Hi {{candidate_name}},\n\nI reviewed your profile and would like to discuss the {{job_title}} role.\n\nBest regards,");
  const [result, setResult] = useState<BatchActionResponse | null>(null);

  useEffect(() => {
    if (open) setResult(null);
  }, [open]);

  const mutation = useMutation({
    mutationFn: () =>
      api.shortlist.dispatch.createOutreachDrafts(collectionId, {
        candidate_profile_ids: candidates.map((candidate) => candidate.candidate_profile_id),
        subject_template: subjectTemplate,
        body_template: bodyTemplate,
        content_source: "template",
      }),
    onSuccess: (response) => {
      setResult(response);
      toast.success(`Created ${response.created_count} outreach draft${response.created_count === 1 ? "" : "s"}`);
      onComplete();
    },
    onError: () => toast.error("Failed to create outreach drafts"),
  });

  return (
    <Modal open={open} onOpenChange={onOpenChange}>
      <ModalContent size="large">
        <ModalHeader>
          <ModalTitle>Create outreach drafts</ModalTitle>
          <ModalDescription>
            Review selected candidates before creating draft messages.
          </ModalDescription>
        </ModalHeader>
        <div className="space-y-4">
          <div className="max-h-40 overflow-y-auto rounded-[var(--radius-md)] border border-[color:var(--hairline)]">
            {candidates.map((candidate) => (
              <div key={candidate.candidate_profile_id} className="flex items-center justify-between gap-3 px-3 py-2 hairline-b last:border-b-0">
                <span className="text-sm text-fg">{candidate.full_name}</span>
                <span className="text-xs text-fg-muted">
                  {candidate.email || "missing email"}
                </span>
              </div>
            ))}
          </div>
          <label className="space-y-1.5 block">
            <span className="text-xs font-medium uppercase tracking-wide text-fg-muted">Subject template</span>
            <input
              value={subjectTemplate}
              onChange={(event) => setSubjectTemplate(event.target.value)}
              className="h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
            />
          </label>
          <label className="space-y-1.5 block">
            <span className="text-xs font-medium uppercase tracking-wide text-fg-muted">Body template</span>
            <textarea
              value={bodyTemplate}
              onChange={(event) => setBodyTemplate(event.target.value)}
              rows={7}
              className="w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2 text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
            />
          </label>
          <BatchResultSummary result={result} />
        </div>
        <ModalFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>Close</Button>
          <Button
            loading={mutation.isPending}
            disabled={!candidates.length || !subjectTemplate.trim() || !bodyTemplate.trim()}
            onClick={() => mutation.mutate()}
          >
            Create drafts
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

function InterviewInviteModal({
  open,
  onOpenChange,
  collectionId,
  jobId,
  candidates,
  onComplete,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  collectionId: string;
  jobId: string | null;
  candidates: DispatchCandidateResponse[];
  onComplete: () => void;
}) {
  const [templateId, setTemplateId] = useState("");
  const [expiresInHours, setExpiresInHours] = useState("72");
  const [result, setResult] = useState<BatchActionResponse | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["interview-templates", jobId],
    queryFn: () => api.interviewTemplates.list(jobId!),
    enabled: open && !!jobId,
  });

  useEffect(() => {
    if (open) {
      setTemplateId("");
      setExpiresInHours("72");
      setResult(null);
    }
  }, [open]);

  const templates = data?.items.filter((template) => template.status === "active") ?? [];
  const mutation = useMutation({
    mutationFn: () =>
      api.shortlist.dispatch.createInterviewInvitations(collectionId, {
        candidate_profile_ids: candidates.map((candidate) => candidate.candidate_profile_id),
        job_id: jobId!,
        interview_template_id: templateId,
        expires_in_hours: expiresInHours ? Number(expiresInHours) : null,
      }),
    onSuccess: (response) => {
      setResult(response);
      toast.success(`Created ${response.created_count} interview invitation${response.created_count === 1 ? "" : "s"}`);
      onComplete();
    },
    onError: () => toast.error("Failed to create interview invitations"),
  });

  return (
    <Modal open={open} onOpenChange={onOpenChange}>
      <ModalContent size="large">
        <ModalHeader>
          <ModalTitle>Send interview invites</ModalTitle>
          <ModalDescription>
            Select an active interview template and review the selected candidates.
          </ModalDescription>
        </ModalHeader>
        <div className="space-y-4">
          <div className="max-h-40 overflow-y-auto rounded-[var(--radius-md)] border border-[color:var(--hairline)]">
            {candidates.map((candidate) => (
              <div key={candidate.candidate_profile_id} className="flex items-center justify-between gap-3 px-3 py-2 hairline-b last:border-b-0">
                <span className="text-sm text-fg">{candidate.full_name}</span>
                <span className="text-xs text-fg-muted">
                  {candidate.email || "missing email"}
                </span>
              </div>
            ))}
          </div>
          <label className="space-y-1.5 block">
            <span className="text-xs font-medium uppercase tracking-wide text-fg-muted">Interview template</span>
            <select
              value={templateId}
              onChange={(event) => setTemplateId(event.target.value)}
              className="h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
            >
              <option value="">{isLoading ? "Loading templates..." : "Select template..."}</option>
              {templates.map((template) => (
                <option key={template.id} value={template.id}>{template.name}</option>
              ))}
            </select>
          </label>
          <label className="space-y-1.5 block">
            <span className="text-xs font-medium uppercase tracking-wide text-fg-muted">Expires in hours</span>
            <input
              value={expiresInHours}
              onChange={(event) => setExpiresInHours(event.target.value)}
              inputMode="numeric"
              className="h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
            />
          </label>
          {!jobId && (
            <p className="rounded-[var(--radius-md)] border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
              This shortlist has no job context, so interview invitations are blocked.
            </p>
          )}
          {jobId && !isLoading && templates.length === 0 && (
            <p className="rounded-[var(--radius-md)] border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
              Create an active interview template for this job before sending invites.
            </p>
          )}
          <BatchResultSummary result={result} />
        </div>
        <ModalFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>Close</Button>
          <Button
            loading={mutation.isPending}
            disabled={!jobId || !templateId || !candidates.length}
            onClick={() => mutation.mutate()}
          >
            Create invitations
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

// ── main component ─────────────────────────────────────────────────────────────

export default function ShortlistCollectionRoute() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();

  const [page, setPage] = useState(1);
  const [editing, setEditing] = useState(false);
  const [nameConflict, setNameConflict] = useState(false);
  const [removeTarget, setRemoveTarget] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [outreachOpen, setOutreachOpen] = useState(false);
  const [interviewOpen, setInterviewOpen] = useState(false);

  // ── collection data ────────────────────────────────────────────────────────

  const { data: collection, isLoading: colLoading, error: colError } = useQuery({
    queryKey: ["collection", id],
    queryFn: () => api.shortlist.collections.get(id!),
    enabled: !!id,
  });

  const { data: dispatchSummary, isLoading: dispatchLoading } = useQuery({
    queryKey: ["collection-dispatch", id],
    queryFn: () => api.shortlist.dispatch.summary(id!),
    enabled: !!id,
    staleTime: 30_000,
  });

  const candidates = useMemo(() => dispatchSummary?.candidates ?? [], [dispatchSummary]);
  const total = candidates.length;
  const pagedCandidates = candidates.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const selectedCandidates = useMemo(
    () => selectedCandidateList(candidates, selectedIds),
    [candidates, selectedIds],
  );
  const allPageSelected = pagedCandidates.length > 0 && pagedCandidates.every((candidate) => selectedIds.has(candidate.candidate_profile_id));

  useEffect(() => {
    setSelectedIds((current) => {
      const validIds = new Set(candidates.map((candidate) => candidate.candidate_profile_id));
      const next = new Set([...current].filter((candidateId) => validIds.has(candidateId)));
      return next.size === current.size ? current : next;
    });
  }, [candidates]);

  function toggleSelected(candidateId: string) {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(candidateId)) {
        next.delete(candidateId);
      } else {
        next.add(candidateId);
      }
      return next;
    });
  }

  function togglePageSelected() {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (allPageSelected) {
        pagedCandidates.forEach((candidate) => next.delete(candidate.candidate_profile_id));
      } else {
        pagedCandidates.forEach((candidate) => next.add(candidate.candidate_profile_id));
      }
      return next;
    });
  }

  // ── mutations ──────────────────────────────────────────────────────────────

  const renameMutation = useMutation({
    mutationFn: (name: string) => api.shortlist.collections.update(id!, { name }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["collection", id] });
      qc.invalidateQueries({ queryKey: ["collections"] });
      toast.success("Collection renamed");
      setEditing(false);
      setNameConflict(false);
    },
    onError: (err: unknown) => {
      if (parseAxiosError(err).status === 409) {
        setNameConflict(true);
      } else {
        toast.error("Failed to rename collection");
      }
    },
  });

  const removeMutation = useMutation({
    mutationFn: (candidateId: string) =>
      api.shortlist.items.remove(id!, candidateId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["collection-dispatch", id] });
      qc.invalidateQueries({ queryKey: ["collection", id] });
      qc.invalidateQueries({ queryKey: ["collections"] });
      toast.success("Candidate removed from collection");
      setRemoveTarget(null);
    },
    onError: () => toast.error("Failed to remove candidate"),
  });

  // ── loading / error states ─────────────────────────────────────────────────

  if (colLoading) {
    return (
      <div className="px-8 py-8">
        <Skeleton className="h-5 w-32 mb-6" />
        <Skeleton className="h-9 w-64 mb-2" />
        <Skeleton className="h-4 w-40 mb-8" />
        <div className="space-y-3">
          {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-14 w-full" />)}
        </div>
      </div>
    );
  }

  if (colError || !collection) {
    return (
      <div className="px-8 py-8">
        <EmptyState
          heading="Collection not found"
          body="This collection does not exist or has been deleted."
          action={{ label: "Back to shortlists", onClick: () => navigate(routes.shortlists) }}
        />
      </div>
    );
  }

  return (
    <div className="px-8 py-8 min-h-full">
      {/* Back nav */}
      <Link
        to={routes.shortlists}
        className="inline-flex items-center gap-1.5 text-sm font-sans text-fg-muted hover:text-fg transition-colors mb-6"
      >
        <ArrowLeft size={14} strokeWidth={2} />
        Shortlists
      </Link>

      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-6 mb-8">
        <div className="flex items-start gap-4 flex-1 min-w-0">
          <div className="h-11 w-11 rounded-[var(--radius-lg)] bg-accent/10 flex items-center justify-center shrink-0 mt-0.5">
            <Layers size={20} strokeWidth={1.75} className="text-accent" />
          </div>
          <div className="flex-1 min-w-0">
            {editing ? (
              <InlineRename
                value={collection.name}
                onSave={(name) => renameMutation.mutate(name)}
                onCancel={() => { setEditing(false); setNameConflict(false); }}
                loading={renameMutation.isPending}
                conflict={nameConflict}
              />
            ) : (
              <div className="flex items-center gap-2 group/title">
                <h1 className="font-display text-[2rem] font-medium text-fg leading-tight truncate">
                  {collection.name}
                </h1>
                <button
                  type="button"
                  onClick={() => setEditing(true)}
                  aria-label="Rename collection"
                  className={cn(
                    "opacity-0 group-hover/title:opacity-100 transition-opacity",
                    "h-7 w-7 rounded-[var(--radius-sm)] flex items-center justify-center",
                    "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors",
                  )}
                >
                  <Pencil size={13} strokeWidth={1.75} />
                </button>
              </div>
            )}

            <div className="flex items-center gap-3 mt-1.5 flex-wrap">
              <div className="flex items-center gap-1.5 text-sm font-sans text-fg-muted">
                <Users size={13} strokeWidth={1.75} />
                <span className="tabular-nums">
                  {collection.item_count} candidate{collection.item_count !== 1 ? "s" : ""}
                </span>
              </div>
              <span className="text-sm font-sans text-fg-subtle">
                Created {relativeTime(collection.created_at)}
              </span>
              {collection.source_query_turn_id && (
                <Tooltip
                  content={`Linked to query turn #${collection.source_query_turn_id.slice(0, 8)}…`}
                >
                  <div className="inline-flex items-center gap-1.5 text-[11px] font-sans text-accent border border-accent/20 rounded-full px-2 py-0.5 bg-accent/5 cursor-default">
                    <MessageSquare size={10} strokeWidth={1.75} />
                    From query
                  </div>
                </Tooltip>
              )}
            </div>
          </div>
        </div>
      </div>

      {selectedIds.size > 0 && (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated px-4 py-3">
          <div className="text-sm font-sans text-fg">
            <span className="font-medium tabular-nums">{selectedIds.size}</span> selected
            {dispatchSummary?.job && (
              <span className="ml-2 text-fg-muted">for {dispatchSummary.job.title}</span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              icon={<Mail size={14} strokeWidth={2} />}
              onClick={() => setOutreachOpen(true)}
            >
              Create outreach drafts
            </Button>
            <Button
              variant="secondary"
              size="sm"
              icon={<Mic2 size={14} strokeWidth={2} />}
              onClick={() => setInterviewOpen(true)}
            >
              Send interview invites
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setSelectedIds(new Set())}>
              Clear
            </Button>
          </div>
        </div>
      )}

      {/* ── Members table ── */}
      <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] overflow-hidden">
        {/* Table header */}
        <div className="flex items-center gap-4 px-5 py-3 bg-[color:var(--hairline)]/40 hairline-b">
          <input
            type="checkbox"
            checked={allPageSelected}
            onChange={togglePageSelected}
            aria-label="Select all candidates on this page"
            className="h-4 w-4 shrink-0 accent-[color:var(--accent)]"
          />
          <div className="w-8 shrink-0" />
          <div className="flex-1 text-[11px] font-sans font-semibold uppercase tracking-wider text-fg-muted">
            Candidate
          </div>
          <div className="hidden xl:block flex-1 text-[11px] font-sans font-semibold uppercase tracking-wider text-fg-muted">
            Skills
          </div>
          <div className="hidden md:block text-[11px] font-sans font-semibold uppercase tracking-wider text-fg-muted w-28 text-right">
            Contact
          </div>
          <div className="hidden xl:block text-[11px] font-sans font-semibold uppercase tracking-wider text-fg-muted w-32 text-right">
            Outreach
          </div>
          <div className="hidden xl:block text-[11px] font-sans font-semibold uppercase tracking-wider text-fg-muted w-32 text-right">
            Interview
          </div>
          <div className="w-7" />
        </div>

        {dispatchLoading ? (
          <div className="divide-y divide-[color:var(--hairline)]">
            {[0, 1, 2, 3, 4].map((i) => (
              <div key={i} className="flex items-center gap-4 px-5 py-3.5">
                <Skeleton width={32} height={32} rounded />
                <div className="flex-1 space-y-1.5">
                  <Skeleton className="h-3.5 w-40" />
                  <Skeleton className="h-3 w-60" />
                </div>
                <Skeleton className="h-3 w-20 hidden md:block" />
              </div>
            ))}
          </div>
        ) : pagedCandidates.length === 0 ? (
          <div className="py-16">
            <EmptyState
              icon={<Users size={28} strokeWidth={1.25} />}
              heading="No candidates in this collection"
              body="Add candidates to this collection from the Candidates list or via AI Chat shortlisting."
              action={{ label: "Go to Candidates", onClick: () => navigate(routes.candidates) }}
            />
          </div>
        ) : (
          <div>
            {pagedCandidates.map((candidate) => (
              <MemberRow
                key={candidate.candidate_profile_id}
                candidate={candidate}
                selected={selectedIds.has(candidate.candidate_profile_id)}
                onToggleSelected={toggleSelected}
                onRemove={setRemoveTarget}
                removing={removeMutation.isPending && removeTarget === candidate.candidate_profile_id}
              />
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {!dispatchLoading && total > PAGE_SIZE && (
        <div className="mt-4 hairline-t pt-3">
          <Pagination
            total={total}
            page={page}
            pageSize={PAGE_SIZE}
            onPageChange={setPage}
            onPageSizeChange={() => {}}
          />
        </div>
      )}

      {/* ── Remove confirmation modal ── */}
      <Modal open={!!removeTarget} onOpenChange={(o) => !o && setRemoveTarget(null)}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Remove candidate</ModalTitle>
            <ModalDescription>
              Remove candidate{" "}
              <span className="font-mono text-xs bg-[color:var(--hairline)] px-1 py-0.5 rounded">
                {removeTarget ? truncateId(removeTarget) : ""}
              </span>{" "}
              from this collection? They will not be deleted from the system.
            </ModalDescription>
          </ModalHeader>
          <ModalFooter>
            <Button variant="ghost" onClick={() => setRemoveTarget(null)}>Cancel</Button>
            <Button
              variant="danger"
              loading={removeMutation.isPending}
              onClick={() => removeTarget && removeMutation.mutate(removeTarget)}
            >
              Remove
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {id && (
        <OutreachDraftModal
          open={outreachOpen}
          onOpenChange={setOutreachOpen}
          collectionId={id}
          candidates={selectedCandidates}
          onComplete={() => {
            qc.invalidateQueries({ queryKey: ["collection-dispatch", id] });
            qc.invalidateQueries({ queryKey: ["outreach"] });
          }}
        />
      )}

      {id && (
        <InterviewInviteModal
          open={interviewOpen}
          onOpenChange={setInterviewOpen}
          collectionId={id}
          jobId={dispatchSummary?.job?.id ?? null}
          candidates={selectedCandidates}
          onComplete={() => {
            qc.invalidateQueries({ queryKey: ["collection-dispatch", id] });
            if (dispatchSummary?.job?.id) {
              qc.invalidateQueries({ queryKey: ["interview-invitations", dispatchSummary.job.id] });
            }
          }}
        />
      )}
    </div>
  );
}
