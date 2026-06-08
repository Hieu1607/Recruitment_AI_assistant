import type { CandidateProfileResponse, ContentSource, OutreachResponse, SentStatus } from "@/api";
import { api } from "@/api";
import { parseAxiosError } from "@/api/errors";
import { Badge, Button, EmptyState, Modal, ModalContent, ModalDescription, ModalFooter, ModalHeader, ModalTitle, Skeleton } from "@/components/ui";
import { useAuthStore, useSelectedJobId, useUserId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Inbox, Mail } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router";
import { toast } from "sonner";

type FolderKey = "all" | "not_sent" | "sent" | "failed";

const FOLDERS: { key: FolderKey; label: string; sentStatus: SentStatus | undefined; badgeVariant: "neutral" | "warning" | "success" | "danger" }[] = [
  { key: "all",      label: "All",      sentStatus: undefined,   badgeVariant: "neutral"  },
  { key: "not_sent", label: "Not sent", sentStatus: "not_sent",  badgeVariant: "warning"  },
  { key: "sent",     label: "Sent",     sentStatus: "sent",      badgeVariant: "success"  },
  { key: "failed",   label: "Failed",   sentStatus: "failed",    badgeVariant: "danger"   },
];

const STATUS_VARIANT: Record<SentStatus, "neutral" | "success" | "danger"> = {
  not_sent: "neutral",
  sent:     "success",
  failed:   "danger",
};

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

function useOutreachParams() {
  const [params, setParams] = useSearchParams();
  const folder = (params.get("folder") ?? "all") as FolderKey;
  const candidate = params.get("candidate") ?? undefined;
  const messageId = params.get("message") ?? undefined;

  const setFolder = useCallback((key: FolderKey) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("folder", key);
      next.delete("message");
      return next;
    });
  }, [setParams]);

  const setCandidate = useCallback((id: string | undefined) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      if (id) next.set("candidate", id); else next.delete("candidate");
      next.delete("message");
      return next;
    });
  }, [setParams]);

  const setMessage = useCallback((id: string | undefined) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      if (id) next.set("message", id); else next.delete("message");
      return next;
    });
  }, [setParams]);

  return { params, setParams, folder, candidate, messageId, setFolder, setCandidate, setMessage };
}

function FolderSidebar({
  folder,
  counts,
  candidate,
  onFolderChange,
  onCandidateChange,
  onNewMessage,
  candidates,
}: {
  folder: FolderKey;
  counts: Record<FolderKey, number>;
  candidate: string | undefined;
  onFolderChange: (key: FolderKey) => void;
  onCandidateChange: (id: string | undefined) => void;
  onNewMessage: () => void;
  candidates: CandidateProfileResponse[];
}) {
  return (
    <div className="w-[200px] shrink-0 border-r border-[color:var(--hairline)] flex flex-col bg-bg-sidebar">
      <div className="px-4 pt-4 pb-3 border-b border-[color:var(--hairline)]">
        <p className="font-display text-sm font-semibold text-fg mb-2">Outreach</p>
        <Button variant="primary" size="sm" className="w-full" onClick={onNewMessage}>
          + New message
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto py-2">
        {FOLDERS.map((f) => {
          const isActive = f.key === folder;
          return (
            <button
              key={f.key}
              onClick={() => onFolderChange(f.key)}
              className={cn(
                "h-10 w-full flex items-center justify-between px-3 text-sm font-sans transition-colors duration-[120ms]",
                isActive
                  ? "relative font-medium text-fg bg-[color:var(--hairline)] before:absolute before:left-0 before:top-0 before:bottom-0 before:w-[3px] before:bg-accent before:rounded-r"
                  : "text-fg-muted hover:bg-[color:var(--hairline)]"
              )}
            >
              <span>{f.label}</span>
              <Badge variant={f.badgeVariant} size="sm" dot={false}>
                {counts[f.key] ?? 0}
              </Badge>
            </button>
          );
        })}
      </div>
      <div className="px-3 py-3 border-t border-[color:var(--hairline)]">
        <p className="text-[11px] font-sans text-fg-subtle mb-1.5 font-medium uppercase tracking-wide">Filter by candidate</p>
        <select
          value={candidate ?? ""}
          onChange={(e) => onCandidateChange(e.target.value || undefined)}
          className="w-full text-sm bg-bg border border-[color:var(--hairline)] rounded-[var(--radius-md)] px-2 py-1.5 outline-none focus:ring-2 focus:ring-accent/50"
        >
          <option value="">All candidates</option>
          {candidates.map((candidate) => {
            const label = candidate.full_name || candidate.current_job_title || candidate.email || candidate.id;
            return (
              <option key={candidate.id} value={candidate.id}>
                {label}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  );
}

function MessageListItem({
  message,
  isSelected,
  onClick,
}: {
  message: OutreachResponse;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "h-[72px] w-full flex flex-col justify-center px-3 py-4 border-b border-[color:var(--hairline)] text-left transition-colors duration-[120ms]",
        isSelected
          ? "bg-[color:var(--hairline-strong)] relative before:absolute before:left-0 before:top-0 before:bottom-0 before:w-[3px] before:bg-accent before:rounded-r"
          : "hover:bg-[color:var(--hairline)]"
      )}
    >
      <div className="flex justify-between items-center w-full mb-0.5">
        <span className="text-sm font-sans font-semibold text-fg truncate">
          {message.candidate_full_name ?? "Unknown"}
        </span>
        <Badge variant={STATUS_VARIANT[message.sent_status]} size="sm" dot={false}>
          {message.sent_status.replace("_", " ")}
        </Badge>
      </div>
      <span className="text-[13px] font-sans text-fg truncate block w-full mb-0.5">
        {message.subject}
      </span>
      <div className="flex justify-between items-center gap-2 w-full">
        <span className="text-xs font-sans text-fg-muted truncate flex-1">
          {message.body.slice(0, 80)}
        </span>
        <span className="text-[11px] font-sans text-fg-subtle shrink-0 tabular-nums">
          {relativeTime(message.created_at)}
        </span>
      </div>
    </button>
  );
}

function MessageList({
  messages,
  isLoading,
  selectedId,
  onSelect,
  folderLabel,
  total,
}: {
  messages: OutreachResponse[];
  isLoading: boolean;
  selectedId?: string;
  onSelect: (id: string) => void;
  folderLabel: string;
  total: number;
}) {
  return (
    <div className="w-[320px] shrink-0 border-r border-[color:var(--hairline)] flex flex-col bg-bg">
      <div className="px-3 py-3 border-b border-[color:var(--hairline)] flex items-center gap-2">
        <span className="text-[13px] font-sans font-semibold text-fg">{folderLabel}</span>
        <span className="text-[12px] font-sans text-fg-muted tabular-nums">{total}</span>
      </div>
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <>
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-[72px] w-full border-b border-[color:var(--hairline)] rounded-none" />
            ))}
          </>
        ) : messages.length === 0 ? (
          <div className="py-12">
            <EmptyState
              icon={<Inbox size={24} strokeWidth={1.25} />}
              heading="No messages here yet"
              body="Messages you compose will appear in this folder. Start with + New message."
            />
          </div>
        ) : (
          messages.map((m) => (
            <MessageListItem
              key={m.id}
              message={m}
              isSelected={m.id === selectedId}
              onClick={() => onSelect(m.id)}
            />
          ))
        )}
      </div>
    </div>
  );
}

function MessageDetailPanel({
  messageId,
  onClose,
}: {
  messageId: string | undefined;
  onClose: () => void;
}) {
  const qc = useQueryClient();

  // Fetch the selected message
  const { data: message, isLoading, isError } = useQuery({
    queryKey: ["outreach-message", messageId],
    queryFn: () => api.outreach.get(messageId!),
    enabled: !!messageId,
    staleTime: 30_000,
  });

  // Local edit state
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  // Sync local state when message loads or messageId changes
  useEffect(() => {
    if (message) {
      setSubject(message.subject);
      setBody(message.body);
      setDeleteConfirm(false);
    }
  }, [message]);

  const isDirty = message ? (subject !== message.subject || body !== message.body) : false;

  // Edit save mutation
  const editMutation = useMutation({
    mutationFn: () => api.outreach.update(messageId!, { subject: subject.trim(), body: body.trim() }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["outreach"] });
      qc.invalidateQueries({ queryKey: ["outreach-message", messageId] });
      toast.success("Message saved");
    },
    onError: (err: unknown) => {
      if (parseAxiosError(err).status === 404) {
        toast.error("Message no longer exists");
        onClose();
      } else {
        toast.error("Something went wrong. Please try again.");
      }
    },
  });

  // Send email mutation
  const sendMutation = useMutation({
    mutationFn: () => api.outreach.send(messageId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["outreach"] });
      qc.invalidateQueries({ queryKey: ["outreach-count"] });
      qc.invalidateQueries({ queryKey: ["outreach-message", messageId] });
      toast.success("Email queued for sending");
    },
    onError: (err: unknown) => {
      if (parseAxiosError(err).status === 404) {
        toast.error("Message no longer exists");
        onClose();
      } else {
        toast.error("Could not queue email. Check Google/Gmail setup and candidate email.");
      }
    },
  });

  // Delete mutation — optimistic remove
  const deleteMutation = useMutation({
    mutationFn: () => api.outreach.remove(messageId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["outreach"] });
      qc.invalidateQueries({ queryKey: ["outreach-count"] });
      onClose();
      toast.success("Message deleted");
    },
    onError: () => {
      toast.error("Something went wrong. Please try again.");
    },
  });

  // GET 404: message was deleted elsewhere; show empty state + toast once
  useEffect(() => {
    if (isError) {
      toast.error("This message no longer exists");
      onClose();
    }
  }, [isError, onClose]);

  // Empty state when nothing is selected
  if (!messageId) {
    return (
      <div className="flex-1 flex flex-col min-w-0 bg-bg items-center justify-center">
        <EmptyState
          icon={<Mail size={28} strokeWidth={1.25} />}
          heading="Select a message"
          body="Pick any message from the list to read or edit it."
        />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex-1 flex flex-col min-w-0 bg-bg items-center justify-center">
        <EmptyState
          icon={<Mail size={28} strokeWidth={1.25} />}
          heading="Message not found"
          body="This message may have been deleted. Return to the message list."
        />
      </div>
    );
  }

  if (isLoading || !message) {
    return (
      <div className="flex-1 flex flex-col min-w-0 bg-bg">
        <div className="h-16 border-b border-[color:var(--hairline)] bg-bg-elevated px-6 flex items-center">
          <div className="h-5 w-48 bg-[color:var(--hairline)] rounded animate-pulse" />
        </div>
        <div className="p-6 space-y-4">
          {[0, 1, 2, 3].map((i) => <div key={i} className="h-4 bg-[color:var(--hairline)] rounded animate-pulse" style={{ width: `${80 - i * 15}%` }} />)}
        </div>
      </div>
    );
  }

  const canMarkSent = message.sent_status === "not_sent";

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-bg">
      {/* Header strip — 64px */}
      <div className="h-16 shrink-0 border-b border-[color:var(--hairline)] bg-bg-elevated flex items-center justify-between gap-3 px-6">
        <span className="font-display text-base font-semibold text-fg truncate flex-1">
          {message.subject}
        </span>
        <div className="flex items-center gap-2 shrink-0">
          {canMarkSent && (
            <Button
              variant="secondary"
              size="sm"
              loading={sendMutation.isPending}
              onClick={() => sendMutation.mutate()}
            >
              Send email
            </Button>
          )}
          {!deleteConfirm ? (
            <Button variant="danger" size="sm" onClick={() => setDeleteConfirm(true)}>
              Delete
            </Button>
          ) : (
            <div className="flex items-center gap-2 animate-in slide-in-from-top-2 duration-200">
              <span className="text-xs font-sans text-fg-muted">Delete this message permanently?</span>
              <Button
                variant="danger"
                size="sm"
                loading={deleteMutation.isPending}
                onClick={() => deleteMutation.mutate()}
              >
                Delete
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setDeleteConfirm(false)}>
                Cancel
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Body area */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Meta row */}
        <div className="flex items-center gap-3 mb-4">
          <span className="font-display text-sm font-semibold text-fg">
            {message.candidate_full_name ?? "Unknown candidate"}
          </span>
          <Badge variant="neutral" size="sm" dot={false}>
            {message.content_source === "ai_draft" ? "AI Draft" : "Template"}
          </Badge>
          <Badge variant={STATUS_VARIANT[message.sent_status]} size="sm" dot={false}>
            {message.sent_status.replace("_", " ")}
          </Badge>
          <span className="text-xs font-sans text-fg-subtle tabular-nums">
            {relativeTime(message.created_at)}
          </span>
          {message.sent_at && (
            <span className="text-xs font-sans text-fg-subtle tabular-nums">
              Sent {relativeTime(message.sent_at)}
            </span>
          )}
        </div>
        <div className="border-b border-[color:var(--hairline)] mb-4" />

        {/* Subject input */}
        <div className="mb-4">
          <label className="block text-[11px] font-sans font-medium text-fg-subtle uppercase tracking-wide mb-1.5">
            Subject
          </label>
          <input
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            className={cn(
              "w-full h-9 px-3 text-[13px] font-sans text-fg bg-bg rounded-[var(--radius-md)]",
              "border border-[color:var(--hairline)] focus:border-[color:var(--hairline-strong)]",
              "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent outline-none",
            )}
          />
        </div>

        {/* Body textarea */}
        <div className="mb-4">
          <label className="block text-[11px] font-sans font-medium text-fg-subtle uppercase tracking-wide mb-1.5">
            Body
          </label>
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={12}
            className={cn(
              "w-full px-3 py-2 text-[14px] font-sans text-fg bg-bg",
              "border-none outline-none resize-none leading-[1.6]",
              "placeholder:text-fg-subtle",
            )}
            placeholder="Write your message here…"
            style={{ fieldSizing: "content" } as React.CSSProperties}
          />
        </div>

        {/* Save button — only when dirty */}
        {isDirty && (
          <div className="flex justify-end">
            <Button
              variant="primary"
              size="sm"
              loading={editMutation.isPending}
              onClick={() => editMutation.mutate()}
            >
              Save
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

function ComposeModal({
  candidates,
  onClose,
}: {
  candidates: CandidateProfileResponse[];
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const userId = useUserId();
  const [candidateId, setCandidateId] = useState("");
  const [contentSource, setContentSource] = useState<ContentSource>("ai_draft");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [discardWarning, setDiscardWarning] = useState(false);
  const [candidateError, setCandidateError] = useState(false);

  const hasContent = !!subject.trim() || !!body.trim() || !!candidateId;
  const canSave = !!candidateId && !!subject.trim() && !!body.trim();

  const composeMutation = useMutation({
    mutationFn: () =>
      api.outreach.create({
        candidate_profile_id: candidateId,
        created_by_user_id: userId ?? "",
        content_source: contentSource,
        subject: subject.trim(),
        body: body.trim(),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["outreach"] });
      qc.invalidateQueries({ queryKey: ["outreach-count"] });
      toast.success("Message saved to drafts");
      onClose();
    },
    onError: (err: unknown) => {
      if (parseAxiosError(err).status === 404) {
        setCandidateError(true);
      } else {
        toast.error("Something went wrong. Please try again.");
      }
    },
  });

  function handleDiscard() {
    if (hasContent) {
      setDiscardWarning(true);
    } else {
      onClose();
    }
  }

  function candidateLabel(candidate: CandidateProfileResponse): string {
    return candidate.full_name || candidate.current_job_title || candidate.email || candidate.id;
  }

  return (
    <Modal open onOpenChange={(open) => !open && handleDiscard()}>
      <ModalContent className="w-[560px] max-h-[80vh] overflow-y-auto rounded-[var(--radius-lg)]">
        <ModalHeader>
          <ModalTitle>New message</ModalTitle>
          <ModalDescription>Compose an outreach message for a candidate.</ModalDescription>
        </ModalHeader>

        <div className="mt-4 space-y-4 px-1">
          {/* Candidate selector */}
          <div>
            <label className="block text-[11px] font-sans font-medium text-fg-subtle uppercase tracking-wide mb-1.5">
              Candidate
            </label>
            <select
              value={candidateId}
              onChange={(e) => { setCandidateId(e.target.value); setCandidateError(false); }}
              className={cn(
                "w-full h-9 px-3 text-sm font-sans text-fg bg-bg rounded-[var(--radius-md)]",
                "border focus:outline focus:outline-2 focus:outline-offset-1 outline-none",
                candidateError
                  ? "border-danger focus:outline-danger"
                  : "border-[color:var(--hairline)] focus:outline-accent",
              )}
            >
              <option value="">Select a candidate…</option>
              {candidates.map((r) => (
                <option key={r.id} value={r.id}>
                  {candidateLabel(r)}
                </option>
              ))}
            </select>
            {candidateError && (
              <p className="mt-1 text-xs font-sans text-danger">
                Candidate not found. Please select a different candidate.
              </p>
            )}
          </div>

          {/* Content source toggle */}
          <div>
            <label className="block text-[11px] font-sans font-medium text-fg-subtle uppercase tracking-wide mb-1.5">
              Content source
            </label>
            <div className="flex rounded-[var(--radius-md)] border border-[color:var(--hairline)] overflow-hidden">
              {(["ai_draft", "template"] as ContentSource[]).map((src) => (
                <button
                  key={src}
                  type="button"
                  onClick={() => setContentSource(src)}
                  className={cn(
                    "flex-1 px-4 py-2 text-sm font-sans transition-colors",
                    contentSource === src
                      ? "bg-accent text-accent-fg font-medium"
                      : "bg-bg text-fg-muted hover:bg-[color:var(--hairline)]",
                  )}
                >
                  {src === "ai_draft" ? "AI Draft" : "Template"}
                </button>
              ))}
            </div>
          </div>

          {/* Subject */}
          <div>
            <label className="block text-[11px] font-sans font-medium text-fg-subtle uppercase tracking-wide mb-1.5">
              Subject
            </label>
            <input
              type="text"
              value={subject}
              onChange={(e) => setSubject(e.target.value.slice(0, 255))}
              maxLength={255}
              placeholder="Subject line…"
              className={cn(
                "w-full h-9 px-3 text-[13px] font-sans text-fg bg-bg rounded-[var(--radius-md)]",
                "border border-[color:var(--hairline)] focus:border-[color:var(--hairline-strong)]",
                "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent outline-none",
              )}
            />
          </div>

          {/* Body */}
          <div>
            <label className="block text-[11px] font-sans font-medium text-fg-subtle uppercase tracking-wide mb-1.5">
              Body
            </label>
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              placeholder="Write your message here…"
              className={cn(
                "w-full px-3 py-2 text-sm font-sans text-fg bg-bg rounded-[var(--radius-md)]",
                "border border-[color:var(--hairline)] focus:border-[color:var(--hairline-strong)]",
                "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent outline-none",
                "resize-none leading-[1.6] min-h-[120px] max-h-[300px] overflow-y-auto",
              )}
              style={{ fieldSizing: "content" } as React.CSSProperties}
            />
          </div>

          {/* Discard warning — inline, not a modal */}
          {discardWarning && (
            <div className="flex items-center gap-2 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated px-3 py-2">
              <span className="text-xs font-sans text-fg-muted flex-1">Discard draft? Your changes will be lost.</span>
              <Button variant="danger" size="sm" onClick={onClose}>Discard</Button>
              <Button variant="ghost" size="sm" onClick={() => setDiscardWarning(false)}>Keep editing</Button>
            </div>
          )}
        </div>

        <ModalFooter>
          <Button variant="ghost" size="sm" onClick={handleDiscard}>
            Discard draft
          </Button>
          <Button
            variant="primary"
            size="sm"
            disabled={!canSave}
            loading={composeMutation.isPending}
            onClick={() => composeMutation.mutate()}
          >
            Save draft
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

function GmailOnboardingPanel({
  isPending,
  onConnect,
}: {
  isPending: boolean;
  onConnect: () => void;
}) {
  return (
    <div className="flex overflow-hidden" style={{ height: "calc(100vh - var(--topbar-height))" }}>
      <div className="flex-1 bg-bg-elevated p-6 md:p-8">
        <div className="flex h-full items-center justify-center rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg">
          <div className="max-w-md px-6 py-10 text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-[color:var(--hairline)] text-accent">
              <Mail size={24} strokeWidth={1.5} />
            </div>
            <p className="font-display text-2xl font-semibold text-fg">Connect Gmail to start outreach</p>
            <p className="mt-3 text-sm font-sans leading-6 text-fg-muted">
              Outreach needs Gmail permission before you can view drafts, edit messages, or send candidate emails.
            </p>
            <Button
              variant="primary"
              size="sm"
              className="mt-6"
              loading={isPending}
              disabled={isPending}
              onClick={onConnect}
            >
              Connect Gmail
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function OutreachRoute() {
  const { params, setParams, folder, candidate, messageId, setFolder, setCandidate, setMessage } = useOutreachParams();
  const [composeOpen, setComposeOpen] = useState(false);
  const [isConnectPending, setIsConnectPending] = useState(false);
  const selectedJobId = useSelectedJobId();
  const user = useAuthStore((s) => s.user);
  const setUser = useAuthStore((s) => s.setUser);
  const needsGmailOnboarding = user?.gmail_connected === false;

  useEffect(() => {
    const gmailConnected = params.get("gmail_connected") === "1";
    const gmailConsentDenied = params.get("error") === "gmail_consent_denied";

    if (!gmailConnected && !gmailConsentDenied) return;

    if (gmailConsentDenied) {
      toast("Gmail connection was canceled. You can try again anytime.");
    }

    if (gmailConnected && user?.gmail_connected !== true) {
      if (user) {
        setUser({
          id: user.id,
          email: user.email,
          display_name: user.display_name,
          gmail_connected: true,
        });
      }
      void api.auth.me().then(setUser).catch(() => {
        toast.error("Gmail connected, but your account state could not be refreshed.");
      });
    }

    setParams((prev) => {
      const next = new URLSearchParams(prev);
      next.delete("gmail_connected");
      if (next.get("error") === "gmail_consent_denied") {
        next.delete("error");
      }
      return next;
    }, { replace: true });
  }, [params, setParams, setUser, user?.gmail_connected]);

  const handleConnectGmail = useCallback(async () => {
    if (isConnectPending) return;

    setIsConnectPending(true);
    try {
      const redirectParams = new URLSearchParams(window.location.search);
      redirectParams.delete("gmail_connected");
      redirectParams.delete("error");
      const redirect = redirectParams.toString()
        ? `${window.location.pathname}?${redirectParams.toString()}`
        : window.location.pathname;
      const url = await api.auth.getGoogleConnectGmailUrl(redirect);
      window.location.assign(url);
    } catch {
      toast.error("Could not start Gmail connection. Please try again.");
      setIsConnectPending(false);
    }
  }, [isConnectPending]);

  // Fetch candidates for the filter combobox
  const { data: candidateData } = useQuery({
    queryKey: ["outreach-candidates", selectedJobId],
    queryFn: () =>
      selectedJobId
        ? api.jobs.listCandidates(selectedJobId)
        : Promise.resolve({ items: [], total: 0 }),
    enabled: !!selectedJobId && !needsGmailOnboarding,
    staleTime: 60_000,
  });
  const candidates = candidateData?.items ?? [];

  // Derive sent_status filter from folder
  const folderDef = FOLDERS.find((f) => f.key === folder) ?? FOLDERS[0];

  // Fetch messages for the current folder + candidate filter
  const { data: listData, isLoading } = useQuery({
    queryKey: ["outreach", folder, candidate ?? null],
    queryFn: () =>
      api.outreach.list({
        sent_status: folderDef.sentStatus,
        candidate_profile_id: candidate,
        limit: 100,
      }),
    enabled: !needsGmailOnboarding,
    staleTime: 30_000,
  });
  const messages = listData?.items ?? [];

  // Fetch per-folder counts (4 parallel queries, each for one status)
  const { data: allData } = useQuery({ queryKey: ["outreach-count", "all", candidate ?? null], queryFn: () => api.outreach.list({ candidate_profile_id: candidate, limit: 1 }), enabled: !needsGmailOnboarding, staleTime: 30_000 });
  const { data: notSentData } = useQuery({ queryKey: ["outreach-count", "not_sent", candidate ?? null], queryFn: () => api.outreach.list({ sent_status: "not_sent", candidate_profile_id: candidate, limit: 1 }), enabled: !needsGmailOnboarding, staleTime: 30_000 });
  const { data: sentData } = useQuery({ queryKey: ["outreach-count", "sent", candidate ?? null], queryFn: () => api.outreach.list({ sent_status: "sent", candidate_profile_id: candidate, limit: 1 }), enabled: !needsGmailOnboarding, staleTime: 30_000 });
  const { data: failedData } = useQuery({ queryKey: ["outreach-count", "failed", candidate ?? null], queryFn: () => api.outreach.list({ sent_status: "failed", candidate_profile_id: candidate, limit: 1 }), enabled: !needsGmailOnboarding, staleTime: 30_000 });

  const counts = {
    all: allData?.total ?? 0,
    not_sent: notSentData?.total ?? 0,
    sent: sentData?.total ?? 0,
    failed: failedData?.total ?? 0,
  };

  if (needsGmailOnboarding) {
    return <GmailOnboardingPanel isPending={isConnectPending} onConnect={handleConnectGmail} />;
  }

  return (
    <div className="flex overflow-hidden" style={{ height: "calc(100vh - var(--topbar-height))" }}>
      <FolderSidebar
        folder={folder}
        counts={counts}
        candidate={candidate}
        candidates={candidates}
        onFolderChange={setFolder}
        onCandidateChange={setCandidate}
        onNewMessage={() => setComposeOpen(true)}
      />
      <MessageList
        messages={messages}
        isLoading={isLoading}
        selectedId={messageId}
        onSelect={(id) => setMessage(id)}
        folderLabel={folderDef.label}
        total={listData?.total ?? 0}
      />
      <MessageDetailPanel messageId={messageId} onClose={() => setMessage(undefined)} />
      {composeOpen && <ComposeModal candidates={candidates} onClose={() => setComposeOpen(false)} />}
    </div>
  );
}
