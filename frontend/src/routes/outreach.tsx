import type { OutreachResponse, SentStatus, ResumeResponse } from "@/api";
import { api } from "@/api";
import { Badge, Button, EmptyState, Skeleton } from "@/components/ui";
import { cn } from "@/lib/cn";
import { useQuery } from "@tanstack/react-query";
import { Inbox, Mail } from "lucide-react";
import { useCallback } from "react";
import { useSearchParams } from "react-router";

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

  return { folder, candidate, messageId, setFolder, setCandidate, setMessage };
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
  candidates: ResumeResponse[];
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
          {candidates.map((r) => {
            const label = r.original_file_name.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ").trim() || r.original_file_name;
            return (
              <option key={r.id} value={r.id}>
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

function DetailPanelPlaceholder() {
  return (
    <div className="flex-1 flex flex-col min-w-0 bg-bg">
      <div className="flex-1 flex items-center justify-center">
        <EmptyState
          icon={<Mail size={28} strokeWidth={1.25} />}
          heading="Select a message"
          body="Pick any message from the list to read or edit it."
        />
      </div>
    </div>
  );
}

export default function OutreachRoute() {
  const { folder, candidate, messageId, setFolder, setCandidate, setMessage } = useOutreachParams();

  // Fetch candidates for the filter combobox
  const { data: resumeData } = useQuery({
    queryKey: ["resumes-outreach"],
    queryFn: () => api.upload.list({ limit: 200 }),
    staleTime: 60_000,
  });
  const candidates = resumeData?.items ?? [];

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
    staleTime: 30_000,
  });
  const messages = listData?.items ?? [];

  // Fetch per-folder counts (4 parallel queries, each for one status)
  const { data: allData } = useQuery({ queryKey: ["outreach-count", "all", candidate ?? null], queryFn: () => api.outreach.list({ candidate_profile_id: candidate, limit: 1 }), staleTime: 30_000 });
  const { data: notSentData } = useQuery({ queryKey: ["outreach-count", "not_sent", candidate ?? null], queryFn: () => api.outreach.list({ sent_status: "not_sent", candidate_profile_id: candidate, limit: 1 }), staleTime: 30_000 });
  const { data: sentData } = useQuery({ queryKey: ["outreach-count", "sent", candidate ?? null], queryFn: () => api.outreach.list({ sent_status: "sent", candidate_profile_id: candidate, limit: 1 }), staleTime: 30_000 });
  const { data: failedData } = useQuery({ queryKey: ["outreach-count", "failed", candidate ?? null], queryFn: () => api.outreach.list({ sent_status: "failed", candidate_profile_id: candidate, limit: 1 }), staleTime: 30_000 });

  const counts = {
    all: allData?.total ?? 0,
    not_sent: notSentData?.total ?? 0,
    sent: sentData?.total ?? 0,
    failed: failedData?.total ?? 0,
  };

  return (
    <div className="flex overflow-hidden" style={{ height: "calc(100vh - var(--topbar-height))" }}>
      <FolderSidebar
        folder={folder}
        counts={counts}
        candidate={candidate}
        candidates={candidates}
        onFolderChange={setFolder}
        onCandidateChange={setCandidate}
        onNewMessage={() => {/* Plan 02 will wire this */}}
      />
      <MessageList
        messages={messages}
        isLoading={isLoading}
        selectedId={messageId}
        onSelect={(id) => setMessage(id)}
        folderLabel={folderDef.label}
        total={listData?.total ?? 0}
      />
      <DetailPanelPlaceholder />
    </div>
  );
}
