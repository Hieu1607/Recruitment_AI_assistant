import type { ChatResponse, ResumeResponse } from "@/api";
import { api } from "@/api";
import { parseAxiosError } from "@/api/errors";
import { Avatar } from "@/components/ui/avatar";
import { useSelectedJobId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
    Edit2,
    ExternalLink,
    MessageSquare,
    Plus,
    Send,
    Trash2,
    Users,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { toast } from "sonner";

// ── constants ─────────────────────────────────────────────────────────────────

const STORAGE_KEY = "recruit-ai-chat-sessions";
const PENDING_CHAT_SESSION_ID = "__pending__";

const PROMPT_SUGGESTIONS = [
  "Who has 5+ years of Python experience?",
  "Find candidates with machine learning skills",
  "Show candidates available in New York",
  "Who has a Master's degree in Computer Science?",
];

// ── types ─────────────────────────────────────────────────────────────────────

interface LocalSession {
  id: string;
  title: string;
  chatSessionId: string | null;
  createdAt: string;
}

interface ChatMsg {
  id: string;
  role: "human" | "ai";
  content: string;
  candidatesInScope?: number;
}

// ── localStorage helpers ──────────────────────────────────────────────────────

function loadSessions(): LocalSession[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function saveSessions(sessions: LocalSession[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function newSessionId() {
  return crypto.randomUUID();
}

// ── helpers ───────────────────────────────────────────────────────────────────

function fileToName(f: string) {
  return (
    f
      .replace(/\.pdf$/i, "")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .trim() || f
  );
}

function groupSessionsByDate(sessions: LocalSession[]) {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const weekStart = new Date(todayStart);
  weekStart.setDate(weekStart.getDate() - 7);

  const today: LocalSession[] = [];
  const week: LocalSession[] = [];
  const older: LocalSession[] = [];

  for (const s of sessions) {
    const d = new Date(s.createdAt);
    if (d >= todayStart) today.push(s);
    else if (d >= weekStart) week.push(s);
    else older.push(s);
  }
  return { today, week, older };
}

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDateLabel(iso: string) {
  const d = new Date(iso);
  const now = new Date();
  const isToday =
    d.getDate() === now.getDate() &&
    d.getMonth() === now.getMonth() &&
    d.getFullYear() === now.getFullYear();
  if (isToday) return `Today, ${formatTime(iso)}`;
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

// ── sub-components ────────────────────────────────────────────────────────────

function AiAvatar() {
  return (
    <span
      className="inline-flex items-center justify-center shrink-0 h-8 w-8 rounded-full bg-accent text-accent-fg text-[10px] font-semibold font-sans select-none"
      aria-label="AI"
    >
      AI
    </span>
  );
}

function CandidateCards({
  count,
  candidates,
}: {
  count: number;
  candidates: ResumeResponse[];
}) {
  const shown = candidates.slice(0, 4);
  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-center gap-1.5 text-xs text-fg-muted font-sans">
        <Users size={12} strokeWidth={2} />
        <span>
          Found <span className="font-medium text-fg tabular-nums">{count}</span> candidate
          {count !== 1 ? "s" : ""} in scope
        </span>
      </div>
      {shown.length > 0 && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {shown.map((c) => {
            const name = fileToName(c.original_file_name);
            return (
              <a
                key={c.id}
                href={`/candidates`}
                className={cn(
                  "flex items-center gap-2.5 shrink-0 px-3 py-2 rounded-[var(--radius-md)]",
                  "border border-[color:var(--hairline)] bg-bg-elevated",
                  "hover:border-[color:var(--hairline-strong)] hover:shadow-[var(--shadow-sm)]",
                  "transition-all duration-[var(--duration-fast)]"
                )}
              >
                <Avatar name={name} size="sm" />
                <div>
                  <p className="text-xs font-sans font-medium text-fg leading-none mb-0.5">
                    {name.split(" ")[0]}
                  </p>
                  <p className="text-[10px] text-fg-subtle font-sans leading-none">
                    {c.upload_status}
                  </p>
                </div>
                <ExternalLink size={10} strokeWidth={1.75} className="text-fg-subtle ml-1" />
              </a>
            );
          })}
        </div>
      )}
      <a
        href="/candidates"
        className="inline-flex items-center gap-1 text-xs text-accent hover:underline font-sans"
      >
        View all candidates →
      </a>
    </div>
  );
}

function MessageBubble({
  msg,
  candidates,
}: {
  msg: ChatMsg;
  candidates: ResumeResponse[];
}) {
  if (msg.role === "human") {
    return (
      <div className="flex justify-end">
        <div
          className={cn(
            "max-w-[70%] px-4 py-2.5 rounded-[var(--radius-lg)] rounded-tr-[var(--radius-sm)]",
            "bg-[color:var(--hairline)] text-fg text-sm font-sans leading-relaxed"
          )}
        >
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 items-start">
      <AiAvatar />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-sans text-fg leading-relaxed whitespace-pre-wrap">
          {msg.content}
        </p>
        {msg.candidatesInScope !== undefined && msg.candidatesInScope > 0 && (
          <CandidateCards count={msg.candidatesInScope} candidates={candidates} />
        )}
      </div>
    </div>
  );
}

function SessionItem({
  session,
  isActive,
  onSelect,
  onRename,
  onDelete,
}: {
  session: LocalSession;
  isActive: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.focus();
  }, [editing]);

  function commitRename() {
    const t = draft.trim();
    if (t && t !== session.title) onRename(t);
    else setDraft(session.title);
    setEditing(false);
  }

  return (
    <div
      className={cn(
        "group flex items-center gap-2 px-2.5 py-2 rounded-[var(--radius-md)] cursor-pointer",
        "transition-colors duration-[var(--duration-fast)]",
        isActive
          ? "bg-accent text-accent-fg"
          : "hover:bg-[color:var(--hairline)] text-fg"
      )}
      onClick={() => !editing && onSelect()}
    >
      <MessageSquare
        size={13}
        strokeWidth={1.75}
        className={cn("shrink-0", isActive ? "text-accent-fg/70" : "text-fg-muted")}
      />
      {editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitRename}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitRename();
            if (e.key === "Escape") { setDraft(session.title); setEditing(false); }
          }}
          onClick={(e) => e.stopPropagation()}
          className={cn(
            "flex-1 min-w-0 bg-transparent text-sm font-sans outline-none border-b",
            isActive ? "border-accent-fg/40 text-accent-fg" : "border-accent text-fg"
          )}
        />
      ) : (
        <span
          className={cn(
            "flex-1 min-w-0 text-sm font-sans truncate",
            isActive ? "text-accent-fg" : "text-fg"
          )}
        >
          {session.title}
        </span>
      )}
      {!editing && (
        <div
          className={cn(
            "hidden group-hover:flex items-center gap-0.5 shrink-0",
            isActive && "flex"
          )}
        >
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); setEditing(true); }}
            className={cn(
              "p-1 rounded transition-colors",
              isActive
                ? "hover:bg-accent-fg/10 text-accent-fg/70"
                : "hover:bg-[color:var(--hairline-strong)] text-fg-muted"
            )}
            aria-label="Rename"
          >
            <Edit2 size={11} strokeWidth={2} />
          </button>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            className={cn(
              "p-1 rounded transition-colors",
              isActive
                ? "hover:bg-danger/20 text-accent-fg/70 hover:text-danger"
                : "hover:bg-[color:var(--hairline-strong)] text-fg-muted hover:text-danger"
            )}
            aria-label="Delete"
          >
            <Trash2 size={11} strokeWidth={2} />
          </button>
        </div>
      )}
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export default function ChatRoute() {
  const { sessionId: urlSessionId } = useParams<{ sessionId?: string }>();
  const navigate = useNavigate();
  const selectedJobId = useSelectedJobId();

  const [sessions, setSessions] = useState<LocalSession[]>(() => loadSessions());
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const activeSession = sessions.find((s) => s.id === urlSessionId) ?? null;

  const { data: resumeData } = useQuery({
    queryKey: ["resumes-chat"],
    queryFn: () => (selectedJobId ? api.jobs.resumes.list(selectedJobId, { limit: 100, upload_status: "processed" }) : Promise.resolve({ items: [], total: 0 })),
    staleTime: 60_000,
  });
  const processedCandidates = resumeData?.items ?? [];

  // ── load session history when active session changes ──────────────────────

  useEffect(() => {
    setHistoryLoading(false);
    if (!activeSession) {
      setMessages([]);
    }
  }, [activeSession?.id, activeSession?.chatSessionId]);

  // ── auto-scroll to bottom ─────────────────────────────────────────────────

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // ── send message mutation ─────────────────────────────────────────────────

  const sendMutation = useMutation<ChatResponse, Error, string>({
    mutationFn: (text: string) =>
      api.jobs.chat.send(selectedJobId!, {
        message: text,
        session_id:
          activeSession?.chatSessionId &&
          activeSession.chatSessionId !== PENDING_CHAT_SESSION_ID
            ? activeSession.chatSessionId
            : undefined,
        candidate_limit: 500,
      }),
    onSuccess: (res: ChatResponse, text: string) => {
      const aiMsg: ChatMsg = {
        id: `ai-${Date.now()}`,
        role: "ai",
        content: res.answer,
        candidatesInScope: res.candidates_in_scope,
      };
      setMessages((prev) => [...prev, aiMsg]);

      if (activeSession) {
        setSessions((prev) => {
          const updated = prev.map((s) => {
            if (s.id !== activeSession.id) return s;
            const isUntitled = s.title === "New Conversation" || s.title.startsWith("New Conversation");
            return {
              ...s,
              chatSessionId: res.session_id,
              title: isUntitled
                ? text.slice(0, 48) + (text.length > 48 ? "…" : "")
                : s.title,
            };
          });
          saveSessions(updated);
          return updated;
        });
      }
    },
    onError: (err: unknown) => {
      const apiError = parseAxiosError(err);
      if (apiError.status === 404) {
        toast.info("Session expired — started new conversation");
        if (activeSession) {
          setSessions((prev) => {
            const updated = prev.map((s) =>
              s.id === activeSession.id ? { ...s, chatSessionId: null } : s
            );
            saveSessions(updated);
            return updated;
          });
        }
        setMessages((prev) => prev.slice(0, -1));
      } else {
        toast.error(apiError.detail || "Failed to send message — please try again");
        setMessages((prev) => prev.slice(0, -1));
      }
    },
  });

  // ── handlers ──────────────────────────────────────────────────────────────

  function handleSend(text?: string) {
    const msg = (text ?? input).trim();
    if (!msg || sendMutation.isPending) return;
    if (!selectedJobId) {
      toast.error("Select a job before using chat.");
      return;
    }
    localStorage.setItem("recruiter_onboarding_chatted", "true");

    if (!activeSession) {
      const newSession: LocalSession = {
        id: newSessionId(),
        title: msg.slice(0, 48) + (msg.length > 48 ? "…" : ""),
        chatSessionId: PENDING_CHAT_SESSION_ID,
        createdAt: new Date().toISOString(),
      };
      const updated = [newSession, ...sessions];
      setSessions(updated);
      saveSessions(updated);
      navigate(`/chat/${newSession.id}`);
      const userMsg: ChatMsg = { id: `u-${Date.now()}`, role: "human", content: msg };
      setMessages([userMsg]);
      setInput("");
      sendMutation.mutate(msg);
      return;
    }

    const userMsg: ChatMsg = { id: `u-${Date.now()}`, role: "human", content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    sendMutation.mutate(msg);
  }

  function handleNewSession() {
    navigate("/chat");
    setMessages([]);
    setInput("");
    setTimeout(() => textareaRef.current?.focus(), 50);
  }

  function handleSelectSession(session: LocalSession) {
    navigate(`/chat/${session.id}`);
  }

  function handleRenameSession(session: LocalSession, title: string) {
    setSessions((prev) => {
      const updated = prev.map((s) => (s.id === session.id ? { ...s, title } : s));
      saveSessions(updated);
      return updated;
    });
  }

  function handleDeleteSession(session: LocalSession) {
    if (session.chatSessionId && session.chatSessionId !== PENDING_CHAT_SESSION_ID) {
      api.chat.deleteSession(session.chatSessionId).catch(() => {});
    }
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== session.id);
      saveSessions(updated);
      return updated;
    });
    if (session.id === urlSessionId) {
      navigate("/chat");
      setMessages([]);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const grouped = groupSessionsByDate(sessions);
  const isPending = sendMutation.isPending;

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div
      className="flex overflow-hidden"
      style={{ height: "calc(100vh - var(--topbar-height))" }}
    >
      {/* ── Sessions sidebar ── */}
      <div className="w-64 shrink-0 border-r border-[color:var(--hairline)] flex flex-col bg-bg-sidebar">
        {/* New conversation */}
        <div className="p-3 border-b border-[color:var(--hairline)]">
          <button
            type="button"
            onClick={handleNewSession}
            className={cn(
              "w-full flex items-center justify-between gap-2 px-3 py-2 rounded-[var(--radius-md)]",
              "text-sm font-sans text-fg border border-[color:var(--hairline)]",
              "bg-bg-elevated hover:border-[color:var(--hairline-strong)] hover:shadow-[var(--shadow-sm)]",
              "transition-all duration-[var(--duration-fast)]"
            )}
          >
            <span className="font-medium">New Conversation</span>
            <Plus size={14} strokeWidth={2} className="text-fg-muted shrink-0" />
          </button>
        </div>

        {/* Session list */}
        <div className="flex-1 overflow-y-auto p-2 space-y-4">
          {sessions.length === 0 && (
            <p className="px-2 py-3 text-xs text-fg-subtle font-sans text-center">
              No conversations yet
            </p>
          )}

          {grouped.today.length > 0 && (
            <div>
              <p className="px-2.5 mb-1 text-[10px] font-sans font-semibold uppercase tracking-widest text-fg-subtle">
                Today
              </p>
              <div className="space-y-0.5">
                {grouped.today.map((s) => (
                  <SessionItem
                    key={s.id}
                    session={s}
                    isActive={s.id === urlSessionId}
                    onSelect={() => handleSelectSession(s)}
                    onRename={(t) => handleRenameSession(s, t)}
                    onDelete={() => handleDeleteSession(s)}
                  />
                ))}
              </div>
            </div>
          )}

          {grouped.week.length > 0 && (
            <div>
              <p className="px-2.5 mb-1 text-[10px] font-sans font-semibold uppercase tracking-widest text-fg-subtle">
                Previous 7 Days
              </p>
              <div className="space-y-0.5">
                {grouped.week.map((s) => (
                  <SessionItem
                    key={s.id}
                    session={s}
                    isActive={s.id === urlSessionId}
                    onSelect={() => handleSelectSession(s)}
                    onRename={(t) => handleRenameSession(s, t)}
                    onDelete={() => handleDeleteSession(s)}
                  />
                ))}
              </div>
            </div>
          )}

          {grouped.older.length > 0 && (
            <div>
              <p className="px-2.5 mb-1 text-[10px] font-sans font-semibold uppercase tracking-widest text-fg-subtle">
                Older
              </p>
              <div className="space-y-0.5">
                {grouped.older.map((s) => (
                  <SessionItem
                    key={s.id}
                    session={s}
                    isActive={s.id === urlSessionId}
                    onSelect={() => handleSelectSession(s)}
                    onRename={(t) => handleRenameSession(s, t)}
                    onDelete={() => handleDeleteSession(s)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Chat panel ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Messages area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {historyLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 rounded-full border-2 border-accent border-t-transparent animate-spin" />
                <p className="text-sm text-fg-muted font-sans">Loading history…</p>
              </div>
            </div>
          ) : messages.length === 0 ? (
            /* Empty state */
            <div className="flex flex-col items-center justify-center h-full px-8 gap-8">
              <div className="text-center">
                <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-accent/10 mb-5">
                  <MessageSquare size={28} strokeWidth={1.5} className="text-accent" />
                </div>
                <h1 className="font-display text-3xl font-medium text-fg mb-3">
                  Ask anything about your candidates
                </h1>
                <p className="text-sm text-fg-muted font-sans max-w-sm">
                  Search, compare, and analyse your candidate pool using natural language.
                </p>
              </div>

              <div className="flex flex-wrap justify-center gap-2 max-w-lg">
                {PROMPT_SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => handleSend(s)}
                    className={cn(
                      "px-3 py-1.5 rounded-full text-xs font-sans text-fg",
                      "border border-[color:var(--hairline)] bg-bg-elevated",
                      "hover:border-[color:var(--hairline-strong)] hover:shadow-[var(--shadow-sm)]",
                      "transition-all duration-[var(--duration-fast)]"
                    )}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Message list */
            <div className="max-w-3xl mx-auto px-6 py-6 space-y-6">
              {activeSession && (
                <div className="flex justify-center">
                  <span className="text-xs text-fg-subtle font-sans bg-bg-sidebar px-3 py-1 rounded-full border border-[color:var(--hairline)]">
                    {formatDateLabel(activeSession.createdAt)}
                  </span>
                </div>
              )}
              {messages.map((msg) => (
                <MessageBubble
                  key={msg.id}
                  msg={msg}
                  candidates={processedCandidates}
                />
              ))}
              {isPending && (
                <div className="flex gap-3 items-start">
                  <AiAvatar />
                  <div className="flex items-center gap-1.5 py-2">
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        className="h-1.5 w-1.5 rounded-full bg-fg-subtle"
                        style={{
                          animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
                        }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="border-t border-[color:var(--hairline)] bg-bg px-6 py-4">
          <div className="max-w-3xl mx-auto">
            <div
              className={cn(
                "flex items-end gap-3 rounded-[var(--radius-lg)] border",
                "bg-bg-elevated px-4 py-3 transition-all duration-[var(--duration-fast)]",
                isPending
                  ? "border-[color:var(--hairline)] opacity-60"
                  : "border-[color:var(--hairline-strong)] focus-within:border-accent"
              )}
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message the recruiter assistant…"
                rows={1}
                disabled={isPending}
                className={cn(
                  "flex-1 resize-none bg-transparent text-sm font-sans text-fg",
                  "placeholder:text-fg-subtle outline-none leading-relaxed",
                  "max-h-36 overflow-y-auto"
                )}
                style={{ fieldSizing: "content" } as React.CSSProperties}
              />
              <button
                type="button"
                onClick={() => handleSend()}
                disabled={!input.trim() || isPending}
                className={cn(
                  "shrink-0 h-8 w-8 rounded-[var(--radius-md)] flex items-center justify-center",
                  "transition-colors duration-[var(--duration-fast)]",
                  input.trim() && !isPending
                    ? "bg-accent text-accent-fg hover:bg-accent-hover"
                    : "bg-[color:var(--hairline)] text-fg-subtle cursor-not-allowed"
                )}
                aria-label="Send message"
              >
                <Send size={14} strokeWidth={2} />
              </button>
            </div>
            <p className="text-[11px] text-fg-subtle font-sans text-center mt-2">
              AI Assistant may generate inaccurate information. Please verify critical data.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
