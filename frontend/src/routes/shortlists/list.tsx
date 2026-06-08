import { api, type CollectionResponse, type SessionResponse, type TurnResponse } from "@/api";
import { parseAxiosError } from "@/api/errors";
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
    Tooltip,
} from "@/components/ui";
import { useUserId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    ChevronDown,
    ChevronRight,
    Layers,
    MessageSquare,
    MoreHorizontal,
    Plus,
    Users,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import { toast } from "sonner";

// ── constants ─────────────────────────────────────────────────────────────────

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

// ── CollectionCard ─────────────────────────────────────────────────────────────

function CollectionCard({
  col,
  onRename,
  onDelete,
}: {
  col: CollectionResponse;
  onRename: (col: CollectionResponse) => void;
  onDelete: (col: CollectionResponse) => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    if (menuOpen) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [menuOpen]);

  return (
    <div
      className={cn(
        "group relative rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
        "bg-bg-elevated p-5 hover:shadow-[var(--shadow-md)] transition-shadow duration-200",
      )}
    >
      {/* Menu button */}
      <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity" ref={menuRef}>
        <button
          type="button"
          onClick={() => setMenuOpen((v) => !v)}
          className={cn(
            "h-7 w-7 rounded-[var(--radius-sm)] flex items-center justify-center",
            "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors",
          )}
          aria-label="More options"
        >
          <MoreHorizontal size={14} strokeWidth={1.75} />
        </button>
        {menuOpen && (
          <div
            className={cn(
              "absolute right-0 top-full mt-1 z-20 w-36 py-1",
              "rounded-[var(--radius-md)] bg-bg-elevated",
              "border border-[color:var(--hairline)] shadow-[var(--shadow-md)]",
            )}
          >
            <Link
              to={routes.shortlistCollection(col.id)}
              onClick={() => setMenuOpen(false)}
              className="block px-3 py-2 text-sm font-sans text-fg hover:bg-[color:var(--hairline)] transition-colors"
            >
              View
            </Link>
            <button
              type="button"
              onClick={() => { setMenuOpen(false); onRename(col); }}
              className="w-full text-left px-3 py-2 text-sm font-sans text-fg hover:bg-[color:var(--hairline)] transition-colors"
            >
              Rename
            </button>
            <button
              type="button"
              onClick={() => { setMenuOpen(false); onDelete(col); }}
              className="w-full text-left px-3 py-2 text-sm font-sans text-danger hover:bg-[color:var(--hairline)] transition-colors"
            >
              Delete
            </button>
          </div>
        )}
      </div>

      {/* Card body */}
      <Link to={routes.shortlistCollection(col.id)} className="block">
        <div className="flex items-start gap-3 mb-3">
          <div className="h-9 w-9 rounded-[var(--radius-md)] bg-accent/10 flex items-center justify-center shrink-0">
            <Layers size={16} strokeWidth={1.75} className="text-accent" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-display text-base font-medium text-fg leading-snug line-clamp-2 pr-6">
              {col.name}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center gap-1.5 text-xs font-sans text-fg-muted">
            <Users size={11} strokeWidth={1.75} />
            <span className="tabular-nums">{col.item_count} candidate{col.item_count !== 1 ? "s" : ""}</span>
          </div>
          <span className="text-xs font-sans text-fg-subtle tabular-nums">
            {relativeTime(col.created_at)}
          </span>
        </div>

        {col.source_query_turn_id && (
          <Tooltip content={`Created from query turn #${col.source_query_turn_id.slice(0, 8)}…`} side="bottom">
            <div className="mt-3 inline-flex items-center gap-1.5 text-[11px] font-sans text-accent border border-accent/20 rounded-full px-2 py-0.5 bg-accent/5 cursor-default">
              <MessageSquare size={10} strokeWidth={1.75} />
              From query
            </div>
          </Tooltip>
        )}
      </Link>
    </div>
  );
}

// ── RenameModal ────────────────────────────────────────────────────────────────

function RenameModal({
  col,
  onClose,
  onSuccess,
}: {
  col: CollectionResponse;
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [name, setName] = useState(col.name);
  const [conflict, setConflict] = useState(false);
  const qc = useQueryClient();

  const renameMutation = useMutation({
    mutationFn: () => api.shortlist.collections.update(col.id, { name: name.trim() }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["collections"] });
      toast.success("Collection renamed");
      onSuccess();
      onClose();
    },
    onError: (err: unknown) => {
      if (parseAxiosError(err).status === 409) {
        setConflict(true);
      } else {
        toast.error("Failed to rename collection");
      }
    },
  });

  return (
    <Modal open onOpenChange={(o) => !o && onClose()}>
      <ModalContent>
        <ModalHeader>
          <ModalTitle>Rename collection</ModalTitle>
        </ModalHeader>
        <div className="mt-2">
          <input
            type="text"
            value={name}
            onChange={(e) => { setName(e.target.value); setConflict(false); }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && name.trim() && name.trim() !== col.name) renameMutation.mutate();
              if (e.key === "Escape") onClose();
            }}
            autoFocus
            className={cn(
              "w-full h-9 px-3 text-sm font-sans rounded-[var(--radius-md)]",
              "border bg-bg text-fg",
              conflict
                ? "border-danger focus:outline-danger"
                : "border-[color:var(--hairline-strong)] focus:outline-accent",
              "focus:outline focus:outline-2 focus:outline-offset-1 outline-none",
            )}
          />
          {conflict && (
            <p className="mt-1.5 text-xs font-sans text-danger">
              A collection with this name already exists. Please choose a different name.
            </p>
          )}
        </div>
        <ModalFooter>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button
            variant="primary"
            loading={renameMutation.isPending}
            disabled={!name.trim() || name.trim() === col.name}
            onClick={() => renameMutation.mutate()}
          >
            Rename
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

// ── NewCollectionModal ─────────────────────────────────────────────────────────

function NewCollectionModal({
  onClose,
  sourceTurnId,
}: {
  onClose: (created?: CollectionResponse) => void;
  sourceTurnId?: string;
}) {
  const [name, setName] = useState("");
  const qc = useQueryClient();
  const userId = useUserId();

  const createMutation = useMutation({
    mutationFn: () =>
      api.shortlist.collections.create({
        created_by_user_id: userId ?? "",
        name: name.trim(),
        source_query_turn_id: sourceTurnId,
      }),
    onSuccess: (col) => {
      qc.invalidateQueries({ queryKey: ["collections"] });
      toast.success("Collection created");
      onClose(col);
    },
    onError: () => toast.error("Failed to create collection"),
  });

  return (
    <Modal open onOpenChange={(o) => !o && onClose()}>
      <ModalContent>
        <ModalHeader>
          <ModalTitle>New collection</ModalTitle>
          {sourceTurnId && (
            <ModalDescription>
              This collection will be linked to the selected query turn.
            </ModalDescription>
          )}
        </ModalHeader>
        <div className="mt-2">
          <input
            type="text"
            placeholder="Collection name…"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && name.trim()) createMutation.mutate();
              if (e.key === "Escape") onClose();
            }}
            autoFocus
            className={cn(
              "w-full h-9 px-3 text-sm font-sans rounded-[var(--radius-md)]",
              "border border-[color:var(--hairline-strong)] bg-bg text-fg",
              "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent outline-none",
            )}
          />
        </div>
        <ModalFooter>
          <Button variant="ghost" onClick={() => onClose()}>Cancel</Button>
          <Button
            variant="primary"
            loading={createMutation.isPending}
            disabled={!name.trim()}
            onClick={() => createMutation.mutate()}
          >
            Create
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

// ── TurnItem ───────────────────────────────────────────────────────────────────

function TurnItem({
  turn,
  onCreateCollection,
}: {
  turn: TurnResponse;
  onCreateCollection: (turnId: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="py-4 hairline-b last:border-b-0">
      {/* User question */}
      <p className="font-display text-sm italic text-fg leading-snug mb-2">
        "{turn.user_question}"
      </p>

      {/* AI answer */}
      <p className="text-sm font-sans text-fg-muted leading-relaxed line-clamp-3 mb-3">
        {turn.answer_text}
      </p>

      {/* Footer row */}
      <div className="flex items-center gap-3 flex-wrap">
        {turn.matched_count !== null && turn.matched_count !== undefined && (
          <Badge variant="neutral" size="sm" dot={false}>
            {turn.matched_count} matched
          </Badge>
        )}
        <span className="text-[11px] font-sans text-fg-subtle tabular-nums">
          {relativeTime(turn.created_at)}
        </span>

        {turn.matched_count !== null && turn.matched_count !== undefined && turn.matched_count > 0 && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="inline-flex items-center gap-1 text-[11px] font-sans text-accent hover:underline"
          >
            {expanded ? <ChevronDown size={11} strokeWidth={2} /> : <ChevronRight size={11} strokeWidth={2} />}
            {expanded ? "Hide" : "Show"} matched candidates
          </button>
        )}

        <button
          type="button"
          onClick={() => onCreateCollection(turn.id)}
          className="ml-auto inline-flex items-center gap-1 text-[11px] font-sans text-fg-muted hover:text-fg transition-colors"
        >
          <Plus size={11} strokeWidth={2} />
          Create collection from this turn
        </button>
      </div>

      {/* Expanded candidate IDs */}
      {expanded && turn.matched_candidate_ids && turn.matched_candidate_ids.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {turn.matched_candidate_ids.map((id) => (
            <span
              key={id}
              className="inline-block px-2 py-0.5 text-[11px] font-mono text-fg-muted rounded-[var(--radius-sm)] border border-[color:var(--hairline)] bg-bg"
            >
              #{id.slice(0, 8)}…
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── QueryHistoryTab ────────────────────────────────────────────────────────────

function QueryHistoryTab({
  onCreateCollection,
}: {
  onCreateCollection: (turnId: string) => void;
}) {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const userId = useUserId();

  const { data: sessionsData, isLoading: sessionsLoading } = useQuery({
    queryKey: ["shortlist-sessions"],
    queryFn: () => api.shortlist.sessions.list({ user_id: userId ?? "", limit: 100 }),
    staleTime: 30_000,
  });
  const sessions = sessionsData?.items ?? [];

  const selected = selectedSessionId ?? sessions[0]?.id ?? null;

  const { data: turnsData, isLoading: turnsLoading } = useQuery({
    queryKey: ["shortlist-turns", selected],
    queryFn: () => api.shortlist.turns.listForSession(selected!, { limit: 100 }),
    enabled: !!selected,
    staleTime: 30_000,
  });
  const turns = turnsData?.items ?? [];

  const selectedSession = sessions.find((s) => s.id === selected);

  if (sessionsLoading) {
    return (
      <div className="grid grid-cols-3 gap-6 h-[560px]">
        <div className="space-y-2">
          {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-14 w-full" />)}
        </div>
        <div className="col-span-2 space-y-4">
          {[0, 1, 2].map((i) => <Skeleton key={i} className="h-24 w-full" />)}
        </div>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <EmptyState
        icon={<MessageSquare size={28} strokeWidth={1.25} />}
        heading="No query sessions yet"
        body="Start a conversation in AI Chat to build query history that you can convert to collections."
        action={{ label: "Open AI Chat", onClick: () => { window.location.href = "/chat"; } }}
      />
    );
  }

  return (
    <div className="grid grid-cols-3 gap-6" style={{ minHeight: 480 }}>
      {/* Session list (left) */}
      <div className="border-r border-[color:var(--hairline)] pr-4 overflow-y-auto space-y-0.5">
        {sessions.map((s) => (
          <SessionItem
            key={s.id}
            session={s}
            isActive={s.id === selected}
            onClick={() => setSelectedSessionId(s.id)}
          />
        ))}
      </div>

      {/* Turn timeline (right) */}
      <div className="col-span-2 overflow-y-auto">
        {selectedSession && (
          <div className="mb-4">
            <h3 className="font-display text-base font-medium text-fg">
              {selectedSession.session_title ?? "Untitled session"}
            </h3>
            <p className="text-xs font-sans text-fg-muted mt-0.5">
              {selectedSession.turn_count} turn{selectedSession.turn_count !== 1 ? "s" : ""} ·{" "}
              {relativeTime(selectedSession.created_at)}
            </p>
          </div>
        )}
        {turnsLoading ? (
          <div className="space-y-4">
            {[0, 1, 2].map((i) => <Skeleton key={i} className="h-24 w-full" />)}
          </div>
        ) : turns.length === 0 ? (
          <p className="text-sm font-sans text-fg-muted py-8 text-center">No turns in this session.</p>
        ) : (
          <div>
            {turns.map((t) => (
              <TurnItem key={t.id} turn={t} onCreateCollection={onCreateCollection} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SessionItem({
  session,
  isActive,
  onClick,
}: {
  session: SessionResponse;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full text-left px-3 py-2.5 rounded-[var(--radius-md)] transition-colors",
        isActive
          ? "bg-accent text-accent-fg"
          : "hover:bg-[color:var(--hairline)] text-fg",
      )}
    >
      <p className={cn("text-sm font-sans font-medium truncate", isActive ? "text-accent-fg" : "text-fg")}>
        {session.session_title ?? "Untitled session"}
      </p>
      <p className={cn("text-[11px] font-sans tabular-nums", isActive ? "text-accent-fg/70" : "text-fg-subtle")}>
        {session.turn_count} turn{session.turn_count !== 1 ? "s" : ""} · {relativeTime(session.created_at)}
      </p>
    </button>
  );
}

// ── main component ─────────────────────────────────────────────────────────────

type Tab = "collections" | "history";

export default function ShortlistsListRoute() {
  const qc = useQueryClient();
  const userId = useUserId();
  const [activeTab, setActiveTab] = useState<Tab>("collections");
  const [newColOpen, setNewColOpen] = useState(false);
  const [newColTurnId, setNewColTurnId] = useState<string | undefined>();
  const [renameTarget, setRenameTarget] = useState<CollectionResponse | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<CollectionResponse | null>(null);

  const { data: collectionsData, isLoading } = useQuery({
    queryKey: ["collections"],
    queryFn: () => api.shortlist.collections.list({ user_id: userId ?? "", limit: 100 }),
    staleTime: 30_000,
  });
  const collections = collectionsData?.items ?? [];

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.shortlist.collections.remove(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["collections"] });
      toast.success("Collection deleted");
      setDeleteTarget(null);
    },
    onError: () => toast.error("Failed to delete collection"),
  });

  function handleCreateFromTurn(turnId: string) {
    setNewColTurnId(turnId);
    setNewColOpen(true);
    setActiveTab("collections");
  }

  return (
    <div className="px-8 py-8 min-h-full">
      {/* Page header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-display text-[2rem] font-medium text-fg leading-tight">Shortlists</h1>
          <p className="text-sm text-fg-muted mt-1 font-sans">
            Saved candidate collections and query session history
          </p>
        </div>
        <Button
          variant="primary"
          icon={<Plus size={15} strokeWidth={2} />}
          onClick={() => { setNewColTurnId(undefined); setNewColOpen(true); }}
        >
          New collection
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-0 border-b border-[color:var(--hairline)] mb-6">
        {(["collections", "history"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setActiveTab(t)}
            className={cn(
              "px-4 py-3 text-sm font-sans border-b-2 -mb-px transition-colors capitalize",
              activeTab === t
                ? "border-accent text-fg font-medium"
                : "border-transparent text-fg-muted hover:text-fg",
            )}
          >
            {t === "collections" ? "Collections" : "Query History"}
            {t === "collections" && collections.length > 0 && (
              <span
                className={cn(
                  "ml-1.5 inline-flex items-center justify-center h-4 min-w-4 px-1 rounded-full text-[10px] font-sans font-semibold tabular-nums",
                  activeTab === t ? "bg-accent text-accent-fg" : "bg-[color:var(--hairline)] text-fg-muted",
                )}
              >
                {collections.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ── Collections tab ── */}
      {activeTab === "collections" && (
        <>
          {isLoading ? (
            <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
              {[0, 1, 2, 3, 4, 5].map((i) => (
                <Skeleton key={i} className="h-36 w-full" />
              ))}
            </div>
          ) : collections.length === 0 ? (
            <EmptyState
              icon={<Layers size={28} strokeWidth={1.25} />}
              heading="No collections yet"
              body="Create a collection to save and organize candidate shortlists."
              action={{ label: "New collection", onClick: () => setNewColOpen(true) }}
            />
          ) : (
            <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
              {collections.map((col) => (
                <CollectionCard
                  key={col.id}
                  col={col}
                  onRename={setRenameTarget}
                  onDelete={setDeleteTarget}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* ── Query History tab ── */}
      {activeTab === "history" && (
        <QueryHistoryTab onCreateCollection={handleCreateFromTurn} />
      )}

      {/* ── Modals ── */}

      {newColOpen && (
        <NewCollectionModal
          sourceTurnId={newColTurnId}
          onClose={(created) => {
            setNewColOpen(false);
            setNewColTurnId(undefined);
            if (created) setActiveTab("collections");
          }}
        />
      )}

      {renameTarget && (
        <RenameModal
          col={renameTarget}
          onClose={() => setRenameTarget(null)}
          onSuccess={() => setRenameTarget(null)}
        />
      )}

      <Modal open={!!deleteTarget} onOpenChange={(o) => !o && setDeleteTarget(null)}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Delete collection</ModalTitle>
            <ModalDescription>
              Are you sure you want to delete{" "}
              <span className="font-medium text-fg">"{deleteTarget?.name}"</span>? This will remove all
              {deleteTarget?.item_count ? ` ${deleteTarget.item_count}` : ""} candidate
              {deleteTarget?.item_count !== 1 ? "s" : ""} from the collection. This cannot be undone.
            </ModalDescription>
          </ModalHeader>
          <ModalFooter>
            <Button variant="ghost" onClick={() => setDeleteTarget(null)}>Cancel</Button>
            <Button
              variant="danger"
              loading={deleteMutation.isPending}
              onClick={() => deleteTarget && deleteMutation.mutate(deleteTarget.id)}
            >
              Delete
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}
