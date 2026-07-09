import { api, type CollectionResponse } from "@/api";
import { parseAxiosError } from "@/api/errors";
import {
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
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Layers, MessageSquare, MoreHorizontal, Plus, Users } from "lucide-react";
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
}: {
  onClose: (created?: CollectionResponse) => void;
}) {
  const [name, setName] = useState("");
  const qc = useQueryClient();
  const createMutation = useMutation({
    mutationFn: () =>
      api.shortlist.collections.create({
        name: name.trim(),
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

// ── main component ─────────────────────────────────────────────────────────────

export default function ShortlistsListRoute() {
  const qc = useQueryClient();
  const [newColOpen, setNewColOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<CollectionResponse | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<CollectionResponse | null>(null);

  const { data: collectionsData, isLoading } = useQuery({
    queryKey: ["collections"],
    queryFn: () => api.shortlist.collections.list({ limit: 100 }),
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

  return (
    <div className="px-8 py-8 min-h-full">
      {/* Page header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-display text-[2rem] font-medium text-fg leading-tight">Shortlists</h1>
          <p className="text-sm text-fg-muted mt-1 font-sans">
            Saved candidate collections
          </p>
        </div>
        <Button
          variant="primary"
          icon={<Plus size={15} strokeWidth={2} />}
          onClick={() => setNewColOpen(true)}
        >
          New collection
        </Button>
      </div>

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

      {/* ── Modals ── */}

      {newColOpen && (
        <NewCollectionModal
          onClose={(created) => {
            setNewColOpen(false);
            if (created) qc.invalidateQueries({ queryKey: ["collections"] });
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
