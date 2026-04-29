import { api, type ShortlistItemResponse } from "@/api";
import {
    Avatar,
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
    MessageSquare,
    Pencil,
    Trash2,
    Users,
    X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
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
  item,
  onRemove,
  removing,
}: {
  item: ShortlistItemResponse;
  onRemove: (item: ShortlistItemResponse) => void;
  removing: boolean;
}) {
  const displayId = truncateId(item.candidate_profile_id);

  return (
    <div className="group flex items-center gap-4 px-5 py-3.5 hairline-b last:border-b-0 hover:bg-[color:var(--hairline)]/30 transition-colors">
      <Avatar name={displayId} size="md" />

      <div className="flex-1 min-w-0">
        <p className="text-sm font-sans font-medium text-fg truncate">
          Candidate {displayId}
        </p>
        <p className="text-xs font-mono text-fg-subtle mt-0.5 truncate">
          {item.candidate_profile_id}
        </p>
      </div>

      {/* Placeholder skill chips */}
      <div className="hidden xl:flex items-center gap-1.5 flex-1 min-w-0">
        {["Profile", "Available"].map((s) => (
          <span
            key={s}
            className="px-2 py-0.5 text-[11px] font-sans text-fg-muted rounded-[var(--radius-sm)] border border-[color:var(--hairline)] bg-bg shrink-0"
          >
            {s}
          </span>
        ))}
      </div>

      {/* Added-at */}
      <span className="text-xs font-sans text-fg-subtle tabular-nums whitespace-nowrap hidden md:block">
        Added {relativeTime(item.added_at)}
      </span>

      {/* Remove button */}
      <button
        type="button"
        disabled={removing}
        onClick={() => onRemove(item)}
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

// ── main component ─────────────────────────────────────────────────────────────

export default function ShortlistCollectionRoute() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();

  const [page, setPage] = useState(1);
  const [editing, setEditing] = useState(false);
  const [nameConflict, setNameConflict] = useState(false);
  const [removeTarget, setRemoveTarget] = useState<ShortlistItemResponse | null>(null);

  // ── collection data ────────────────────────────────────────────────────────

  const { data: collection, isLoading: colLoading, error: colError } = useQuery({
    queryKey: ["collection", id],
    queryFn: () => api.shortlist.collections.get(id!),
    enabled: !!id,
  });

  const { data: itemsData, isLoading: itemsLoading } = useQuery({
    queryKey: ["collection-items", id, page],
    queryFn: () =>
      api.shortlist.items.listForCollection(id!, {
        limit: PAGE_SIZE,
        offset: (page - 1) * PAGE_SIZE,
      }),
    enabled: !!id,
    staleTime: 30_000,
  });

  const items = itemsData?.items ?? [];
  const total = itemsData?.total ?? 0;

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
    onError: (err: any) => {
      if (err?.response?.status === 409) {
        setNameConflict(true);
      } else {
        toast.error("Failed to rename collection");
      }
    },
  });

  const removeMutation = useMutation({
    mutationFn: (item: ShortlistItemResponse) =>
      api.shortlist.items.remove(id!, item.candidate_profile_id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["collection-items", id] });
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

      {/* ── Members table ── */}
      <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] overflow-hidden">
        {/* Table header */}
        <div className="flex items-center gap-4 px-5 py-3 bg-[color:var(--hairline)]/40 hairline-b">
          <div className="w-8 shrink-0" />
          <div className="flex-1 text-[11px] font-sans font-semibold uppercase tracking-wider text-fg-muted">
            Candidate
          </div>
          <div className="hidden xl:block flex-1 text-[11px] font-sans font-semibold uppercase tracking-wider text-fg-muted">
            Skills
          </div>
          <div className="hidden md:block text-[11px] font-sans font-semibold uppercase tracking-wider text-fg-muted w-28 text-right">
            Added
          </div>
          <div className="w-7" />
        </div>

        {itemsLoading ? (
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
        ) : items.length === 0 ? (
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
            {items.map((item) => (
              <MemberRow
                key={item.id}
                item={item}
                onRemove={setRemoveTarget}
                removing={removeMutation.isPending && removeTarget?.id === item.id}
              />
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {!itemsLoading && total > PAGE_SIZE && (
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
                {removeTarget ? truncateId(removeTarget.candidate_profile_id) : ""}
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
    </div>
  );
}
