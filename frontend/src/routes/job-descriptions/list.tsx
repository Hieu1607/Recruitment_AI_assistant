import { api, type JobDescriptionResponse } from "@/api";
import {
    Badge,
    Button,
    EmptyState,
    FilterChip,
    Modal,
    ModalContent,
    ModalDescription,
    ModalFooter,
    ModalHeader,
    ModalTitle,
    Skeleton,
} from "@/components/ui";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BarChart2, Eye, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { toast } from "sonner";

// ─── helpers ────────────────────────────────────────────────────────────────

function relativeTime(iso: string): string {
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

function truncateBody(text: string, max = 120): string {
  const flat = text.replace(/\n+/g, " ").trim();
  return flat.length > max ? flat.slice(0, max) + "…" : flat;
}

// ─── main component ──────────────────────────────────────────────────────────

export default function JobDescriptionsListRoute() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();

  const activeFilter = params.get("active"); // "true" | "false" | null
  const [deleteTarget, setDeleteTarget] = useState<JobDescriptionResponse | null>(null);

  function setParam(key: string, value: string) {
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      if (value) next.set(key, value);
      else next.delete(key);
      return next;
    });
  }

  // ── data ──────────────────────────────────────────────────────────────────

  const isActiveParam =
    activeFilter === "true" ? true : activeFilter === "false" ? false : undefined;

  const { data, isLoading } = useQuery({
    queryKey: ["jobDescriptions", isActiveParam],
    queryFn: () => api.jobDescriptions.list({ is_active: isActiveParam, limit: 200 }),
  });

  const items: JobDescriptionResponse[] = data?.items ?? [];

  // ── mutations ─────────────────────────────────────────────────────────────

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.jobDescriptions.remove(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobDescriptions"] });
      toast.success("Job description deleted");
      setDeleteTarget(null);
    },
    onError: () => toast.error("Failed to delete job description"),
  });

  // ── render ────────────────────────────────────────────────────────────────

  const FILTERS = [
    { label: "All", value: "" },
    { label: "Active", value: "true" },
    { label: "Inactive", value: "false" },
  ];

  return (
    <div className="px-8 py-8 min-h-full">

      {/* Page header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-display text-[2rem] font-medium text-fg leading-tight">
            Job Descriptions
          </h1>
          <p className="text-sm text-fg-muted mt-1 font-sans">
            Create and manage positions to score candidates against
          </p>
        </div>
        <Button
          variant="primary"
          icon={<Plus size={15} strokeWidth={2} />}
          onClick={() => navigate(routes.jobDescriptionNew)}
        >
          Create JD
        </Button>
      </div>

      {/* Filter chips */}
      <div className="flex items-center gap-1.5 mb-6">
        {FILTERS.map((f) => (
          <FilterChip
            key={f.value}
            selected={(activeFilter ?? "") === f.value}
            onClick={() => setParam("active", f.value)}
          >
            {f.label}
          </FilterChip>
        ))}
        {!isLoading && (
          <span className="ml-2 text-xs text-fg-muted tabular-nums">
            {items.length} {items.length === 1 ? "result" : "results"}
          </span>
        )}
      </div>

      {/* Grid */}
      {isLoading ? (
        <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 rounded-[var(--radius-lg)]" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          heading="No job descriptions yet"
          body="Create a JD to start scoring candidates against positions."
          action={{ label: "Create JD", onClick: () => navigate(routes.jobDescriptionNew) }}
        />
      ) : (
        <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
          {items.map((jd) => (
            <JDCard
              key={jd.id}
              jd={jd}
              onDelete={() => setDeleteTarget(jd)}
            />
          ))}
        </div>
      )}

      {/* Delete confirmation */}
      <Modal open={!!deleteTarget} onOpenChange={(o) => !o && setDeleteTarget(null)}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Delete job description</ModalTitle>
            <ModalDescription>
              Are you sure you want to delete{" "}
              <span className="font-medium text-fg">
                {deleteTarget?.title ?? "Untitled position"}
              </span>
              ? This cannot be undone.
            </ModalDescription>
          </ModalHeader>
          <ModalFooter>
            <Button variant="ghost" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
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

// ─── JD card ─────────────────────────────────────────────────────────────────

function JDCard({
  jd,
  onDelete,
}: {
  jd: JobDescriptionResponse;
  onDelete: () => void;
}) {
  const navigate = useNavigate();

  return (
    <div
      className={cn(
        "group relative rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
        "bg-bg-elevated p-5 flex flex-col gap-3",
        "hover:shadow-[var(--shadow-md)] transition-all duration-200",
      )}
    >
      {/* Top row: title + active badge */}
      <div className="flex items-start justify-between gap-2">
        <Link
          to={routes.jobDescriptionEdit(jd.id)}
          className="font-display text-[0.9375rem] font-medium text-fg hover:text-accent transition-colors line-clamp-2 flex-1 min-w-0 leading-snug"
        >
          {jd.title ?? "Untitled position"}
        </Link>
        <Badge variant={jd.is_active ? "success" : "neutral"} size="sm" dot>
          {jd.is_active ? "Active" : "Inactive"}
        </Badge>
      </div>

      {/* Body preview */}
      <p className="text-sm text-fg-muted font-sans leading-relaxed line-clamp-3 flex-1">
        {truncateBody(jd.jd_text)}
      </p>

      {/* Footer row: timestamp + hover actions */}
      <div className="flex items-center justify-between pt-1 border-t border-[color:var(--hairline)]">
        <span
          className="text-xs text-fg-muted tabular-nums"
          title={new Date(jd.created_at).toUTCString()}
        >
          Created {relativeTime(jd.created_at)}
        </span>

        {/* Hover-reveal CTAs */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
          <button
            type="button"
            onClick={() => navigate(routes.jobDescriptionEdit(jd.id))}
            className={cn(
              "inline-flex items-center gap-1.5 h-7 px-2.5 text-xs font-sans font-medium rounded-[var(--radius-sm)]",
              "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors",
            )}
          >
            <Eye size={12} strokeWidth={1.75} />
            View
          </button>
          <button
            type="button"
            onClick={() => navigate(`${routes.scoring}?jd=${jd.id}`)}
            className={cn(
              "inline-flex items-center gap-1.5 h-7 px-2.5 text-xs font-sans font-medium rounded-[var(--radius-sm)]",
              "text-accent hover:bg-[rgba(31,58,46,0.08)] transition-colors",
            )}
          >
            <BarChart2 size={12} strokeWidth={1.75} />
            Score candidates
          </button>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            className={cn(
              "inline-flex items-center justify-center h-7 w-7 rounded-[var(--radius-sm)]",
              "text-fg-muted hover:text-danger hover:bg-[rgba(184,68,46,0.08)] transition-colors",
            )}
            aria-label="Delete JD"
          >
            <Trash2 size={12} strokeWidth={1.75} />
          </button>
        </div>
      </div>
    </div>
  );
}
