import { api, type ResumeResponse, type UploadStatus } from "@/api";
import { UploadModal } from "@/components/candidates/UploadModal";
import {
    Badge,
    Button,
    DataTable,
    EmptyState,
    FilterChip,
    Modal,
    ModalContent,
    ModalDescription,
    ModalFooter,
    ModalHeader,
    ModalTitle,
    Pagination,
    type ColumnDef,
} from "@/components/ui";
import { cn } from "@/lib/cn";
import { useSelectedJobId, useUserId } from "@/lib/auth";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    ChevronDown,
    Eye,
    LayoutGrid,
    LayoutList,
    Layers,
    Pencil,
    Search,
    Trash2,
    Upload,
    X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { toast } from "sonner";

// ─── helpers ────────────────────────────────────────────────────────────────

function fileToDisplayName(filename: string): string {
  return (
    filename
      .replace(/\.pdf$/i, "")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .trim() || filename
  );
}

function relativeTime(iso: string | null): string {
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

function truncateUserId(id: string): string {
  return `#${id.slice(0, 6)}…${id.slice(-4)}`;
}

function candidateDisplayName(resume: ResumeResponse): string {
  return resume.candidate_display_name?.trim() || fileToDisplayName(resume.original_file_name);
}

function uploaderDisplayName(resume: ResumeResponse): string {
  return resume.uploader_display_name?.trim() || truncateUserId(resume.uploaded_by_user_id);
}

type UploadStatusVariant = "neutral" | "warning" | "success" | "danger";

function uploadVariant(status: UploadStatus): UploadStatusVariant {
  switch (status) {
    case "processed":
      return "success";
    case "processing":
      return "warning";
    case "failed":
      return "danger";
    default:
      return "neutral";
  }
}

function UploadStatusBadge({ status }: { status: UploadStatus }) {
  return (
    <Badge variant={uploadVariant(status)}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  );
}

// ─── constants ───────────────────────────────────────────────────────────────

const STATUS_FILTERS: { label: string; value: UploadStatus | "" }[] = [
  { label: "All", value: "" },
  { label: "Uploaded", value: "uploaded" },
  { label: "Processing", value: "processing" },
  { label: "Processed", value: "processed" },
  { label: "Failed", value: "failed" },
];

type SortOption = {
  label: string;
  key: keyof ResumeResponse;
  dir: "asc" | "desc";
};

const SORT_OPTIONS: SortOption[] = [
  { label: "Newest first", key: "uploaded_at", dir: "desc" },
  { label: "Oldest first", key: "uploaded_at", dir: "asc" },
  { label: "Status", key: "upload_status", dir: "asc" },
  { label: "Name A→Z", key: "original_file_name", dir: "asc" },
  { label: "Name Z→A", key: "original_file_name", dir: "desc" },
];

type ShortlistMode = "create" | "add";

function errorStatus(err: unknown): number | undefined {
  if (
    typeof err === "object" &&
    err !== null &&
    "response" in err &&
    typeof err.response === "object" &&
    err.response !== null &&
    "status" in err.response &&
    typeof err.response.status === "number"
  ) {
    return err.response.status;
  }
  return undefined;
}

// ─── main component ───────────────────────────────────────────────────────────

export default function CandidatesListRoute() {
  const qc = useQueryClient();
  const selectedJobId = useSelectedJobId();
  const userId = useUserId();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();

  const view = (params.get("view") as "table" | "grid") ?? "table";
  const statusFilter = (params.get("status") ?? "") as UploadStatus | "";
  const page = Math.max(1, parseInt(params.get("page") ?? "1", 10));
  const pageSize = [50, 100, 200].includes(parseInt(params.get("pageSize") ?? "50", 10))
    ? parseInt(params.get("pageSize") ?? "50", 10)
    : 50;
  const sortLabel = params.get("sort") ?? "Newest first";
  const searchQuery = params.get("q") ?? "";

  const sortOption = SORT_OPTIONS.find((o) => o.label === sortLabel) ?? SORT_OPTIONS[0];

  const [uploadOpen, setUploadOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<(string | number)[]>([]);
  const [editTarget, setEditTarget] = useState<ResumeResponse | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ResumeResponse | null>(null);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [shortlistOpen, setShortlistOpen] = useState(false);
  const [shortlistMode, setShortlistMode] = useState<ShortlistMode>("create");
  const [shortlistName, setShortlistName] = useState("");
  const [shortlistConflict, setShortlistConflict] = useState(false);
  const [selectedCollectionId, setSelectedCollectionId] = useState("");
  const [collectionSearch, setCollectionSearch] = useState("");
  const [editName, setEditName] = useState("");
  const [editStatus, setEditStatus] = useState<UploadStatus>("uploaded");
  const [sortOpen, setSortOpen] = useState(false);
  const sortRef = useRef<HTMLDivElement>(null);

  // Close sort dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (sortRef.current && !sortRef.current.contains(e.target as Node)) {
        setSortOpen(false);
      }
    }
    if (sortOpen) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [sortOpen]);

  function setParam(key: string, value: string) {
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      if (value) next.set(key, value);
      else next.delete(key);
      if (key !== "page") next.set("page", "1");
      return next;
    });
  }

  // ── data ──────────────────────────────────────────────────────────────────

  const { data, isLoading } = useQuery({
    queryKey: ["candidates", selectedJobId, statusFilter, page, pageSize],
    queryFn: () =>
      selectedJobId
        ? api.jobs.resumes.list(selectedJobId, {
            upload_status: statusFilter || undefined,
            limit: pageSize,
            offset: (page - 1) * pageSize,
          })
        : Promise.resolve({ items: [], total: 0 }),
    refetchInterval: (query) => {
      const items = query.state.data?.items ?? [];
      return items.some(
        (resume) =>
          resume.upload_status === "uploaded" || resume.upload_status === "processing",
      )
        ? 3000
        : false;
    },
  });

  const { data: collectionsData, isLoading: isCollectionsLoading } = useQuery({
    queryKey: ["collections", userId],
    queryFn: () => api.shortlist.collections.list({ user_id: userId ?? "", limit: 100 }),
    enabled: shortlistOpen && !!userId,
    staleTime: 30_000,
  });

  const allItems: ResumeResponse[] = useMemo(() => data?.items ?? [], [data?.items]);
  const collections = useMemo(() => collectionsData?.items ?? [], [collectionsData?.items]);
  const serverTotal: number = data?.total ?? 0;

  const filteredItems = useMemo(() => {
    if (!searchQuery.trim()) return allItems;
    const q = searchQuery.toLowerCase();
    return allItems.filter(
      (r) =>
        candidateDisplayName(r).toLowerCase().includes(q) ||
        uploaderDisplayName(r).toLowerCase().includes(q) ||
        r.original_file_name.toLowerCase().includes(q) ||
        fileToDisplayName(r.original_file_name).toLowerCase().includes(q),
    );
  }, [allItems, searchQuery]);

  const sortedItems = useMemo(() => {
    return [...filteredItems].sort((a, b) => {
      const av =
        sortOption.key === "original_file_name"
          ? candidateDisplayName(a)
          : String(a[sortOption.key] ?? "");
      const bv =
        sortOption.key === "original_file_name"
          ? candidateDisplayName(b)
          : String(b[sortOption.key] ?? "");
      const cmp = av.localeCompare(bv, undefined, { numeric: true });
      return sortOption.dir === "asc" ? cmp : -cmp;
    });
  }, [filteredItems, sortOption]);

  const displayTotal = searchQuery.trim() ? filteredItems.length : serverTotal;
  const selectedResumeIdSet = useMemo(
    () => new Set(selectedIds.map(String)),
    [selectedIds],
  );
  const selectedResumes = useMemo(
    () => allItems.filter((resume) => selectedResumeIdSet.has(String(resume.id))),
    [allItems, selectedResumeIdSet],
  );
  const selectedCandidateProfileIds = useMemo(
    () =>
      [...new Set(
        selectedResumes
          .map((resume) => resume.candidate_profile_id)
          .filter((id): id is string => Boolean(id)),
      )],
    [selectedResumes],
  );
  const selectedMissingProfileCount = selectedResumes.length - selectedCandidateProfileIds.length;
  const filteredCollections = useMemo(() => {
    const query = collectionSearch.trim().toLowerCase();
    if (!query) return collections;
    return collections.filter((collection) => collection.name.toLowerCase().includes(query));
  }, [collectionSearch, collections]);

  function closeShortlistModal() {
    setShortlistOpen(false);
    setShortlistMode("create");
    setShortlistName("");
    setShortlistConflict(false);
    setSelectedCollectionId("");
    setCollectionSearch("");
  }

  // ── mutations ─────────────────────────────────────────────────────────────

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.jobs.resumes.remove(selectedJobId!, id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["candidates", selectedJobId] });
      toast.success("Resume deleted");
      setDeleteTarget(null);
    },
  });

  const bulkDeleteMutation = useMutation({
    mutationFn: (ids: string[]) => Promise.all(ids.map((id) => api.jobs.resumes.remove(selectedJobId!, id))),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["candidates", selectedJobId] });
      toast.success(`${selectedIds.length} resumes deleted`);
      setSelectedIds([]);
      setBulkDeleteOpen(false);
    },
  });

  const editMutation = useMutation({
    mutationFn: ({
      id,
      name,
      status,
    }: {
      id: string;
      name: string;
      status: UploadStatus;
    }) => api.jobs.resumes.update(selectedJobId!, id, { original_file_name: name, upload_status: status }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["candidates", selectedJobId] });
      toast.success("Resume updated");
      setEditTarget(null);
    },
  });

  const createShortlistMutation = useMutation({
    mutationFn: async (name: string) => {
      if (!userId) {
        throw new Error("Missing user session");
      }
      if (selectedCandidateProfileIds.length === 0) {
        throw new Error("No candidate profiles available for shortlisting");
      }

      const collection = await api.shortlist.collections.create({
        created_by_user_id: userId,
        name: name.trim(),
      });

      await Promise.all(
        selectedCandidateProfileIds.map((candidateProfileId) =>
          api.shortlist.items.add(collection.id, {
            candidate_profile_id: candidateProfileId,
          }),
        ),
      );

      return collection;
    },
    onSuccess: (collection) => {
      qc.invalidateQueries({ queryKey: ["collections"] });
      qc.invalidateQueries({ queryKey: ["collection", collection.id] });
      qc.invalidateQueries({ queryKey: ["collection-items", collection.id] });

      const addedCount = selectedCandidateProfileIds.length;
      if (selectedMissingProfileCount > 0) {
        toast.success(
          `Collection created with ${addedCount} candidate${addedCount !== 1 ? "s" : ""}; ${selectedMissingProfileCount} skipped without candidate profiles`,
        );
      } else {
        toast.success(
          `Collection created with ${addedCount} candidate${addedCount !== 1 ? "s" : ""}`,
        );
      }

      setSelectedIds([]);
      closeShortlistModal();
      navigate(routes.shortlistCollection(collection.id));
    },
    onError: (err: unknown) => {
      const status = errorStatus(err);
      if (status === 409) {
        setShortlistConflict(true);
        return;
      }
      toast.error(err instanceof Error ? err.message : "Failed to create shortlist collection");
    },
  });

  const addToShortlistMutation = useMutation({
    mutationFn: async (collectionId: string) => {
      if (!collectionId) {
        throw new Error("Choose a collection");
      }
      if (selectedCandidateProfileIds.length === 0) {
        throw new Error("No candidate profiles available for shortlisting");
      }

      const results = await Promise.allSettled(
        selectedCandidateProfileIds.map((candidateProfileId) =>
          api.shortlist.items.add(collectionId, {
            candidate_profile_id: candidateProfileId,
          }),
        ),
      );

      let added = 0;
      let duplicates = 0;
      let failed = 0;
      for (const result of results) {
        if (result.status === "fulfilled") {
          added += 1;
          continue;
        }
        const status = errorStatus(result.reason);
        if (status === 409) duplicates += 1;
        else failed += 1;
      }

      if (failed > 0 && added === 0 && duplicates === 0) {
        throw new Error("Failed to add candidates to collection");
      }

      return { collectionId, added, duplicates, failed };
    },
    onSuccess: ({ collectionId, added, duplicates, failed }) => {
      qc.invalidateQueries({ queryKey: ["collections"] });
      qc.invalidateQueries({ queryKey: ["collection", collectionId] });
      qc.invalidateQueries({ queryKey: ["collection-items", collectionId] });

      const summary: string[] = [];
      if (added > 0) summary.push(`added ${added}`);
      if (duplicates > 0) summary.push(`skipped ${duplicates} duplicate${duplicates !== 1 ? "s" : ""}`);
      if (selectedMissingProfileCount > 0) {
        summary.push(`skipped ${selectedMissingProfileCount} without profiles`);
      }
      if (failed > 0) summary.push(`${failed} failed`);

      toast.success(`Updated shortlist: ${summary.join(", ")}`);
      if (failed > 0) {
        toast.error(`${failed} candidate${failed !== 1 ? "s" : ""} could not be added`);
      }

      setSelectedIds([]);
      closeShortlistModal();
      navigate(routes.shortlistCollection(collectionId));
    },
    onError: (err: unknown) => {
      toast.error(err instanceof Error ? err.message : "Failed to update shortlist collection");
    },
  });

  const openEdit = useCallback((resume: ResumeResponse) => {
    setEditTarget(resume);
    setEditName(resume.original_file_name);
    setEditStatus(resume.upload_status);
  }, []);

  // ── table columns ─────────────────────────────────────────────────────────

  const tableColumns: ColumnDef<ResumeResponse>[] = useMemo(
    () => [
      {
        key: "name",
        header: "Candidate",
        render: (row) => (
          <div className="flex flex-col gap-0.5 min-w-0">
            <Link
              to={routes.candidateDetail(row.id)}
              className="font-sans text-sm font-medium text-fg hover:text-accent transition-colors truncate"
            >
              {candidateDisplayName(row)}
            </Link>
            {candidateDisplayName(row) !== row.original_file_name && (
              <span className="font-mono text-[0.6875rem] text-fg-subtle truncate">
                {row.original_file_name}
              </span>
            )}
          </div>
        ),
      },
      {
        key: "upload_status",
        header: "Status",
        width: 140,
        render: (row) => <UploadStatusBadge status={row.upload_status} />,
      },
      {
        key: "uploaded_by_user_id",
        header: "Uploaded by",
        width: 180,
        render: (row) => (
          <span className="text-sm text-fg-muted truncate">
            {uploaderDisplayName(row)}
          </span>
        ),
      },
      {
        key: "uploaded_at",
        header: "Uploaded",
        width: 120,
        render: (row) => (
          <span
            className="text-sm text-fg-muted tabular-nums"
            title={row.uploaded_at ? new Date(row.uploaded_at).toUTCString() : ""}
          >
            {relativeTime(row.uploaded_at)}
          </span>
        ),
      },
      {
        key: "retention_expires_at",
        header: "Expires",
        width: 120,
        render: (row) => (
          <span
            className="text-sm text-fg-muted tabular-nums"
            title={
              row.retention_expires_at
                ? new Date(row.retention_expires_at).toUTCString()
                : ""
            }
          >
            {relativeTime(row.retention_expires_at)}
          </span>
        ),
      },
      {
        key: "actions",
        header: "",
        width: 100,
        className: "text-right",
        render: (row) => (
          <div className="flex items-center justify-end gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <Link
              to={routes.candidateDetail(row.id)}
              aria-label="View candidate"
              className={cn(
                "inline-flex items-center justify-center h-7 w-7 rounded-[var(--radius-sm)]",
                "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors",
              )}
            >
              <Eye size={14} strokeWidth={1.75} />
            </Link>
            <Button variant="icon" size="sm" aria-label="Edit" onClick={() => openEdit(row)}>
              <Pencil size={14} strokeWidth={1.75} />
            </Button>
            <Button
              variant="icon"
              size="sm"
              aria-label="Delete"
              onClick={() => setDeleteTarget(row)}
            >
              <Trash2 size={14} strokeWidth={1.75} />
            </Button>
          </div>
        ),
      },
    ],
    [openEdit],
  );

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="px-8 py-8 min-h-full">

      {/* Page header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-display text-[2rem] font-medium text-fg leading-tight">
            Candidates
          </h1>
          <p className="text-sm text-fg-muted mt-1 font-sans">
            Manage parsed resumes and candidate profiles
          </p>
        </div>
        <Button
          variant="primary"
          icon={<Upload size={15} strokeWidth={2} />}
          onClick={() => setUploadOpen(true)}
        >
          Upload resumes
        </Button>
      </div>

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3 mb-5">
        {/* Search */}
        <div className="relative min-w-[200px] max-w-[300px] flex-1">
          <Search
            size={14}
            strokeWidth={1.75}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-muted pointer-events-none"
          />
          <input
            type="text"
            placeholder="Search by candidate, uploader, or file…"
            value={searchQuery}
            onChange={(e) => setParam("q", e.target.value)}
            className={cn(
              "w-full h-9 pl-8 pr-3 text-sm font-sans rounded-[var(--radius-md)]",
              "border border-[color:var(--hairline-strong)] bg-bg text-fg placeholder:text-fg-subtle",
              "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
            )}
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setParam("q", "")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-fg-muted hover:text-fg transition-colors"
              aria-label="Clear search"
            >
              <X size={13} strokeWidth={2} />
            </button>
          )}
        </div>

        {/* Status filter chips */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {STATUS_FILTERS.map((f) => (
            <FilterChip
              key={f.value}
              selected={statusFilter === f.value}
              onClick={() => setParam("status", f.value)}
            >
              {f.label}
            </FilterChip>
          ))}
        </div>

        <div className="flex-1" />

        {/* Sort dropdown */}
        <div className="relative" ref={sortRef}>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setSortOpen((v) => !v)}
          >
            {sortOption.label}
            <ChevronDown
              size={13}
              strokeWidth={2}
              className={cn(
                "ml-1 transition-transform duration-150",
                sortOpen && "rotate-180",
              )}
            />
          </Button>
          {sortOpen && (
            <div
              className={cn(
                "absolute right-0 top-full mt-1 z-20 w-44",
                "rounded-[var(--radius-md)] bg-bg-elevated",
                "border border-[color:var(--hairline)] shadow-[var(--shadow-md)]",
                "py-1 flex flex-col",
              )}
            >
              {SORT_OPTIONS.map((opt) => (
                <button
                  key={opt.label}
                  type="button"
                  className={cn(
                    "px-3 py-2 text-left text-sm font-sans transition-colors",
                    opt.label === sortLabel
                      ? "text-accent font-medium bg-[rgba(31,58,46,0.06)]"
                      : "text-fg hover:bg-[color:var(--hairline)]",
                  )}
                  onClick={() => {
                    setParam("sort", opt.label);
                    setSortOpen(false);
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* View toggle */}
        <div className="flex items-center h-9 rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] overflow-hidden">
          <button
            type="button"
            aria-label="Table view"
            aria-pressed={view === "table"}
            onClick={() => setParam("view", "table")}
            className={cn(
              "h-full px-2.5 transition-colors",
              view === "table"
                ? "bg-accent text-accent-fg"
                : "bg-bg text-fg-muted hover:bg-[color:var(--hairline)]",
            )}
          >
            <LayoutList size={15} strokeWidth={1.75} />
          </button>
          <button
            type="button"
            aria-label="Grid view"
            aria-pressed={view === "grid"}
            onClick={() => setParam("view", "grid")}
            className={cn(
              "h-full px-2.5 transition-colors",
              view === "grid"
                ? "bg-accent text-accent-fg"
                : "bg-bg text-fg-muted hover:bg-[color:var(--hairline)]",
            )}
          >
            <LayoutGrid size={15} strokeWidth={1.75} />
          </button>
        </div>
      </div>

      {/* Bulk action bar */}
      {selectedIds.length > 0 && (
        <div
          className={cn(
            "fixed bottom-6 left-1/2 -translate-x-1/2 z-30",
            "flex items-center gap-3 px-5 py-3 rounded-[var(--radius-lg)]",
            "bg-fg text-bg shadow-[var(--shadow-lg)]",
            "animate-in fade-in slide-in-from-bottom-2 duration-200",
          )}
        >
          <span className="font-sans text-sm font-medium tabular-nums">
            {selectedIds.length} selected
          </span>
          <div className="w-px h-4 bg-current opacity-20" />
          <Button
            variant="ghost"
            size="sm"
            className="text-bg hover:bg-white/10 hover:text-bg"
            icon={<Layers size={14} strokeWidth={2} />}
            disabled={selectedCandidateProfileIds.length === 0}
            onClick={() => {
              setShortlistConflict(false);
              setShortlistMode("create");
              setShortlistOpen(true);
            }}
          >
            Create/Add to shortlist
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-bg hover:bg-white/10 hover:text-bg"
            icon={<Trash2 size={14} strokeWidth={2} />}
            onClick={() => setBulkDeleteOpen(true)}
          >
            Delete
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-bg/60 hover:bg-white/10 hover:text-bg"
            onClick={() => setSelectedIds([])}
          >
            Clear
          </Button>
        </div>
      )}

      {/* ── TABLE VIEW ── */}
      {view === "table" && (
        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] overflow-hidden">
          <DataTable
            columns={tableColumns}
            data={sortedItems}
            loading={isLoading}
            selectable
            onSelectionChange={setSelectedIds}
            rowClassName={() => "group"}
            emptyState={
              <EmptyState
                heading="No candidates yet"
                body="Upload PDF resumes to start building candidate profiles."
                action={{ label: "Upload resumes", onClick: () => setUploadOpen(true) }}
              />
            }
          />
        </div>
      )}

      {/* ── GRID VIEW ── */}
      {view === "grid" && (
        <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
          {isLoading
            ? Array.from({ length: 6 }).map((_, i) => (
                <div
                  key={i}
                  className="h-40 rounded-[var(--radius-lg)] bg-gradient-to-r from-[color:var(--hairline)] via-[color:var(--hairline-strong)] to-[color:var(--hairline)] bg-[length:200%_100%]"
                  style={{ animation: "skeleton-shimmer 1.5s ease-in-out infinite" }}
                />
              ))
            : sortedItems.length === 0 ? (
                <div className="col-span-full">
                  <EmptyState
                    heading="No candidates yet"
                    body="Upload PDF resumes to start building candidate profiles."
                    action={{ label: "Upload resumes", onClick: () => setUploadOpen(true) }}
                  />
                </div>
              ) : (
                sortedItems.map((resume) => (
                  <CandidateCard
                    key={resume.id}
                    resume={resume}
                    onEdit={() => openEdit(resume)}
                    onDelete={() => setDeleteTarget(resume)}
                  />
                ))
              )}
        </div>
      )}

      {/* Pagination */}
      {!isLoading && displayTotal > 0 && (
        <div className="mt-4 hairline-t pt-3">
          <Pagination
            total={displayTotal}
            page={page}
            pageSize={pageSize}
            onPageChange={(p) =>
              setParams((prev) => {
                const next = new URLSearchParams(prev);
                next.set("page", String(p));
                return next;
              })
            }
            onPageSizeChange={(s) =>
              setParams((prev) => {
                const next = new URLSearchParams(prev);
                next.set("pageSize", String(s));
                next.set("page", "1");
                return next;
              })
            }
          />
        </div>
      )}

      {/* ── MODALS ── */}

      {/* Upload */}
      <UploadModal
        open={uploadOpen}
        onOpenChange={setUploadOpen}
        onComplete={() => qc.invalidateQueries({ queryKey: ["candidates", selectedJobId] })}
      />

      {/* Delete single */}
      <Modal open={!!deleteTarget} onOpenChange={(o) => !o && setDeleteTarget(null)}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Delete resume</ModalTitle>
            <ModalDescription>
              Are you sure you want to delete{" "}
              <span className="font-mono text-xs bg-[color:var(--hairline)] px-1 py-0.5 rounded">
                {deleteTarget?.original_file_name}
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

      {/* Bulk delete */}
      <Modal open={bulkDeleteOpen} onOpenChange={setBulkDeleteOpen}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Delete {selectedIds.length} resumes</ModalTitle>
            <ModalDescription>
              Are you sure? This will permanently delete {selectedIds.length} resume
              {selectedIds.length !== 1 ? "s" : ""} and cannot be undone.
            </ModalDescription>
          </ModalHeader>
          <ModalFooter>
            <Button variant="ghost" onClick={() => setBulkDeleteOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              loading={bulkDeleteMutation.isPending}
              onClick={() => bulkDeleteMutation.mutate(selectedIds.map(String))}
            >
              Delete {selectedIds.length}
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* Create shortlist collection */}
      <Modal open={shortlistOpen} onOpenChange={(o) => !o && closeShortlistModal()}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Create/Add to shortlist</ModalTitle>
            <ModalDescription>
              Save {selectedIds.length} selected resume
              {selectedIds.length !== 1 ? "s" : ""}.
              {selectedMissingProfileCount > 0
                ? ` ${selectedMissingProfileCount} selected resume${selectedMissingProfileCount !== 1 ? "s do" : " does"} not have a candidate profile yet and will be skipped.`
                : ""}
            </ModalDescription>
          </ModalHeader>
          <div className="mt-2 space-y-4">
            <div className="grid grid-cols-2 gap-2 rounded-[var(--radius-md)] bg-[color:var(--hairline)] p-1">
              <button
                type="button"
                onClick={() => setShortlistMode("create")}
                className={cn(
                  "h-9 rounded-[var(--radius-sm)] text-sm font-sans font-medium transition-colors",
                  shortlistMode === "create"
                    ? "bg-bg text-fg shadow-[var(--shadow-sm)]"
                    : "text-fg-muted hover:text-fg",
                )}
              >
                Create new
              </button>
              <button
                type="button"
                onClick={() => setShortlistMode("add")}
                className={cn(
                  "h-9 rounded-[var(--radius-sm)] text-sm font-sans font-medium transition-colors",
                  shortlistMode === "add"
                    ? "bg-bg text-fg shadow-[var(--shadow-sm)]"
                    : "text-fg-muted hover:text-fg",
                )}
              >
                Add to existing
              </button>
            </div>

            {shortlistMode === "create" ? (
              <div>
                <input
                  type="text"
                  placeholder="Collection name…"
                  value={shortlistName}
                  onChange={(e) => {
                    setShortlistName(e.target.value);
                    setShortlistConflict(false);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && shortlistName.trim() && selectedCandidateProfileIds.length > 0) {
                      createShortlistMutation.mutate(shortlistName);
                    }
                    if (e.key === "Escape") closeShortlistModal();
                  }}
                  autoFocus
                  className={cn(
                    "w-full h-9 px-3 text-sm font-sans rounded-[var(--radius-md)]",
                    "border bg-bg text-fg",
                    shortlistConflict
                      ? "border-danger focus:outline-danger"
                      : "border-[color:var(--hairline-strong)] focus:outline-accent",
                    "focus:outline focus:outline-2 focus:outline-offset-1 outline-none",
                  )}
                />
                {shortlistConflict && (
                  <p className="mt-1.5 text-xs font-sans text-danger">
                    A collection with this name already exists. Please choose a different name.
                  </p>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <div className="relative">
                  <Search
                    size={14}
                    strokeWidth={1.75}
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-muted pointer-events-none"
                  />
                  <input
                    type="text"
                    placeholder="Search collections…"
                    value={collectionSearch}
                    onChange={(e) => setCollectionSearch(e.target.value)}
                    className={cn(
                      "w-full h-9 pl-8 pr-3 text-sm font-sans rounded-[var(--radius-md)]",
                      "border border-[color:var(--hairline-strong)] bg-bg text-fg placeholder:text-fg-subtle",
                      "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                    )}
                  />
                </div>

                <div className="max-h-64 overflow-y-auto rounded-[var(--radius-md)] border border-[color:var(--hairline)]">
                  {isCollectionsLoading ? (
                    <div className="px-4 py-6 text-sm text-fg-muted">Loading collections…</div>
                  ) : filteredCollections.length === 0 ? (
                    <div className="px-4 py-6 text-sm text-fg-muted">
                      {collectionSearch.trim() ? "No matching collections." : "No collections yet."}
                    </div>
                  ) : (
                    filteredCollections.map((collection) => (
                      <label
                        key={collection.id}
                        className={cn(
                          "flex cursor-pointer items-start gap-3 px-4 py-3 transition-colors",
                          "border-b border-[color:var(--hairline)] last:border-b-0 hover:bg-[color:var(--hairline)]",
                          selectedCollectionId === collection.id && "bg-[rgba(31,58,46,0.06)]",
                        )}
                      >
                        <input
                          type="radio"
                          name="selected-collection"
                          value={collection.id}
                          checked={selectedCollectionId === collection.id}
                          onChange={() => setSelectedCollectionId(collection.id)}
                          className="mt-0.5 h-4 w-4 accent-accent"
                        />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-medium text-fg">
                            {collection.name}
                          </span>
                          <span className="mt-0.5 block text-xs text-fg-muted">
                            {collection.item_count} candidate{collection.item_count !== 1 ? "s" : ""}
                          </span>
                        </span>
                      </label>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
          <ModalFooter>
            <Button variant="ghost" onClick={closeShortlistModal}>
              Cancel
            </Button>
            <Button
              variant="primary"
              loading={
                shortlistMode === "create"
                  ? createShortlistMutation.isPending
                  : addToShortlistMutation.isPending
              }
              disabled={
                selectedCandidateProfileIds.length === 0 ||
                (shortlistMode === "create"
                  ? !shortlistName.trim()
                  : !selectedCollectionId)
              }
              onClick={() =>
                shortlistMode === "create"
                  ? createShortlistMutation.mutate(shortlistName)
                  : addToShortlistMutation.mutate(selectedCollectionId)
              }
            >
              {shortlistMode === "create" ? "Create" : "Add to collection"}
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* Edit */}
      <Modal open={!!editTarget} onOpenChange={(o) => !o && setEditTarget(null)}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Edit resume</ModalTitle>
            <ModalDescription>Update the filename or processing status.</ModalDescription>
          </ModalHeader>

          <div className="space-y-4 mt-2">
            <div>
              <label
                htmlFor="edit-filename"
                className="block text-xs font-medium text-fg-muted mb-1.5"
              >
                Filename
              </label>
              <input
                id="edit-filename"
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                className={cn(
                  "w-full h-9 px-3 text-sm font-mono rounded-[var(--radius-md)]",
                  "border border-[color:var(--hairline-strong)] bg-bg text-fg",
                  "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                )}
              />
            </div>
            <div>
              <label
                htmlFor="edit-status"
                className="block text-xs font-medium text-fg-muted mb-1.5"
              >
                Status
              </label>
              <select
                id="edit-status"
                value={editStatus}
                onChange={(e) => setEditStatus(e.target.value as UploadStatus)}
                className={cn(
                  "w-full h-9 px-3 text-sm font-sans rounded-[var(--radius-md)]",
                  "border border-[color:var(--hairline-strong)] bg-bg text-fg appearance-none",
                  "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                )}
              >
                <option value="uploaded">Uploaded</option>
                <option value="processing">Processing</option>
                <option value="processed">Processed</option>
                <option value="failed">Failed</option>
              </select>
            </div>
          </div>

          <ModalFooter>
            <Button variant="ghost" onClick={() => setEditTarget(null)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              loading={editMutation.isPending}
              disabled={!editName.trim()}
              onClick={() =>
                editTarget &&
                editMutation.mutate({
                  id: editTarget.id,
                  name: editName.trim(),
                  status: editStatus,
                })
              }
            >
              Save changes
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}

// ─── grid card ────────────────────────────────────────────────────────────────

function CandidateCard({
  resume,
  onEdit,
  onDelete,
}: {
  resume: ResumeResponse;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div
      className={cn(
        "group relative rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
        "bg-bg-elevated p-5 hover:shadow-[var(--shadow-md)] transition-shadow duration-200",
      )}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <Link
          to={routes.candidateDetail(resume.id)}
          className="font-display text-[0.9375rem] font-medium text-fg hover:text-accent transition-colors line-clamp-1 flex-1 min-w-0"
        >
          {candidateDisplayName(resume)}
        </Link>
        <UploadStatusBadge status={resume.upload_status} />
      </div>

      {candidateDisplayName(resume) !== resume.original_file_name && (
        <p className="font-mono text-[0.6875rem] text-fg-subtle truncate mb-3">
          {resume.original_file_name}
        </p>
      )}

      <p className="text-xs text-fg-muted truncate mb-4">
        Uploaded by {uploaderDisplayName(resume)}
      </p>

      <div className="flex items-center justify-between">
        <span
          className="text-xs text-fg-muted"
          title={resume.uploaded_at ? new Date(resume.uploaded_at).toUTCString() : ""}
        >
          {relativeTime(resume.uploaded_at)}
        </span>

        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <Link
            to={routes.candidateDetail(resume.id)}
            aria-label="View candidate"
            className={cn(
              "inline-flex items-center justify-center h-7 w-7 rounded-[var(--radius-sm)]",
              "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors",
            )}
          >
            <Eye size={13} strokeWidth={1.75} />
          </Link>
          <Button variant="icon" size="sm" aria-label="Edit" onClick={onEdit}>
            <Pencil size={13} strokeWidth={1.75} />
          </Button>
          <Button variant="icon" size="sm" aria-label="Delete" onClick={onDelete}>
            <Trash2 size={13} strokeWidth={1.75} />
          </Button>
        </div>
      </div>
    </div>
  );
}
