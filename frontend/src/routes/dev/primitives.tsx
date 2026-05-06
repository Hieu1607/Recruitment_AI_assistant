import { Avatar } from "@/components/ui/avatar";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DataTable, type ColumnDef } from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
import { FilterChip } from "@/components/ui/filter-chip";
import {
    Modal,
    ModalClose,
    ModalContent,
    ModalDescription,
    ModalFooter,
    ModalHeader,
    ModalTitle,
} from "@/components/ui/modal";
import { Pagination } from "@/components/ui/pagination";
import { ScoreBar, ScoreDonut, ScoreRadar } from "@/components/ui/score-visualization";
import { Skeleton, SkeletonAvatar, SkeletonTableRow, SkeletonText } from "@/components/ui/skeleton";
import { Tooltip, TooltipProvider } from "@/components/ui/tooltip";
import {
    Briefcase,
    Code2,
    FileText,
    Plus,
    Search,
    Star,
    Trash2,
    Upload,
    Users,
} from "lucide-react";
import { useState, type ReactNode } from "react";

/* ── Fake data ─────────────────────────────────────────────────── */

type FakeRow = { id: number; name: string; role: string; score: number; status: string };

function makeFakeData(n: number): FakeRow[] {
  const names = [
    "Alice Chen", "Bob Martinez", "Carol Liu", "David Kim",
    "Emma Wilson", "Frank Nguyen", "Grace Park", "Hieu Le",
    "Ivan Petrov", "Julia Santos",
  ];
  const roles = ["Frontend Engineer", "Backend Engineer", "ML Engineer", "Product Designer", "Data Scientist"];
  const statuses = ["completed", "processing", "pending", "failed"] as const;
  return Array.from({ length: n }, (_, i) => ({
    id: i + 1,
    name: names[i % names.length],
    role: roles[i % roles.length],
    score: Math.round(40 + (i * 37 + 13) % 60),
    status: statuses[i % statuses.length],
  }));
}

const FAKE_DATA = makeFakeData(50);

const TABLE_COLS: ColumnDef<FakeRow>[] = [
  {
    key: "name",
    header: "Name",
    sortable: true,
    render: (row) => (
      <div className="flex items-center gap-2.5">
        <Avatar name={row.name} size="sm" />
        <span className="font-medium text-fg">{row.name}</span>
      </div>
    ),
  },
  { key: "role", header: "Role", sortable: true },
  {
    key: "score",
    header: "Score",
    sortable: true,
    render: (row) => (
      <div className="flex items-center gap-2">
        <ScoreBar score={row.score} size="sm" showLabel={false} />
        <span className="tabular-nums font-mono text-sm">{row.score}</span>
      </div>
    ),
  },
  {
    key: "status",
    header: "Status",
    render: (row) => <StatusBadge status={row.status as "pending"} />,
  },
];

const RADAR_DATA = [
  { subject: "Technical", value: 85, fullMark: 100 },
  { subject: "Experience", value: 72, fullMark: 100 },
  { subject: "Leadership", value: 60, fullMark: 100 },
  { subject: "Communication", value: 90, fullMark: 100 },
  { subject: "Culture", value: 78, fullMark: 100 },
];

const DONUT_SEGMENTS = [
  { label: "Technical", value: 35, color: "#1F3A2E" },
  { label: "Experience", value: 25, color: "#2A5A78" },
  { label: "Leadership", value: 20, color: "#5A3A7E" },
  { label: "Communication", value: 20, color: "#7A3A3A" },
];

/* ── Section wrapper ────────────────────────────────────────────── */
function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-6">
      <h2 className="font-display text-2xl font-medium text-fg">{title}</h2>
      {children}
      <hr className="hairline mt-2" />
    </section>
  );
}

/* ── Main Page ──────────────────────────────────────────────────── */
export default function PrimitivesShowcase() {
  const [chipSelected, setChipSelected] = useState<string[]>([]);
  const [showEmptyTable, setShowEmptyTable] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);

  const toggleChip = (val: string) =>
    setChipSelected((prev) =>
      prev.includes(val) ? prev.filter((v) => v !== val) : [...prev, val]
    );

  return (
    <TooltipProvider>
      <div className="px-8 py-12 max-w-5xl mx-auto flex flex-col gap-12">
        <div>
          <h1 className="font-display text-4xl font-medium text-fg">Primitives Showcase</h1>
          <p className="mt-2 text-fg-muted font-sans text-base">
            Phase 2 — Internal design system reference. Route: <code className="font-mono text-sm">/dev/primitives</code>
          </p>
        </div>

        {/* 1. Buttons */}
        <Section title="Buttons">
          <div className="flex flex-col gap-4">
            {(["primary", "secondary", "ghost", "danger"] as const).map((variant) => (
              <div key={variant} className="flex items-center gap-3 flex-wrap">
                <span className="text-xs text-fg-subtle font-mono w-20 shrink-0">{variant}</span>
                <Button variant={variant} size="sm">{variant} sm</Button>
                <Button variant={variant} size="md">{variant} md</Button>
                <Button variant={variant} size="lg">{variant} lg</Button>
                <Button variant={variant} size="md" loading>Loading</Button>
                <Button variant={variant} size="md" icon={<Upload size={16} />}>With icon</Button>
              </div>
            ))}
            <div className="flex items-center gap-3">
              <span className="text-xs text-fg-subtle font-mono w-20 shrink-0">icon-only</span>
              <Button variant="icon" size="sm" icon={<Search size={14} />} aria-label="Search" />
              <Button variant="icon" size="md" icon={<Plus size={16} />} aria-label="Add" />
              <Button variant="icon" size="lg" icon={<Trash2 size={18} />} aria-label="Delete" />
            </div>
          </div>
        </Section>

        {/* 2. Status Badges */}
        <Section title="Status Badges">
          <div className="flex flex-wrap gap-3">
            {(["pending", "processing", "completed", "failed", "sent", "not_sent", "queued", "running", "active"] as const).map(
              (s) => (
                <StatusBadge key={s} status={s} />
              )
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {(["neutral", "warning", "success", "danger"] as const).map((v) => (
              <Badge key={v} variant={v} size="sm">{v} sm</Badge>
            ))}
            {(["neutral", "warning", "success", "danger"] as const).map((v) => (
              <Badge key={v + "md"} variant={v} size="md">{v} md</Badge>
            ))}
          </div>
        </Section>

        {/* 3. Avatars */}
        <Section title="Avatars">
          <div className="flex items-end gap-4 flex-wrap">
            {(["sm", "md", "lg", "xl"] as const).map((size) => (
              <div key={size} className="flex flex-col items-center gap-2">
                <Avatar name="Hieu Le" size={size} />
                <span className="text-xs text-fg-subtle font-mono">{size}</span>
              </div>
            ))}
            <div className="flex flex-col items-center gap-2">
              <Avatar src="https://picsum.photos/seed/avatar1/64/64" name="Photo User" size="lg" />
              <span className="text-xs text-fg-subtle font-mono">photo</span>
            </div>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            {["Alice Chen", "Bob Martinez", "Carol Liu", "David Kim", "Emma Wilson", "Frank Nguyen"].map((name) => (
              <Avatar key={name} name={name} size="md" />
            ))}
          </div>
        </Section>

        {/* 4. Filter Chips */}
        <Section title="Filter Chips">
          <div className="flex flex-wrap gap-2">
            {[
              { val: "frontend", label: "Frontend", icon: <Code2 size={13} /> },
              { val: "backend", label: "Backend" },
              { val: "design", label: "Design", icon: <Star size={13} /> },
              { val: "ml", label: "Machine Learning" },
              { val: "pm", label: "Product", icon: <Briefcase size={13} /> },
            ].map(({ val, label, icon }) => (
              <FilterChip
                key={val}
                selected={chipSelected.includes(val)}
                icon={icon}
                onClick={() => toggleChip(val)}
              >
                {label}
              </FilterChip>
            ))}
          </div>
          {chipSelected.length > 0 && (
            <p className="text-sm text-fg-muted">
              Selected: <span className="font-medium text-fg">{chipSelected.join(", ")}</span>
            </p>
          )}
        </Section>

        {/* 5. DataTable */}
        <Section title="DataTable">
          <div className="flex items-center gap-3 mb-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setShowEmptyTable((v) => !v)}
            >
              {showEmptyTable ? "Show data" : "Show empty state"}
            </Button>
          </div>
          <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] overflow-hidden">
            <DataTable
              columns={TABLE_COLS}
              data={showEmptyTable ? [] : FAKE_DATA.slice((page - 1) * pageSize, page * pageSize)}
              selectable
              emptyState={
                <EmptyState
                  icon={<Users size={36} />}
                  heading="No candidates yet"
                  body="Upload resumes to get started."
                  action={{ label: "Upload resumes", onClick: () => {} }}
                />
              }
            />
            <div className="px-4 hairline-t">
              <Pagination
                total={FAKE_DATA.length}
                page={page}
                pageSize={pageSize}
                onPageChange={setPage}
                onPageSizeChange={setPageSize}
              />
            </div>
          </div>
        </Section>

        {/* 6. Modal */}
        <Section title="Modal">
          <div className="flex gap-3">
            <Button variant="secondary" size="md" onClick={() => setModalOpen(true)}>
              Open modal
            </Button>
          </div>
          <Modal open={modalOpen} onOpenChange={setModalOpen}>
            <ModalContent>
              <ModalHeader>
                <ModalTitle>Confirm action</ModalTitle>
                <ModalDescription>
                  This is a sample modal. Focus is trapped inside. Press <kbd className="font-mono text-xs px-1 py-0.5 rounded bg-bg-sidebar hairline">Esc</kbd> or click the backdrop to close.
                </ModalDescription>
              </ModalHeader>
              <p className="text-sm text-fg-muted leading-relaxed mt-2">
                Modals use Radix Dialog under the hood — accessible, keyboard-navigable, and focus-trapped out of the box. The editorial treatment uses Fraunces for the title and hairline borders for structure.
              </p>
              <ModalFooter>
                <ModalClose asChild>
                  <Button variant="ghost" size="sm">Cancel</Button>
                </ModalClose>
                <Button variant="primary" size="sm" onClick={() => setModalOpen(false)}>
                  Confirm
                </Button>
              </ModalFooter>
            </ModalContent>
          </Modal>
        </Section>

        {/* 7. Tooltip */}
        <Section title="Tooltip">
          <div className="flex items-center gap-6 flex-wrap">
            {(["top", "right", "bottom", "left"] as const).map((side) => (
              <Tooltip key={side} content={`Tooltip on the ${side}`} side={side}>
                <Button variant="secondary" size="sm">{side}</Button>
              </Tooltip>
            ))}
            <Tooltip content={<span>Rich <strong>HTML</strong> tooltip content</span>}>
              <span className="underline decoration-dotted text-fg-muted cursor-help text-sm">
                hover me
              </span>
            </Tooltip>
          </div>
        </Section>

        {/* 8. Empty States */}
        <Section title="Empty State">
          <div className="flex gap-8 flex-wrap">
            <div className="flex-1 min-w-[280px] border border-[color:var(--hairline)] rounded-[var(--radius-lg)]">
              <EmptyState
                heading="No job descriptions"
                body="Create your first JD to start scoring candidates against it."
                action={{ label: "Create JD", onClick: () => {} }}
              />
            </div>
            <div className="flex-1 min-w-[280px] border border-[color:var(--hairline)] rounded-[var(--radius-lg)]">
              <EmptyState
                icon={<FileText size={36} />}
                heading="No results found"
                body="Try adjusting your filters or search query."
              />
            </div>
          </div>
        </Section>

        {/* 9. Skeleton */}
        <Section title="Skeleton">
          <div className="flex flex-col gap-4 max-w-lg">
            <div className="flex items-center gap-3">
              <SkeletonAvatar size="md" />
              <SkeletonText lines={2} className="flex-1" />
            </div>
            <Skeleton className="h-6 w-1/3" />
            <SkeletonText lines={3} />
            <Skeleton className="h-48 w-full rounded-[var(--radius-lg)]" />
          </div>
          <div className="border border-[color:var(--hairline)] rounded-[var(--radius-lg)] overflow-hidden max-w-2xl">
            {Array.from({ length: 5 }).map((_, i) => (
              <SkeletonTableRow key={i} cols={4} />
            ))}
          </div>
        </Section>

        {/* 10. Pagination (standalone) */}
        <Section title="Pagination">
          <div className="border border-[color:var(--hairline)] rounded-[var(--radius-md)] max-w-xl">
            <Pagination
              total={234}
              page={page}
              pageSize={pageSize}
              onPageChange={setPage}
              onPageSizeChange={setPageSize}
            />
          </div>
        </Section>

        {/* 11. Score Visualization */}
        <Section title="Score Visualization">
          <div className="flex flex-col gap-8">
            <div>
              <p className="text-xs font-mono text-fg-subtle mb-3 uppercase tracking-wide">Mini Bar</p>
              <div className="flex flex-col gap-2">
                {[100, 85, 72, 55, 30].map((s) => (
                  <div key={s} className="flex items-center gap-4">
                    <span className="font-mono text-xs text-fg-muted w-8">{s}</span>
                    <ScoreBar score={s} />
                  </div>
                ))}
              </div>
            </div>
            <div className="flex gap-12 flex-wrap items-start">
              <div>
                <p className="text-xs font-mono text-fg-subtle mb-3 uppercase tracking-wide">Donut (200px, segments)</p>
                <ScoreDonut score={85} segments={DONUT_SEGMENTS} size={200} />
              </div>
              <div>
                <p className="text-xs font-mono text-fg-subtle mb-3 uppercase tracking-wide">Donut (140px, simple)</p>
                <ScoreDonut score={72} size={140} />
              </div>
            </div>
            <div>
              <p className="text-xs font-mono text-fg-subtle mb-3 uppercase tracking-wide">Radar (400px)</p>
              <ScoreRadar data={RADAR_DATA} size={400} />
            </div>
          </div>
        </Section>
      </div>
    </TooltipProvider>
  );
}
