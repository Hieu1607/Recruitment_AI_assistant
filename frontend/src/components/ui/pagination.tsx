import { cn } from "@/lib/cn";
import { ChevronLeft, ChevronRight } from "lucide-react";

export interface PaginationProps {
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (size: number) => void;
  pageSizeOptions?: number[];
  className?: string;
}

export function Pagination({
  total,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [50, 100, 200],
  className,
}: PaginationProps) {
  const start = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, total);
  const totalPages = Math.ceil(total / pageSize);
  const isFirst = page <= 1;
  const isLast = page >= totalPages;

  return (
    <div
      className={cn(
        "flex items-center justify-between gap-4 py-3 px-1 font-sans text-sm text-fg-muted",
        className
      )}
    >
      {/* Count */}
      <span className="tabular-nums whitespace-nowrap">
        {total === 0
          ? "No results"
          : `Showing ${start}–${end} of ${total.toLocaleString()}`}
      </span>

      <div className="flex items-center gap-3">
        {/* Page-size selector */}
        {onPageSizeChange && (
          <div className="flex items-center gap-2">
            <span className="text-fg-subtle text-xs">Rows</span>
            <select
              value={pageSize}
              onChange={(e) => {
                onPageSizeChange(Number(e.target.value));
                onPageChange(1);
              }}
              className={cn(
                "h-8 pl-2 pr-6 text-sm rounded-[var(--radius-sm)] border border-[color:var(--hairline-strong)]",
                "bg-bg text-fg appearance-none",
                "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
              )}
              aria-label="Rows per page"
            >
              {pageSizeOptions.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Prev / Next */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => onPageChange(page - 1)}
            disabled={isFirst}
            aria-label="Previous page"
            className={cn(
              "h-8 w-8 inline-flex items-center justify-center rounded-[var(--radius-sm)]",
              "border border-[color:var(--hairline)] transition-colors",
              "hover:bg-[color:var(--hairline)] hover:border-[color:var(--hairline-strong)]",
              "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent",
              "disabled:opacity-40 disabled:cursor-not-allowed"
            )}
          >
            <ChevronLeft size={14} strokeWidth={2} />
          </button>

          <span className="tabular-nums text-xs min-w-[4rem] text-center">
            {totalPages === 0 ? "—" : `${page} / ${totalPages}`}
          </span>

          <button
            type="button"
            onClick={() => onPageChange(page + 1)}
            disabled={isLast}
            aria-label="Next page"
            className={cn(
              "h-8 w-8 inline-flex items-center justify-center rounded-[var(--radius-sm)]",
              "border border-[color:var(--hairline)] transition-colors",
              "hover:bg-[color:var(--hairline)] hover:border-[color:var(--hairline-strong)]",
              "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent",
              "disabled:opacity-40 disabled:cursor-not-allowed"
            )}
          >
            <ChevronRight size={14} strokeWidth={2} />
          </button>
        </div>
      </div>
    </div>
  );
}
