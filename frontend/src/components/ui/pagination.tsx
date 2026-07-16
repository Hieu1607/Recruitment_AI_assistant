import { cn } from "@/lib/cn";
import { ChevronDown, ChevronLeft, ChevronRight } from "lucide-react";

export interface PaginationProps {
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (size: number) => void;
  pageSizeOptions?: number[];
  className?: string;
}

type PageItem = number | "ellipsis-start" | "ellipsis-end";

function getPageItems(page: number, totalPages: number): PageItem[] {
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_, index) => index + 1);
  }

  if (page <= 4) return [1, 2, 3, 4, 5, "ellipsis-end", totalPages];
  if (page >= totalPages - 3) {
    return [1, "ellipsis-start", totalPages - 4, totalPages - 3, totalPages - 2, totalPages - 1, totalPages];
  }
  return [1, "ellipsis-start", page - 1, page, page + 1, "ellipsis-end", totalPages];
}

export function Pagination({
  total,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [10, 20, 50, 100, 200],
  className,
}: PaginationProps) {
  const start = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, total);
  const totalPages = Math.ceil(total / pageSize);
  const isFirst = page <= 1;
  const isLast = page >= totalPages;
  const pageItems = getPageItems(page, totalPages);

  return (
    <div
      className={cn(
        "flex flex-wrap items-center justify-between gap-4 py-3 px-1 font-sans text-sm text-fg-muted",
        className
      )}
    >
      {/* Count */}
      <span className="tabular-nums whitespace-nowrap">
        {total === 0
          ? "No results"
          : `Showing ${start}–${end} of ${total.toLocaleString()}`}
      </span>

      <div className="flex flex-wrap items-center gap-3">
        {/* Page-size selector */}
        {onPageSizeChange && (
          <div className="flex items-center gap-2">
            <span className="text-fg-subtle text-xs">Rows</span>
            <div className="relative">
              <select
                value={pageSize}
                onChange={(e) => onPageSizeChange(Number(e.target.value))}
                className={cn(
                  "h-8 pl-2 pr-7 text-sm rounded-[var(--radius-sm)] border border-[color:var(--hairline-strong)]",
                  "bg-bg text-fg appearance-none cursor-pointer",
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
              <ChevronDown
                size={14}
                strokeWidth={2}
                aria-hidden="true"
                className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-fg-muted"
              />
            </div>
          </div>
        )}

        {/* Page navigation */}
        <div className="flex flex-wrap items-center gap-1">
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

          {pageItems.map((item) =>
            typeof item === "number" ? (
              <button
                key={item}
                type="button"
                onClick={() => onPageChange(item)}
                aria-label={`Go to page ${item}`}
                aria-current={item === page ? "page" : undefined}
                className={cn(
                  "h-8 min-w-8 px-2 inline-flex items-center justify-center rounded-[var(--radius-sm)]",
                  "border text-xs tabular-nums transition-colors",
                  "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent",
                  item === page
                    ? "border-accent bg-accent text-white"
                    : "border-[color:var(--hairline)] text-fg hover:bg-[color:var(--hairline)] hover:border-[color:var(--hairline-strong)]",
                )}
              >
                {item}
              </button>
            ) : (
              <span
                key={item}
                className="h-8 min-w-6 inline-flex items-center justify-center text-fg-subtle"
                aria-hidden="true"
              >
                …
              </span>
            ),
          )}

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
