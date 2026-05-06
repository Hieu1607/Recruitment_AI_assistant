import { useState, useMemo, useCallback, type ReactNode } from "react";
import { ChevronUp, ChevronDown, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/cn";

export type SortDirection = "asc" | "desc";

export interface ColumnDef<T> {
  key: string;
  header: string;
  render?: (row: T, index: number) => ReactNode;
  sortable?: boolean;
  width?: number | string;
  className?: string;
}

export interface DataTableProps<T extends { id: string | number }> {
  columns: ColumnDef<T>[];
  data: T[];
  loading?: boolean;
  emptyState?: ReactNode;
  selectable?: boolean;
  onSelectionChange?: (ids: (string | number)[]) => void;
  className?: string;
  rowClassName?: (row: T) => string;
  onRowClick?: (row: T) => void;
  skeletonRows?: number;
}

type SortState = { key: string; dir: SortDirection } | null;

function ShimmerRow({ cols }: { cols: number }) {
  return (
    <tr aria-hidden="true">
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <span className="block h-4 rounded-[var(--radius-sm)] skeleton-shimmer" />
        </td>
      ))}
    </tr>
  );
}

export function DataTable<T extends { id: string | number }>({
  columns,
  data,
  loading = false,
  emptyState,
  selectable = false,
  onSelectionChange,
  className,
  rowClassName,
  onRowClick,
  skeletonRows = 8,
}: DataTableProps<T>) {
  const [sort, setSort] = useState<SortState>(null);
  const [selected, setSelected] = useState<Set<string | number>>(new Set());

  const totalCols = selectable ? columns.length + 1 : columns.length;

  const sortedData = useMemo(() => {
    if (!sort) return data;
    return [...data].sort((a, b) => {
      const av = (a as Record<string, unknown>)[sort.key];
      const bv = (b as Record<string, unknown>)[sort.key];
      if (av === bv) return 0;
      const cmp = String(av).localeCompare(String(bv), undefined, { numeric: true });
      return sort.dir === "asc" ? cmp : -cmp;
    });
  }, [data, sort]);

  const handleSort = useCallback((key: string) => {
    setSort((prev) => {
      if (!prev || prev.key !== key) return { key, dir: "asc" };
      if (prev.dir === "asc") return { key, dir: "desc" };
      return null;
    });
  }, []);

  const allSelected =
    data.length > 0 && data.every((r) => selected.has(r.id));
  const someSelected = !allSelected && data.some((r) => selected.has(r.id));

  const toggleAll = useCallback(() => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (allSelected) {
        data.forEach((r) => next.delete(r.id));
      } else {
        data.forEach((r) => next.add(r.id));
      }
      onSelectionChange?.([...next]);
      return next;
    });
  }, [allSelected, data, onSelectionChange]);

  const toggleRow = useCallback(
    (id: string | number) => {
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        onSelectionChange?.([...next]);
        return next;
      });
    },
    [onSelectionChange]
  );

  return (
    <div className={cn("w-full overflow-x-auto", className)}>
      <table className="w-full border-collapse font-sans text-sm">
        <thead className="sticky top-0 z-10 bg-bg">
          <tr className="hairline-b">
            {selectable && (
              <th className="w-10 px-4 py-2.5 text-left">
                <input
                  type="checkbox"
                  aria-label="Select all"
                  checked={allSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = someSelected;
                  }}
                  onChange={toggleAll}
                  className="h-4 w-4 rounded-[var(--radius-sm)] accent-accent cursor-pointer"
                />
              </th>
            )}
            {columns.map((col) => (
              <th
                key={col.key}
                style={{ width: col.width }}
                className={cn(
                  "px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wide text-fg-subtle select-none",
                  col.sortable && "cursor-pointer hover:text-fg transition-colors",
                  col.className
                )}
                onClick={col.sortable ? () => handleSort(col.key) : undefined}
              >
                <span className="inline-flex items-center gap-1">
                  {col.header}
                  {col.sortable &&
                    (sort?.key === col.key ? (
                      sort.dir === "asc" ? (
                        <ChevronUp size={12} strokeWidth={2} />
                      ) : (
                        <ChevronDown size={12} strokeWidth={2} />
                      )
                    ) : (
                      <ChevronsUpDown size={12} strokeWidth={1.5} className="opacity-40" />
                    ))}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading ? (
            Array.from({ length: skeletonRows }).map((_, i) => (
              <ShimmerRow key={i} cols={totalCols} />
            ))
          ) : data.length === 0 ? (
            <tr>
              <td colSpan={totalCols} className="py-16 text-center">
                {emptyState ?? (
                  <span className="text-fg-muted text-sm">No results</span>
                )}
              </td>
            </tr>
          ) : (
            sortedData.map((row, i) => (
              <tr
                key={row.id}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                className={cn(
                  "hairline-b transition-colors",
                  onRowClick && "cursor-pointer",
                  "hover:bg-[color:var(--hairline)]",
                  selected.has(row.id) && "bg-[rgba(31,58,46,0.04)]",
                  rowClassName?.(row)
                )}
              >
                {selectable && (
                  <td className="w-10 px-4 py-3">
                    <input
                      type="checkbox"
                      aria-label="Select row"
                      checked={selected.has(row.id)}
                      onChange={() => toggleRow(row.id)}
                      onClick={(e) => e.stopPropagation()}
                      className="h-4 w-4 rounded-[var(--radius-sm)] accent-accent cursor-pointer"
                    />
                  </td>
                )}
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={cn("px-4 py-3 text-fg", col.className)}
                  >
                    {col.render
                      ? col.render(row, i)
                      : String((row as Record<string, unknown>)[col.key] ?? "")}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
