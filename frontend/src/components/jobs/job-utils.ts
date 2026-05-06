import type { JobResponse } from "@/api";

export type JobFilter = "all" | "active" | "archived";

function toDate(iso: string | null): Date | null {
  if (!iso) return null;
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function formatJobStatus(status: string): string {
  return status.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function formatRelativeDate(iso: string | null): string {
  const date = toDate(iso);
  if (!date) return "Unknown";

  const diffMs = Date.now() - date.getTime();
  const diffMinutes = Math.floor(diffMs / 60_000);
  if (diffMinutes < 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 30) return `${diffDays}d ago`;

  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: date.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
  });
}

export function formatAbsoluteDate(iso: string | null): string {
  const date = toDate(iso);
  if (!date) return "Unknown";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function isArchivedJob(job: JobResponse): boolean {
  return job.status.toLowerCase() === "archived" || Boolean(job.archived_at);
}

export function jobMatchesFilter(job: JobResponse, filter: JobFilter): boolean {
  if (filter === "all") return true;
  return filter === "archived" ? isArchivedJob(job) : !isArchivedJob(job);
}

export const fieldClasses =
  "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated px-3 py-2.5 text-sm text-fg outline-none transition-colors placeholder:text-fg-subtle focus:border-[color:var(--hairline-strong)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

export const panelClasses =
  "rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated";

