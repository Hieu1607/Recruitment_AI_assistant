import { Badge, type BadgeVariant } from "@/components/ui";
import { cn } from "@/lib/cn";
import { formatJobStatus } from "./job-utils";

function statusVariant(status: string): BadgeVariant {
  const normalized = status.toLowerCase();
  if (normalized === "active") return "success";
  if (normalized === "archived") return "warning";
  if (normalized === "deleted") return "danger";
  return "neutral";
}

export function JobStatusBadge({ status, className }: { status: string; className?: string }) {
  return (
    <Badge variant={statusVariant(status)} size="sm" className={className}>
      {formatJobStatus(status)}
    </Badge>
  );
}

export function CurrentWorkspaceBadge({ className }: { className?: string }) {
  return (
    <Badge
      variant="neutral"
      size="sm"
      dot={false}
      className={cn(
        "border border-[color:var(--hairline-strong)] bg-[rgba(31,58,46,0.08)] text-fg",
        className,
      )}
    >
      Current workspace
    </Badge>
  );
}
