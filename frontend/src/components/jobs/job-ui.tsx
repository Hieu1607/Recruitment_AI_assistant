import { Badge, type BadgeVariant } from "@/components/ui";
import { cn } from "@/lib/cn";
import { CheckCircle2 } from "lucide-react";
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
        "h-6 gap-1.5 border border-[rgba(74,124,89,0.38)] bg-[rgba(74,124,89,0.14)] px-2.5 text-[0.72rem] font-semibold text-success shadow-[0_1px_0_rgba(255,255,255,0.55)_inset]",
        className,
      )}
    >
      <CheckCircle2 size={13} strokeWidth={2} aria-hidden="true" />
      Selected job
    </Badge>
  );
}
