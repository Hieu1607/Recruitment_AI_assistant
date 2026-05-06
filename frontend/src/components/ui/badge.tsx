import { cn } from "@/lib/cn";
import * as React from "react";

export type BadgeVariant = "neutral" | "warning" | "success" | "danger";
export type BadgeSize = "sm" | "md";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: BadgeSize;
  dot?: boolean;
}

const variantClasses: Record<BadgeVariant, { wrap: string; dot: string }> = {
  neutral: {
    wrap: "bg-[rgba(138,138,133,0.12)] text-fg-muted",
    dot: "bg-neutral",
  },
  warning: {
    wrap: "bg-[rgba(184,138,62,0.14)] text-warning",
    dot: "bg-warning",
  },
  success: {
    wrap: "bg-[rgba(74,124,89,0.14)] text-success",
    dot: "bg-success",
  },
  danger: {
    wrap: "bg-[rgba(184,68,46,0.14)] text-danger",
    dot: "bg-danger",
  },
};

const sizeClasses: Record<BadgeSize, { wrap: string; dot: string }> = {
  sm: { wrap: "text-[0.6875rem] px-2 py-0.5 gap-1.5 h-5", dot: "w-1.5 h-1.5" },
  md: { wrap: "text-xs px-2.5 py-1 gap-1.5 h-6", dot: "w-2 h-2" },
};

export function Badge({
  variant = "neutral",
  size = "md",
  dot = true,
  children,
  className,
  ...props
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full font-sans font-medium whitespace-nowrap",
        variantClasses[variant].wrap,
        sizeClasses[size].wrap,
        className
      )}
      {...props}
    >
      {dot && (
        <span
          aria-hidden="true"
          className={cn(
            "shrink-0 rounded-full",
            variantClasses[variant].dot,
            sizeClasses[size].dot
          )}
        />
      )}
      {children}
    </span>
  );
}

export type UploadStatus = "pending" | "processing" | "completed" | "failed";
export type ProfileStatus = "pending" | "processing" | "completed" | "failed" | "active";
export type MatchRunStatus = "queued" | "running" | "completed" | "failed";
export type SentStatus = "not_sent" | "sent" | "failed";

export type AnyStatus = UploadStatus | ProfileStatus | MatchRunStatus | SentStatus;

function statusToVariant(status: AnyStatus): BadgeVariant {
  switch (status) {
    case "completed":
    case "sent":
    case "active":
      return "success";
    case "processing":
    case "running":
      return "warning";
    case "failed":
      return "danger";
    case "pending":
    case "queued":
    case "not_sent":
    default:
      return "neutral";
  }
}

function statusToLabel(status: AnyStatus): string {
  return status
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export interface StatusBadgeProps extends Omit<BadgeProps, "variant" | "children"> {
  status: AnyStatus;
  label?: string;
}

export function StatusBadge({ status, label, ...props }: StatusBadgeProps) {
  return (
    <Badge variant={statusToVariant(status)} {...props}>
      {label ?? statusToLabel(status)}
    </Badge>
  );
}
