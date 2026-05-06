import { type ReactNode } from "react";
import { cn } from "@/lib/cn";
import { Button } from "./button";

function DefaultIcon() {
  return (
    <svg
      width="40"
      height="40"
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect x="6" y="8" width="28" height="24" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <path d="M14 16h12M14 22h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M20 28v4M16 32h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export interface EmptyStateAction {
  label: string;
  onClick: () => void;
}

export interface EmptyStateProps {
  icon?: ReactNode;
  heading: string;
  body?: string;
  action?: EmptyStateAction;
  className?: string;
}

export function EmptyState({ icon, heading, body, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center gap-4 py-16 max-w-sm mx-auto",
        className
      )}
    >
      <span className="text-fg-subtle">{icon ?? <DefaultIcon />}</span>
      <div className="flex flex-col gap-1.5">
        <p className="font-display text-xl font-medium text-fg">{heading}</p>
        {body && <p className="font-sans text-sm text-fg-muted leading-relaxed">{body}</p>}
      </div>
      {action && (
        <Button variant="primary" size="sm" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </div>
  );
}
