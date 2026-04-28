import * as React from "react";
import { cn } from "@/lib/cn";

export interface FilterChipProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  icon?: React.ReactNode;
}

export function FilterChip({
  selected = false,
  icon,
  children,
  className,
  ...props
}: FilterChipProps) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={selected}
      className={cn(
        "inline-flex items-center gap-1.5 h-7 px-3 rounded-full font-sans text-[0.8125rem] transition-colors",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        selected
          ? "border border-accent bg-[rgba(31,58,46,0.08)] text-fg font-medium"
          : "border border-[color:var(--hairline)] bg-transparent text-fg-muted hover:text-fg hover:border-[color:var(--hairline-strong)]",
        className
      )}
      {...props}
    >
      {icon && (
        <span className="shrink-0" aria-hidden="true">
          {icon}
        </span>
      )}
      {children}
    </button>
  );
}

export interface FilterChipGroupProps {
  value: string | string[];
  onChange: (value: string | string[]) => void;
  multiple?: boolean;
  children: React.ReactNode;
  className?: string;
}

export interface FilterChipOptionProps extends Omit<FilterChipProps, "selected" | "onClick"> {
  value: string;
}

function FilterChipOption({ value: _value, ...props }: FilterChipOptionProps) {
  return <FilterChip {...props} />;
}

FilterChipOption.displayName = "FilterChipOption";

export { FilterChipOption };
