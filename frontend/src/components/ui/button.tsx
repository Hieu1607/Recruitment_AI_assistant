import * as React from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "icon";
export type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  icon?: React.ReactNode;
  asChild?: boolean;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-accent-fg hover:bg-accent-hover focus-visible:outline-accent",
  secondary:
    "bg-bg-elevated text-fg border border-hairline-strong hover:bg-bg-sidebar focus-visible:outline-accent",
  ghost:
    "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] focus-visible:outline-accent",
  danger:
    "bg-[rgba(184,68,46,0.10)] text-danger border border-[rgba(184,68,46,0.30)] hover:bg-[rgba(184,68,46,0.18)] focus-visible:outline-danger",
  icon:
    "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] focus-visible:outline-accent",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "h-7 px-3 text-xs gap-1.5 rounded-[var(--radius-sm)]",
  md: "h-9 px-4 text-sm gap-2 rounded-[var(--radius-md)]",
  lg: "h-11 px-5 text-[0.9375rem] gap-2.5 rounded-[var(--radius-md)]",
};

const iconSizeClasses: Record<ButtonSize, string> = {
  sm: "h-7 w-7 rounded-[var(--radius-sm)]",
  md: "h-9 w-9 rounded-[var(--radius-md)]",
  lg: "h-11 w-11 rounded-[var(--radius-md)]",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = "primary",
      size = "md",
      loading = false,
      icon,
      children,
      className,
      disabled,
      ...props
    },
    ref
  ) => {
    const isIcon = variant === "icon" || (!children && icon);
    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={cn(
          "inline-flex items-center justify-center font-sans font-medium transition-colors",
          "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          variantClasses[variant],
          isIcon ? iconSizeClasses[size] : sizeClasses[size],
          className
        )}
        {...props}
      >
        {loading ? (
          <Loader2
            className="animate-spin"
            size={size === "sm" ? 14 : size === "lg" ? 18 : 16}
            strokeWidth={2}
          />
        ) : (
          icon && (
            <span className="shrink-0" aria-hidden="true">
              {icon}
            </span>
          )
        )}
        {children && !isIcon && <span className="inline-flex items-center">{children}</span>}
      </button>
    );
  }
);
Button.displayName = "Button";
