import { type ReactNode } from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cn } from "@/lib/cn";

export const TooltipProvider = TooltipPrimitive.Provider;
export const TooltipRoot = TooltipPrimitive.Root;
export const TooltipTrigger = TooltipPrimitive.Trigger;

export interface TooltipContentProps {
  children: ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  className?: string;
  sideOffset?: number;
}

export function TooltipContent({
  children,
  side = "top",
  align = "center",
  className,
  sideOffset = 6,
}: TooltipContentProps) {
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        side={side}
        align={align}
        sideOffset={sideOffset}
        className={cn(
          "z-50 max-w-[240px] px-2.5 py-1.5",
          "bg-bg-elevated border border-[color:var(--hairline-strong)]",
          "rounded-[var(--radius-sm)] shadow-[var(--shadow-md)]",
          "font-sans text-xs text-fg leading-snug",
          "data-[state=delayed-open]:animate-in data-[state=delayed-open]:fade-in-0",
          "data-[state=delayed-open]:zoom-in-95",
          "data-[state=closed]:animate-out data-[state=closed]:fade-out-0",
          "data-[state=closed]:zoom-out-95",
          className
        )}
      >
        {children}
        <TooltipPrimitive.Arrow
          className="fill-bg-elevated"
          style={{ filter: "drop-shadow(0 1px 0 var(--hairline-strong))" }}
          width={8}
          height={4}
        />
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  );
}

export interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  side?: TooltipContentProps["side"];
  align?: TooltipContentProps["align"];
  delayDuration?: number;
  disabled?: boolean;
}

export function Tooltip({
  content,
  children,
  side = "top",
  align = "center",
  delayDuration = 400,
  disabled = false,
}: TooltipProps) {
  if (disabled) return <>{children}</>;
  return (
    <TooltipPrimitive.Root delayDuration={delayDuration}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipContent side={side} align={align}>
        {content}
      </TooltipContent>
    </TooltipPrimitive.Root>
  );
}
