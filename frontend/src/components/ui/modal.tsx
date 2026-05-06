import { type ReactNode } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "@/lib/cn";

export { Dialog as DialogPrimitive };

export type ModalSize = "default" | "large";

export interface ModalProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
}

export function Modal({ open, onOpenChange, children }: ModalProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      {children}
    </Dialog.Root>
  );
}

export const ModalTrigger = Dialog.Trigger;

export interface ModalContentProps {
  children: ReactNode;
  size?: ModalSize;
  className?: string;
  showClose?: boolean;
}

export function ModalContent({
  children,
  size = "default",
  className,
  showClose = true,
}: ModalContentProps) {
  return (
    <Dialog.Portal>
      <Dialog.Overlay
        className={cn(
          "fixed inset-0 z-40 bg-black/40 backdrop-blur-sm",
          "data-[state=open]:animate-in data-[state=closed]:animate-out",
          "data-[state=open]:fade-in-0 data-[state=closed]:fade-out-0"
        )}
      />
      <Dialog.Content
        className={cn(
          "fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2",
          "w-[calc(100%-2rem)] bg-bg-elevated rounded-[var(--radius-lg)]",
          "shadow-[var(--shadow-lg)] border border-[color:var(--hairline)]",
          "p-6 focus:outline-none",
          "data-[state=open]:animate-in data-[state=closed]:animate-out",
          "data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95",
          "data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
          "data-[state=open]:slide-in-from-bottom-2",
          size === "default" ? "max-w-[560px]" : "max-w-[720px]",
          className
        )}
      >
        {children}
        {showClose && (
          <Dialog.Close
            className={cn(
              "absolute right-4 top-4 inline-flex h-7 w-7 items-center justify-center rounded-[var(--radius-sm)]",
              "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors",
              "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
            )}
            aria-label="Close"
          >
            <X size={16} strokeWidth={1.75} />
          </Dialog.Close>
        )}
      </Dialog.Content>
    </Dialog.Portal>
  );
}

export interface ModalHeaderProps {
  children: ReactNode;
  className?: string;
}

export function ModalHeader({ children, className }: ModalHeaderProps) {
  return (
    <div className={cn("mb-4 pr-8", className)}>
      {children}
    </div>
  );
}

export function ModalTitle({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <Dialog.Title
      className={cn("font-display text-xl font-medium text-fg leading-tight", className)}
    >
      {children}
    </Dialog.Title>
  );
}

export function ModalDescription({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <Dialog.Description
      className={cn("mt-1 text-sm text-fg-muted font-sans", className)}
    >
      {children}
    </Dialog.Description>
  );
}

export interface ModalFooterProps {
  children: ReactNode;
  className?: string;
}

export function ModalFooter({ children, className }: ModalFooterProps) {
  return (
    <div
      className={cn(
        "mt-6 flex items-center justify-end gap-2 pt-4 border-t border-[color:var(--hairline)]",
        className
      )}
    >
      {children}
    </div>
  );
}

export const ModalClose = Dialog.Close;
