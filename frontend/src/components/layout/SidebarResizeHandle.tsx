import { cn } from "@/lib/cn";

export function SidebarResizeHandle({
  className,
  onPointerDown,
  testId,
}: {
  className?: string;
  onPointerDown: (event: React.MouseEvent<HTMLDivElement> | React.PointerEvent<HTMLDivElement>) => void;
  testId: string;
}) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      data-testid={testId}
      onPointerDown={onPointerDown}
      onMouseDown={onPointerDown}
      className={cn(
        "relative z-10 h-full w-3 shrink-0 cursor-col-resize touch-none",
        className,
      )}
    >
      <div className="mx-auto h-full w-px bg-[color:var(--hairline)] transition-colors hover:bg-[color:var(--hairline-strong)]" />
    </div>
  );
}
