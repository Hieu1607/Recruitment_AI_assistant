import { cn } from "@/lib/cn";
import type { CSSProperties } from "react";

export interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  rounded?: boolean;
  style?: CSSProperties;
}

export function Skeleton({ className, width, height, rounded = false, style }: SkeletonProps) {
  return (
    <span
      className={cn(
        "block animate-pulse bg-gradient-to-r from-[color:var(--hairline)] via-[color:var(--hairline-strong)] to-[color:var(--hairline)]",
        "bg-[length:200%_100%]",
        rounded ? "rounded-full" : "rounded-[var(--radius-sm)]",
        className
      )}
      style={{
        width: width !== undefined ? (typeof width === "number" ? `${width}px` : width) : undefined,
        height: height !== undefined ? (typeof height === "number" ? `${height}px` : height) : undefined,
        animation: "skeleton-shimmer 1.5s ease-in-out infinite",
        ...style,
      }}
      aria-hidden="true"
    />
  );
}

export function SkeletonText({
  lines = 3,
  className,
}: {
  lines?: number;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className="h-4"
          style={{ width: i === lines - 1 && lines > 1 ? "60%" : "100%" } as CSSProperties}
        />
      ))}
    </div>
  );
}

export function SkeletonAvatar({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const sz = { sm: 24, md: 32, lg: 40 }[size];
  return <Skeleton width={sz} height={sz} rounded className="shrink-0" />;
}

export function SkeletonTableRow({ cols = 4 }: { cols?: number }) {
  return (
    <div className="flex items-center gap-4 px-4 h-12 hairline-b">
      {Array.from({ length: cols }).map((_, i) => (
        <Skeleton key={i} className="h-4 flex-1" />
      ))}
    </div>
  );
}
