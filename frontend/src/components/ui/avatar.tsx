import * as React from "react";
import { cn } from "@/lib/cn";

export type AvatarSize = "sm" | "md" | "lg" | "xl";

export interface AvatarProps extends React.HTMLAttributes<HTMLSpanElement> {
  src?: string;
  name?: string;
  size?: AvatarSize;
  alt?: string;
}

const sizeClasses: Record<AvatarSize, { wrap: string; text: string; img: string }> = {
  sm: { wrap: "h-6 w-6", text: "text-[10px]", img: "h-6 w-6" },
  md: { wrap: "h-8 w-8", text: "text-[13px]", img: "h-8 w-8" },
  lg: { wrap: "h-10 w-10", text: "text-base", img: "h-10 w-10" },
  xl: { wrap: "h-14 w-14", text: "text-[22px]", img: "h-14 w-14" },
};

const PALETTE = [
  "#1F3A2E",
  "#2A5A78",
  "#5A3A7E",
  "#7A3A3A",
  "#3A5A3A",
  "#5A4A2A",
];

function hashName(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = (h * 31 + name.charCodeAt(i)) >>> 0;
  }
  return h;
}

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function getColor(name: string): string {
  return PALETTE[hashName(name) % PALETTE.length];
}

export function Avatar({ src, name, size = "md", alt, className, ...props }: AvatarProps) {
  const { wrap, text, img } = sizeClasses[size];
  const initials = name ? getInitials(name) : "?";
  const bg = name ? getColor(name) : "#1F3A2E";

  return (
    <span
      role="img"
      aria-label={alt ?? name ?? "avatar"}
      className={cn(
        "relative inline-flex items-center justify-center shrink-0 rounded-full overflow-hidden select-none",
        wrap,
        className
      )}
      style={!src ? { backgroundColor: bg } : undefined}
      {...props}
    >
      {src ? (
        <img
          src={src}
          alt={alt ?? name ?? "avatar"}
          className={cn("object-cover rounded-full", img)}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = "none";
          }}
        />
      ) : (
        <span
          className={cn("font-sans font-semibold leading-none text-white", text)}
          aria-hidden="true"
        >
          {initials}
        </span>
      )}
    </span>
  );
}
