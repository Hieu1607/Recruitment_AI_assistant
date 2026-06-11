import { useEffect, useRef, useState } from "react";

type UseResizableSidebarOptions = {
  storageKey: string;
  defaultWidth: number;
  minWidth: number;
  maxWidth: number;
};

type StoredSidebarState = {
  width?: number;
  collapsed?: boolean;
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function readStoredState(storageKey: string): StoredSidebarState | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return null;
    return JSON.parse(raw) as StoredSidebarState;
  } catch {
    return null;
  }
}

export function useResizableSidebar({
  storageKey,
  defaultWidth,
  minWidth,
  maxWidth,
}: UseResizableSidebarOptions) {
  const stored = readStoredState(storageKey);
  const initialWidth = clamp(stored?.width ?? defaultWidth, minWidth, maxWidth);

  const [width, setWidth] = useState(initialWidth);
  const [isCollapsed, setIsCollapsed] = useState(Boolean(stored?.collapsed));
  const dragCleanupRef = useRef<null | (() => void)>(null);

  useEffect(() => {
    window.localStorage.setItem(
      storageKey,
      JSON.stringify({
        width,
        collapsed: isCollapsed,
      }),
    );
  }, [isCollapsed, storageKey, width]);

  useEffect(() => () => dragCleanupRef.current?.(), []);

  function collapse() {
    setIsCollapsed(true);
  }

  function expand() {
    setIsCollapsed(false);
  }

  function toggle() {
    setIsCollapsed((prev) => !prev);
  }

  function startResize(event: React.MouseEvent<HTMLElement> | React.PointerEvent<HTMLElement>) {
    event.preventDefault();

    const startX = event.clientX;
    const startWidth = width;
    const previousCursor = document.body.style.cursor;
    const previousUserSelect = document.body.style.userSelect;

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const handleMove = (moveEvent: MouseEvent | PointerEvent) => {
      const nextWidth = clamp(startWidth + (moveEvent.clientX - startX), minWidth, maxWidth);
      setWidth(nextWidth);
      setIsCollapsed(false);
    };

    const cleanup = () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("mouseup", cleanup);
      window.removeEventListener("pointerup", cleanup);
      window.removeEventListener("mouseleave", cleanup);
      window.removeEventListener("pointercancel", cleanup);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousUserSelect;
      dragCleanupRef.current = null;
    };

    dragCleanupRef.current?.();
    dragCleanupRef.current = cleanup;

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("mouseup", cleanup);
    window.addEventListener("pointerup", cleanup);
    window.addEventListener("mouseleave", cleanup);
    window.addEventListener("pointercancel", cleanup);
  }

  return {
    width,
    isCollapsed,
    currentWidth: isCollapsed ? 0 : width,
    collapse,
    expand,
    toggle,
    startResize,
  };
}
