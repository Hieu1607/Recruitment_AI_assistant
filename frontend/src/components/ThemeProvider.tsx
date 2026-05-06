import { useTheme } from "@/lib/theme";
import { useEffect } from "react";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const applied = useTheme((s) => s.applied);
  useEffect(() => {
    // Update meta theme-color for mobile browser chrome
    const meta = document.querySelector('meta[name="theme-color"]');
    const color = applied === "dark" ? "#0F1012" : "#FAFAF7";
    if (meta) meta.setAttribute("content", color);
    else {
      const m = document.createElement("meta");
      m.name = "theme-color";
      m.content = color;
      document.head.appendChild(m);
    }
  }, [applied]);
  return <>{children}</>;
}
