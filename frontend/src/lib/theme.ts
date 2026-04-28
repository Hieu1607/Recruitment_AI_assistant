import { create } from "zustand";

export type Theme = "light" | "dark" | "system";
const STORAGE_KEY = "recruitai.theme";

function loadStoredTheme(): Theme {
  if (typeof localStorage === "undefined") return "system";
  const v = localStorage.getItem(STORAGE_KEY);
  if (v === "light" || v === "dark" || v === "system") return v;
  return "system";
}

function resolveAppliedTheme(theme: Theme): "light" | "dark" {
  if (theme === "system") {
    if (typeof matchMedia === "undefined") return "light";
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return theme;
}

function applyToDocument(applied: "light" | "dark") {
  if (typeof document === "undefined") return;
  document.documentElement.setAttribute("data-theme", applied);
}

interface ThemeState {
  theme: Theme;          // user preference (light | dark | system)
  applied: "light" | "dark"; // resolved value applied to <html data-theme>
  setTheme: (theme: Theme) => void;
}

export const useTheme = create<ThemeState>((set) => {
  const initial = loadStoredTheme();
  const applied = resolveAppliedTheme(initial);
  applyToDocument(applied);

  return {
    theme: initial,
    applied,
    setTheme: (theme: Theme) => {
      localStorage.setItem(STORAGE_KEY, theme);
      const applied = resolveAppliedTheme(theme);
      applyToDocument(applied);
      set({ theme, applied });
    }
  };
});

// Listen for system preference changes when user picked "system"
if (typeof matchMedia !== "undefined") {
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    const { theme, setTheme } = useTheme.getState();
    if (theme === "system") setTheme("system"); // re-resolve and re-apply
  });
}
