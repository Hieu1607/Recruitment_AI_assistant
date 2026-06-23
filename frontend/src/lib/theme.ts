import { create } from "zustand";

export type Theme = "light" | "dark" | "system";
const STORAGE_KEY = "easyhr.theme";
type AppliedTheme = "light" | "dark";
type RouteThemeOverride = AppliedTheme | null;

function loadStoredTheme(): Theme {
  if (typeof localStorage === "undefined") return "system";
  const v = localStorage.getItem(STORAGE_KEY);
  if (v === "light" || v === "dark" || v === "system") return v;
  return "system";
}

function resolveAppliedTheme(theme: Theme): AppliedTheme {
  if (theme === "system") {
    if (typeof matchMedia === "undefined") return "light";
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return theme;
}

function applyToDocument(applied: AppliedTheme) {
  if (typeof document === "undefined") return;
  document.documentElement.setAttribute("data-theme", applied);
}

export function shouldForceLightTheme(pathname: string): boolean {
  return pathname === "/" || pathname.startsWith("/apply/") || /^\/interviews\/[^/]+$/.test(pathname);
}

function resolveRouteThemeOverride(pathname: string): RouteThemeOverride {
  return shouldForceLightTheme(pathname) ? "light" : null;
}

function loadInitialRouteThemeOverride(): RouteThemeOverride {
  if (typeof window === "undefined") return null;
  return resolveRouteThemeOverride(window.location.pathname);
}

function resolveDocumentTheme(theme: Theme, routeThemeOverride: RouteThemeOverride): AppliedTheme {
  return routeThemeOverride ?? resolveAppliedTheme(theme);
}

interface ThemeState {
  theme: Theme; // user preference (light | dark | system)
  applied: AppliedTheme; // resolved value applied to <html data-theme>
  routeThemeOverride: RouteThemeOverride;
  setTheme: (theme: Theme) => void;
  setRouteThemeOverride: (routeThemeOverride: RouteThemeOverride) => void;
}

export const useTheme = create<ThemeState>((set, get) => {
  const initial = loadStoredTheme();
  const routeThemeOverride = loadInitialRouteThemeOverride();
  const applied = resolveDocumentTheme(initial, routeThemeOverride);
  applyToDocument(applied);

  return {
    theme: initial,
    applied,
    routeThemeOverride,
    setTheme: (theme: Theme) => {
      localStorage.setItem(STORAGE_KEY, theme);
      const applied = resolveDocumentTheme(theme, get().routeThemeOverride);
      applyToDocument(applied);
      set({ theme, applied });
    },
    setRouteThemeOverride: (nextRouteThemeOverride: RouteThemeOverride) => {
      const applied = resolveDocumentTheme(get().theme, nextRouteThemeOverride);
      applyToDocument(applied);
      set({ routeThemeOverride: nextRouteThemeOverride, applied });
    },
  };
});

// Listen for system preference changes when user picked "system"
if (typeof matchMedia !== "undefined") {
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    const { theme, routeThemeOverride, setTheme } = useTheme.getState();
    if (theme === "system" && routeThemeOverride === null) setTheme("system"); // re-resolve and re-apply
  });
}
