import { useTheme } from "@/lib/theme";
import { useEffect } from "react";

export function PublicThemeBoundary({ children }: { children: React.ReactNode }) {
  const setRouteThemeOverride = useTheme((s) => s.setRouteThemeOverride);

  useEffect(() => {
    setRouteThemeOverride("light");
    return () => setRouteThemeOverride(null);
  }, [setRouteThemeOverride]);

  return <>{children}</>;
}
