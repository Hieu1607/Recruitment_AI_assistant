import { Sun, Moon, Monitor, LogOut, Settings as SettingsIcon } from "lucide-react";
import { Link } from "react-router";
import { useTheme, type Theme } from "@/lib/theme";
import { routes } from "@/routes";
import { cn } from "@/lib/cn";

const themeOptions: { value: Theme; label: string; icon: typeof Sun }[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor }
];

export function UserMenu() {
  const theme = useTheme((s) => s.theme);
  const setTheme = useTheme((s) => s.setTheme);

  return (
    <details className="relative group">
      <summary className="list-none cursor-pointer">
        <span
          className="block size-9 rounded-full bg-[color:var(--hairline)] hairline overflow-hidden"
          aria-label="User menu"
        >
          {/* Avatar placeholder until Phase 2's Avatar component lands */}
          <span className="flex items-center justify-center w-full h-full font-mono text-xs text-fg-muted">
            R
          </span>
        </span>
      </summary>
      <div
        className="absolute right-0 top-full mt-2 w-56 bg-bg-elevated rounded-lg hairline shadow-lg p-1.5 z-50"
        role="menu"
      >
        <div className="px-3 py-2 hairline-b">
          <p className="font-sans text-sm font-medium text-fg">Recruiter</p>
          <p className="font-mono text-xs text-fg-subtle">user@recruitai.local</p>
        </div>
        <div className="py-1.5">
          <p className="px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-fg-subtle">
            Theme
          </p>
          <div className="flex gap-1 px-1.5">
            {themeOptions.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                type="button"
                onClick={() => setTheme(value)}
                className={cn(
                  "flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md font-sans text-xs transition-colors",
                  theme === value
                    ? "bg-accent text-accent-fg"
                    : "text-fg-muted hover:bg-[color:var(--hairline)] hover:text-fg"
                )}
                aria-pressed={theme === value}
              >
                <Icon size={12} strokeWidth={1.75} />
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="hairline-t py-1">
          <Link
            to={routes.settings}
            className="flex items-center gap-2 px-3 py-2 rounded-md font-sans text-sm text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors"
            role="menuitem"
          >
            <SettingsIcon size={14} strokeWidth={1.5} />
            Settings
          </Link>
          <button
            type="button"
            className="w-full text-left flex items-center gap-2 px-3 py-2 rounded-md font-sans text-sm text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors"
            role="menuitem"
          >
            <LogOut size={14} strokeWidth={1.5} />
            Sign out
          </button>
        </div>
      </div>
    </details>
  );
}
