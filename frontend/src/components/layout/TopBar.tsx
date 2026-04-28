import { Search, Bell, Command, ChevronRight } from "lucide-react";
import { Link, useMatches, useLocation } from "react-router";
import { UserMenu } from "./UserMenu";
import { routes } from "@/routes";

// Map from URL prefix → human breadcrumb label.
// Order matters: longer prefixes first so /shortlists/collections/:id wins over /shortlists.
const BREADCRUMB_RULES: Array<[string, string]> = [
  ["/dashboard", "Dashboard"],
  ["/candidates", "Candidates"],
  ["/job-descriptions/new", "Job Descriptions / New"],
  ["/job-descriptions", "Job Descriptions"],
  ["/scoring", "Scoring"],
  ["/chat", "AI Chat"],
  ["/shortlists/collections", "Shortlists / Collection"],
  ["/shortlists", "Shortlists"],
  ["/outreach", "Outreach"],
  ["/interview-questions", "Interview Prep"],
  ["/settings", "Settings"]
];

function resolveBreadcrumb(pathname: string): string {
  for (const [prefix, label] of BREADCRUMB_RULES) {
    if (pathname.startsWith(prefix)) return label;
  }
  return "";
}

export function TopBar() {
  const { pathname } = useLocation();
  // useMatches is referenced so future plans can swap to per-route handle metadata
  // (handle: { breadcrumb: string }) without TopBar re-architecture.
  useMatches();
  const crumb = resolveBreadcrumb(pathname);

  return (
    <header
      className="hairline-b bg-bg flex items-center px-6 gap-6"
      style={{ height: "var(--topbar-height)" }}
    >
      {/* Wordmark — keeps brand visible above the content area even though
          the Sidebar already shows it (FOUND-09 / SC#2 require both). */}
      <Link
        to={routes.dashboard}
        className="flex items-baseline gap-2 shrink-0"
        aria-label="RecruitAI — go to dashboard"
      >
        <span className="font-display text-lg font-medium leading-none text-fg">RecruitAI</span>
      </Link>

      {/* Breadcrumb — derived from the active route */}
      <nav aria-label="Breadcrumb" className="flex items-center gap-2 min-w-0 shrink">
        <ChevronRight size={14} strokeWidth={1.5} className="text-fg-subtle shrink-0" aria-hidden="true" />
        <span className="font-sans text-sm text-fg-muted truncate">
          {crumb || "—"}
        </span>
      </nav>

      {/* Search */}
      <div className="flex-1 max-w-xl ml-auto">
        <button
          type="button"
          className="flex items-center gap-3 w-full px-3 py-2 rounded-md hairline text-fg-subtle hover:text-fg-muted transition-colors text-left"
          aria-label="Search (Cmd K)"
        >
          <Search size={14} strokeWidth={1.75} />
          <span className="font-sans text-sm flex-1">Search candidates, JDs…</span>
          <span className="font-mono text-[10px] flex items-center gap-0.5">
            <Command size={10} strokeWidth={2} />K
          </span>
        </button>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-2 shrink-0">
        <button
          type="button"
          aria-label="Notifications"
          className="size-9 rounded-md flex items-center justify-center text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors"
        >
          <Bell size={16} strokeWidth={1.5} />
        </button>
        <button
          type="button"
          aria-label="Command palette"
          className="size-9 rounded-md flex items-center justify-center text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors"
        >
          <Command size={16} strokeWidth={1.5} />
        </button>
        <UserMenu />
      </div>
    </header>
  );
}
