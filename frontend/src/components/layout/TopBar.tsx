import { api } from "@/api";
import { NotificationCenter } from "@/components/notifications/NotificationCenter";
import { useAuthStore } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useQuery } from "@tanstack/react-query";
import { Search, Command, ChevronRight, BriefcaseBusiness, ChevronDown, PanelLeftOpen } from "lucide-react";
import { Link, useMatches, useLocation } from "react-router";
import { UserMenu } from "./UserMenu";
import { routes } from "@/routes";

// Map from URL prefix → human breadcrumb label.
// Order matters: longer prefixes first so /shortlists/collections/:id wins over /shortlists.
const BREADCRUMB_RULES: Array<[string, string]> = [
  ["/dashboard", "Dashboard"],
  ["/jobs/new", "Jobs / New"],
  ["/jobs/", "Jobs / Edit"],
  ["/jobs", "Jobs"],
  ["/candidates", "Candidates"],
  ["/job-descriptions/new", "Jobs / Job Description"],
  ["/job-descriptions", "Jobs / Job Description"],
  ["/scoring", "Scoring"],
  ["/chat", "AI Chat"],
  ["/shortlists/collections", "Shortlists / Collection"],
  ["/shortlists", "Shortlists"],
  ["/outreach", "Outreach"],
  ["/interviews/reports", "Interview Reports"],
  ["/interviews/templates", "Interview Templates"],
  ["/interviews", "Interviews"],
  ["/interview-questions", "Interview Prep"],
  ["/settings", "Settings"]
];

function resolveBreadcrumb(pathname: string): string {
  for (const [prefix, label] of BREADCRUMB_RULES) {
    if (pathname.startsWith(prefix)) return label;
  }
  return "";
}

export function TopBar({
  navSidebarCollapsed,
  onExpandNavSidebar,
}: {
  navSidebarCollapsed: boolean;
  onExpandNavSidebar: () => void;
}) {
  const { pathname } = useLocation();
  const selectedJobId = useAuthStore((s) => s.selectedJobId);
  const setSelectedJobId = useAuthStore((s) => s.setSelectedJobId);
  const { data: jobsData } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.jobs.list(),
    staleTime: 60_000,
  });
  // useMatches is referenced so future plans can swap to per-route handle metadata
  // (handle: { breadcrumb: string }) without TopBar re-architecture.
  useMatches();
  const crumb = resolveBreadcrumb(pathname);
  const jobs = jobsData?.items ?? [];
  const selectedJobExists = selectedJobId ? jobs.some((job) => job.id === selectedJobId) : false;

  return (
    <header
      className="hairline-b bg-bg flex items-center px-6 gap-6"
      style={{ height: "var(--topbar-height)" }}
    >
      {navSidebarCollapsed && (
        <button
          type="button"
          onClick={onExpandNavSidebar}
          aria-label="Expand navigation sidebar"
          className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-fg"
        >
          <PanelLeftOpen size={16} strokeWidth={1.75} />
        </button>
      )}

      {/* Wordmark — keeps brand visible above the content area even though
          the Sidebar already shows it (FOUND-09 / SC#2 require both). */}
      <Link
        to={routes.dashboard}
        className="flex items-baseline gap-2 shrink-0"
        aria-label="EasyHR — go to dashboard"
      >
        <span className="font-display text-lg font-medium leading-none text-fg">EasyHR</span>
      </Link>

      {/* Breadcrumb — derived from the active route */}
      <nav aria-label="Breadcrumb" className="flex items-center gap-2 min-w-0 shrink">
        <ChevronRight size={14} strokeWidth={1.5} className="text-fg-subtle shrink-0" aria-hidden="true" />
        <span className="font-sans text-sm text-fg-muted truncate">
          {crumb || "—"}
        </span>
      </nav>

      {jobs.length > 0 && (
        <div className="relative hidden min-w-[210px] max-w-[280px] shrink-0 sm:block">
          <BriefcaseBusiness
            size={15}
            strokeWidth={1.75}
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-success"
            aria-hidden="true"
          />
          <select
            value={selectedJobExists && selectedJobId ? selectedJobId : ""}
            onChange={(e) => setSelectedJobId(e.target.value || null)}
            className={cn(
              "h-10 w-full appearance-none rounded-[var(--radius-md)] border border-[rgba(74,124,89,0.34)]",
              "bg-bg-elevated py-0 pl-9 pr-9 font-sans text-sm font-medium text-fg shadow-[0_1px_0_rgba(255,255,255,0.55)_inset]",
              "transition-colors hover:border-[rgba(74,124,89,0.55)] hover:bg-bg-sidebar",
              "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
            )}
            aria-label="Selected job"
          >
            <option value="" disabled>
              Select job
            </option>
            {jobs.map((job) => (
              <option key={job.id} value={job.id}>
                {job.title}
              </option>
            ))}
          </select>
          <ChevronDown
            size={15}
            strokeWidth={1.75}
            className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-fg-subtle"
            aria-hidden="true"
          />
        </div>
      )}

      {/* Search */}
      <div className="flex-1 max-w-xl ml-auto">
        <button
          type="button"
          onClick={() => window.dispatchEvent(new Event("easyhr:open-command-palette"))}
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
        <NotificationCenter />
        <UserMenu />
      </div>
    </header>
  );
}
