import { useState } from "react";
import { NavLink } from "react-router";
import {
  LayoutDashboard,
  BriefcaseBusiness,
  Users,
  BarChart3,
  MessageSquare,
  ListChecks,
  Mail,
  Mic2,
  Settings,
  HelpCircle,
  FileUp,
  PanelLeftClose,
} from "lucide-react";
import { routes } from "@/routes";
import { cn } from "@/lib/cn";
import { UploadModal } from "@/components/candidates/UploadModal";
import { Tooltip } from "@/components/ui/tooltip";

const NAV_ITEMS = [
  { to: routes.dashboard, label: "Dashboard", icon: LayoutDashboard },
  { to: routes.jobs, label: "Jobs", icon: BriefcaseBusiness },
  { to: routes.candidates, label: "Candidates", icon: Users },
  { to: routes.scoring, label: "Scoring", icon: BarChart3 },
  { to: routes.chat, label: "AI Chat", icon: MessageSquare },
  { to: routes.shortlists, label: "Shortlists", icon: ListChecks },
  { to: routes.outreach, label: "Outreach", icon: Mail },
  { to: routes.interviewQuestions, label: "Interview Prep", icon: Mic2 }
] as const;

const SECONDARY_ITEMS = [
  { to: routes.settings, label: "Settings", icon: Settings },
  { to: "#support", label: "Support", icon: HelpCircle }
] as const;

export function Sidebar({ onCollapse }: { onCollapse?: () => void }) {
  const [uploadOpen, setUploadOpen] = useState(false);
  return (
    <aside
      className="hairline-r flex h-full w-full flex-col bg-bg-sidebar"
    >
      {/* Brand */}
      <div className="px-5 pt-5 pb-4">
        <div className="flex items-center justify-between gap-3">
          <p className="font-display text-[1.35rem] font-medium leading-none text-fg">
            RecruitAI
          </p>
          {onCollapse && (
            <Tooltip content="Collapse navigation" side="right">
              <button
                type="button"
                onClick={onCollapse}
                aria-label="Collapse navigation sidebar"
                className={cn(
                  "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[var(--radius-md)]",
                  "border border-[color:var(--hairline)] bg-bg-elevated text-fg-muted shadow-[var(--shadow-sm)]",
                  "transition-colors hover:border-[color:var(--hairline-strong)] hover:text-fg"
                )}
              >
                <PanelLeftClose size={15} strokeWidth={1.75} />
              </button>
            </Tooltip>
          )}
        </div>
        <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.18em] text-fg-subtle whitespace-nowrap">
          Editorial Intelligence
        </p>
      </div>

      {/* Primary nav */}
      <nav className="flex-1 px-3">
        <ul className="flex flex-col gap-0.5">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) =>
                  cn(
                    "group relative flex items-center gap-3 px-3 py-2 rounded-md font-sans text-sm transition-colors",
                    // Active/hover backgrounds use the --hairline token so dark-mode parity
                    // holds (hardcoded rgba(0,0,0,0.04) was near-invisible on #14151A).
                    isActive
                      ? "text-fg font-medium bg-[color:var(--hairline)]"
                      : "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)]"
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <span
                        aria-hidden="true"
                        className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent"
                      />
                    )}
                    <Icon size={16} strokeWidth={1.5} />
                    <span>{label}</span>
                  </>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Upload CTA — pinned at the BOTTOM per FOUND-10 / ROADMAP Phase 1 SC#2.
          Sits above the secondary footer so it's the last action a user encounters
          before settings/support. */}
      <div className="hairline-t px-4 pt-3 pb-3">
        <button
          type="button"
          onClick={() => setUploadOpen(true)}
          className="w-full bg-accent text-accent-fg rounded-md px-4 py-2.5 font-sans text-sm font-medium flex items-center gap-2 hover:bg-accent-hover transition-colors"
        >
          <FileUp size={16} strokeWidth={1.75} />
          Upload resume
        </button>
      </div>

      <UploadModal open={uploadOpen} onOpenChange={setUploadOpen} />

      {/* Secondary footer */}
      <div className="hairline-t px-3 py-3">
        <ul className="flex flex-col gap-0.5">
          {SECONDARY_ITEMS.map(({ to, label, icon: Icon }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md font-sans text-sm transition-colors",
                    isActive
                      ? "text-fg font-medium"
                      : "text-fg-muted hover:text-fg"
                  )
                }
              >
                <Icon size={16} strokeWidth={1.5} />
                <span>{label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
