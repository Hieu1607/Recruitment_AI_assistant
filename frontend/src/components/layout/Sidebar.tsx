import { api } from "@/api";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
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
  FileUp
} from "lucide-react";
import { routes } from "@/routes";
import { useAuthStore } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { UploadModal } from "@/components/candidates/UploadModal";

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

export function Sidebar() {
  const [uploadOpen, setUploadOpen] = useState(false);
  const selectedJobId = useAuthStore((s) => s.selectedJobId);
  const { data: jobsData } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.jobs.list(),
    staleTime: 60_000,
  });
  const activeJob = (jobsData?.items ?? []).find((job) => job.id === selectedJobId);
  return (
    <aside
      className="hairline-r flex flex-col bg-bg-sidebar"
      style={{ width: "var(--sidebar-width)" }}
    >
      {/* Brand */}
      <div className="px-6 pt-6 pb-4">
        <p className="font-display text-xl font-medium leading-none text-fg">RecruitAI</p>
        <p className="font-mono text-[10px] text-fg-subtle uppercase tracking-widest mt-1">
          Editorial Intelligence
        </p>
        <p className="mt-3 text-xs font-sans text-fg-muted truncate">
          {activeJob ? `Job: ${activeJob.title}` : "No job selected"}
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
