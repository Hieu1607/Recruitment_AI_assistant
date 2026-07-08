import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { Mail, ScrollText } from "lucide-react";
import { NavLink } from "react-router";

export function OutreachWorkspaceNav() {
  return (
    <div className="flex items-center gap-2 border-b border-[color:var(--hairline)]">
      <NavLink
        to={routes.outreach}
        end
        className={({ isActive }) =>
          cn(
            "inline-flex items-center gap-2 border-b-2 px-1 py-3 text-sm transition-colors",
            isActive ? "border-accent font-medium text-fg" : "border-transparent text-fg-muted hover:text-fg",
          )
        }
      >
        <Mail size={14} strokeWidth={1.75} />
        Messages
      </NavLink>
      <NavLink
        to={routes.outreachTemplates}
        className={({ isActive }) =>
          cn(
            "inline-flex items-center gap-2 border-b-2 px-1 py-3 text-sm transition-colors",
            isActive ? "border-accent font-medium text-fg" : "border-transparent text-fg-muted hover:text-fg",
          )
        }
      >
        <ScrollText size={14} strokeWidth={1.75} />
        Templates
      </NavLink>
    </div>
  );
}
