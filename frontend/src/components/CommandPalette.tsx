import { api, type CandidateProfileResponse, type JobDescriptionResponse, type JobResponse } from "@/api";
import { useSelectedJobId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useQuery } from "@tanstack/react-query";
import { BriefcaseBusiness, Command, FileText, LayoutDashboard, Search, User } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

type PaletteItem = {
  id: string;
  label: string;
  description: string;
  icon: typeof Search;
  path: string;
  searchText: string;
};

function compactText(value: string | null | undefined) {
  return (value ?? "").replace(/\s+/g, " ").trim();
}

function snippet(value: string, query: string) {
  const text = compactText(value);
  if (!query.trim() || text.length <= 90) return text;
  const index = text.toLowerCase().indexOf(query.toLowerCase());
  if (index < 0) return `${text.slice(0, 90)}…`;
  const start = Math.max(0, index - 30);
  const end = Math.min(text.length, index + query.length + 60);
  return `${start > 0 ? "…" : ""}${text.slice(start, end)}${end < text.length ? "…" : ""}`;
}

function candidateName(candidate: CandidateProfileResponse) {
  return compactText(candidate.full_name) || compactText(candidate.submitted_full_name) || "Unnamed candidate";
}

function buildCandidateItem(candidate: CandidateProfileResponse, query: string): PaletteItem {
  const detail = [
    candidate.current_job_title,
    candidate.email,
    candidate.skills_text,
    candidate.summary_text,
  ]
    .map(compactText)
    .filter(Boolean)
    .join(" · ");

  return {
    id: `candidate:${candidate.id}`,
    label: candidateName(candidate),
    description: snippet(detail || "Candidate profile", query),
    icon: User,
    path: routes.candidateDetail(candidate.resume_document_id),
    searchText: [
      candidate.full_name,
      candidate.submitted_full_name,
      candidate.email,
      candidate.submitted_email,
      candidate.current_job_title,
      candidate.skills_text,
      candidate.summary_text,
      candidate.experience_text,
      candidate.education_text,
    ]
      .map(compactText)
      .join(" ")
      .toLowerCase(),
  };
}

function buildJobItem(job: JobResponse): PaletteItem {
  return {
    id: `job:${job.id}`,
    label: job.title,
    description: `Workspace · ${job.status}`,
    icon: BriefcaseBusiness,
    path: routes.jobEdit(job.id),
    searchText: `${job.title} ${job.status}`.toLowerCase(),
  };
}

function buildJobDescriptionItem(jd: JobDescriptionResponse, query: string): PaletteItem {
  const label = compactText(jd.title) || "Current job description";
  return {
    id: `jd:${jd.id}`,
    label,
    description: snippet(`${jd.jd_text} ${jd.hidden_text}`, query) || "Workspace job description",
    icon: FileText,
    path: routes.jobDescriptions,
    searchText: `${label} ${jd.jd_text} ${jd.hidden_text}`.toLowerCase(),
  };
}

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const navigate = useNavigate();
  const selectedJobId = useSelectedJobId();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((value) => !value);
      }
      if (e.key === "Escape") {
        setOpen(false);
      }
    };
    const openPalette = () => setOpen(true);

    document.addEventListener("keydown", down);
    window.addEventListener("easyhr:open-command-palette", openPalette);
    return () => {
      document.removeEventListener("keydown", down);
      window.removeEventListener("easyhr:open-command-palette", openPalette);
    };
  }, []);

  const { data: jobsData } = useQuery({
    queryKey: ["jobs", "command-palette"],
    queryFn: () => api.jobs.list(),
    enabled: open,
    staleTime: 60_000,
  });

  const { data: candidatesData } = useQuery({
    queryKey: ["jobs", selectedJobId, "candidates", "command-palette"],
    queryFn: () => api.jobs.listCandidates(selectedJobId!),
    enabled: open && !!selectedJobId,
    staleTime: 30_000,
  });

  const { data: jobDescription } = useQuery({
    queryKey: ["jobs", selectedJobId, "job-description", "command-palette"],
    queryFn: () => api.jobs.jobDescription.get(selectedJobId!),
    enabled: open && !!selectedJobId,
    retry: false,
    staleTime: 30_000,
  });

  const items = useMemo(() => {
    const actions: PaletteItem[] = [
      {
        id: "dashboard",
        label: "Go to Dashboard",
        description: "Open the workspace overview",
        icon: LayoutDashboard,
        path: routes.dashboard,
        searchText: "dashboard overview home",
      },
      {
        id: "candidates",
        label: "View Candidates",
        description: "Open candidate management",
        icon: User,
        path: routes.candidates,
        searchText: "candidates resumes applicants",
      },
      {
        id: "jd",
        label: "Open Job Description",
        description: "Edit the current workspace JD",
        icon: FileText,
        path: routes.jobDescriptions,
        searchText: "job description jd",
      },
      {
        id: "scoring",
        label: "Score Candidates",
        description: "Run candidate scoring",
        icon: Command,
        path: routes.scoring,
        searchText: "score candidates scoring",
      },
    ];

    const dynamicItems = [
      ...(jobsData?.items ?? []).map(buildJobItem),
      ...(candidatesData?.items ?? []).map((candidate) => buildCandidateItem(candidate, query)),
      ...(jobDescription ? [buildJobDescriptionItem(jobDescription, query)] : []),
    ];

    return [...actions, ...dynamicItems];
  }, [candidatesData?.items, jobDescription, jobsData?.items, query]);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return items.slice(0, 8);
    return items.filter((item) => item.searchText.includes(normalized)).slice(0, 20);
  }, [items, query]);

  const handleSelect = (path: string) => {
    setOpen(false);
    setQuery("");
    navigate(path);
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[18vh]">
      <button
        type="button"
        aria-label="Close command palette"
        className="fixed inset-0 bg-forest-900/35 backdrop-blur-sm"
        onClick={() => setOpen(false)}
      />
      <div className="relative w-[min(42rem,calc(100vw-2rem))] overflow-hidden rounded-lg border border-[color:var(--hairline)] bg-bg-elevated shadow-2xl">
        <div className="flex items-center px-4 hairline-b">
          <Search className="h-5 w-5 text-fg-subtle" />
          <input
            autoFocus
            className="w-full bg-transparent px-4 py-4 text-sm text-fg placeholder:text-fg-subtle focus:outline-none"
            placeholder="Search candidates, JDs, jobs, or actions..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <kbd className="hidden rounded border border-[color:var(--hairline)] bg-[color:var(--hairline)] px-2 py-0.5 font-mono text-xs text-fg-muted sm:inline-block">
            ESC
          </kbd>
        </div>
        <div className="max-h-[60vh] overflow-y-auto py-2">
          {filtered.length === 0 ? (
            <div className="p-8 text-center text-sm text-fg-muted">No results found.</div>
          ) : (
            <div className="px-2">
              <div className="px-3 py-2 text-xs font-medium text-fg-subtle">Results</div>
              {filtered.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleSelect(item.path)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-md px-3 py-3 text-left transition-colors",
                    "text-fg-muted hover:bg-[color:var(--hairline)] hover:text-fg",
                  )}
                >
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-[color:var(--hairline)]">
                    <item.icon className="h-4 w-4" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-fg">{item.label}</span>
                    <span className="mt-0.5 block truncate text-xs text-fg-muted">{item.description}</span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
