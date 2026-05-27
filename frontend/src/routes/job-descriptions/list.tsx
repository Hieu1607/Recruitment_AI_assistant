import { ApiError, api, type JobResponse } from "@/api";
import {
  Badge,
  Button,
  EmptyState,
  Skeleton,
} from "@/components/ui";
import { cn } from "@/lib/cn";
import { useSelectedJobId } from "@/lib/auth";
import { routes } from "@/routes";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, BarChart2, PencilLine } from "lucide-react";
import { useNavigate } from "react-router";

const EMPTY_JOBS: JobResponse[] = [];

function truncateBody(text: string, max = 320): string {
  const flat = text.replace(/\n+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}...` : flat;
}

export default function JobDescriptionsListRoute() {
  const navigate = useNavigate();
  const selectedJobId = useSelectedJobId();

  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.jobs.list(),
    staleTime: 60_000,
  });

  const selectedJob = (jobsQuery.data?.items ?? EMPTY_JOBS).find((job) => job.id === selectedJobId) ?? null;

  const jdQuery = useQuery({
    queryKey: ["jobs", selectedJobId, "job-description", "workspace-alias"],
    enabled: !!selectedJobId,
    queryFn: async () => {
      try {
        return await api.jobs.jobDescription.get(selectedJobId!);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
  });

  if (!selectedJobId) {
    return (
      <div className="px-8 py-8 min-h-full">
        <EmptyState
          heading="No workspace selected"
          body="Select a job workspace first. Job description management now follows the active workspace."
          action={{ label: "Open jobs", onClick: () => navigate(routes.jobs) }}
        />
      </div>
    );
  }

  if (jobsQuery.isLoading || jdQuery.isLoading) {
    return (
      <div className="px-8 py-8 min-h-full space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full rounded-[var(--radius-lg)]" />
        <Skeleton className="h-80 w-full rounded-[var(--radius-lg)]" />
      </div>
    );
  }

  const jd = jdQuery.data;

  return (
    <div className="px-8 py-8 min-h-full">
      <div className="flex flex-col gap-4 border-b border-[color:var(--hairline)] pb-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl">
          <h1 className="font-display text-[2rem] font-medium leading-tight text-fg">
            Workspace job description
          </h1>
          <p className="mt-2 text-sm leading-6 text-fg-muted">
            This page follows the selected workspace. Manage the current job description from the active job context instead of treating it as a standalone resource.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="ghost"
            icon={<ArrowUpRight size={15} strokeWidth={1.75} />}
            onClick={() => navigate(routes.jobs)}
          >
            Open workspace
          </Button>
          <Button
            variant="primary"
            icon={<PencilLine size={15} strokeWidth={1.75} />}
            onClick={() => navigate(routes.jobDescriptionNew)}
          >
            {jd ? "Edit job description" : "Create job description"}
          </Button>
        </div>
      </div>

      <section
        className={cn(
          "mt-8 rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6",
        )}
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Selected workspace</p>
            <h2 className="mt-3 font-display text-3xl leading-tight text-fg">
              {selectedJob?.title ?? "Current workspace"}
            </h2>
          </div>
          {jd && (
            <Badge variant={jd.is_active ? "success" : "neutral"} size="sm" dot>
              {jd.is_active ? "Active" : "Inactive"}
            </Badge>
          )}
        </div>

        {!jd ? (
          <EmptyState
            heading="No job description yet"
            body="This workspace does not have a current JD yet. Create one here, then continue to scoring and interview preparation."
            action={{ label: "Create job description", onClick: () => navigate(routes.jobDescriptionNew) }}
          />
        ) : (
          <div className="mt-6 space-y-5">
            <div>
              <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Current JD</p>
              <h3 className="mt-3 font-display text-2xl leading-tight text-fg">
                {jd.title ?? "Untitled position"}
              </h3>
              <p className="mt-4 max-w-3xl text-sm leading-7 text-fg-muted">
                {truncateBody(jd.jd_text)}
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <Button
                variant="secondary"
                icon={<PencilLine size={15} strokeWidth={1.75} />}
                onClick={() => navigate(routes.jobDescriptionNew)}
              >
                Edit job description
              </Button>
              <Button
                variant="ghost"
                icon={<BarChart2 size={15} strokeWidth={1.75} />}
                onClick={() => navigate(`${routes.scoring}?jd=${jd.id}`)}
              >
                Score candidates
              </Button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
