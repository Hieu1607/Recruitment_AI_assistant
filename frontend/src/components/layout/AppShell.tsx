import { api, type JobResponse } from "@/api";
import { JobWorkspaceGate } from "@/components/jobs/JobWorkspaceGate";
import { useAuthStore } from "@/lib/auth";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { Outlet } from "react-router";
import { TopBar } from "./TopBar";
import { Sidebar } from "./Sidebar";
import { CommandPalette } from "../CommandPalette";

const EMPTY_JOBS: JobResponse[] = [];

export function AppShell() {
  const qc = useQueryClient();
  const selectedJobId = useAuthStore((s) => s.selectedJobId);
  const setSelectedJobId = useAuthStore((s) => s.setSelectedJobId);
  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.jobs.list(),
    staleTime: 60_000,
  });

  const jobsData = jobsQuery.data;
  const jobs = jobsData?.items ?? EMPTY_JOBS;

  useEffect(() => {
    if (!jobsData) return;

    const hasMatchingSelection = !!selectedJobId && jobs.some((job) => job.id === selectedJobId);
    if (hasMatchingSelection) return;

    if (jobs.length === 1) {
      setSelectedJobId(jobs[0].id);
      return;
    }

    if (selectedJobId && !jobs.some((job) => job.id === selectedJobId)) {
      setSelectedJobId(null);
    }
  }, [jobs, jobsData, selectedJobId, setSelectedJobId]);

  const hasValidSelection = !!selectedJobId && jobs.some((job) => job.id === selectedJobId);
  const shouldAutoSelectSingleJob =
    jobsData !== undefined && jobs.length === 1 && !hasValidSelection;
  const showWorkspaceGate =
    (!selectedJobId && jobsQuery.isLoading) ||
    Boolean(jobsQuery.error) ||
    (jobsData !== undefined && !shouldAutoSelectSingleJob && (jobs.length === 0 || !hasValidSelection));

  return (
    <>
      <div className="flex h-full">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <TopBar />
          <div className="flex-1 overflow-y-auto">
            <div
              className="mx-auto w-full"
              style={{ maxWidth: "var(--content-max)" }}
            >
              <Outlet />
            </div>
          </div>
        </div>
        <CommandPalette />
      </div>
      {showWorkspaceGate && (
        <JobWorkspaceGate
          jobs={jobs}
          isLoading={jobsQuery.isLoading}
          error={jobsQuery.error}
          onRetry={() => qc.invalidateQueries({ queryKey: ["jobs"] })}
        />
      )}
    </>
  );
}
