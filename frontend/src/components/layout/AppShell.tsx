import { api } from "@/api";
import { Button, Modal, ModalContent, ModalDescription, ModalFooter, ModalHeader, ModalTitle } from "@/components/ui";
import { useAuthStore } from "@/lib/auth";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Outlet } from "react-router";
import { TopBar } from "./TopBar";
import { Sidebar } from "./Sidebar";
import { CommandPalette } from "../CommandPalette";

export function AppShell() {
  const qc = useQueryClient();
  const selectedJobId = useAuthStore((s) => s.selectedJobId);
  const setSelectedJobId = useAuthStore((s) => s.setSelectedJobId);
  const [jobTitle, setJobTitle] = useState("");
  const { data: jobsData } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.jobs.list(),
    staleTime: 60_000,
  });
  const createJob = useMutation({
    mutationFn: () => api.jobs.create({ title: jobTitle.trim() || "New Job" }),
    onSuccess: (job) => {
      setSelectedJobId(job.id);
      setJobTitle("");
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const jobs = jobsData?.items ?? [];
  useEffect(() => {
    if (jobs.length === 0) return;
    if (!selectedJobId || !jobs.some((job) => job.id === selectedJobId)) {
      setSelectedJobId(jobs[0].id);
    }
  }, [jobs, selectedJobId, setSelectedJobId]);

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
      <Modal open={jobsData !== undefined && jobs.length === 0} onOpenChange={() => {}}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Create your first job</ModalTitle>
            <ModalDescription>
              Jobs are now the primary workspace boundary. Create one job before adding a JD or resumes.
            </ModalDescription>
          </ModalHeader>
          <div className="space-y-2">
            <label htmlFor="job-title" className="text-sm font-sans text-fg-muted">
              Job title
            </label>
            <input
              id="job-title"
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              placeholder="Senior Backend Engineer"
              className="w-full h-10 px-3 rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg text-fg"
            />
          </div>
          <ModalFooter>
            <Button variant="primary" loading={createJob.isPending} onClick={() => createJob.mutate()}>
              Create job
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
}
