import {
  api,
  type InterviewInvitationResponse,
} from "@/api";
import { InterviewLinkComposerModal } from "@/components/interviews/InterviewLinkComposerModal";
import {
  Badge,
  Button,
  DataTable,
  EmptyState,
  Modal,
  ModalContent,
  ModalDescription,
  ModalFooter,
  ModalHeader,
  ModalTitle,
  type ColumnDef,
} from "@/components/ui";
import { useSelectedJobId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Calendar, Copy, ExternalLink, Link2, Mic2, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link, NavLink } from "react-router";
import { toast } from "sonner";

function formatDateTime(value: string | null) {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}

function statusVariant(status: string): "neutral" | "warning" | "success" | "danger" {
  if (status === "completed") return "success";
  if (status === "pending" || status === "opened" || status === "in_progress") return "warning";
  if (status === "cancelled" || status === "expired" || status === "failed") return "danger";
  return "neutral";
}

function isRevocable(status: string) {
  return !["cancelled", "completed", "expired"].includes(status);
}

export default function InterviewsRoute() {
  const queryClient = useQueryClient();
  const selectedJobId = useSelectedJobId();
  const [createOpen, setCreateOpen] = useState(false);
  const [revokeTarget, setRevokeTarget] = useState<InterviewInvitationResponse | null>(null);

  const { data: invitationsData, isLoading: invitationsLoading } = useQuery({
    queryKey: ["interview-invitations", selectedJobId],
    queryFn: () => api.interviewInvitations.list(selectedJobId!),
    enabled: !!selectedJobId,
  });

  const revokeMutation = useMutation({
    mutationFn: (invitationId: string) => api.interviewInvitations.revoke(invitationId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["interview-invitations", selectedJobId] });
      toast.success("Interview link revoked");
      setRevokeTarget(null);
    },
    onError: (error: Error) => toast.error(error.message || "Failed to revoke interview link"),
  });

  const columns: ColumnDef<InterviewInvitationResponse>[] = [
    {
      key: "candidate_full_name",
      header: "Candidate",
      render: (row) => <span className="font-medium text-fg">{row.candidate_full_name || "Unknown candidate"}</span>,
    },
    {
      key: "interview_template_name",
      header: "Template",
      render: (row) => row.interview_template_name || "Interview template",
    },
    {
      key: "status",
      header: "Status",
      render: (row) => <Badge variant={statusVariant(row.status)}>{row.status}</Badge>,
    },
    {
      key: "created_at",
      header: "Created",
      render: (row) => (
        <span className="flex items-center gap-1.5 text-fg-muted">
          <Calendar size={14} strokeWidth={1.5} />
          {formatDateTime(row.created_at)}
        </span>
      ),
    },
    {
      key: "expires_at",
      header: "Expires",
      render: (row) => formatDateTime(row.expires_at),
    },
    {
      key: "completed_at",
      header: "Completed",
      render: (row) => formatDateTime(row.completed_at),
    },
    {
      key: "attempt_count",
      header: "Attempts",
      render: (row) => `${row.attempt_count}/${row.max_attempts}`,
    },
    {
      key: "actions",
      header: "Actions",
      className: "w-[260px]",
      render: (row) => (
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            icon={<Copy size={13} strokeWidth={1.75} />}
            onClick={async (event) => {
              event.stopPropagation();
              await navigator.clipboard.writeText(row.public_url);
              toast.success("Interview link copied");
            }}
          >
            Copy link
          </Button>
          {row.latest_interview_session_id && (
            <Link
              to={routes.interviewReport(row.latest_interview_session_id)}
              onClick={(event) => event.stopPropagation()}
              className="inline-flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] px-3 text-xs font-medium text-accent transition-colors hover:bg-[color:var(--hairline)]"
            >
              <ExternalLink size={13} strokeWidth={1.75} />
              Report
            </Link>
          )}
          {isRevocable(row.status) && (
            <Button
              variant="ghost"
              size="sm"
              icon={<Trash2 size={13} strokeWidth={1.75} />}
              onClick={(event) => {
                event.stopPropagation();
                setRevokeTarget(row);
              }}
            >
              Revoke
            </Button>
          )}
        </div>
      ),
    },
  ];
  return (
    <div className="px-8 py-8 min-h-full">
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="font-display text-[2rem] font-medium text-fg">Interviews</h1>
            <p className="mt-1 text-sm text-fg-muted">
              Manage interview links, invitation status, and recruiter report access for the selected job.
            </p>
          </div>
          <Button
            onClick={() => setCreateOpen(true)}
            disabled={!selectedJobId}
            icon={<Plus size={15} strokeWidth={2} />}
          >
            New interview link
          </Button>
        </div>

        <div className="flex items-center gap-2 border-b border-[color:var(--hairline)]">
          <NavLink
            to={routes.interviews}
            end
            className={({ isActive }) =>
              cn(
                "inline-flex items-center gap-2 border-b-2 px-1 py-3 text-sm transition-colors",
                isActive ? "border-accent font-medium text-fg" : "border-transparent text-fg-muted hover:text-fg",
              )
            }
          >
            <Link2 size={14} strokeWidth={1.75} />
            Interview links
          </NavLink>
          <NavLink
            to={routes.interviewTemplates}
            className={({ isActive }) =>
              cn(
                "inline-flex items-center gap-2 border-b-2 px-1 py-3 text-sm transition-colors",
                isActive ? "border-accent font-medium text-fg" : "border-transparent text-fg-muted hover:text-fg",
              )
            }
          >
            <Mic2 size={14} strokeWidth={1.75} />
            Templates
          </NavLink>
        </div>

        {!selectedJobId ? (
          <EmptyState
            heading="Select a job first"
            body="Interview links are scoped to the active job in the top bar."
          />
        ) : (
          <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg overflow-hidden">
            <DataTable
              columns={columns}
              data={invitationsData?.items ?? []}
              loading={invitationsLoading}
              emptyState={
                <EmptyState
                  heading="No interview links"
                  body="Create the first invitation for this job to start sending interview links."
                />
              }
            />
          </div>
        )}
      </div>

      <InterviewLinkComposerModal open={createOpen} onOpenChange={setCreateOpen} jobId={selectedJobId} />

      <Modal open={!!revokeTarget} onOpenChange={(open) => !open && setRevokeTarget(null)}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Revoke interview link</ModalTitle>
            <ModalDescription>
              {revokeTarget
                ? `Revoke the interview link for ${revokeTarget.candidate_full_name || "this candidate"}?`
                : "Revoke the selected interview link?"}
            </ModalDescription>
          </ModalHeader>
          <p className="text-sm text-fg-muted">
            Candidates will no longer be able to start the interview from this link after revocation.
          </p>
          <ModalFooter>
            <Button variant="ghost" onClick={() => setRevokeTarget(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              loading={revokeMutation.isPending}
              onClick={() => revokeTarget && revokeMutation.mutate(revokeTarget.id)}
            >
              Confirm revoke
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}
