import { api } from "@/api";
import { Button, EmptyState, Modal, ModalContent, ModalDescription, ModalFooter, ModalHeader, ModalTitle } from "@/components/ui";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Send } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

export function InvitationSendDialog({
  open,
  onOpenChange,
  jobId,
  candidateProfileId,
  candidateName,
  onSent,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  jobId: string | null;
  candidateProfileId: string | null;
  candidateName: string;
  onSent?: () => void;
}) {
  const queryClient = useQueryClient();
  const [templateId, setTemplateId] = useState("");
  const [expiresInHours, setExpiresInHours] = useState("72");

  const { data, isLoading } = useQuery({
    queryKey: ["interview-templates", jobId],
    queryFn: () => api.interviewTemplates.list(jobId!),
    enabled: open && !!jobId,
  });

  useEffect(() => {
    if (!open) {
      setTemplateId("");
      setExpiresInHours("72");
    }
  }, [open]);

  const sendMutation = useMutation({
    mutationFn: () =>
      api.interviewInvitations.create({
        job_id: jobId!,
        candidate_profile_id: candidateProfileId!,
        interview_template_id: templateId,
        expires_in_hours: expiresInHours ? Number(expiresInHours) : undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["interview-invitations", jobId] });
      toast.success("Interview invitation sent");
      onSent?.();
      onOpenChange(false);
    },
    onError: (error: Error) => {
      toast.error(error.message || "Failed to send interview invitation");
    },
  });

  const templates = data?.items ?? [];

  return (
    <Modal open={open} onOpenChange={onOpenChange}>
      <ModalContent>
        <ModalHeader>
          <ModalTitle>Send interview invitation</ModalTitle>
          <ModalDescription>
            Send a voice interview link for {candidateName}.
          </ModalDescription>
        </ModalHeader>

        {isLoading ? (
          <div className="py-8 text-center text-sm text-fg-muted">Loading templates…</div>
        ) : templates.length === 0 ? (
          <EmptyState
            icon={<Send size={28} strokeWidth={1.5} />}
            heading="No interview templates"
            body="Create a template for the selected job before sending an interview invitation."
          />
        ) : (
          <div className="space-y-4">
            <label className="space-y-1.5">
              <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
                Interview template
              </span>
              <select
                aria-label="Interview template"
                value={templateId}
                onChange={(event) => setTemplateId(event.target.value)}
                className={cn(
                  "h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3",
                  "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                )}
              >
                <option value="">Select template…</option>
                {templates.map((template) => (
                  <option key={template.id} value={template.id}>
                    {template.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-1.5">
              <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
                Expires in hours
              </span>
              <input
                aria-label="Expires in hours"
                inputMode="numeric"
                value={expiresInHours}
                onChange={(event) => setExpiresInHours(event.target.value)}
                className={cn(
                  "h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3",
                  "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                )}
              />
            </label>
          </div>
        )}

        <ModalFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => sendMutation.mutate()}
            loading={sendMutation.isPending}
            disabled={!jobId || !candidateProfileId || !templateId || templates.length === 0}
          >
            Send invitation
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
