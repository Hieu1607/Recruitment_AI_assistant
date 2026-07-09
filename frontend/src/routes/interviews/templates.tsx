import { api, type InterviewTemplateCreateRequest, type InterviewTemplateResponse } from "@/api";
import { TemplateEditor } from "@/components/interviews/TemplateEditor";
import {
  Button,
  DataTable,
  EmptyState,
  Modal,
  ModalContent,
  ModalDescription,
  ModalHeader,
  ModalTitle,
  type ColumnDef,
} from "@/components/ui";
import { useSelectedJobId } from "@/lib/auth";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Calendar, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, NavLink, useNavigate, useSearchParams } from "react-router";
import { toast } from "sonner";

function formatDate(value: string) {
  return new Date(value).toLocaleDateString();
}

export default function InterviewTemplatesRoute() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const selectedJobId = useSelectedJobId();
  const [searchParams, setSearchParams] = useSearchParams();
  const [createOpen, setCreateOpen] = useState(false);
  const shouldAutoOpenCreate = searchParams.get("create") === "1";

  const { data, isLoading } = useQuery({
    queryKey: ["interview-templates", selectedJobId],
    queryFn: () => api.interviewTemplates.list(selectedJobId!),
    enabled: !!selectedJobId,
  });

  const createMutation = useMutation({
    mutationFn: (payload: InterviewTemplateCreateRequest) =>
      api.interviewTemplates.create(selectedJobId!, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["interview-templates", selectedJobId] });
      toast.success("Interview template created");
      handleCreateOpenChange(false);
    },
    onError: (error: Error) => toast.error(error.message || "Failed to create template"),
  });

  useEffect(() => {
    if (selectedJobId && shouldAutoOpenCreate) {
      setCreateOpen(true);
    }
  }, [selectedJobId, shouldAutoOpenCreate]);

  function handleCreateOpenChange(nextOpen: boolean) {
    setCreateOpen(nextOpen);
    if (!nextOpen && shouldAutoOpenCreate) {
      const nextParams = new URLSearchParams(searchParams);
      nextParams.delete("create");
      setSearchParams(nextParams, { replace: true });
    }
  }

  const columns: ColumnDef<InterviewTemplateResponse>[] = [
    {
      key: "name",
      header: "Template",
      render: (row) => (
        <Link className="font-medium text-accent hover:underline" to={routes.interviewTemplateDetail(row.id)}>
          {row.name}
        </Link>
      ),
    },
    {
      key: "status",
      header: "Status",
    },
    {
      key: "language_code",
      header: "Language",
    },
    {
      key: "updated_at",
      header: "Updated",
      render: (row) => (
        <span className="flex items-center gap-1.5 text-fg-muted">
          <Calendar size={14} strokeWidth={1.5} />
          {formatDate(row.updated_at)}
        </span>
      ),
    },
  ];

  return (
    <div className="px-8 py-8 min-h-full">
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="font-display text-[2rem] font-medium text-fg">Interview Templates</h1>
            <p className="mt-1 text-sm text-fg-muted">
              Create and maintain job-scoped interview scripts, questions, and report rubrics.
            </p>
          </div>
          <Button
            onClick={() => handleCreateOpenChange(true)}
            disabled={!selectedJobId}
            icon={<Plus size={15} strokeWidth={2} />}
          >
            New template
          </Button>
        </div>

        <div className="flex items-center gap-2 border-b border-[color:var(--hairline)]">
          <NavLink
            to={routes.interviews}
            end
            className={({ isActive }) =>
              isActive
                ? "inline-flex border-b-2 border-accent px-1 py-3 text-sm font-medium text-fg"
                : "inline-flex border-b-2 border-transparent px-1 py-3 text-sm text-fg-muted transition-colors hover:text-fg"
            }
          >
            Interview links
          </NavLink>
          <NavLink
            to={routes.interviewTemplates}
            className={({ isActive }) =>
              isActive
                ? "inline-flex border-b-2 border-accent px-1 py-3 text-sm font-medium text-fg"
                : "inline-flex border-b-2 border-transparent px-1 py-3 text-sm text-fg-muted transition-colors hover:text-fg"
            }
          >
            Templates
          </NavLink>
        </div>

        {!selectedJobId ? (
          <EmptyState
            heading="Select a job first"
            body="Interview templates are scoped to the active job in the top bar."
          />
        ) : (
          <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg overflow-hidden">
            <DataTable
              columns={columns}
              data={data?.items ?? []}
              loading={isLoading}
              onRowClick={(row) => navigate(routes.interviewTemplateDetail(row.id))}
              emptyState={
                <EmptyState
                  heading="No interview templates"
                  body="Create the first template for this job to start sending candidate interview invitations."
                />
              }
            />
          </div>
        )}
      </div>

      <Modal open={createOpen} onOpenChange={handleCreateOpenChange}>
        <ModalContent size="large">
          <ModalHeader>
            <ModalTitle>Create Interview Template</ModalTitle>
            <ModalDescription>
              Define the recruiter-facing template before sending interview invitations.
            </ModalDescription>
          </ModalHeader>
          <TemplateEditor
            mode="create"
            submitLabel="Create template"
            loading={createMutation.isPending}
            onSubmit={async (payload) => {
              await createMutation.mutateAsync(payload as InterviewTemplateCreateRequest);
            }}
          />
        </ModalContent>
      </Modal>
    </div>
  );
}
