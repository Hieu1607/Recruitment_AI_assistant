import { api } from "@/api";
import type { InterviewTemplateUpdateRequest } from "@/api";
import { TemplateEditor } from "@/components/interviews/TemplateEditor";
import { Button, EmptyState, Skeleton } from "@/components/ui";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { Link, useParams } from "react-router";
import { toast } from "sonner";

export default function InterviewTemplateDetailRoute() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();

  const { data: template, isLoading } = useQuery({
    queryKey: ["interview-template", id],
    queryFn: () => api.interviewTemplates.get(id!),
    enabled: !!id,
  });

  const updateMutation = useMutation({
    mutationFn: (payload: Parameters<typeof api.interviewTemplates.update>[1]) =>
      api.interviewTemplates.update(id!, payload),
    onSuccess: (updated) => {
      queryClient.invalidateQueries({ queryKey: ["interview-template", id] });
      queryClient.invalidateQueries({ queryKey: ["interview-templates", updated.job_id] });
      toast.success("Interview template saved");
    },
    onError: (error: Error) => toast.error(error.message || "Failed to save template"),
  });

  if (isLoading) {
    return (
      <div className="px-8 py-8 space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (!template) {
    return (
      <div className="px-8 py-8">
        <EmptyState heading="Template not found" body="The selected interview template could not be loaded." />
      </div>
    );
  }

  return (
    <div className="px-8 py-8 min-h-full">
      <div className="mx-auto max-w-5xl space-y-6">
        <Link
          to={routes.interviewTemplates}
          className="inline-flex items-center gap-1.5 text-sm text-fg-muted hover:text-fg"
        >
          <ArrowLeft size={14} strokeWidth={2} />
          Interview templates
        </Link>

        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6">
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <h1 className="font-display text-[2rem] font-medium text-fg">{template.name}</h1>
              <p className="mt-1 text-sm text-fg-muted">
                Version {template.version} · {template.language_code} · {template.status}
              </p>
            </div>
            <Button
              variant="secondary"
              onClick={() => queryClient.invalidateQueries({ queryKey: ["interview-template", id] })}
            >
              Refresh
            </Button>
          </div>

          <TemplateEditor
            mode="edit"
            template={template}
            submitLabel="Save changes"
            loading={updateMutation.isPending}
            onSubmit={async (payload) => {
              await updateMutation.mutateAsync(payload as InterviewTemplateUpdateRequest);
            }}
          />
        </div>
      </div>
    </div>
  );
}
