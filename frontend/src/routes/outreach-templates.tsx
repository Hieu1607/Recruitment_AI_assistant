import { api, type OutreachTemplateResponse } from "@/api";
import { parseAxiosError } from "@/api/errors";
import { OutreachWorkspaceNav } from "@/components/outreach/OutreachWorkspaceNav";
import { OutreachRichEditor } from "@/components/outreach/OutreachRichEditor";
import {
  TEMPLATE_DEFAULT_VARIABLES,
  TEMPLATE_VARIABLES,
  detectUsedVariables,
  missingRequiredTemplateDefaults,
} from "@/components/outreach/outreach-constants";
import { htmlToPlainText } from "@/components/outreach/rich-text";
import {
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
import { useSelectedJobId, useUserId } from "@/lib/auth";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Calendar, Plus } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

function formatDate(value: string) {
  return new Date(value).toLocaleDateString();
}

function TemplateEditorModal({
  template,
  open,
  onOpenChange,
}: {
  template: OutreachTemplateResponse | null;
  open: boolean;
  onOpenChange: (nextOpen: boolean) => void;
}) {
  const queryClient = useQueryClient();
  const userId = useUserId();
  const selectedJobId = useSelectedJobId();
  const isEditing = !!template;
  const [name, setName] = useState("");
  const [aiBrief, setAiBrief] = useState("");
  const [subject, setSubject] = useState("");
  const [bodyHtml, setBodyHtml] = useState("");
  const [bodyText, setBodyText] = useState("");
  const [defaultVariables, setDefaultVariables] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!open) return;
    setName(template?.name ?? "");
    setAiBrief("");
    setSubject(template?.subject_template ?? "");
    setBodyHtml(template?.body_html_template ?? "");
    setBodyText(template?.body_text_template ?? "");
    setDefaultVariables(template?.default_variables ? { ...template.default_variables } : {});
  }, [open, template]);

  // Which variables the template content actually references, detected from
  // {{variable}} placeholders rather than assumed — keeps variables_used
  // truthful and lets the missing-defaults warning below reflect reality.
  const liveVariablesUsed = useMemo(
    () =>
      detectUsedVariables(
        `${subject}\n${bodyText || htmlToPlainText(bodyHtml)}`,
        TEMPLATE_VARIABLES.map((item) => item.key),
      ),
    [subject, bodyText, bodyHtml],
  );
  const missingDefaults = useMemo(
    () => missingRequiredTemplateDefaults(liveVariablesUsed, defaultVariables),
    [liveVariablesUsed, defaultVariables],
  );

  const saveMutation = useMutation({
    mutationFn: async () => {
      const cleanedDefaults: Record<string, string> = {};
      for (const { key } of TEMPLATE_DEFAULT_VARIABLES) {
        const value = (defaultVariables[key] ?? "").trim();
        if (value) cleanedDefaults[key] = value;
      }

      const payload = {
        name: name.trim(),
        subject_template: subject.trim(),
        body_html_template: bodyHtml.trim(),
        body_text_template: bodyText.trim() || htmlToPlainText(bodyHtml),
        variables_used: liveVariablesUsed,
        default_variables: cleanedDefaults,
      };

      if (template) {
        return api.outreach.updateTemplate(template.id, payload);
      }

      return api.outreach.createTemplate({
        created_by_user_id: userId ?? "",
        job_id: selectedJobId ?? null,
        content_source: "template",
        ...payload,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["outreach-templates", userId, selectedJobId] });
      toast.success(template ? "Template updated" : "Template created");
      onOpenChange(false);
    },
    onError: () => toast.error("Failed to save template"),
  });

  const generateMutation = useMutation({
    mutationFn: () =>
      api.outreach.generateTemplateDraft({
        job_id: selectedJobId ?? "",
        brief: aiBrief.trim(),
        variables_allowed: TEMPLATE_VARIABLES.map((item) => item.key),
      }),
    onSuccess: (draft) => {
      setSubject(draft.subject);
      setBodyHtml(draft.body_html);
      setBodyText(draft.body_text);
      toast.success("Draft added to the editor");
    },
    onError: (error: unknown) => {
      const parsed = parseAxiosError(error);
      if (parsed.status === 422) {
        toast.error("Write a short AI brief first");
        return;
      }
      toast.error("AI draft could not be generated");
    },
  });

  const canSave = !!name.trim() && !!subject.trim() && !!bodyHtml.trim();
  const canGenerate = !!selectedJobId && !!aiBrief.trim();

  return (
    <Modal open={open} onOpenChange={onOpenChange}>
      <ModalContent className="w-[760px] max-h-[85vh] overflow-y-auto rounded-[var(--radius-lg)]">
        <ModalHeader>
          <ModalTitle>{isEditing ? "Edit template" : "Create Outreach Template"}</ModalTitle>
          <ModalDescription>
            Draft reusable recruiter copy here. AI only helps in the Templates workspace.
          </ModalDescription>
        </ModalHeader>

        <div className="mt-4 space-y-5 px-1">
          <div>
            <label className="mb-1.5 block text-[11px] font-sans font-medium uppercase tracking-wide text-fg-subtle">
              Template name
            </label>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Warm intro for shortlisted candidates"
              className="h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg px-3 text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
            />
          </div>

          <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-fg">AI brief</p>
                <p className="mt-1 text-xs text-fg-muted">
                  Describe the tone, key points, and variables the recruiter should mention.
                </p>
              </div>
              <Button
                type="button"
                size="sm"
                variant="secondary"
                disabled={!canGenerate}
                loading={generateMutation.isPending}
                onClick={() => generateMutation.mutate()}
              >
                Generate once
              </Button>
            </div>
            <textarea
              value={aiBrief}
              onChange={(event) => setAiBrief(event.target.value)}
              placeholder="Write a concise recruiter email for a promising candidate. Mention the role, keep it warm, and leave placeholders for the candidate name and job title."
              className="mt-3 min-h-28 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg px-3 py-2 text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
            />
          </div>

          <div>
            <label className="mb-1.5 block text-[11px] font-sans font-medium uppercase tracking-wide text-fg-subtle">
              Subject
            </label>
            <input
              value={subject}
              onChange={(event) => setSubject(event.target.value.slice(0, 255))}
              placeholder="Subject line…"
              maxLength={255}
              className="h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg px-3 text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
            />
          </div>

          <div>
            <label className="mb-1.5 block text-[11px] font-sans font-medium uppercase tracking-wide text-fg-subtle">
              Body
            </label>
            <OutreachRichEditor
              value={bodyHtml}
              onChange={({ html, text }) => {
                setBodyHtml(html);
                setBodyText(text);
              }}
              variableOptions={TEMPLATE_VARIABLES}
            />
          </div>

          <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
            <p className="text-sm font-medium text-fg">Configure Variables</p>
            <p className="mt-1 text-xs text-fg-muted">
              Set default values so Messages can auto-fill them for this template. Candidate Name and Candidate
              Email always resolve from the candidate picked in New message — they can&apos;t be configured here.
            </p>
            <div className="mt-3 grid grid-cols-2 gap-3">
              {TEMPLATE_DEFAULT_VARIABLES.map((item) => (
                <div key={item.key}>
                  <label className="mb-1.5 block text-[11px] font-sans font-medium uppercase tracking-wide text-fg-subtle">
                    {item.label} default
                  </label>
                  <input
                    value={defaultVariables[item.key] ?? ""}
                    onChange={(event) =>
                      setDefaultVariables((prev) => ({ ...prev, [item.key]: event.target.value }))
                    }
                    placeholder={item.key === "job_title" ? "e.g. Backend Engineer" : "e.g. EasyHR"}
                    className="h-9 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg px-3 text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent"
                  />
                </div>
              ))}
            </div>
            {missingDefaults.length > 0 && (
              <p className="mt-3 text-xs text-danger">
                This template uses {missingDefaults.map((key) => `{{${key}}}`).join(", ")} but no default value is
                set yet. Messages composed from this template will be blocked until it&apos;s configured.
              </p>
            )}
          </div>
        </div>

        <ModalFooter>
          <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            disabled={!canSave}
            loading={saveMutation.isPending}
            onClick={() => saveMutation.mutate()}
          >
            {isEditing ? "Save changes" : "Save template"}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

export default function OutreachTemplatesRoute() {
  const selectedJobId = useSelectedJobId();
  const userId = useUserId();
  const [createOpen, setCreateOpen] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<OutreachTemplateResponse | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["outreach-templates", userId, selectedJobId],
    queryFn: () =>
      api.outreach.listTemplates({
        created_by_user_id: userId ?? undefined,
        job_id: selectedJobId ?? undefined,
        limit: 100,
      }),
    enabled: !!userId && !!selectedJobId,
  });

  const columns: ColumnDef<OutreachTemplateResponse>[] = [
    {
      key: "name",
      header: "Template",
      render: (row) => <span className="font-medium text-fg">{row.name}</span>,
    },
    {
      key: "variables_used",
      header: "Variables",
      render: (row) => row.variables_used.join(", ") || "—",
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
    <div className="pl-4 pr-8 py-8 min-h-full space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-[2rem] font-medium text-fg">Outreach Templates</h1>
          <p className="mt-1 text-sm text-fg-muted">
            Create reusable recruiter email templates first, then apply them from the Messages workspace.
          </p>
        </div>
        <Button
          onClick={() => {
            setEditingTemplate(null);
            setCreateOpen(true);
          }}
          disabled={!selectedJobId}
          icon={<Plus size={15} strokeWidth={2} />}
        >
          New template
        </Button>
      </div>

      <OutreachWorkspaceNav />

      {!selectedJobId ? (
        <EmptyState
          heading="Select a job first"
          body="Outreach templates are scoped to the active job in the top bar."
        />
      ) : (
        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg overflow-hidden">
          <DataTable
            columns={columns}
            data={data?.items ?? []}
            loading={isLoading}
            onRowClick={(row) => {
              setEditingTemplate(row);
              setCreateOpen(true);
            }}
            emptyState={
              <EmptyState
                heading="No outreach templates"
                body="Create the first reusable email template for this job before drafting messages."
              />
            }
          />
        </div>
      )}

      <TemplateEditorModal
        template={editingTemplate}
        open={createOpen}
        onOpenChange={(nextOpen) => {
          setCreateOpen(nextOpen);
          if (!nextOpen) {
            setEditingTemplate(null);
          }
        }}
      />
    </div>
  );
}
