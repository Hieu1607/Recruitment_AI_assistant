import type {
  InterviewTemplateCreateRequest,
  InterviewTemplateResponse,
  InterviewTemplateUpdateRequest,
} from "@/api";
import { Button } from "@/components/ui";
import { cn } from "@/lib/cn";
import { useEffect, useState } from "react";

type TemplateFormState = {
  name: string;
  language_code: string;
  status: string;
  intro_script: string;
  closing_script: string;
  question_payload: string;
  report_rubric: string;
};

function toPrettyJson(value: Record<string, unknown> | undefined) {
  return JSON.stringify(value ?? {}, null, 2);
}

function buildInitialState(template?: InterviewTemplateResponse): TemplateFormState {
  return {
    name: template?.name ?? "",
    language_code: template?.language_code ?? "vi-VN",
    status: template?.status ?? "draft",
    intro_script: template?.intro_script ?? "",
    closing_script: template?.closing_script ?? "",
    question_payload: toPrettyJson(template?.question_payload),
    report_rubric: toPrettyJson(template?.report_rubric),
  };
}

function parseJsonField(value: string, field: string) {
  try {
    const parsed = JSON.parse(value || "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { error: `${field} must be a JSON object.` };
    }
    return { value: parsed as Record<string, unknown> };
  } catch {
    return { error: `${field} must be valid JSON.` };
  }
}

export function TemplateEditor({
  mode,
  template,
  submitLabel,
  loading = false,
  onSubmit,
}: {
  mode: "create" | "edit";
  template?: InterviewTemplateResponse;
  submitLabel: string;
  loading?: boolean;
  onSubmit: (
    payload: InterviewTemplateCreateRequest | InterviewTemplateUpdateRequest,
  ) => Promise<void> | void;
}) {
  const [form, setForm] = useState<TemplateFormState>(() => buildInitialState(template));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setForm(buildInitialState(template));
    setError(null);
  }, [template]);

  const statusOptions = mode === "create"
    ? ["draft", "active"]
    : ["draft", "active", "archived"];

  function updateField<K extends keyof TemplateFormState>(key: K, value: TemplateFormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const questionPayload = parseJsonField(form.question_payload, "Question payload");
    if (questionPayload.error) {
      setError(questionPayload.error);
      return;
    }

    const reportRubric = parseJsonField(form.report_rubric, "Report rubric");
    if (reportRubric.error) {
      setError(reportRubric.error);
      return;
    }

    setError(null);
    await onSubmit({
      name: form.name.trim(),
      language_code: form.language_code.trim(),
      status: form.status.trim(),
      intro_script: form.intro_script.trim() || null,
      closing_script: form.closing_script.trim() || null,
      question_payload: questionPayload.value,
      report_rubric: reportRubric.value,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        <label className="space-y-1.5">
          <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
            Template name
          </span>
          <input
            aria-label="Template name"
            value={form.name}
            onChange={(event) => updateField("name", event.target.value)}
            required
            className={cn(
              "h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3",
              "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
            )}
          />
        </label>
        <label className="space-y-1.5">
          <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
            Language code
          </span>
          <input
            aria-label="Language code"
            value={form.language_code}
            onChange={(event) => updateField("language_code", event.target.value)}
            required
            className={cn(
              "h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3",
              "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
            )}
          />
        </label>
      </div>

      <label className="space-y-1.5">
        <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">Status</span>
        <select
          aria-label="Status"
          value={form.status}
          onChange={(event) => updateField("status", event.target.value)}
          className={cn(
            "h-10 w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3",
            "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
          )}
        >
          {statusOptions.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>

      <label className="space-y-1.5">
        <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
          Intro script
        </span>
        <textarea
          aria-label="Intro script"
          value={form.intro_script}
          onChange={(event) => updateField("intro_script", event.target.value)}
          rows={4}
          className={cn(
            "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2.5",
            "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
          )}
        />
      </label>

      <label className="space-y-1.5">
        <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
          Closing script
        </span>
        <textarea
          aria-label="Closing script"
          value={form.closing_script}
          onChange={(event) => updateField("closing_script", event.target.value)}
          rows={4}
          className={cn(
            "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2.5",
            "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
          )}
        />
      </label>

      <div className="grid gap-4 lg:grid-cols-2">
        <label className="space-y-1.5">
          <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
            Question payload
          </span>
          <textarea
            aria-label="Question payload"
            value={form.question_payload}
            onChange={(event) => updateField("question_payload", event.target.value)}
            rows={12}
            className={cn(
              "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2.5",
              "font-mono text-xs text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
            )}
          />
        </label>

        <label className="space-y-1.5">
          <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
            Report rubric
          </span>
          <textarea
            aria-label="Report rubric"
            value={form.report_rubric}
            onChange={(event) => updateField("report_rubric", event.target.value)}
            rows={12}
            className={cn(
              "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2.5",
              "font-mono text-xs text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
            )}
          />
        </label>
      </div>

      {error && (
        <p className="rounded-[var(--radius-md)] border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
          {error}
        </p>
      )}

      <div className="flex justify-end">
        <Button type="submit" loading={loading}>
          {submitLabel}
        </Button>
      </div>
    </form>
  );
}
