import type {
  InterviewTemplateCreateRequest,
  InterviewTemplateResponse,
  InterviewTemplateUpdateRequest,
} from "@/api";
import { Button } from "@/components/ui";
import { cn } from "@/lib/cn";
import { Plus, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type QuestionDraft = {
  id: string;
  prompt: string;
};

type TemplateFormState = {
  name: string;
  language_code: string;
  status: string;
  intro_script: string;
  closing_script: string;
  report_rubric: string;
};

function toPrettyJson(value: Record<string, unknown> | undefined) {
  return JSON.stringify(value ?? {}, null, 2);
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

function toQuestionDrafts(payload: Record<string, unknown> | undefined): QuestionDraft[] {
  const rawQuestions = Array.isArray(payload?.questions) ? payload.questions : [];

  return rawQuestions
    .map((item, index) => {
      if (!item || typeof item !== "object") return null;
      const candidate = item as Record<string, unknown>;
      const prompt = typeof candidate.prompt === "string"
        ? candidate.prompt
        : typeof candidate.text === "string"
          ? candidate.text
          : "";
      if (!prompt.trim()) return null;
      return {
        id: `question-${index + 1}`,
        prompt: prompt.trim(),
      } satisfies QuestionDraft;
    })
    .filter((item): item is QuestionDraft => item !== null);
}

function withoutQuestions(payload: Record<string, unknown> | undefined) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return {};
  }

  const { questions: _questions, ...rest } = payload;
  return rest;
}

function buildInitialState(template?: InterviewTemplateResponse): TemplateFormState {
  return {
    name: template?.name ?? "",
    language_code: template?.language_code ?? "vi-VN",
    status: template?.status ?? "draft",
    intro_script: template?.intro_script ?? "",
    closing_script: template?.closing_script ?? "",
    report_rubric: toPrettyJson(template?.report_rubric),
  };
}

function buildQuestionPayload(basePayload: Record<string, unknown>, questions: QuestionDraft[]) {
  return {
    ...basePayload,
    questions: questions
      .map((question, index) => ({
        key: `question_${index + 1}`,
        prompt: question.prompt.trim(),
      }))
      .filter((question) => question.prompt),
  };
}

function buildQuestionPreview(basePayload: Record<string, unknown>, questions: QuestionDraft[]) {
  return JSON.stringify(buildQuestionPayload(basePayload, questions), null, 2);
}

function parseQuestionImport(value: string) {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^(\d+[\.\)]|[-*])\s*/, "").trim())
    .filter(Boolean);
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
  const [questions, setQuestions] = useState<QuestionDraft[]>(() => toQuestionDrafts(template?.question_payload));
  const [questionImport, setQuestionImport] = useState("");
  const [error, setError] = useState<string | null>(null);

  const baseQuestionPayload = useMemo(
    () => withoutQuestions(template?.question_payload),
    [template],
  );

  useEffect(() => {
    setForm(buildInitialState(template));
    setQuestions(toQuestionDrafts(template?.question_payload));
    setQuestionImport("");
    setError(null);
  }, [template]);

  const statusOptions = mode === "create"
    ? ["draft", "active"]
    : ["draft", "active", "archived"];

  const questionPreview = useMemo(
    () => buildQuestionPreview(baseQuestionPayload, questions),
    [baseQuestionPayload, questions],
  );

  function updateField<K extends keyof TemplateFormState>(key: K, value: TemplateFormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function addQuestion() {
    setQuestions((current) => [
      ...current,
      {
        id: `question-${Date.now()}-${current.length + 1}`,
        prompt: "",
      },
    ]);
  }

  function updateQuestion(id: string, prompt: string) {
    setQuestions((current) =>
      current.map((question) => (question.id === id ? { ...question, prompt } : question)),
    );
  }

  function removeQuestion(id: string) {
    setQuestions((current) => current.filter((question) => question.id !== id));
  }

  function importQuestions() {
    const parsedQuestions = parseQuestionImport(questionImport);
    if (!parsedQuestions.length) {
      setError("Question import must contain at least one non-empty line.");
      return;
    }

    setQuestions(
      parsedQuestions.map((prompt, index) => ({
        id: `question-import-${index + 1}`,
        prompt,
      })),
    );
    setError(null);
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const reportRubric = parseJsonField(form.report_rubric, "Report rubric");
    if (reportRubric.error) {
      setError(reportRubric.error);
      return;
    }

    const normalizedQuestions = questions
      .map((question) => ({
        ...question,
        prompt: question.prompt.trim(),
      }))
      .filter((question) => question.prompt);

    if (!normalizedQuestions.length) {
      setError("Add at least one interview question before saving the template.");
      return;
    }

    setError(null);
    await onSubmit({
      name: form.name.trim(),
      language_code: form.language_code.trim(),
      status: form.status.trim(),
      intro_script: form.intro_script.trim() || null,
      closing_script: form.closing_script.trim() || null,
      question_payload: buildQuestionPayload(baseQuestionPayload, normalizedQuestions),
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

      <section className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-medium text-fg">Interview questions</h2>
            <p className="text-sm text-fg-muted">
              Add each question one by one or import a numbered list.
            </p>
          </div>
          <Button type="button" variant="secondary" onClick={addQuestion} icon={<Plus size={15} strokeWidth={2} />}>
            Add question
          </Button>
        </div>

        {questions.length ? (
          <div className="space-y-3">
            {questions.map((question, index) => (
              <div
                key={question.id}
                className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-3"
              >
                <div className="mb-2 flex items-center justify-between gap-3">
                  <span className="text-sm font-medium text-fg">Question {index + 1}</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => removeQuestion(question.id)}
                    icon={<Trash2 size={14} strokeWidth={2} />}
                  >
                    Remove
                  </Button>
                </div>
                <textarea
                  aria-label={`Question ${index + 1}`}
                  value={question.prompt}
                  onChange={(event) => updateQuestion(question.id, event.target.value)}
                  rows={3}
                  placeholder={`Question ${index + 1}`}
                  className={cn(
                    "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2.5",
                    "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                  )}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-[var(--radius-md)] border border-dashed border-[color:var(--hairline-strong)] px-4 py-6 text-sm text-fg-muted">
            No questions yet. Add one manually or import from a sample list below.
          </div>
        )}

        <div className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-4 space-y-3">
          <div>
            <h3 className="text-sm font-medium text-fg">Import from list</h3>
            <p className="text-sm text-fg-muted">
              Paste one question per line. Supported formats: <code>1.</code>, <code>1)</code>, or <code>-</code>.
            </p>
          </div>
          <textarea
            aria-label="Question list import"
            value={questionImport}
            onChange={(event) => setQuestionImport(event.target.value)}
            rows={6}
            placeholder={"1. Ban hay gioi thieu ngan ve ban than.\n2. Tai sao ban muon ung tuyen vi tri nay?\n3. Ban da xu ly tinh huong kho voi khach hang nhu the nao?"}
            className={cn(
              "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2.5",
              "text-sm text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
            )}
          />
          <div className="flex justify-end">
            <Button type="button" variant="secondary" onClick={importQuestions}>
              Import questions
            </Button>
          </div>
        </div>

        <label className="space-y-1.5">
          <span className="block text-xs font-medium uppercase tracking-wide text-fg-muted">
            Question payload preview
          </span>
          <textarea
            aria-label="Question payload preview"
            value={questionPreview}
            readOnly
            rows={10}
            className={cn(
              "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated px-3 py-2.5",
              "font-mono text-xs text-fg-muted outline-none",
            )}
          />
        </label>
      </section>

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
