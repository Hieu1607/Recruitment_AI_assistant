import type { InterviewReportResponse } from "@/api";
import { Badge } from "@/components/ui";

function prettyDate(value: string) {
  return new Date(value).toLocaleString();
}

export function ReportView({ report }: { report: InterviewReportResponse }) {
  const payload = report.report_payload;
  const status = payload?.status ?? "pending";
  const summary = payload?.summary ?? null;
  const failure = payload?.failure ?? null;

  return (
    <div className="space-y-6">
      <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h1 className="font-display text-[2rem] font-medium text-fg">Interview Report</h1>
            <p className="mt-1 text-sm text-fg-muted">
              Session {report.interview_session_id}
            </p>
          </div>
          <Badge variant={status === "failed" ? "danger" : status === "pending" ? "warning" : "success"}>
            {status}
          </Badge>
        </div>
        <div className="mt-4 grid gap-3 text-sm text-fg-muted md:grid-cols-2">
          <div>Created {prettyDate(report.created_at)}</div>
          <div>Updated {prettyDate(report.updated_at)}</div>
        </div>
      </div>

      {status === "failed" && (
        <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
          <h2 className="font-display text-xl font-medium text-fg">Report Generation Failed</h2>
          <p className="mt-3 text-sm leading-relaxed text-danger">
            {failure?.message || "The report could not be generated."}
          </p>
        </section>
      )}

      {status === "pending" && (
        <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
          <p className="text-sm leading-relaxed text-fg-muted">
            The report is still being generated. Check back shortly.
          </p>
        </section>
      )}

      {status === "completed" && summary && (
        <>
          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
            <h2 className="font-display text-xl font-medium text-fg">Candidate Overview</h2>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">
              {summary.candidate_overview}
            </p>
          </section>

          <section className="space-y-4">
            <h2 className="font-display text-xl font-medium text-fg">Questions &amp; Answers</h2>
            {summary.questions.map((item, index) => (
              <div
                key={item.question_transcript_turn_id || index}
                className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5"
              >
                <p className="text-sm font-medium text-fg">
                  {index + 1}. {item.question_text}
                </p>
                <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg-muted">
                  {item.answer_text}
                </p>
                <div className="mt-4 rounded-[var(--radius-md)] bg-bg-elevated p-3">
                  <p className="text-xs font-medium uppercase tracking-wide text-fg-muted">Evaluation</p>
                  <p className="mt-1 whitespace-pre-wrap text-sm leading-relaxed text-fg">{item.evaluation}</p>
                </div>
              </div>
            ))}
          </section>

          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
            <h2 className="font-display text-xl font-medium text-fg">Overall Summary</h2>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">
              {summary.overall_summary}
            </p>
          </section>
        </>
      )}

      {status === "completed" && !summary && (
        <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
          <p className="text-sm leading-relaxed text-fg-muted">
            {report.summary_text || "No summary is available for this report yet."}
          </p>
        </section>
      )}
    </div>
  );
}
