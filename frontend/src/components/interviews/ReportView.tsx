import type { InterviewReportResponse } from "@/api";
import { Badge } from "@/components/ui";

function prettyDate(value: string) {
  return new Date(value).toLocaleString();
}

export function ReportView({ report }: { report: InterviewReportResponse }) {
  const status =
    typeof report.report_payload.status === "string"
      ? String(report.report_payload.status)
      : "ready";

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
          <Badge variant={status === "failed" ? "danger" : "success"}>{status}</Badge>
        </div>
        <div className="mt-4 grid gap-3 text-sm text-fg-muted md:grid-cols-2">
          <div>Created {prettyDate(report.created_at)}</div>
          <div>Updated {prettyDate(report.updated_at)}</div>
        </div>
      </div>

      <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
        <h2 className="font-display text-xl font-medium text-fg">Summary</h2>
        <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">
          {report.summary_text || "No summary text is available for this report yet."}
        </p>
      </section>

      <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
        <h2 className="font-display text-xl font-medium text-fg">Report Payload</h2>
        <pre className="mt-3 overflow-x-auto rounded-[var(--radius-md)] bg-bg-elevated p-4 text-xs text-fg">
          {JSON.stringify(report.report_payload, null, 2)}
        </pre>
      </section>
    </div>
  );
}
