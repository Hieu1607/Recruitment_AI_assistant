import { ApiError, api } from "@/api";
import { Button, Skeleton } from "@/components/ui";
import { fieldClasses } from "@/components/jobs/job-utils";
import { cn } from "@/lib/cn";
import { useMutation, useQuery } from "@tanstack/react-query";
import { CheckCircle2, FileText, UploadCloud } from "lucide-react";
import { useMemo, useState } from "react";
import { useParams } from "react-router";

export default function PublicApplyRoute() {
  const { token = "" } = useParams();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitted, setSubmitted] = useState(false);

  const jobQuery = useQuery({
    queryKey: ["public-jobs", token],
    enabled: token.length > 0,
    queryFn: () => api.publicJobs.get(token),
    retry: false,
  });

  const uploadResume = useMutation({
    mutationFn: () =>
      api.publicJobs.uploadResume(token, {
        fullName: fullName.trim(),
        email: email.trim(),
        file: file!,
      }),
    onSuccess: () => setSubmitted(true),
  });

  const errorMessage = useMemo(() => {
    if (!jobQuery.error) return null;
    if (jobQuery.error instanceof ApiError) {
      if (jobQuery.error.status === 410) return "This application link is currently disabled.";
      if (jobQuery.error.status === 404) return "This application link was not found.";
      return jobQuery.error.message;
    }
    return jobQuery.error instanceof Error ? jobQuery.error.message : "Unable to load this application link.";
  }, [jobQuery.error]);

  const uploadError = useMemo(() => {
    if (!uploadResume.error) return null;
    return uploadResume.error instanceof Error ? uploadResume.error.message : "Unable to submit your resume.";
  }, [uploadResume.error]);

  const canSubmit =
    fullName.trim().length > 0 &&
    email.trim().length > 0 &&
    Boolean(file) &&
    !uploadResume.isPending;

  return (
    <main className="min-h-full bg-bg px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-4xl flex-col justify-center">
        <div className="border-b border-[color:var(--hairline)] pb-6">
          <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">RecruitAI application</p>
          {jobQuery.isLoading ? (
            <div className="mt-4 space-y-3">
              <Skeleton className="h-12 w-2/3" />
              <Skeleton className="h-5 w-full max-w-xl" />
            </div>
          ) : errorMessage ? (
            <>
              <h1 className="mt-4 font-display text-4xl leading-tight text-fg">Application unavailable</h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-fg-muted">{errorMessage}</p>
            </>
          ) : (
            <>
              <h1 className="mt-4 font-display text-4xl leading-tight text-fg">
                {jobQuery.data?.job_title ?? "Submit your resume"}
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-fg-muted">
                {jobQuery.data?.candidate_message ||
                  "Share your contact details and upload a PDF resume for this role."}
              </p>
            </>
          )}
        </div>

        {!errorMessage && (
          <section className="mt-8 rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6 sm:p-8">
            {submitted ? (
              <div className="flex flex-col items-start gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-[var(--radius-md)] bg-[rgba(47,111,78,0.10)] text-success">
                  <CheckCircle2 size={24} strokeWidth={1.75} />
                </div>
                <div>
                  <h2 className="font-display text-3xl leading-tight text-fg">Resume submitted</h2>
                  <p className="mt-3 max-w-xl text-sm leading-6 text-fg-muted">
                    Your resume was received. The hiring team will review it inside this job workspace.
                  </p>
                </div>
              </div>
            ) : (
              <form
                className="space-y-5"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (canSubmit) uploadResume.mutate();
                }}
              >
                <div className="grid gap-5 sm:grid-cols-2">
                  <div>
                    <label htmlFor="full-name" className="text-sm font-medium text-fg-muted">
                      Full name
                    </label>
                    <input
                      id="full-name"
                      value={fullName}
                      onChange={(event) => setFullName(event.target.value)}
                      autoComplete="name"
                      className={cn(fieldClasses, "mt-2")}
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="text-sm font-medium text-fg-muted">
                      Email
                    </label>
                    <input
                      id="email"
                      type="email"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      autoComplete="email"
                      className={cn(fieldClasses, "mt-2")}
                      required
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="resume-file" className="text-sm font-medium text-fg-muted">
                    Resume PDF
                  </label>
                  <label
                    htmlFor="resume-file"
                    className="mt-2 flex cursor-pointer flex-col items-center justify-center rounded-[var(--radius-lg)] border border-dashed border-[color:var(--hairline-strong)] bg-bg px-6 py-8 text-center transition-colors hover:bg-bg-sidebar"
                  >
                    <UploadCloud size={28} strokeWidth={1.5} className="text-fg-subtle" />
                    <span className="mt-3 text-sm font-medium text-fg">
                      {file ? file.name : "Choose a PDF resume"}
                    </span>
                    <span className="mt-1 text-xs text-fg-muted">PDF only</span>
                    <input
                      id="resume-file"
                      type="file"
                      accept="application/pdf,.pdf"
                      className="sr-only"
                      onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                      required
                    />
                  </label>
                </div>

                {uploadError && (
                  <div className="rounded-[var(--radius-md)] border border-[rgba(184,68,46,0.24)] bg-[rgba(184,68,46,0.06)] p-4 text-sm text-danger">
                    {uploadError}
                  </div>
                )}

                <div className="flex flex-col gap-3 border-t border-[color:var(--hairline)] pt-5 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2 text-sm text-fg-muted">
                    <FileText size={15} strokeWidth={1.75} />
                    {file ? `${Math.max(1, Math.round(file.size / 1024))} KB selected` : "No file selected"}
                  </div>
                  <Button loading={uploadResume.isPending} disabled={!canSubmit}>
                    Submit resume
                  </Button>
                </div>
              </form>
            )}
          </section>
        )}
      </div>
    </main>
  );
}
