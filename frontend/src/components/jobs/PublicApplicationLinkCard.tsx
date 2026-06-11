import { api, type JobApplicationLinkResponse, type JobResponse } from "@/api";
import { Button, Skeleton } from "@/components/ui";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, ExternalLink, QrCode, RefreshCw } from "lucide-react";
import QRCode from "qrcode";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { panelClasses } from "./job-utils";

async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

export function PublicApplicationLinkCard({
  job,
  showRotate = false,
  className,
}: {
  job: JobResponse | null | undefined;
  showRotate?: boolean;
  className?: string;
}) {
  const qc = useQueryClient();
  const [qrDataUrl, setQrDataUrl] = useState("");
  const linkQuery = useQuery({
    queryKey: ["jobs", job?.id, "application-link"],
    enabled: Boolean(job?.id),
    queryFn: () => api.jobs.applicationLink.get(job!.id),
    staleTime: 60_000,
  });

  const rotateLink = useMutation({
    mutationFn: () => api.jobs.applicationLink.rotate(job!.id),
    onSuccess: (data) => {
      qc.setQueryData<JobApplicationLinkResponse>(["jobs", job?.id, "application-link"], data);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      if (job?.id) qc.invalidateQueries({ queryKey: ["jobs", job.id] });
      toast.success("Candidate upload link rotated");
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Unable to rotate link");
    },
  });

  const linkData = linkQuery.data ?? (job
    ? {
        public_apply_enabled: job.public_apply_enabled,
        public_apply_url: job.public_apply_url,
        candidate_message: job.candidate_message,
      }
    : null);
  const publicApplyUrl = linkData?.public_apply_url ?? "";
  const isDisabled = linkData ? !linkData.public_apply_enabled : false;

  useEffect(() => {
    let cancelled = false;
    if (!publicApplyUrl) {
      setQrDataUrl("");
      return;
    }

    QRCode.toDataURL(publicApplyUrl, {
      errorCorrectionLevel: "M",
      margin: 2,
      width: 220,
      color: {
        dark: "#1f3a2e",
        light: "#ffffff",
      },
    })
      .then((dataUrl) => {
        if (!cancelled) setQrDataUrl(dataUrl);
      })
      .catch(() => {
        if (!cancelled) setQrDataUrl("");
      });

    return () => {
      cancelled = true;
    };
  }, [publicApplyUrl]);

  return (
    <section className={cn(panelClasses, "p-6", className)}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-fg-subtle">Candidate upload</p>
          <p className="mt-3 font-display text-3xl leading-tight text-fg">Link and QR</p>
        </div>
        <QrCode size={18} strokeWidth={1.5} className="mt-1 text-fg-subtle" />
      </div>

      {!job ? (
        <p className="mt-4 text-sm leading-6 text-fg-muted">
          Select a job to share its public resume upload link.
        </p>
      ) : linkQuery.isLoading && !linkData ? (
        <div className="mt-5 space-y-3">
          <Skeleton className="h-44 w-full rounded-[var(--radius-lg)]" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : linkQuery.error ? (
        <div className="mt-5 rounded-[var(--radius-md)] border border-[rgba(184,68,46,0.24)] bg-[rgba(184,68,46,0.06)] p-4 text-sm text-danger">
          Unable to load the candidate upload link.
        </div>
      ) : (
        <>
          <div className="mt-5 flex flex-col items-center gap-4">
            <div className="flex h-44 w-44 shrink-0 items-center justify-center rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-white p-3">
              {qrDataUrl ? (
                <img
                  src={qrDataUrl}
                  alt={`QR code for ${job.title} resume upload`}
                  className="h-full w-full object-contain"
                />
              ) : (
                <QrCode size={48} strokeWidth={1.5} className="text-fg-subtle" />
              )}
            </div>
            <div className="w-full min-w-0 space-y-3">
              <div className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg px-3 py-2">
                <p className="truncate font-mono text-xs text-fg-muted" title={publicApplyUrl}>
                  {publicApplyUrl || "No public URL available"}
                </p>
              </div>
              <p className="max-w-[56ch] text-sm leading-6 text-fg-muted sm:text-center">
                {isDisabled
                  ? "This link is currently disabled for candidates."
                  : "Candidates can open this link or scan the QR code to submit a PDF resume."}
              </p>
            </div>
          </div>

          <div className="mt-5 flex flex-wrap gap-2">
            <Button
              variant="secondary"
              icon={<Copy size={15} strokeWidth={1.75} />}
              disabled={!publicApplyUrl}
              onClick={() => {
                copyText(publicApplyUrl)
                  .then(() => toast.success("Candidate upload link copied"))
                  .catch(() => toast.error("Unable to copy link"));
              }}
            >
              Copy link
            </Button>
            <a
              href={publicApplyUrl || undefined}
              target="_blank"
              rel="noreferrer"
              aria-disabled={!publicApplyUrl}
              className={cn(
                "inline-flex h-9 items-center justify-center gap-2 rounded-[var(--radius-md)] border border-hairline-strong bg-bg-elevated px-4 text-sm font-medium text-fg transition-colors hover:bg-bg-sidebar",
                !publicApplyUrl && "pointer-events-none opacity-50",
              )}
            >
              <ExternalLink size={15} strokeWidth={1.75} />
              Open
            </a>
            {showRotate && (
              <Button
                variant="ghost"
                icon={<RefreshCw size={15} strokeWidth={1.75} />}
                loading={rotateLink.isPending}
                disabled={!job}
                onClick={() => {
                  if (window.confirm("Rotate this candidate upload link? The old QR code and link will stop working.")) {
                    rotateLink.mutate();
                  }
                }}
              >
                Rotate
              </Button>
            )}
          </div>
        </>
      )}
    </section>
  );
}
