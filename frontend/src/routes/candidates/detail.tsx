import type {
  CandidateProfileResponse,
  InterviewInvitationResponse,
  OutreachResponse,
  ResumeResponse,
  StructuredEntry,
  StructuredLink,
  StructuredSection,
  StructuredSummary,
} from "@/api";
import { api } from "@/api";
import { InvitationSendDialog } from "@/components/interviews/InvitationSendDialog";
import { Avatar } from "@/components/ui/avatar";
import { Badge, Button, EmptyState, Skeleton } from "@/components/ui";
import { useSelectedJobId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, BookOpen, Calendar, ExternalLink, Mail } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router";

type Tab = "overview" | "resume" | "outreach" | "interview";

function fileToName(value: string) {
  return value.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ").replace(/\b\w/g, (match) => match.toUpperCase());
}

function relativeTime(value: string | null) {
  if (!value) return "—";
  const diff = Date.now() - new Date(value).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function absoluteDate(value: string | null) {
  if (!value) return "No infomation";
  return new Date(value).toLocaleDateString();
}

function statusVariant(status: string): "neutral" | "warning" | "success" | "danger" {
  if (status === "processed" || status === "completed" || status === "sent") return "success";
  if (status === "processing" || status === "pending") return "warning";
  if (status === "failed" || status === "cancelled") return "danger";
  return "neutral";
}

const missingDisplayValues = new Set(["-", "--", "n/a", "na", "none", "null", "notapplicable"]);

function cleanDisplayText(value: string | null | undefined) {
  const normalized = value?.trim();
  if (!normalized) return null;
  const comparable = normalized.toLowerCase().replace(/\./g, "").replace(/\s+/g, "");
  return missingDisplayValues.has(comparable) ? null : normalized;
}

function cleanDisplayList(values: string[]) {
  return values.map(cleanDisplayText).filter((item): item is string => item !== null);
}

function cleanDisplayLinks(links: StructuredLink[]) {
  return links
    .map((link) => {
      const url = cleanDisplayText(link.url);
      if (!url) return null;
      return { url, label: cleanDisplayText(link.label) };
    })
    .filter((link): link is StructuredLink => link !== null);
}

function cleanStructuredEntry(entry: StructuredEntry): StructuredEntry | null {
  const cleaned: StructuredEntry = {
    title: cleanDisplayText(entry.title),
    subtitle: cleanDisplayText(entry.subtitle),
    role: cleanDisplayText(entry.role),
    location: cleanDisplayText(entry.location),
    dateRange: cleanDisplayText(entry.dateRange),
    description: cleanDisplayText(entry.description),
    bullets: cleanDisplayList(entry.bullets),
    links: cleanDisplayLinks(entry.links),
    metadata: cleanDisplayList(entry.metadata),
  };

  if (
    cleaned.title ||
    cleaned.subtitle ||
    cleaned.role ||
    cleaned.location ||
    cleaned.dateRange ||
    cleaned.description ||
    cleaned.bullets.length > 0 ||
    cleaned.links.length > 0 ||
    cleaned.metadata.length > 0
  ) {
    return cleaned;
  }
  return null;
}

function displayValue(value: string | number | boolean | null | undefined) {
  if (value === null || value === undefined) return "No infomation";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return String(value);
  return cleanDisplayText(value) || "No infomation";
}

function renderLinks(links: StructuredLink[]) {
  const visibleLinks = cleanDisplayLinks(links);
  if (visibleLinks.length === 0) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {visibleLinks.map((link, index) => (
        <a
          key={`${link.url}-${index}`}
          href={link.url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 rounded-full border border-[color:var(--hairline)] px-3 py-1 text-xs text-accent hover:bg-[rgba(31,58,46,0.05)]"
        >
          {link.label || link.url}
          <ExternalLink size={11} strokeWidth={1.75} />
        </a>
      ))}
    </div>
  );
}

function hasStructuredSection(section: StructuredSection | null | undefined) {
  return !!section && (section.entries.length > 0 || !!section.rawText);
}

function StructuredSummaryBlock({ summary, fallbackText }: { summary?: StructuredSummary | null; fallbackText?: string | null }) {
  const text = cleanDisplayText(summary?.text) || cleanDisplayText(fallbackText);
  const links = summary?.links ?? [];
  return (
    <>
      <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">{displayValue(text)}</p>
      {renderLinks(links)}
    </>
  );
}

function StructuredSectionBlock({
  section,
  fallbackText,
}: {
  section?: StructuredSection | null;
  fallbackText?: string | null;
}) {
  if (!hasStructuredSection(section)) {
    return <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">{displayValue(fallbackText)}</p>;
  }

  const entries = section!.entries.map(cleanStructuredEntry).filter((entry): entry is StructuredEntry => entry !== null);
  const rawText = cleanDisplayText(section?.rawText) || cleanDisplayText(fallbackText);

  if (entries.length === 0 && !rawText) {
    return <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">No infomation</p>;
  }

  return (
    <div className="mt-4 space-y-4">
      {entries.map((entry, index) => (
        (() => {
          const hasListContent = entry.metadata.length > 0 || entry.bullets.length > 0;

          return (
            <article
              key={`${entry.title || entry.subtitle || "entry"}-${index}`}
              className="rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg px-4 py-3"
            >
              <div className="space-y-2">
                {(entry.title || entry.subtitle) && (
                  <div>
                    {entry.title && <h3 className="text-sm font-medium text-fg">{entry.title}</h3>}
                    {entry.subtitle && <p className="text-sm text-fg-muted">{entry.subtitle}</p>}
                  </div>
                )}
                {(entry.role || entry.location || entry.dateRange) && (
                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-fg-muted">
                    {entry.role && <span>{entry.role}</span>}
                    {entry.location && <span>{entry.location}</span>}
                    {entry.dateRange && <span>{entry.dateRange}</span>}
                  </div>
                )}
                {entry.description && !hasListContent && (
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-fg">{entry.description}</p>
                )}
                {entry.metadata.length > 0 && (
                  <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-fg">
                    {entry.metadata.map((item, metaIndex) => (
                      <li key={`${item}-${metaIndex}`}>{item}</li>
                    ))}
                  </ul>
                )}
                {entry.bullets.length > 0 && (
                  <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-fg">
                    {entry.bullets.map((item, bulletIndex) => (
                      <li key={`${item}-${bulletIndex}`}>{item}</li>
                    ))}
                  </ul>
                )}
                {renderLinks(entry.links)}
              </div>
            </article>
          );
        })()
      ))}
      {rawText && entries.length === 0 && (
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-fg">{rawText}</p>
      )}
    </div>
  );
}

function TabButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "border-b-2 px-4 py-3 text-sm transition-colors",
        active ? "border-accent font-medium text-fg" : "border-transparent text-fg-muted hover:text-fg",
      )}
    >
      {children}
    </button>
  );
}

function OverviewTab({
  resume,
  profile,
}: {
  resume: ResumeResponse;
  profile: CandidateProfileResponse | undefined;
}) {
  const displayName = profile?.full_name || profile?.submitted_full_name;
  const displayEmail = profile?.email || profile?.submitted_email;
  const sectionFields = [
    { title: "Name", kind: "plain" as const, content: displayName },
    { title: "Current Role", kind: "plain" as const, content: profile?.current_job_title },
    { title: "Summary", kind: "summary" as const, content: profile?.summary_text, structured: profile?.structured_profile?.summary },
    { title: "Skills", kind: "structured" as const, content: profile?.skills_text, structured: profile?.structured_profile?.skills },
    { title: "Experience", kind: "structured" as const, content: profile?.experience_text, structured: profile?.structured_profile?.experience },
    { title: "Education", kind: "structured" as const, content: profile?.education_text, structured: profile?.structured_profile?.education },
    { title: "Projects", kind: "structured" as const, content: profile?.projects_text, structured: profile?.structured_profile?.projects },
    { title: "Languages", kind: "structured" as const, content: profile?.languages_text, structured: profile?.structured_profile?.languages },
    { title: "Achievements", kind: "structured" as const, content: profile?.achievements_text, structured: profile?.structured_profile?.achievements },
    { title: "Publications", kind: "structured" as const, content: profile?.publications_text, structured: profile?.structured_profile?.publications },
    { title: "Certifications", kind: "structured" as const, content: profile?.certifications_text, structured: profile?.structured_profile?.certifications },
    { title: "References", kind: "structured" as const, content: profile?.references_text, structured: profile?.structured_profile?.references },
    { title: "Other", kind: "structured" as const, content: profile?.other_text, structured: profile?.structured_profile?.other },
  ];

  const resumeInfoRows = [
    { label: "Status", value: resume.upload_status, badge: true },
    { label: "Uploaded", value: absoluteDate(resume.uploaded_at) },
    { label: "Processed", value: absoluteDate(resume.processed_at) },
    { label: "Extraction mode", value: profile?.extraction_mode },
    { label: "Email", value: displayEmail },
    { label: "Phone", value: profile?.phone },
    { label: "Location", value: profile?.location_normalized },
    { label: "Contact", value: profile?.contact },
    { label: "Experience years", value: profile?.experience_years },
    { label: "Major", value: profile?.major },
    { label: "CPA", value: profile?.cpa },
    { label: "Graduation status", value: profile?.graduation_status },
    { label: "Studied abroad", value: profile?.ever_studied_abroad },
  ];

  return (
    <div className="grid gap-8 lg:grid-cols-[2fr_1fr]">
      <div className="space-y-5">
        {!profile && (
          <EmptyState
            heading="Candidate profile unavailable"
            body={`This resume is ${resume.upload_status}. Sections are shown with fallback values until profile data is available.`}
          />
        )}

        {sectionFields.map((section) => (
          <section
            key={section.title}
            className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5"
          >
            <h2 className="font-display text-xl font-medium text-fg">{section.title}</h2>
            {section.kind === "plain" && (
              <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">
                {displayValue(section.content)}
              </p>
            )}
            {section.kind === "summary" && (
              <StructuredSummaryBlock summary={section.structured} fallbackText={section.content} />
            )}
            {section.kind === "structured" && (
              <StructuredSectionBlock section={section.structured} fallbackText={section.content} />
            )}
          </section>
        ))}
      </div>

      <aside className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
        <h2 className="font-display text-xl font-medium text-fg">Resume Info</h2>
        <div className="mt-4 space-y-3 text-sm">
          {resumeInfoRows.map((row) => (
            <div key={row.label} className="flex items-center justify-between gap-3">
              <span className="text-fg-muted">{row.label}</span>
              {row.badge ? (
                <Badge variant={statusVariant(String(row.value ?? ""))}>{displayValue(row.value)}</Badge>
              ) : (
                <span className="text-right text-fg">{displayValue(row.value)}</span>
              )}
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

function ResumeTab({ resume }: { resume: ResumeResponse }) {
  const { data: pdfBlob, isLoading, isError } = useQuery({
    queryKey: ["candidate-resume-file", resume.id],
    queryFn: () => api.upload.getFile(resume.id),
    enabled: resume.upload_status === "processed",
    staleTime: 5 * 60 * 1000,
  });
  const pdfUrl = useMemo(() => (pdfBlob ? URL.createObjectURL(pdfBlob) : null), [pdfBlob]);

  useEffect(() => {
    return () => {
      if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    };
  }, [pdfUrl]);

  if (resume.upload_status !== "processed") {
    return (
      <EmptyState
        heading="Resume preview unavailable"
        body={`Preview will appear after processing finishes. Current status: ${resume.upload_status}.`}
      />
    );
  }

  if (isLoading) {
    return <Skeleton className="h-[640px] w-full rounded-[var(--radius-lg)]" />;
  }

  if (isError || !pdfUrl) {
    return (
      <EmptyState
        heading="Resume preview unavailable"
        body="The PDF could not be loaded from storage right now."
      />
    );
  }

  return (
    <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated overflow-hidden">
      <div className="flex items-center justify-between border-b border-[color:var(--hairline)] px-4 py-3">
        <span className="text-sm text-fg">{resume.original_file_name}</span>
        <a href={pdfUrl} target="_blank" rel="noreferrer" className="text-sm text-accent hover:underline">
          Open PDF
        </a>
      </div>
      <iframe src={pdfUrl} title="Resume PDF" className="h-[640px] w-full" />
    </div>
  );
}

function OutreachTab({ items }: { items: OutreachResponse[] }) {
  if (items.length === 0) {
    return (
      <EmptyState
        heading="No outreach messages"
        body="Draft recruiter outreach from the Outreach workspace when you want to contact this candidate."
      />
    );
  }

  return (
    <div className="space-y-3">
      {items.map((item) => (
        <section key={item.id} className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="font-medium text-fg">{item.subject}</h3>
              <p className="mt-2 text-sm text-fg-muted">{item.body}</p>
            </div>
            <Badge variant={statusVariant(item.sent_status)}>{item.sent_status}</Badge>
          </div>
        </section>
      ))}
    </div>
  );
}

function InterviewsTab({ items }: { items: InterviewInvitationResponse[] }) {
  if (items.length === 0) {
    return (
      <EmptyState
        heading="No interview invitations"
        body="Send a recruiter interview invitation for this candidate to create the interview link and track status."
      />
    );
  }

  return (
    <div className="space-y-3">
      {items.map((item) => (
        <section key={item.id} className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-2">
              <h3 className="font-medium text-fg">{item.interview_template_name || "Interview template"}</h3>
              <div className="flex flex-wrap items-center gap-3 text-sm text-fg-muted">
                <span>{item.status}</span>
                <span>Expires {absoluteDate(item.expires_at)}</span>
                <span>Attempts {item.attempt_count}/{item.max_attempts}</span>
              </div>
              <a
                href={item.public_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-sm text-accent hover:underline"
              >
                Open interview link
                <ExternalLink size={12} strokeWidth={1.75} />
              </a>
            </div>
            <Badge variant={statusVariant(item.status)}>{item.status}</Badge>
          </div>
        </section>
      ))}
    </div>
  );
}

export default function CandidateDetailRoute() {
  const { id } = useParams<{ id: string }>();
  const selectedJobId = useSelectedJobId();
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [invitationDialogOpen, setInvitationDialogOpen] = useState(false);

  const { data: resume, isLoading: resumeLoading } = useQuery({
    queryKey: ["candidate", id],
    queryFn: () => api.upload.get(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.upload_status;
      return status === "uploaded" || status === "processing" ? 3000 : false;
    },
  });

  const { data: profile } = useQuery({
    queryKey: ["candidate-profile", id],
    queryFn: () => api.upload.getProfile(id!),
    enabled: !!id && resume?.upload_status === "processed",
    retry: false,
  });

  const { data: outreachData } = useQuery({
    queryKey: ["outreach-candidate", profile?.id],
    queryFn: () => api.outreach.list({ candidate_profile_id: profile!.id, limit: 50 }),
    enabled: !!profile?.id,
  });

  const { data: invitationData } = useQuery({
    queryKey: ["interview-invitations", selectedJobId],
    queryFn: () => api.interviewInvitations.list(selectedJobId!),
    enabled: !!selectedJobId,
  });

  const invitations = useMemo(
    () => (invitationData?.items ?? []).filter((item) => item.candidate_profile_id === profile?.id),
    [invitationData?.items, profile?.id],
  );

  const outreachItems = outreachData?.items ?? [];

  if (resumeLoading) {
    return (
      <div className="px-8 py-8 space-y-4">
        <Skeleton className="h-10 w-72" />
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-80 w-full" />
      </div>
    );
  }

  if (!resume) {
    return (
      <div className="px-8 py-8">
        <EmptyState heading="Candidate not found" body="The candidate record could not be loaded." />
      </div>
    );
  }

  const candidateName = profile?.full_name || profile?.submitted_full_name || fileToName(resume.original_file_name);

  return (
    <div className="px-8 py-8 min-h-full">
      <div className="mx-auto max-w-6xl space-y-6">
        <Link to={routes.candidates} className="inline-flex items-center gap-1.5 text-sm text-fg-muted hover:text-fg">
          <ArrowLeft size={14} strokeWidth={2} />
          Candidates
        </Link>

        <header className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-4">
            <Avatar name={candidateName} size="xl" />
            <div>
              <h1 className="font-display text-[2.5rem] font-medium text-fg">{candidateName}</h1>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <Badge variant={statusVariant(resume.upload_status)}>{resume.upload_status}</Badge>
                <span className="inline-flex items-center gap-1.5 text-sm text-fg-muted">
                  <Calendar size={12} strokeWidth={1.75} />
                  Uploaded {relativeTime(resume.uploaded_at)}
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              icon={<BookOpen size={14} strokeWidth={1.75} />}
              onClick={() => setInvitationDialogOpen(true)}
              disabled={!selectedJobId || !profile?.id}
            >
              Send interview invitation
            </Button>
            <Link to={routes.outreach}>
              <Button variant="secondary" icon={<Mail size={14} strokeWidth={1.75} />}>
                Draft outreach
              </Button>
            </Link>
          </div>
        </header>

        <div className="border-b border-[color:var(--hairline)]">
          <TabButton active={activeTab === "overview"} onClick={() => setActiveTab("overview")}>
            Overview
          </TabButton>
          <TabButton active={activeTab === "resume"} onClick={() => setActiveTab("resume")}>
            Resume PDF
          </TabButton>
          <TabButton active={activeTab === "outreach"} onClick={() => setActiveTab("outreach")}>
            Outreach
          </TabButton>
          <TabButton active={activeTab === "interview"} onClick={() => setActiveTab("interview")}>
            Interviews
          </TabButton>
        </div>

        <div>
          {activeTab === "overview" && <OverviewTab resume={resume} profile={profile} />}
          {activeTab === "resume" && <ResumeTab resume={resume} />}
          {activeTab === "outreach" && <OutreachTab items={outreachItems} />}
          {activeTab === "interview" && <InterviewsTab items={invitations} />}
        </div>
      </div>

      <InvitationSendDialog
        open={invitationDialogOpen}
        onOpenChange={setInvitationDialogOpen}
        jobId={selectedJobId}
        candidateProfileId={profile?.id ?? null}
        candidateName={candidateName}
        onSent={() => setActiveTab("interview")}
      />
    </div>
  );
}
