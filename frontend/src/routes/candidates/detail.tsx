import type {
  CandidateProfileResponse,
  InterviewInvitationResponse,
  OutreachResponse,
  ResumeResponse,
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
import { useMemo, useState, type ReactNode } from "react";
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
  if (!value) return "—";
  return new Date(value).toLocaleDateString();
}

function statusVariant(status: string): "neutral" | "warning" | "success" | "danger" {
  if (status === "processed" || status === "completed" || status === "sent") return "success";
  if (status === "processing" || status === "pending") return "warning";
  if (status === "failed" || status === "cancelled") return "danger";
  return "neutral";
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
  return (
    <div className="grid gap-8 lg:grid-cols-[2fr_1fr]">
      <div className="space-y-5">
        {profile?.summary_text ? (
          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
            <h2 className="font-display text-xl font-medium text-fg">Summary</h2>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">
              {profile.summary_text}
            </p>
          </section>
        ) : (
          <EmptyState
            heading="Profile summary unavailable"
            body={`This resume is ${resume.upload_status}. Candidate profile data appears after processing completes.`}
          />
        )}

        {profile?.skills_text && (
          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
            <h2 className="font-display text-xl font-medium text-fg">Skills</h2>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-fg">{profile.skills_text}</p>
          </section>
        )}
      </div>

      <aside className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
        <h2 className="font-display text-xl font-medium text-fg">Resume Info</h2>
        <div className="mt-4 space-y-3 text-sm">
          <div className="flex items-center justify-between gap-3">
            <span className="text-fg-muted">Status</span>
            <Badge variant={statusVariant(resume.upload_status)}>{resume.upload_status}</Badge>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-fg-muted">Uploaded</span>
            <span className="text-fg">{absoluteDate(resume.uploaded_at)}</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-fg-muted">Processed</span>
            <span className="text-fg">{absoluteDate(resume.processed_at)}</span>
          </div>
          {profile?.email && (
            <div className="flex items-center justify-between gap-3">
              <span className="text-fg-muted">Email</span>
              <span className="text-fg">{profile.email}</span>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

function ResumeTab({ resume }: { resume: ResumeResponse }) {
  const pdfUrl = `http://localhost:8000${resume.storage_uri}`;

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

  const candidateName = profile?.full_name || fileToName(resume.original_file_name);

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
