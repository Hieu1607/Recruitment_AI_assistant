import type {
    CandidateProfileResponse,
    OutreachResponse,
    QuestionSetResponse,
    ResumeResponse
} from "@/api";
import { api } from "@/api";
import { Avatar } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { Modal, ModalContent, ModalFooter, ModalHeader, ModalTitle } from "@/components/ui/modal";
import { Skeleton } from "@/components/ui/skeleton";
import { useUserId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    ArrowLeft,
    BarChart2,
    BookOpen,
    Calendar,
    CheckCircle2,
    ClipboardList,
    ExternalLink,
    Mail,
    Plus,
    Send
} from "lucide-react";
import { type ReactNode, useState } from "react";
import { Link, useParams } from "react-router";
import { toast } from "sonner";

// ── types ─────────────────────────────────────────────────────────────────────

type Tab = "overview" | "resume" | "scoring" | "outreach" | "interview";

// ── helpers ───────────────────────────────────────────────────────────────────

function fileToName(f: string) {
  return (
    f
      .replace(/\.pdf$/i, "")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .trim() || f
  );
}

function relativeTime(iso: string | null): string {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d}d ago`;
  return new Date(iso).toLocaleDateString();
}

function absoluteDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function statusVariant(status: string): "neutral" | "warning" | "success" | "danger" {
  if (status === "processed") return "success";
  if (status === "processing") return "warning";
  if (status === "failed") return "danger";
  return "neutral";
}

// ── sub-components ────────────────────────────────────────────────────────────

function TabBar({
  active,
  onChange,
  outreachCount,
  interviewCount,
}: {
  active: Tab;
  onChange: (t: Tab) => void;
  outreachCount: number;
  interviewCount: number;
}) {
  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: "overview", label: "Overview" },
    { id: "resume", label: "Resume PDF" },
    { id: "scoring", label: "Scoring History" },
    { id: "outreach", label: "Outreach", count: outreachCount },
    { id: "interview", label: "Interview Questions", count: interviewCount },
  ];
  return (
    <div className="flex items-center gap-0 border-b border-[color:var(--hairline)]">
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => onChange(t.id)}
          className={cn(
            "flex items-center gap-1.5 px-4 py-3 text-sm font-sans border-b-2 -mb-px transition-colors",
            active === t.id
              ? "border-accent text-fg font-medium"
              : "border-transparent text-fg-muted hover:text-fg"
          )}
        >
          {t.label}
          {t.count !== undefined && t.count > 0 && (
            <span
              className={cn(
                "inline-flex items-center justify-center h-4 min-w-4 px-1 rounded-full text-[10px] font-sans font-semibold tabular-nums",
                active === t.id
                  ? "bg-accent text-accent-fg"
                  : "bg-[color:var(--hairline)] text-fg-muted"
              )}
            >
              {t.count}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start gap-4 py-3 hairline-b last:border-0">
      <span className="w-36 shrink-0 text-xs font-sans text-fg-muted">{label}</span>
      <span className="text-sm font-sans text-fg">{value}</span>
    </div>
  );
}

function ProfileSection({ label, text }: { label: string; text: string | null | undefined }) {
  if (!text) return null;
  return (
    <div>
      <h3 className="font-display text-lg font-medium text-fg mb-3">{label}</h3>
      <div className="p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
        <p className="text-sm font-sans text-fg leading-relaxed whitespace-pre-wrap">{text}</p>
      </div>
    </div>
  );
}

function OverviewTab({
  resume,
  profile,
  profileLoading,
}: {
  resume: ResumeResponse;
  profile: CandidateProfileResponse | undefined;
  profileLoading: boolean;
}) {
  const skills = profile?.skills_text
    ? profile.skills_text.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean)
    : [];

  return (
    <div className="grid grid-cols-3 gap-8">
      {/* Left: profile data */}
      <div className="col-span-2 space-y-8">
        {profileLoading ? (
          <div className="space-y-4">
            {[0, 1, 2].map((i) => <Skeleton key={i} className="h-24 w-full" />)}
          </div>
        ) : !profile ? (
          <EmptyState
            heading="Profile not yet analysed"
            body={`This resume is ${resume.upload_status}. Profile data will appear once processing completes.`}
          />
        ) : (
          <>
            {/* Basic contact / identity info — always shown when profile exists */}
            {(profile.current_job_title || profile.email || profile.phone || profile.location_normalized || profile.experience_years != null) && (
              <div className="grid grid-cols-2 gap-3">
                {profile.current_job_title && (
                  <div className="col-span-2 p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
                    <p className="text-xs text-fg-muted mb-0.5">Current role</p>
                    <p className="text-sm font-medium text-fg">{profile.current_job_title}</p>
                  </div>
                )}
                {profile.email && (
                  <div className="p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
                    <p className="text-xs text-fg-muted mb-0.5">Email</p>
                    <p className="text-sm text-fg break-all">{profile.email}</p>
                  </div>
                )}
                {profile.phone && (
                  <div className="p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
                    <p className="text-xs text-fg-muted mb-0.5">Phone</p>
                    <p className="text-sm text-fg">{profile.phone}</p>
                  </div>
                )}
                {profile.location_normalized && (
                  <div className="p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
                    <p className="text-xs text-fg-muted mb-0.5">Location</p>
                    <p className="text-sm text-fg">{profile.location_normalized}</p>
                  </div>
                )}
                {profile.experience_years != null && (
                  <div className="p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
                    <p className="text-xs text-fg-muted mb-0.5">Experience</p>
                    <p className="text-sm text-fg">{profile.experience_years} years</p>
                  </div>
                )}
              </div>
            )}

            {/* Summary */}
            {profile.summary_text && (
              <div>
                <h3 className="font-display text-lg font-medium text-fg mb-3">Summary</h3>
                <div className="p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
                  <p className="text-sm font-sans text-fg leading-relaxed">{profile.summary_text}</p>
                </div>
              </div>
            )}

            {/* Skills */}
            {skills.length > 0 && (
              <div>
                <h3 className="font-display text-lg font-medium text-fg mb-3">Key Skills</h3>
                <div className="p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
                  <div className="flex flex-wrap gap-2">
                    {skills.map((skill) => (
                      <span
                        key={skill}
                        className="px-2.5 py-1 text-xs font-sans text-fg rounded-[var(--radius-sm)] border border-[color:var(--hairline)] bg-bg"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <ProfileSection label="Experience" text={profile.experience_text} />
            <ProfileSection label="Education" text={profile.education_text} />
            <ProfileSection label="Projects" text={profile.projects_text} />
            <ProfileSection label="Certifications" text={profile.certifications_text} />
            <ProfileSection label="Achievements" text={profile.achievements_text} />

            {/* Fallback when all text sections are empty */}
            {!profile.summary_text && skills.length === 0 && !profile.experience_text &&
              !profile.education_text && !profile.projects_text && (
              <div className="p-4 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated">
                <p className="text-sm font-sans text-fg-muted italic">
                  Detailed sections (summary, skills, experience, education) were not extracted for this
                  candidate. Re-upload the resume to trigger a fresh extraction.
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Right: metadata card */}
      <div className="space-y-6">
        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
          <h4 className="text-xs font-sans font-semibold uppercase tracking-widest text-fg-subtle mb-4">
            Resume Info
          </h4>
          <div className="divide-y divide-[color:var(--hairline)]">
            <InfoRow label="Status" value={
              <Badge variant={statusVariant(resume.upload_status)} size="sm">
                {resume.upload_status.charAt(0).toUpperCase() + resume.upload_status.slice(1)}
              </Badge>
            } />
            <InfoRow label="File name" value={
              <span className="font-mono text-xs break-all">{resume.original_file_name}</span>
            } />
            <InfoRow label="Uploaded" value={
              <span title={resume.uploaded_at ?? ""} className="tabular-nums">
                {absoluteDate(resume.uploaded_at)}
              </span>
            } />
            <InfoRow label="Processed" value={
              <span title={resume.processed_at ?? ""} className="tabular-nums">
                {absoluteDate(resume.processed_at)}
              </span>
            } />
            <InfoRow label="Expires" value={
              <span className="tabular-nums">{relativeTime(resume.retention_expires_at)}</span>
            } />
          </div>
        </div>

        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-5">
          <h4 className="text-xs font-sans font-semibold uppercase tracking-widest text-fg-subtle mb-3">
            Quick Actions
          </h4>
          <div className="space-y-2">
            <Link
              to={routes.scoring}
              className={cn(
                "flex items-center gap-2.5 px-3 py-2.5 rounded-[var(--radius-md)] w-full",
                "border border-[color:var(--hairline)] bg-bg hover:border-[color:var(--hairline-strong)]",
                "text-sm font-sans text-fg transition-all duration-[var(--duration-fast)] hover:shadow-[var(--shadow-sm)]"
              )}
            >
              <BarChart2 size={14} strokeWidth={1.75} className="text-fg-muted shrink-0" />
              Score against a JD
            </Link>
            <Link
              to={routes.outreach}
              className={cn(
                "flex items-center gap-2.5 px-3 py-2.5 rounded-[var(--radius-md)] w-full",
                "border border-[color:var(--hairline)] bg-bg hover:border-[color:var(--hairline-strong)]",
                "text-sm font-sans text-fg transition-all duration-[var(--duration-fast)] hover:shadow-[var(--shadow-sm)]"
              )}
            >
              <Mail size={14} strokeWidth={1.75} className="text-fg-muted shrink-0" />
              Draft outreach
            </Link>
            <Link
              to={routes.interviewQuestions}
              className={cn(
                "flex items-center gap-2.5 px-3 py-2.5 rounded-[var(--radius-md)] w-full",
                "border border-[color:var(--hairline)] bg-bg hover:border-[color:var(--hairline-strong)]",
                "text-sm font-sans text-fg transition-all duration-[var(--duration-fast)] hover:shadow-[var(--shadow-sm)]"
              )}
            >
              <BookOpen size={14} strokeWidth={1.75} className="text-fg-muted shrink-0" />
              Generate interview questions
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

function ResumePdfTab({ resume }: { resume: ResumeResponse }) {
  const pdfUrl = `http://localhost:8000${resume.storage_uri}`;
  return (
    <div className="flex gap-6 min-h-[560px]">
      {/* PDF pane */}
      <div className="flex-1 rounded-[var(--radius-lg)] border border-[color:var(--hairline)] overflow-hidden bg-bg-elevated flex flex-col">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-[color:var(--hairline)]">
          <span className="text-xs font-sans text-fg-muted truncate">{resume.original_file_name}</span>
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-accent hover:underline font-sans shrink-0 ml-3"
          >
            Open <ExternalLink size={11} strokeWidth={1.75} />
          </a>
        </div>
        <div className="flex-1 relative">
          <iframe
            src={pdfUrl}
            title="Resume PDF"
            className="w-full h-full min-h-[500px]"
          />
        </div>
      </div>

      {/* Parsed data pane */}
      <div className="w-72 shrink-0 space-y-4">
        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
          <h4 className="text-xs font-sans font-semibold uppercase tracking-widest text-fg-subtle mb-3">
            Document Details
          </h4>
          <div className="space-y-2 text-sm font-sans text-fg">
            <div className="flex justify-between">
              <span className="text-fg-muted text-xs">Status</span>
              <Badge variant={statusVariant(resume.upload_status)} size="sm">
                {resume.upload_status.charAt(0).toUpperCase() + resume.upload_status.slice(1)}
              </Badge>
            </div>
            <div className="flex justify-between items-start gap-2">
              <span className="text-fg-muted text-xs shrink-0">Storage path</span>
              <span className="font-mono text-[10px] text-fg-subtle text-right break-all">{resume.storage_uri}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-fg-muted text-xs">Uploaded</span>
              <span className="text-xs tabular-nums">{absoluteDate(resume.uploaded_at)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-fg-muted text-xs">Processed</span>
              <span className="text-xs tabular-nums">{absoluteDate(resume.processed_at)}</span>
            </div>
          </div>
        </div>
        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-4">
          <p className="text-xs text-fg-muted font-sans leading-relaxed">
            The PDF viewer attempts to load from the backend file server.
            If it appears blank, confirm the backend is running and serving static files at{" "}
            <span className="font-mono text-[10px] bg-[color:var(--hairline)] px-1 py-0.5 rounded">
              /app/pdfs/
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}

function ScoringTab({ candidateId }: { candidateId: string }) {
  return (
    <div className="max-w-xl">
      <EmptyState
        icon={<BarChart2 size={28} strokeWidth={1.25} />}
        heading="No scoring history"
        body="Run a scoring session to evaluate this candidate against a job description. Results will be linked here."
        action={{
          label: "Start scoring",
          onClick: () => {
            window.location.href = routes.scoring;
          },
        }}
      />
      <p className="text-xs text-fg-subtle font-sans text-center mt-3">
        Candidate ID: <span className="font-mono">{candidateId}</span>
      </p>
    </div>
  );
}

function OutreachTab({
  items,
  loading,
}: {
  items: OutreachResponse[];
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="space-y-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-20 rounded-[var(--radius-md)] overflow-hidden">
            <Skeleton className="h-full w-full" />
          </div>
        ))}
      </div>
    );
  }
  if (items.length === 0) {
    return (
      <EmptyState
        icon={<Mail size={28} strokeWidth={1.25} />}
        heading="No outreach messages"
        body="Draft and track outreach messages for this candidate from the Outreach screen."
        action={{
          label: "Go to Outreach",
          onClick: () => { window.location.href = routes.outreach; },
        }}
      />
    );
  }
  return (
    <div className="space-y-3 max-w-2xl">
      {items.map((msg) => (
        <div
          key={msg.id}
          className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-4"
        >
          <div className="flex items-start justify-between gap-3 mb-2">
            <p className="text-sm font-sans font-medium text-fg truncate">{msg.subject}</p>
            <Badge
              variant={
                msg.sent_status === "sent"
                  ? "success"
                  : msg.sent_status === "failed"
                  ? "danger"
                  : "neutral"
              }
              size="sm"
              className="shrink-0"
            >
              {msg.sent_status.replace("_", " ")}
            </Badge>
          </div>
          <p className="text-xs font-sans text-fg-muted line-clamp-2 leading-relaxed mb-3">
            {msg.body}
          </p>
          <div className="flex items-center gap-4 text-[11px] text-fg-subtle font-sans">
            <span className="flex items-center gap-1">
              <Send size={10} strokeWidth={2} />
              {msg.content_source === "ai_draft" ? "AI draft" : "Template"}
            </span>
            <span>{absoluteDate(msg.created_at)}</span>
            {msg.sent_at && <span className="text-success">Sent {relativeTime(msg.sent_at)}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

function InterviewTab({
  items,
  loading,
}: {
  items: QuestionSetResponse[];
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="space-y-3">
        {[0, 1].map((i) => (
          <div key={i} className="h-20 rounded-[var(--radius-md)] overflow-hidden">
            <Skeleton className="h-full w-full" />
          </div>
        ))}
      </div>
    );
  }
  if (items.length === 0) {
    return (
      <EmptyState
        icon={<BookOpen size={28} strokeWidth={1.25} />}
        heading="No interview questions"
        body="Generate a tailored question set for this candidate from the Interview Questions screen."
        action={{
          label: "Go to Interview Questions",
          onClick: () => { window.location.href = routes.interviewQuestions; },
        }}
      />
    );
  }
  return (
    <div className="space-y-3 max-w-2xl">
      {items.map((qs) => {
        const qCount = Array.isArray((qs.question_payload as any)?.questions)
          ? (qs.question_payload as any).questions.length
          : Object.keys(qs.question_payload).length;
        return (
          <Link
            key={qs.id}
            to={routes.interviewQuestionDetail(qs.id)}
            className={cn(
              "flex items-center gap-4 rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
              "bg-bg-elevated p-4 hover:border-[color:var(--hairline-strong)] hover:shadow-[var(--shadow-sm)]",
              "transition-all duration-[var(--duration-fast)]"
            )}
          >
            <div className="h-9 w-9 rounded-[var(--radius-md)] bg-accent/10 flex items-center justify-center shrink-0">
              <ClipboardList size={16} strokeWidth={1.75} className="text-accent" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-sans font-medium text-fg truncate">
                {qs.job_description_title ?? "Untitled position"}
              </p>
              <p className="text-xs text-fg-muted font-sans">
                {absoluteDate(qs.created_at)} · {qCount} questions
              </p>
            </div>
            <ExternalLink size={13} strokeWidth={1.75} className="text-fg-muted shrink-0" />
          </Link>
        );
      })}
    </div>
  );
}

// ── Add to shortlist modal ─────────────────────────────────────────────────────

function AddToShortlistModal({
  open,
  onClose,
  candidateId,
}: {
  open: boolean;
  onClose: () => void;
  candidateId: string;
}) {
  const qc = useQueryClient();
  const userId = useUserId();
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  const { data: collectionsData, isLoading } = useQuery({
    queryKey: ["collections-picker"],
    queryFn: () => api.shortlist.collections.list({ user_id: userId ?? "", limit: 50 }),
    enabled: open,
  });
  const collections = collectionsData?.items ?? [];

  const addItemMutation = useMutation({
    mutationFn: (collectionId: string) =>
      api.shortlist.items.add(collectionId, { candidate_profile_id: candidateId }),
    onSuccess: () => {
      toast.success("Added to shortlist");
      qc.invalidateQueries({ queryKey: ["collections-picker"] });
      onClose();
    },
    onError: () => toast.error("Failed to add to shortlist"),
  });

  const createAndAddMutation = useMutation({
    mutationFn: async () => {
      const collection = await api.shortlist.collections.create({
        created_by_user_id: userId ?? "",
        name: newName.trim(),
      });
      await api.shortlist.items.add(collection.id, { candidate_profile_id: candidateId });
      return collection;
    },
    onSuccess: () => {
      toast.success("Created collection and added candidate");
      qc.invalidateQueries({ queryKey: ["collections-picker"] });
      setNewName("");
      setCreating(false);
      onClose();
    },
    onError: () => toast.error("Failed to create collection"),
  });

  return (
    <Modal open={open} onOpenChange={(o) => !o && onClose()}>
      <ModalContent>
        <ModalHeader>
          <ModalTitle>Add to shortlist</ModalTitle>
        </ModalHeader>
        <div className="space-y-4 mt-1">
          {/* Existing collections */}
          <div>
            <p className="text-xs font-sans font-medium text-fg-muted mb-2">
              Choose a collection
            </p>
            {isLoading ? (
              <div className="space-y-2">
                {[0, 1].map((i) => <Skeleton key={i} className="h-10 w-full" />)}
              </div>
            ) : collections.length === 0 ? (
              <p className="text-sm text-fg-muted font-sans text-center py-3">
                No collections yet. Create one below.
              </p>
            ) : (
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {collections.map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    disabled={addItemMutation.isPending}
                    onClick={() => addItemMutation.mutate(c.id)}
                    className={cn(
                      "w-full flex items-center justify-between px-3 py-2.5 rounded-[var(--radius-md)]",
                      "border border-[color:var(--hairline)] bg-bg hover:border-[color:var(--hairline-strong)]",
                      "transition-all duration-[var(--duration-fast)] text-left"
                    )}
                  >
                    <div>
                      <p className="text-sm font-sans font-medium text-fg">{c.name}</p>
                      <p className="text-xs font-sans text-fg-muted">
                        {c.item_count} candidate{c.item_count !== 1 ? "s" : ""}
                      </p>
                    </div>
                    <CheckCircle2 size={15} strokeWidth={1.75} className="text-fg-muted shrink-0" />
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Create new */}
          <div className="border-t border-[color:var(--hairline)] pt-4">
            {creating ? (
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Collection name…"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && newName.trim()) createAndAddMutation.mutate();
                    if (e.key === "Escape") { setCreating(false); setNewName(""); }
                  }}
                  autoFocus
                  className={cn(
                    "flex-1 h-9 px-3 text-sm font-sans rounded-[var(--radius-md)]",
                    "border border-[color:var(--hairline-strong)] bg-bg text-fg",
                    "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent outline-none"
                  )}
                />
                <Button
                  variant="primary"
                  size="sm"
                  disabled={!newName.trim()}
                  loading={createAndAddMutation.isPending}
                  onClick={() => createAndAddMutation.mutate()}
                >
                  Create
                </Button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setCreating(true)}
                className="flex items-center gap-2 text-sm font-sans text-fg-muted hover:text-fg transition-colors"
              >
                <Plus size={14} strokeWidth={2} />
                Create new collection
              </button>
            )}
          </div>
        </div>
        <ModalFooter>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export default function CandidateDetailRoute() {
  const { id } = useParams<{ id: string }>();
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [shortlistOpen, setShortlistOpen] = useState(false);

  const { data: resume, isLoading: resumeLoading, error: resumeError } = useQuery({
    queryKey: ["candidate", id],
    queryFn: () => api.upload.get(id!),
    enabled: !!id,
  });

  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ["candidate-profile", id],
    queryFn: () => api.upload.getProfile(id!),
    enabled: !!id && resume?.upload_status === "processed",
    retry: false,
  });

  const profileId = profile?.id;

  const { data: outreachData, isLoading: outreachLoading } = useQuery({
    queryKey: ["outreach-candidate", profileId],
    queryFn: () => api.outreach.list({ candidate_profile_id: profileId, limit: 50 }),
    enabled: !!profileId,
  });

  const { data: interviewData, isLoading: interviewLoading } = useQuery({
    queryKey: ["interviews-candidate", profileId],
    queryFn: () => api.interviewQuestions.list({ candidate_profile_id: profileId, limit: 50 }),
    enabled: !!profileId,
  });

  const outreachItems = outreachData?.items ?? [];
  const interviewItems = interviewData?.items ?? [];

  if (resumeLoading) {
    return (
      <div className="px-8 py-8">
        <Skeleton className="h-8 w-48 mb-2" />
        <Skeleton className="h-5 w-64 mb-8" />
        <div className="flex gap-4 mb-8">
          {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-9 w-32" />)}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (resumeError || !resume) {
    return (
      <div className="px-8 py-8">
        <EmptyState
          heading="Candidate not found"
          body="This resume document does not exist or has been deleted."
          action={{ label: "Back to candidates", onClick: () => history.back() }}
        />
      </div>
    );
  }

  const name = profile?.full_name || fileToName(resume.original_file_name);

  return (
    <div className="px-8 py-8 min-h-full">

      {/* Back nav */}
      <Link
        to={routes.candidates}
        className="inline-flex items-center gap-1.5 text-sm font-sans text-fg-muted hover:text-fg transition-colors mb-6"
      >
        <ArrowLeft size={14} strokeWidth={2} />
        Candidates
      </Link>

      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-6 mb-8">
        <div className="flex items-start gap-5">
          <Avatar name={name} size="xl" />
          <div>
            <h1 className="font-display text-[2.5rem] font-medium text-fg leading-tight">
              {name}
            </h1>
            <div className="flex items-center gap-3 mt-1.5">
              <Badge variant={statusVariant(resume.upload_status)} size="sm">
                {resume.upload_status.charAt(0).toUpperCase() + resume.upload_status.slice(1)}
              </Badge>
              <span className="text-sm text-fg-muted font-sans flex items-center gap-1.5">
                <Calendar size={12} strokeWidth={1.75} />
                Uploaded {relativeTime(resume.uploaded_at)}
              </span>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2 shrink-0">
          <Button
            variant="secondary"
            size="sm"
            icon={<BarChart2 size={14} strokeWidth={1.75} />}
            onClick={() => { window.location.href = routes.scoring; }}
          >
            Score against JD
          </Button>
          <Button
            variant="secondary"
            size="sm"
            icon={<BookOpen size={14} strokeWidth={1.75} />}
            onClick={() => { window.location.href = routes.interviewQuestions; }}
          >
            Interview questions
          </Button>
          <Button
            variant="secondary"
            size="sm"
            icon={<Mail size={14} strokeWidth={1.75} />}
            onClick={() => { window.location.href = routes.outreach; }}
          >
            Draft outreach
          </Button>
          <Button
            variant="primary"
            size="sm"
            icon={<Plus size={14} strokeWidth={2} />}
            disabled={!profileId}
            onClick={() => setShortlistOpen(true)}
          >
            Add to shortlist
          </Button>
        </div>
      </div>

      {/* ── Tabs ── */}
      <TabBar
        active={activeTab}
        onChange={setActiveTab}
        outreachCount={outreachItems.length}
        interviewCount={interviewItems.length}
      />

      {/* ── Tab content ── */}
      <div className="mt-8">
        {activeTab === "overview" && (
          <OverviewTab resume={resume} profile={profile} profileLoading={profileLoading} />
        )}
        {activeTab === "resume" && <ResumePdfTab resume={resume} />}
        {activeTab === "scoring" && <ScoringTab candidateId={id!} />}
        {activeTab === "outreach" && (
          <OutreachTab items={outreachItems} loading={outreachLoading} />
        )}
        {activeTab === "interview" && (
          <InterviewTab items={interviewItems} loading={interviewLoading} />
        )}
      </div>

      {/* ── Add to shortlist modal ── */}
      <AddToShortlistModal
        open={shortlistOpen}
        onClose={() => setShortlistOpen(false)}
        candidateId={profileId ?? ""}
      />
    </div>
  );
}
