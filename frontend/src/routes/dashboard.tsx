import { api, type ActivityKind, type ActivityResponse, type CollectionResponse, type ResumeResponse } from "@/api";
import { UploadModal } from "@/components/candidates/UploadModal";
import { Button } from "@/components/ui";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuthStore, useSelectedJobId } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { isVietnameseUi } from "@/lib/ui-language";
import { useQuery } from "@tanstack/react-query";
import {
    ArrowRight,
    CircleAlert,
    BarChart3,
    CheckCircle2,
    Circle,
    FileWarning,
    FileText,
    FileUp,
    Layers,
    Mail,
    MessageSquare,
    Sparkles,
    TrendingDown,
    TrendingUp,
    Upload,
    Users,
    XCircle,
} from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router";

// ── helpers ────────────────────────────────────────────────────────────────────

const viUi = isVietnameseUi();

function timeGreeting(): string {
  const h = new Date().getHours();
  if (viUi) {
    if (h < 12) return "Chào buổi sáng";
    if (h < 17) return "Chào buổi chiều";
    return "Chào buổi tối";
  }
  if (h < 12) return "Good morning";
  if (h < 17) return "Good afternoon";
  return "Good evening";
}

function todayLabel(): string {
  return new Date().toLocaleDateString(viUi ? "vi-VN" : undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

function relativeTime(iso: string | null): string {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return viUi ? "vừa xong" : "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return viUi ? `${m} phút trước` : `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return viUi ? `${h} giờ trước` : `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 30) return viUi ? `${d} ngày trước` : `${d}d ago`;
  return new Date(iso).toLocaleDateString(viUi ? "vi-VN" : undefined);
}

function dayKey(iso: string): string {
  return iso.slice(0, 10);
}

function buildSparkline(dates: (string | null)[], days = 7): number[] {
  const counts: Record<string, number> = {};
  const today = new Date();
  const keys: string[] = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const k = d.toISOString().slice(0, 10);
    keys.push(k);
    counts[k] = 0;
  }
  for (const iso of dates) {
    if (!iso) continue;
    const k = dayKey(iso);
    if (k in counts) counts[k]++;
  }
  return keys.map((k) => counts[k]);
}

function pctChange(current: number, previous: number): number | null {
  if (previous === 0) return current > 0 ? 100 : null;
  return Math.round(((current - previous) / previous) * 100);
}

// ── Sparkline SVG ─────────────────────────────────────────────────────────────

function Sparkline({ values, color = "var(--accent)" }: { values: number[]; color?: string }) {
  if (values.length < 2) return null;
  const max = Math.max(...values, 1);
  const W = 64;
  const H = 24;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * W;
    const y = H - (v / max) * (H - 2) - 1;
    return `${x},${y}`;
  });
  const d = `M${pts.join(" L")}`;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} fill="none" aria-hidden="true">
      <path d={d} stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── MetricCard ─────────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  change,
  sparkValues,
  icon: Icon,
  loading,
}: {
  label: string;
  value: number;
  change: number | null;
  sparkValues: number[];
  icon: React.ElementType;
  loading: boolean;
}) {
  const positive = change !== null && change >= 0;
  return (
    <div
      className={cn(
        "rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
        "bg-bg-elevated p-5 flex flex-col gap-3",
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-sans font-medium text-fg-muted uppercase tracking-widest">
          {label}
        </span>
        <span className="h-8 w-8 rounded-[var(--radius-md)] bg-[color:var(--hairline)] flex items-center justify-center">
          <Icon size={15} strokeWidth={1.75} className="text-fg-muted" />
        </span>
      </div>
      {loading ? (
        <>
          <Skeleton className="h-8 w-24" />
          <Skeleton className="h-4 w-16" />
        </>
      ) : (
        <>
          <p className="font-display text-3xl font-medium text-fg tabular-nums leading-none">
            {value.toLocaleString()}
          </p>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1">
              {change !== null ? (
                <>
                  {positive ? (
                    <TrendingUp size={12} strokeWidth={2} className="text-success" />
                  ) : (
                    <TrendingDown size={12} strokeWidth={2} className="text-danger" />
                  )}
                  <span
                    className={cn(
                      "text-xs font-sans font-medium tabular-nums",
                      positive ? "text-success" : "text-danger",
                    )}
                  >
                    {positive ? "+" : ""}
                    {change}% vs last week
                  </span>
                </>
              ) : (
                <span className="text-xs font-sans text-fg-subtle">No prior data</span>
              )}
            </div>
            <Sparkline values={sparkValues} />
          </div>
        </>
      )}
    </div>
  );
}

// ── ActivityItem ──────────────────────────────────────────────────────────────

interface ActivityEntry {
  id: string;
  kind: ActivityKind;
  label: string;
  sub?: string;
  timestamp: string | null;
  targetUrl?: string | null;
}

const ACTIVITY_ICONS: Record<ActivityKind, React.ElementType> = {
  resume_uploaded: Upload,
  resume_processed: CheckCircle2,
  resume_failed: FileWarning,
  shortlist_added: Layers,
  outreach_sent: Mail,
  outreach_failed: XCircle,
  interview_link_created: FileText,
  interview_invitation_sent: Mail,
  interview_completed: Sparkles,
  interview_cancelled: CircleAlert,
  scoring_completed: BarChart3,
};

const ACTIVITY_COLORS: Record<ActivityKind, string> = {
  resume_uploaded: "text-accent",
  resume_processed: "text-success",
  resume_failed: "text-danger",
  shortlist_added: "text-fg-muted",
  outreach_sent: "text-warning",
  outreach_failed: "text-danger",
  interview_link_created: "text-accent",
  interview_invitation_sent: "text-warning",
  interview_completed: "text-success",
  interview_cancelled: "text-danger",
  scoring_completed: "text-fg-muted",
};

function ActivityItem({ entry }: { entry: ActivityEntry }) {
  const Icon = ACTIVITY_ICONS[entry.kind];
  const content = (
    <div className="flex items-start gap-3 py-3 hairline-b last:border-b-0">
      <div
        className={cn(
          "h-7 w-7 shrink-0 rounded-full bg-[color:var(--hairline)] flex items-center justify-center mt-0.5",
        )}
      >
        <Icon size={13} strokeWidth={1.75} className={ACTIVITY_COLORS[entry.kind]} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-sans text-fg leading-snug truncate">{entry.label}</p>
        {entry.sub && (
          <p className="text-xs font-sans text-fg-subtle mt-0.5 truncate">{entry.sub}</p>
        )}
      </div>
      <span
        className="text-xs font-sans text-fg-subtle whitespace-nowrap ml-2 mt-0.5 tabular-nums"
        title={entry.timestamp ? new Date(entry.timestamp).toUTCString() : ""}
      >
        {relativeTime(entry.timestamp)}
      </span>
    </div>
  );
  if (entry.targetUrl) {
    return (
      <Link to={entry.targetUrl} className="block hover:bg-[color:var(--hairline)] transition-colors rounded-[var(--radius-sm)]">
        {content}
      </Link>
    );
  }
  return content;
}

function formatActivityEntry(item: ActivityResponse): ActivityEntry {
  const subject = item.subject_name ?? (viUi ? "mục chưa đặt tên" : "unnamed item");
  const context = item.context_name;
  const message = typeof item.metadata?.message === "string" ? item.metadata.message : null;

  switch (item.kind) {
    case "resume_uploaded":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Đã tải CV lên: ${subject}` : `CV uploaded: ${subject}`,
        sub: viUi ? "Đang chờ phân tích hồ sơ" : "Waiting for resume parsing",
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "resume_processed":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Đã xử lý CV: ${subject}` : `CV processed: ${subject}`,
        sub: context ?? (viUi ? "Hồ sơ ứng viên đã sẵn sàng" : "Candidate profile is ready"),
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "resume_failed":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Xử lý CV thất bại: ${subject}` : `CV processing failed: ${subject}`,
        sub: message ?? context ?? (viUi ? "Cần kiểm tra lại file hoặc thử lại" : "Review the file and retry"),
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "shortlist_added":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Đã thêm ${subject} vào shortlist` : `Added ${subject} to shortlist`,
        sub: context ?? undefined,
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "outreach_sent":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Đã gửi email cho ${subject}` : `Sent email to ${subject}`,
        sub: context ?? undefined,
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "outreach_failed":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Gửi email cho ${subject} thất bại` : `Email send failed for ${subject}`,
        sub: context ?? undefined,
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "interview_link_created":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Đã tạo link phỏng vấn cho ${subject}` : `Created interview link for ${subject}`,
        sub: context ?? undefined,
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "interview_invitation_sent":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Đã gửi lời mời phỏng vấn cho ${subject}` : `Sent interview invitation to ${subject}`,
        sub: context ?? undefined,
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "interview_completed":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `${subject} đã hoàn thành phỏng vấn` : `${subject} completed the interview`,
        sub: context ?? undefined,
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "interview_cancelled":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Đã hủy lời mời phỏng vấn của ${subject}` : `Interview invitation cancelled for ${subject}`,
        sub: context ?? undefined,
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
    case "scoring_completed":
      return {
        id: item.id,
        kind: item.kind,
        label: viUi ? `Đã hoàn tất chấm điểm cho ${subject}` : `Scoring completed for ${subject}`,
        sub: context ?? undefined,
        timestamp: item.timestamp,
        targetUrl: item.target_url,
      };
  }
}

// ── QuickActionButton ─────────────────────────────────────────────────────────

function QuickActionButton({
  icon: Icon,
  label,
  description,
  onClick,
}: {
  icon: React.ElementType;
  label: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 px-4 py-3 rounded-[var(--radius-md)] text-left",
        "border border-[color:var(--hairline)] bg-bg",
        "hover:border-[color:var(--hairline-strong)] hover:shadow-[var(--shadow-sm)] hover:bg-bg-elevated",
        "transition-all duration-[var(--duration-fast)] group",
      )}
    >
      <div className="h-8 w-8 shrink-0 rounded-[var(--radius-md)] bg-accent/10 flex items-center justify-center">
        <Icon size={15} strokeWidth={1.75} className="text-accent" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-sans font-medium text-fg leading-none mb-0.5">{label}</p>
        <p className="text-xs font-sans text-fg-subtle">{description}</p>
      </div>
      <ArrowRight
        size={14}
        strokeWidth={1.75}
        className="text-fg-subtle opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
      />
    </button>
  );
}

// ── CollectionCard ────────────────────────────────────────────────────────────

function CollectionCard({ col }: { col: CollectionResponse }) {
  return (
    <Link
      to="/shortlists"
      className={cn(
        "flex items-center justify-between px-4 py-3 rounded-[var(--radius-md)]",
        "border border-[color:var(--hairline)] bg-bg",
        "hover:border-[color:var(--hairline-strong)] hover:shadow-[var(--shadow-sm)]",
        "transition-all duration-[var(--duration-fast)]",
      )}
    >
      <div className="flex items-center gap-2.5 min-w-0">
        <Layers size={13} strokeWidth={1.75} className="text-fg-muted shrink-0" />
        <span className="text-sm font-sans text-fg truncate">{col.name}</span>
      </div>
      <span
        className={cn(
          "shrink-0 ml-3 text-xs font-sans font-medium tabular-nums px-2 py-0.5",
          "rounded-full bg-[color:var(--hairline)] text-fg-muted",
        )}
      >
        {col.item_count}
      </span>
    </Link>
  );
}

// ── EditorialInsight ──────────────────────────────────────────────────────────

function EditorialInsight({
  totalCandidates,
  processedToday,
  activeJDs,
  pendingOutreach,
}: {
  totalCandidates: number;
  processedToday: number;
  activeJDs: number;
  pendingOutreach: number;
}) {
  let headline = viUi
    ? "Danh sách ứng viên của bạn đang tăng trưởng ổn định."
    : "Your talent pool is growing steadily.";
  let body = viUi
    ? "Hãy tiếp tục tải hồ sơ lên và chấm điểm ứng viên để làm nổi bật các hồ sơ phù hợp nhất."
    : "Keep uploading resumes and scoring candidates to surface top matches.";

  if (totalCandidates === 0) {
    headline = viUi ? "Hãy bắt đầu bằng việc tải những hồ sơ đầu tiên lên." : "Start by uploading your first resumes.";
    body = viUi ? "Phân tích PDF để tạo hồ sơ ứng viên và mở khóa tính năng chấm điểm AI." : "Parse PDFs to build candidate profiles and unlock AI scoring.";
  } else if (pendingOutreach > 5) {
    headline = viUi
      ? `Có ${pendingOutreach} tin nhắn liên hệ đang chờ gửi.`
      : `${pendingOutreach} outreach messages are waiting to be sent.`;
    body = viUi
      ? "Hãy rà soát và gửi các bản nháp để giữ ứng viên luôn tương tác."
      : "Review and send your drafted messages to keep candidates engaged.";
  } else if (processedToday > 0) {
    headline = viUi
      ? `Đã xử lý ${processedToday} hồ sơ hôm nay.`
      : `${processedToday} resume${processedToday > 1 ? "s" : ""} processed today.`;
    body =
      activeJDs > 0
        ? (viUi
            ? `Hãy chấm các hồ sơ này với ${activeJDs} mô tả công việc đang hoạt động để xếp hạng ứng viên.`
            : `Score them against your ${activeJDs} active job description${activeJDs > 1 ? "s" : ""} to rank candidates.`)
        : (viUi
            ? "Hãy tạo một mô tả công việc để bắt đầu chấm điểm các ứng viên mới."
            : "Create a job description to start scoring your new candidates.");
  } else if (activeJDs === 0 && totalCandidates > 0) {
    headline = viUi ? "Chưa có mô tả công việc nào đang hoạt động." : "No active job descriptions yet.";
    body = viUi
      ? "Hãy tạo JD để bắt đầu quy trình chấm điểm AI và làm nổi bật các ứng viên phù hợp nhất."
      : "Create a JD to start the AI scoring workflow and surface your best-fit candidates.";
  }

  return (
    <div
      className={cn(
        "rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
        "bg-bg-elevated p-4",
      )}
    >
      <div className="flex items-center gap-2 mb-2.5">
        <Sparkles size={13} strokeWidth={1.75} className="text-accent shrink-0" />
        <span className="text-[10px] font-sans font-semibold uppercase tracking-widest text-fg-subtle">
          AI Insight
        </span>
      </div>
      <p className="font-display text-sm font-medium text-fg leading-snug mb-1.5">{headline}</p>
      <p className="text-xs font-sans text-fg-muted leading-relaxed">{body}</p>
    </div>
  );
}

// ── OnboardingChecklist ───────────────────────────────────────────────────────

const ONBOARDING_STEPS = [
  {
    id: "upload",
    label: "Upload your first resumes",
    sub: "Parse PDF CVs into structured candidate profiles",
    href: "/candidates",
  },
  {
    id: "jd",
    label: "Add the full job description",
    sub: "Capture responsibilities, requirements, and hiring notes",
    href: "/job-descriptions",
  },
  {
    id: "chat",
    label: "Ask the AI Recruiter",
    sub: "Interrogate your candidate pool after data starts flowing in",
    href: "/chat",
  },
];

function FirstRunDashboardState({
  jobTitle,
  onUpload,
  onAddJobDescription,
}: {
  jobTitle: string;
  onUpload: () => void;
  onAddJobDescription: () => void;
}) {
  return (
    <div className="mx-auto max-w-5xl">
      <div
        className={cn(
          "rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6 sm:p-8",
        )}
      >
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1.3fr)_minmax(280px,1fr)] lg:items-start">
          <div>
            <p className="text-xs font-sans font-semibold uppercase tracking-[0.22em] text-fg-subtle">
              Workspace Ready
            </p>
            <h2 className="mt-3 font-display text-3xl leading-tight text-fg sm:text-4xl">
              {jobTitle ? `"${jobTitle}" is ready.` : "Your workspace is ready."}
            </h2>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-fg-muted sm:text-base">
              The job has been created. Next, upload resumes or add the full job description.
              AI chat becomes much more useful once one of those data sources exists.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Button
                icon={<FileUp size={16} strokeWidth={1.75} />}
                onClick={onUpload}
              >
                Upload resumes
              </Button>
              <Button
                variant="secondary"
                icon={<FileText size={16} strokeWidth={1.75} />}
                onClick={onAddJobDescription}
              >
                Add job description
              </Button>
            </div>
          </div>

          <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-5">
            <p className="text-xs font-sans font-semibold uppercase tracking-[0.22em] text-fg-subtle">
              Recommended Flow
            </p>
            <div className="mt-4 space-y-3">
              {[
                "Upload resumes to build candidate profiles.",
                "Add the full job description to define fit.",
                "Use AI chat to explore the candidate pool once data starts flowing in.",
              ].map((item) => (
                <div key={item} className="flex items-start gap-3">
                  <CheckCircle2
                    size={16}
                    strokeWidth={1.75}
                    className="mt-0.5 shrink-0 text-accent"
                  />
                  <p className="text-sm leading-6 text-fg-muted">{item}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8">
        <OnboardingChecklist
          hasCandidates={false}
          hasJDs={false}
          hasChatted={false}
        />
      </div>
    </div>
  );
}

function OnboardingChecklist({
  hasCandidates,
  hasJDs,
  hasChatted,
}: {
  hasCandidates: boolean;
  hasJDs: boolean;
  hasChatted: boolean;
}) {
  const statuses = {
    upload: hasCandidates,
    jd: hasJDs,
    chat: hasChatted,
  };
  const done = Object.values(statuses).filter(Boolean).length;

  return (
    <div className="max-w-xl mx-auto">
      <div className="mb-6">
        <h2 className="font-display text-xl font-medium text-fg mb-1">Get started</h2>
        <p className="text-sm font-sans text-fg-muted">
          Complete these steps to set up your recruitment workflow.
        </p>
        <div className="mt-3 h-1.5 rounded-full bg-[color:var(--hairline)] overflow-hidden">
          <div
            className="h-full bg-accent rounded-full transition-all duration-500"
            style={{ width: `${(done / ONBOARDING_STEPS.length) * 100}%` }}
          />
        </div>
        <p className="text-xs font-sans text-fg-subtle mt-1.5 tabular-nums">
          {viUi ? `Hoàn thành ${done}/${ONBOARDING_STEPS.length} bước` : `${done} of ${ONBOARDING_STEPS.length} complete`}
        </p>
      </div>
      <div className="space-y-2">
        {ONBOARDING_STEPS.map((step) => {
          const completed = statuses[step.id as keyof typeof statuses];
          return (
            <Link
              key={step.id}
              to={step.href}
              className={cn(
                "flex items-center gap-4 px-5 py-4 rounded-[var(--radius-lg)]",
                "border border-[color:var(--hairline)] transition-all duration-[var(--duration-fast)]",
                completed
                  ? "bg-success/5 border-success/20"
                  : "bg-bg-elevated hover:border-[color:var(--hairline-strong)] hover:shadow-[var(--shadow-sm)]",
              )}
            >
              {completed ? (
                <CheckCircle2 size={20} strokeWidth={1.75} className="text-success shrink-0" />
              ) : (
                <Circle size={20} strokeWidth={1.75} className="text-fg-subtle shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <p
                  className={cn(
                    "text-sm font-sans font-medium",
                    completed ? "text-fg-muted line-through" : "text-fg",
                  )}
                >
                  {step.label}
                </p>
                <p className="text-xs font-sans text-fg-subtle mt-0.5">{step.sub}</p>
              </div>
              {!completed && (
                <ArrowRight size={14} strokeWidth={1.75} className="text-fg-subtle shrink-0" />
              )}
            </Link>
          );
        })}
      </div>
    </div>
  );
}

// ── main component ─────────────────────────────────────────────────────────────

export default function DashboardRoute() {
  const navigate = useNavigate();
  const [uploadOpen, setUploadOpen] = useState(false);
  const selectedJobId = useSelectedJobId();
  const user = useAuthStore((s) => s.user);
  const firstName = user?.display_name?.split(" ")[0] ?? (viUi ? "bạn" : "there");

  // ── data queries ──────────────────────────────────────────────────────────

  const { data: uploadsData, isLoading: uploadsLoading } = useQuery({
    queryKey: ["dashboard-uploads", selectedJobId],
    queryFn: () => (selectedJobId ? api.jobs.resumes.list(selectedJobId, { limit: 500 }) : Promise.resolve({ items: [], total: 0 })),
    enabled: !!selectedJobId,
    staleTime: 60_000,
  });

  const { data: currentJob } = useQuery({
    queryKey: ["dashboard-job", selectedJobId],
    queryFn: () => (selectedJobId ? api.jobs.get(selectedJobId) : Promise.resolve(null)),
    staleTime: 60_000,
  });

  const { data: jdsData, isLoading: jdsLoading } = useQuery({
    queryKey: ["dashboard-jds", selectedJobId],
    queryFn: async () => {
      if (!selectedJobId) return { items: [], total: 0 };
      try {
        const jd = await api.jobs.jobDescription.get(selectedJobId);
        return { items: [jd], total: 1 };
      } catch {
        return { items: [], total: 0 };
      }
    },
    enabled: !!selectedJobId,
    staleTime: 60_000,
  });

  const { data: pendingOutreachData, isLoading: outreachLoading } = useQuery({
    queryKey: ["dashboard-outreach-pending"],
    queryFn: () => api.outreach.list({ sent_status: "not_sent", limit: 1 }),
    staleTime: 60_000,
  });

  const { data: activityData, isLoading: activityLoading } = useQuery({
    queryKey: ["dashboard-activities", selectedJobId],
    queryFn: () =>
      api.activities
        .list({ job_id: selectedJobId ?? undefined, limit: 12 })
        .catch(() => ({ items: [] as ActivityResponse[] })),
    staleTime: 30_000,
  });

  const { data: collectionsData } = useQuery({
    queryKey: ["dashboard-collections"],
    queryFn: () =>
      api.shortlist.collections.list({ limit: 4 }).catch(() => ({
        items: [] as CollectionResponse[],
        total: 0,
      })),
    staleTime: 60_000,
  });

  const { data: setupStatusData } = useQuery({
    queryKey: ["jobs", selectedJobId, "setup-status"],
    queryFn: () => api.jobs.setupStatus.get(selectedJobId!),
    enabled: !!selectedJobId,
    staleTime: 30_000,
  });

  // ── derived values ────────────────────────────────────────────────────────

  const allUploads: ResumeResponse[] = uploadsData?.items ?? [];
  const totalCandidates = setupStatusData?.resume_count ?? uploadsData?.total ?? allUploads.length;

  const todayStr = new Date().toISOString().slice(0, 10);
  const processedToday = allUploads.filter(
    (r) => r.uploaded_at && dayKey(r.uploaded_at) === todayStr,
  ).length;

  const activeJDs = setupStatusData?.has_active_job_description
    ? Math.max((jdsData?.items ?? []).filter((j) => j.is_active).length, 1)
    : (jdsData?.items ?? []).filter((j) => j.is_active).length;
  const pendingOutreach = pendingOutreachData?.total ?? 0;

  const isMetricLoading = uploadsLoading || jdsLoading || outreachLoading;

  // ── sparklines ────────────────────────────────────────────────────────────

  const uploadDates = allUploads.map((r) => r.uploaded_at);
  const candidateSparkline = buildSparkline(uploadDates, 7);

  const jdDates = (jdsData?.items ?? []).filter((j) => j.is_active).map((j) => j.created_at);
  const jdSparkline = buildSparkline(jdDates, 7);

  const outreachDates = (activityData?.items ?? [])
    .filter((item) => item.kind === "outreach_sent" || item.kind === "outreach_failed")
    .map((item) => item.timestamp);
  const outreachSparkline = buildSparkline(outreachDates, 7);

  const todayUploads = buildSparkline(uploadDates, 7)[6] ?? 0;
  const prevUploads = buildSparkline(uploadDates, 14).slice(0, 7).reduce((a, b) => a + b, 0);
  const thisWeekUploads = buildSparkline(uploadDates, 7).reduce((a, b) => a + b, 0);

  // ── % change for metric cards ─────────────────────────────────────────────

  const candidateChange = pctChange(thisWeekUploads, prevUploads);
  const todayChange = (() => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().slice(0, 10);
    const yesterdayCount = allUploads.filter(
      (r) => r.uploaded_at && dayKey(r.uploaded_at) === yesterdayStr,
    ).length;
    return pctChange(processedToday, yesterdayCount);
  })();
  const jdChange = pctChange(
    (jdsData?.items ?? []).filter((j) => j.is_active).length,
    Math.max((jdsData?.items ?? []).length - activeJDs, 0),
  );
  const outreachChange = pctChange(pendingOutreach, 0);

  // ── activity feed ─────────────────────────────────────────────────────────

  const recentActivity = (activityData?.items ?? []).map(formatActivityEntry);

  // ── onboarding tracking ─────────────────────────────────────────────────

  const hasCandidates = setupStatusData?.has_uploaded_resumes ?? totalCandidates > 0;
  const hasJDs = setupStatusData?.has_active_job_description ?? (jdsData?.items ?? []).length > 0;
  const hasChatted = setupStatusData?.has_chat_turn ?? false;
  const onboardingDone = [hasCandidates, hasJDs, hasChatted].filter(Boolean).length;
  const onboardingComplete = onboardingDone === ONBOARDING_STEPS.length;

  // ── empty state detection ─────────────────────────────────────────────────

  const isEmpty =
    !isMetricLoading && !hasCandidates && !hasJDs;

  const collections = collectionsData?.items ?? [];

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="px-8 py-8 min-h-full">
      {/* ── Greeting ── */}
      <div className="mb-8">
        <h1 className="font-display text-[2.25rem] font-medium text-fg leading-tight">
          {timeGreeting()}, {firstName}.
        </h1>
        <p className="text-sm font-sans text-fg-muted mt-1">{todayLabel()}</p>
      </div>

      {isEmpty ? (
        /* ── Onboarding empty state ── */
        <FirstRunDashboardState
          jobTitle={currentJob?.title ?? ""}
          onUpload={() => setUploadOpen(true)}
          onAddJobDescription={() => navigate(routes.jobDescriptions)}
        />
      ) : (
        <>
          {/* ── Onboarding checklist (shown until complete) ── */}
          {!onboardingComplete && (
            <div className="mb-8">
              <OnboardingChecklist
                hasCandidates={hasCandidates}
                hasJDs={hasJDs}
                hasChatted={hasChatted}
              />
            </div>
          )}

          {/* ── Metric cards ── */}
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-8">
            <MetricCard
              label="Total Candidates"
              value={totalCandidates}
              change={candidateChange}
              sparkValues={candidateSparkline}
              icon={Users}
              loading={uploadsLoading}
            />
            <MetricCard
              label="Processed Today"
              value={processedToday}
              change={todayChange}
              sparkValues={candidateSparkline.map((v, i) =>
                i === candidateSparkline.length - 1 ? todayUploads : v,
              )}
              icon={FileUp}
              loading={uploadsLoading}
            />
            <MetricCard
              label="Active JDs"
              value={activeJDs}
              change={jdChange}
              sparkValues={jdSparkline}
              icon={FileText}
              loading={jdsLoading}
            />
            <MetricCard
              label="Pending Outreach"
              value={pendingOutreach}
              change={outreachChange}
              sparkValues={outreachSparkline}
              icon={Mail}
              loading={outreachLoading}
            />
          </div>

          {/* ── Content columns ── */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* ── Left: Activity feed (2/3) ── */}
            <div className="xl:col-span-2">
              <div
                className={cn(
                  "rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
                  "bg-bg-elevated h-full",
                )}
              >
                <div className="px-5 py-4 hairline-b flex items-center justify-between">
                  <h2 className="font-display text-base font-medium text-fg">Recent Activity</h2>
                  <Link
                    to="/candidates"
                    className="text-xs font-sans text-fg-muted hover:text-accent transition-colors"
                  >
                    View all →
                  </Link>
                </div>
                <div className="px-5">
                  {activityLoading ? (
                    <div className="py-4 space-y-4">
                      {Array.from({ length: 6 }).map((_, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <Skeleton width={28} height={28} rounded />
                          <div className="flex-1 space-y-1.5">
                            <Skeleton className="h-3.5 w-3/4" />
                            <Skeleton className="h-3 w-1/2" />
                          </div>
                          <Skeleton className="h-3 w-12" />
                        </div>
                      ))}
                    </div>
                  ) : recentActivity.length === 0 ? (
                    <div className="py-12 text-center">
                      <p className="text-sm font-sans text-fg-muted">
                        {viUi ? "Chưa có hoạt động nào." : "No activity yet."}
                      </p>
                      <p className="text-xs font-sans text-fg-subtle mt-1">
                        {viUi
                          ? "Tải CV, gửi email liên hệ hoặc tạo link phỏng vấn để xem hoạt động tại đây."
                          : "Upload CVs, send outreach, or create interview links to see activity here."}
                      </p>
                    </div>
                  ) : (
                    <div>
                      {recentActivity.map((entry) => (
                        <ActivityItem key={entry.id} entry={entry} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* ── Right column (1/3) ── */}
            <div className="flex flex-col gap-5">
              {/* Quick Actions */}
              <div
                className={cn(
                  "rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
                  "bg-bg-elevated",
                )}
              >
                <div className="px-4 py-3.5 hairline-b">
                  <h2 className="font-display text-base font-medium text-fg">Quick Actions</h2>
                </div>
                <div className="p-3 space-y-2">
                  <QuickActionButton
                    icon={FileUp}
                    label="Upload resumes"
                    description="Parse new PDF CVs"
                    onClick={() => setUploadOpen(true)}
                  />
                  <QuickActionButton
                    icon={FileText}
                    label="Create JD"
                    description="Write a new job description"
                    onClick={() => navigate(routes.jobDescriptions)}
                  />
                  <QuickActionButton
                    icon={BarChart3}
                    label="Start scoring"
                    description="Rank candidates by fit"
                    onClick={() => navigate("/scoring")}
                  />
                  <QuickActionButton
                    icon={MessageSquare}
                    label="Open AI Chat"
                    description="Query your candidate pool"
                    onClick={() => navigate("/chat")}
                  />
                </div>
              </div>

              {/* Editorial Insight */}
              <EditorialInsight
                totalCandidates={totalCandidates}
                processedToday={processedToday}
                activeJDs={activeJDs}
                pendingOutreach={pendingOutreach}
              />

              {/* Top Collections */}
              {collections.length > 0 && (
                <div
                  className={cn(
                    "rounded-[var(--radius-lg)] border border-[color:var(--hairline)]",
                    "bg-bg-elevated",
                  )}
                >
                  <div className="px-4 py-3.5 hairline-b flex items-center justify-between">
                    <h2 className="font-display text-base font-medium text-fg">
                      Top Collections
                    </h2>
                    <Link
                      to="/shortlists"
                      className="text-xs font-sans text-fg-muted hover:text-accent transition-colors"
                    >
                      All →
                    </Link>
                  </div>
                  <div className="p-3 space-y-2">
                    {collections.map((col) => (
                      <CollectionCard key={col.id} col={col} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Upload modal triggered from Quick Actions */}
      <UploadModal
        open={uploadOpen}
        onOpenChange={setUploadOpen}
        onComplete={() => {}}
      />
    </div>
  );
}
