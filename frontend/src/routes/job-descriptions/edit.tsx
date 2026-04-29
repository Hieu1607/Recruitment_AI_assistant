import { api } from "@/api";
import {
    Badge,
    Button,
    Modal,
    ModalContent,
    ModalDescription,
    ModalFooter,
    ModalHeader,
    ModalTitle,
    Skeleton,
} from "@/components/ui";
import { cn } from "@/lib/cn";
import { routes } from "@/routes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    ArrowLeft,
    Bold,
    Heading2,
    Italic,
    List,
    Save,
    ToggleLeft,
    ToggleRight,
    Trash2,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { toast } from "sonner";

const PLACEHOLDER_USER_ID = "00000000-0000-0000-0000-000000000001";

type SaveState = "idle" | "saving" | "saved" | "error";

// ─── main component ──────────────────────────────────────────────────────────

export default function JobDescriptionEditRoute() {
  const { id } = useParams<{ id?: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const isNew = !id;

  const titleRef = useRef<HTMLInputElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  const autosaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [savedAt, setSavedAt] = useState<Date | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [localIsActive, setLocalIsActive] = useState(true);
  const [, forceRender] = useState(0);

  // Tick to refresh "Saved Xs ago" label
  useEffect(() => {
    if (saveState !== "saved") return;
    const t = setInterval(() => forceRender((n) => n + 1), 5000);
    return () => clearInterval(t);
  }, [saveState]);

  // ── load existing JD ─────────────────────────────────────────────────────

  const { data: jd, isLoading } = useQuery({
    queryKey: ["jobDescription", id],
    queryFn: () => api.jobDescriptions.get(id!),
    enabled: !isNew,
  });

  useEffect(() => {
    if (!jd) return;
    if (titleRef.current && !titleRef.current.value) {
      titleRef.current.value = jd.title ?? "";
    }
    if (editorRef.current && !editorRef.current.innerText.trim()) {
      editorRef.current.innerText = jd.jd_text;
    }
    setLocalIsActive(jd.is_active);
  }, [jd]);

  // ── mutations ─────────────────────────────────────────────────────────────

  const createMutation = useMutation({
    mutationFn: (body: { title: string; jd_text: string }) =>
      api.jobDescriptions.create({
        title: body.title || undefined,
        jd_text: body.jd_text,
        created_by_user_id: PLACEHOLDER_USER_ID,
      }),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["jobDescriptions"] });
      setSaveState("saved");
      setSavedAt(new Date());
      navigate(routes.jobDescriptionEdit(data.id), { replace: true });
    },
    onError: () => {
      setSaveState("error");
      toast.error("Failed to save job description");
    },
  });

  const updateMutation = useMutation({
    mutationFn: (body: Partial<{ title: string; jd_text: string; is_active: boolean }>) =>
      api.jobDescriptions.update(id!, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobDescriptions"] });
      qc.invalidateQueries({ queryKey: ["jobDescription", id] });
      setSaveState("saved");
      setSavedAt(new Date());
    },
    onError: () => {
      setSaveState("error");
      toast.error("Failed to save job description");
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.jobDescriptions.remove(id!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobDescriptions"] });
      toast.success("Job description deleted");
      navigate(routes.jobDescriptions);
    },
    onError: () => toast.error("Failed to delete job description"),
  });

  // ── save helpers ──────────────────────────────────────────────────────────

  function getCurrentValues() {
    return {
      title: titleRef.current?.value.trim() ?? "",
      jd_text: editorRef.current?.innerText.trim() ?? "",
    };
  }

  function saveNow(extra?: Partial<{ is_active: boolean }>) {
    const { title, jd_text } = getCurrentValues();
    if (!jd_text && !title) return;
    setSaveState("saving");
    if (isNew) {
      createMutation.mutate({ title, jd_text });
    } else {
      updateMutation.mutate({ title, jd_text, ...extra });
    }
  }

  function scheduleAutosave() {
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    autosaveTimer.current = setTimeout(() => {
      if (!isNew) saveNow();
    }, 1500);
  }

  function cancelAndSaveNow() {
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    if (!isNew) saveNow();
  }

  useEffect(() => () => { if (autosaveTimer.current) clearTimeout(autosaveTimer.current); }, []);

  // ── toolbar ───────────────────────────────────────────────────────────────

  function applyFormat(cmd: string, value?: string) {
    editorRef.current?.focus();
    // eslint-disable-next-line @typescript-eslint/no-deprecated
    document.execCommand(cmd, false, value);
  }

  // ── is_active toggle ──────────────────────────────────────────────────────

  function toggleActive() {
    const next = !localIsActive;
    setLocalIsActive(next);
    if (!isNew) {
      updateMutation.mutate({ is_active: next });
    }
  }

  // ── save indicator ────────────────────────────────────────────────────────

  function savedLabel(): string {
    if (saveState === "saving") return "Saving…";
    if (saveState === "error") return "Save failed";
    if (saveState === "saved" && savedAt) {
      const diff = Math.round((Date.now() - savedAt.getTime()) / 1000);
      if (diff < 5) return "Saved just now";
      if (diff < 60) return `Saved ${diff}s ago`;
      return `Saved ${Math.round(diff / 60)}m ago`;
    }
    return "";
  }

  // ── render ────────────────────────────────────────────────────────────────

  if (!isNew && isLoading) {
    return (
      <div className="px-12 py-10 max-w-4xl space-y-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const isSaving = createMutation.isPending || updateMutation.isPending;

  return (
    <div className="min-h-full flex flex-col">

      {/* Top navigation bar */}
      <div
        className={cn(
          "flex items-center justify-between px-8 py-3.5",
          "border-b border-[color:var(--hairline)] bg-bg-elevated sticky top-0 z-10",
        )}
      >
        <button
          type="button"
          onClick={() => navigate(routes.jobDescriptions)}
          className="inline-flex items-center gap-1.5 text-sm font-sans text-fg-muted hover:text-fg transition-colors"
        >
          <ArrowLeft size={15} strokeWidth={1.75} />
          Job Descriptions
        </button>

        <div className="flex items-center gap-3">
          {savedLabel() && (
            <span
              className={cn(
                "text-xs font-sans tabular-nums transition-colors",
                saveState === "error" ? "text-danger" : "text-fg-muted",
              )}
            >
              {savedLabel()}
            </span>
          )}
          <Button
            variant="primary"
            size="sm"
            icon={<Save size={13} strokeWidth={2} />}
            loading={isSaving}
            onClick={() => saveNow()}
          >
            {isNew ? "Create" : "Save"}
          </Button>
        </div>
      </div>

      {/* Body: editor + settings panel */}
      <div className="flex flex-1 min-h-0">

        {/* Editor column */}
        <div className="flex-1 px-12 py-10 overflow-y-auto">
          <div className="max-w-[720px]">

            {/* Formatting toolbar */}
            <div
              className={cn(
                "inline-flex items-center gap-0.5 mb-6 p-1",
                "rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated",
              )}
            >
              {[
                { cmd: "bold", Icon: Bold, label: "Bold", title: "Bold" },
                { cmd: "italic", Icon: Italic, label: "Italic", title: "Italic" },
                { cmd: "formatBlock", value: "h2", Icon: Heading2, label: "Heading 2", title: "Heading 2" },
                { cmd: "insertUnorderedList", Icon: List, label: "Bullet list", title: "Bullet list" },
              ].map(({ cmd, value, Icon, label, title }) => (
                <button
                  key={`${cmd}${value ?? ""}`}
                  type="button"
                  title={title}
                  aria-label={label}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    applyFormat(cmd, value);
                  }}
                  className={cn(
                    "inline-flex items-center justify-center h-7 w-7 rounded-[var(--radius-sm)]",
                    "text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors",
                  )}
                >
                  <Icon size={13} strokeWidth={1.75} />
                </button>
              ))}
            </div>

            {/* Title input */}
            <input
              ref={titleRef}
              type="text"
              placeholder="Untitled position"
              defaultValue={jd?.title ?? ""}
              className={cn(
                "w-full font-display text-[2.5rem] font-medium leading-tight",
                "text-fg placeholder:text-fg-subtle",
                "bg-transparent border-none outline-none mb-6",
              )}
              onBlur={scheduleAutosave}
            />

            {/* Hairline divider */}
            <div className="h-px bg-[color:var(--hairline)] mb-6" />

            {/* Rich text body */}
            <div
              ref={editorRef}
              contentEditable
              suppressContentEditableWarning
              data-placeholder="Start writing the job description… Use the toolbar to add headings, bold text, and bullet points."
              className={cn(
                "min-h-[400px] text-[0.9375rem] font-sans leading-relaxed text-fg outline-none",
                "[&_h2]:font-display [&_h2]:text-2xl [&_h2]:font-medium [&_h2]:mt-6 [&_h2]:mb-2",
                "[&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2",
                "[&_li]:mb-1",
                "[&_b]:font-semibold [&_strong]:font-semibold",
                "[&_i]:italic [&_em]:italic",
                "empty:before:content-[attr(data-placeholder)]",
                "empty:before:text-fg-subtle empty:before:pointer-events-none",
              )}
              onInput={scheduleAutosave}
              onBlur={cancelAndSaveNow}
            />
          </div>
        </div>

        {/* Settings panel */}
        <div
          className={cn(
            "w-60 shrink-0 border-l border-[color:var(--hairline)]",
            "p-6 flex flex-col gap-6",
          )}
        >
          <div>
            <p className="text-xs font-semibold font-sans uppercase tracking-wider text-fg-muted mb-3">
              Settings
            </p>

            {/* is_active toggle */}
            <div className="flex items-center justify-between py-2">
              <span className="text-sm font-sans text-fg">Active</span>
              <button
                type="button"
                onClick={toggleActive}
                aria-pressed={localIsActive}
                title={localIsActive ? "Deactivate" : "Activate"}
                className="text-fg-muted hover:text-fg transition-colors"
              >
                {localIsActive ? (
                  <ToggleRight size={28} strokeWidth={1.5} className="text-accent" />
                ) : (
                  <ToggleLeft size={28} strokeWidth={1.5} />
                )}
              </button>
            </div>
            <p className="text-xs text-fg-muted font-sans">
              {localIsActive ? "Available for scoring" : "Not available for scoring"}
            </p>
            <div className="mt-2">
              <Badge variant={localIsActive ? "success" : "neutral"} size="sm" dot>
                {localIsActive ? "Active" : "Inactive"}
              </Badge>
            </div>
          </div>

          {/* Created at */}
          {jd && (
            <div>
              <p className="text-xs font-semibold font-sans uppercase tracking-wider text-fg-muted mb-2">
                Created
              </p>
              <p
                className="text-xs text-fg-muted font-sans"
                title={new Date(jd.created_at).toUTCString()}
              >
                {new Date(jd.created_at).toLocaleDateString(undefined, {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })}
              </p>
            </div>
          )}

          {/* Delete */}
          {!isNew && (
            <div className="mt-auto pt-4 border-t border-[color:var(--hairline)]">
              <Button
                variant="danger"
                size="sm"
                className="w-full justify-center"
                icon={<Trash2 size={13} strokeWidth={1.75} />}
                onClick={() => setDeleteOpen(true)}
              >
                Delete JD
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Delete confirmation */}
      <Modal open={deleteOpen} onOpenChange={setDeleteOpen}>
        <ModalContent>
          <ModalHeader>
            <ModalTitle>Delete job description</ModalTitle>
            <ModalDescription>
              Are you sure you want to delete{" "}
              <span className="font-medium text-fg">
                {(jd?.title ?? titleRef.current?.value.trim()) || "Untitled position"}
              </span>
              ? This cannot be undone.
            </ModalDescription>
          </ModalHeader>
          <ModalFooter>
            <Button variant="ghost" onClick={() => setDeleteOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              loading={deleteMutation.isPending}
              onClick={() => deleteMutation.mutate()}
            >
              Delete
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}
