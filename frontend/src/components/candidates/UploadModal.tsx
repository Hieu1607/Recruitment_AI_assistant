import { useState, useCallback, useEffect, useRef } from "react";
import { Upload, FileText, X, CheckCircle, XCircle, AlertTriangle, Clock3 } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ApiError, api, type ResumeBatchParseResponse } from "@/api";
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalTitle,
  ModalDescription,
  ModalFooter,
  Button,
} from "@/components/ui";
import { cn } from "@/lib/cn";
import { useSelectedJobId } from "@/lib/auth";
import { useNavigate } from "react-router";
import { routes } from "@/routes";

const UPLOAD_MESSAGES = [
  "Uploading PDFs...",
  "Saving resume records...",
  "Queueing background parsing...",
  "Preparing candidate extraction...",
];

interface UploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onComplete?: () => void;
}

type ModalState = "idle" | "files-selected" | "processing" | "complete";

interface SelectedFile {
  file: File;
  error?: string;
}

export function UploadModal({ open, onOpenChange, onComplete }: UploadModalProps) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const selectedJobId = useSelectedJobId();
  const [state, setState] = useState<ModalState>("idle");
  const [selectedFiles, setSelectedFiles] = useState<SelectedFile[]>([]);
  const [msgIndex, setMsgIndex] = useState(0);
  const [result, setResult] = useState<ResumeBatchParseResponse | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const { data: jobDescription, isLoading: jobDescriptionLoading } = useQuery({
    queryKey: ["jobs", selectedJobId, "job-description", "upload-gate"],
    enabled: open && !!selectedJobId,
    queryFn: async () => {
      try {
        return await api.jobs.jobDescription.get(selectedJobId!);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    staleTime: 30_000,
  });

  const uploadMutation = useMutation({
    mutationFn: (files: File[]) => {
      if (!selectedJobId) throw new Error("No job selected");
      return api.jobs.resumes.batchParse(selectedJobId, files);
    },
    onSuccess: (data) => {
      setResult(data);
      setState("complete");
      qc.invalidateQueries({ queryKey: ["candidates"] });
      qc.invalidateQueries({ queryKey: ["jobs", selectedJobId, "resumes"] });
    },
    onError: () => {
      setState("files-selected");
      toast.error("Upload failed. Please try again.");
    },
  });

  useEffect(() => {
    if (state === "processing") {
      intervalRef.current = setInterval(() => {
        setMsgIndex((i) => (i + 1) % UPLOAD_MESSAGES.length);
      }, 2400);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [state]);

  const resetModal = useCallback(() => {
    setState("idle");
    setSelectedFiles([]);
    setResult(null);
    setMsgIndex(0);
  }, []);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (state === "processing") return;
      if (!nextOpen) resetModal();
      onOpenChange(nextOpen);
    },
    [state, resetModal, onOpenChange],
  );

  const validateAndAddFiles = useCallback((rawFiles: File[]) => {
    const newFiles: SelectedFile[] = rawFiles.map((file) => {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        return { file, error: `"${file.name}" is not a PDF file` };
      }
      return { file };
    });
    setSelectedFiles((prev) => {
      const existingNames = new Set(prev.map((f) => f.file.name));
      const toAdd = newFiles.filter((f) => !existingNames.has(f.file.name));
      return [...prev, ...toAdd];
    });
    setState("files-selected");
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      validateAndAddFiles(Array.from(e.dataTransfer.files));
    },
    [validateAndAddFiles],
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        validateAndAddFiles(Array.from(e.target.files));
        e.target.value = "";
      }
    },
    [validateAndAddFiles],
  );

  const removeFile = useCallback((name: string) => {
    setSelectedFiles((prev) => {
      const next = prev.filter((f) => f.file.name !== name);
      if (next.length === 0) setState("idle");
      return next;
    });
  }, []);

  const handleSubmit = useCallback(() => {
    const validFiles = selectedFiles.filter((f) => !f.error).map((f) => f.file);
    if (validFiles.length === 0) return;
    if (!selectedJobId) {
      toast.error("Select or create a job before uploading resumes.");
      return;
    }
    setState("processing");
    setMsgIndex(0);
    uploadMutation.mutate(validFiles);
  }, [selectedFiles, selectedJobId, uploadMutation]);

  const validFiles = selectedFiles.filter((f) => !f.error);
  const invalidFiles = selectedFiles.filter((f) => f.error);
  const hasJobDescription = Boolean(jobDescription?.jd_text?.trim());
  const showJobDescriptionGate = open && !jobDescriptionLoading && !!selectedJobId && !hasJobDescription;

  return (
    <Modal open={open} onOpenChange={handleOpenChange}>
      <ModalContent size="large" showClose={state !== "processing"}>
        {open && !selectedJobId && (
          <>
            <ModalHeader>
              <ModalTitle>Select a workspace first</ModalTitle>
              <ModalDescription>
                Choose or create a job workspace before uploading resumes.
              </ModalDescription>
            </ModalHeader>

            <ModalFooter>
              <Button variant="ghost" onClick={() => handleOpenChange(false)}>
                Cancel
              </Button>
            </ModalFooter>
          </>
        )}

        {open && selectedJobId && jobDescriptionLoading && (
          <>
            <ModalHeader>
              <ModalTitle>Checking workspace setup</ModalTitle>
              <ModalDescription>
                Verifying whether this workspace already has a job description.
              </ModalDescription>
            </ModalHeader>

            <div className="flex flex-col items-center gap-4 py-8 text-center">
              <div className="h-10 w-10 animate-spin rounded-full border-2 border-[color:var(--hairline-strong)] border-t-accent" />
              <p className="text-sm text-fg-muted">Loading job description status…</p>
            </div>
          </>
        )}

        {showJobDescriptionGate && (
          <>
            <ModalHeader>
              <ModalTitle>Add a job description first</ModalTitle>
              <ModalDescription>
                This workspace only has the job name so far. Add the JD before uploading CVs so candidate parsing, scoring, and later screening stay anchored to the role.
              </ModalDescription>
            </ModalHeader>

            <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated px-5 py-4">
              <p className="text-xs font-medium uppercase tracking-[0.2em] text-fg-subtle">
                Required next step
              </p>
              <p className="mt-2 text-sm leading-6 text-fg-muted">
                Open the workspace JD page, write the hiring requirements, then come back to upload resumes.
              </p>
            </div>

            <ModalFooter>
              <Button variant="ghost" onClick={() => handleOpenChange(false)}>
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={() => {
                  handleOpenChange(false);
                  navigate(routes.jobDescriptions);
                }}
              >
                Go to job description
              </Button>
            </ModalFooter>
          </>
        )}

        {/* ── IDLE / FILES-SELECTED ── */}
        {!showJobDescriptionGate && !jobDescriptionLoading && selectedJobId && (state === "idle" || state === "files-selected") && (
          <>
            <ModalHeader>
              <ModalTitle>Upload resumes</ModalTitle>
              <ModalDescription>
                Drop PDF files to parse and build candidate profiles.
              </ModalDescription>
            </ModalHeader>

            {/* Dropzone */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={() => setIsDragOver(false)}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
              aria-label="Drop PDFs here or click to browse"
              className={cn(
                "relative flex flex-col items-center justify-center gap-3",
                "rounded-[var(--radius-lg)] border-2 border-dashed cursor-pointer",
                "transition-colors duration-200 py-10",
                isDragOver
                  ? "border-accent bg-[rgba(31,58,46,0.06)]"
                  : "border-[color:var(--hairline-strong)] hover:border-accent hover:bg-[rgba(31,58,46,0.03)]",
              )}
            >
              <Upload size={28} className="text-fg-subtle" strokeWidth={1.5} />
              <div className="text-center">
                <p className="font-sans text-sm font-medium text-fg">
                  Drop PDFs here or{" "}
                  <span className="text-accent underline underline-offset-2">click to browse</span>
                </p>
                <p className="text-xs text-fg-muted mt-1">Only .pdf files accepted</p>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept=".pdf,application/pdf"
                multiple
                className="sr-only"
                onChange={handleFileInput}
              />
            </div>

            {/* File list */}
            {selectedFiles.length > 0 && (
              <div className="mt-4 space-y-2 max-h-48 overflow-y-auto pr-1">
                {selectedFiles.map(({ file, error }) => (
                  <div
                    key={file.name}
                    className={cn(
                      "flex items-center gap-3 px-3 py-2 rounded-[var(--radius-md)] border",
                      error
                        ? "border-[rgba(184,68,46,0.3)] bg-[rgba(184,68,46,0.06)]"
                        : "border-[color:var(--hairline)] bg-bg-elevated",
                    )}
                  >
                    <FileText
                      size={16}
                      strokeWidth={1.5}
                      className={cn(
                        "shrink-0",
                        error ? "text-danger" : "text-fg-muted",
                      )}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="font-mono text-xs text-fg truncate">{file.name}</p>
                      {error ? (
                        <p className="text-xs text-danger mt-0.5">{error}</p>
                      ) : (
                        <p className="text-xs text-fg-muted mt-0.5">
                          {(file.size / 1024).toFixed(0)} KB
                        </p>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(file.name);
                      }}
                      className="shrink-0 text-fg-muted hover:text-fg transition-colors"
                      aria-label={`Remove ${file.name}`}
                    >
                      <X size={14} strokeWidth={2} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {invalidFiles.length > 0 && (
              <p className="mt-3 flex items-center gap-2 text-xs text-danger">
                <AlertTriangle size={13} strokeWidth={2} className="shrink-0" />
                {invalidFiles.length} file{invalidFiles.length > 1 ? "s" : ""} will be skipped
                (not PDF).
              </p>
            )}

            <ModalFooter>
              <Button variant="ghost" onClick={() => handleOpenChange(false)}>
                Cancel
              </Button>
              <Button
                variant="primary"
                disabled={validFiles.length === 0}
                onClick={handleSubmit}
              >
                Upload {validFiles.length > 0 ? `${validFiles.length} ` : ""}
                resume{validFiles.length !== 1 ? "s" : ""}
              </Button>
            </ModalFooter>
          </>
        )}

        {/* ── PROCESSING ── */}
        {!showJobDescriptionGate && !jobDescriptionLoading && selectedJobId && state === "processing" && (
          <div className="flex flex-col items-center gap-6 py-10">
            {/* Editorial SVG spinner */}
            <div className="relative h-20 w-20 shrink-0">
              <svg
                viewBox="0 0 80 80"
                fill="none"
                className="absolute inset-0 animate-spin"
                style={{ animationDuration: "3s" }}
                aria-hidden="true"
              >
                <circle
                  cx="40"
                  cy="40"
                  r="34"
                  stroke="var(--hairline-strong)"
                  strokeWidth="1.5"
                />
                <path
                  d="M40 6 A34 34 0 0 1 74 40"
                  stroke="var(--accent)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <FileText size={22} strokeWidth={1.5} className="text-accent" />
              </div>
            </div>

            <div className="text-center space-y-1.5">
              <p className="font-display text-xl font-medium text-fg">Queueing resumes</p>
              <p
                key={msgIndex}
                className="font-sans text-sm text-fg-muted animate-in fade-in duration-300"
              >
                {UPLOAD_MESSAGES[msgIndex]}
              </p>
            </div>

            {/* Indeterminate progress bar */}
            <div className="w-full max-w-[280px] h-1 rounded-full overflow-hidden bg-[color:var(--hairline-strong)]">
              <div
                className="h-full w-2/5 rounded-full bg-accent"
                style={{ animation: "indeterminate 1.8s ease-in-out infinite" }}
              />
            </div>

            <p className="flex items-center gap-2 text-xs text-fg-muted">
              <AlertTriangle size={13} strokeWidth={2} className="text-warning shrink-0" />
              Upload completes here. Parsing continues in the background.
            </p>
          </div>
        )}

        {/* ── COMPLETE ── */}
        {!showJobDescriptionGate && !jobDescriptionLoading && selectedJobId && state === "complete" && result && (
          <>
            <ModalHeader>
              <ModalTitle>
                {result.queued_files} resume{result.queued_files !== 1 ? "s" : ""} queued
              </ModalTitle>
              <ModalDescription>
                Background parsing has started. Resume status will update to processed or failed when each worker task finishes.
              </ModalDescription>
            </ModalHeader>

            <div className="mt-4 space-y-2 max-h-64 overflow-y-auto pr-1">
              {result.items.map((item) => (
                <div
                  key={item.resume_document_id}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-[var(--radius-md)] border",
                    item.status === "queued"
                      ? "border-[color:var(--hairline)] bg-bg-elevated"
                      : item.status === "processed"
                        ? "border-[rgba(74,124,89,0.3)] bg-[rgba(74,124,89,0.06)]"
                        : "border-[rgba(184,68,46,0.3)] bg-[rgba(184,68,46,0.06)]",
                  )}
                >
                  {item.status === "processed" ? (
                    <CheckCircle size={16} strokeWidth={1.75} className="text-success shrink-0" />
                  ) : item.status === "queued" ? (
                    <Clock3 size={16} strokeWidth={1.75} className="text-fg-muted shrink-0" />
                  ) : (
                    <XCircle size={16} strokeWidth={1.75} className="text-danger shrink-0" />
                  )}
                  <p className="font-mono text-xs text-fg flex-1 truncate">{item.file_name}</p>
                  <span
                    className={cn(
                      "text-xs font-medium font-sans shrink-0",
                      item.status === "queued"
                        ? "text-fg-muted"
                        : item.status === "processed"
                          ? "text-success"
                          : "text-danger",
                    )}
                  >
                    {item.status === "queued"
                      ? "Queued"
                      : item.status === "processed"
                        ? "Parsed"
                        : "Failed"}
                  </span>
                </div>
              ))}
            </div>

            <ModalFooter>
              <Button variant="ghost" onClick={resetModal}>
                Upload more
              </Button>
              <Button
                variant="primary"
                onClick={() => {
                  handleOpenChange(false);
                  onComplete?.();
                }}
              >
                View candidates
              </Button>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}
