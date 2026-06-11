import { api } from "@/api";
import { cn } from "@/lib/cn";
import { Bold, Highlighter, ImagePlus, Italic, Link2, List, ListOrdered, PaintBucket, Underline } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";

import type { OutreachVariableOption } from "./rich-text";
import { htmlToPlainText, normalizeEditorHtml } from "./rich-text";

type Props = {
  value: string;
  onChange: (next: { html: string; text: string }) => void;
  placeholder?: string;
  variableOptions?: OutreachVariableOption[];
  className?: string;
};

function ToolbarButton({
  onClick,
  title,
  children,
}: {
  onClick: () => void;
  title: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className="inline-flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] border border-[color:var(--hairline)] bg-bg text-fg-muted transition-colors hover:text-fg hover:bg-[color:var(--hairline)]"
    >
      {children}
    </button>
  );
}

export function OutreachRichEditor({
  value,
  onChange,
  placeholder = "Write your message here…",
  variableOptions = [],
  className,
}: Props) {
  const editorRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    if (!editorRef.current) return;
    const current = normalizeEditorHtml(editorRef.current.innerHTML);
    const incoming = normalizeEditorHtml(value);
    if (current !== incoming) {
      editorRef.current.innerHTML = incoming;
    }
  }, [value]);

  function emitChange() {
    if (!editorRef.current) return;
    const html = normalizeEditorHtml(editorRef.current.innerHTML);
    onChange({ html, text: htmlToPlainText(html) });
  }

  function runCommand(command: string, ui = false, commandValue?: string) {
    editorRef.current?.focus();
    document.execCommand(command, ui, commandValue);
    emitChange();
  }

  async function uploadImage(file: File) {
    setUploading(true);
    try {
      const result = await api.outreach.uploadImage(file);
      runCommand("insertImage", false, result.asset_url);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to upload image");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <div className={cn("rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg", className)}>
      <div className="flex flex-wrap items-center gap-2 border-b border-[color:var(--hairline)] px-3 py-2">
        <ToolbarButton title="Bold" onClick={() => runCommand("bold")}>
          <Bold size={14} />
        </ToolbarButton>
        <ToolbarButton title="Italic" onClick={() => runCommand("italic")}>
          <Italic size={14} />
        </ToolbarButton>
        <ToolbarButton title="Underline" onClick={() => runCommand("underline")}>
          <Underline size={14} />
        </ToolbarButton>
        <ToolbarButton
          title="Text color"
          onClick={() => {
            const next = window.prompt("Text color (hex or css color)", "#1f3a2e");
            if (next) runCommand("foreColor", false, next);
          }}
        >
          <PaintBucket size={14} />
        </ToolbarButton>
        <ToolbarButton
          title="Highlight"
          onClick={() => {
            const next = window.prompt("Highlight color", "#fff59d");
            if (next) runCommand("hiliteColor", false, next);
          }}
        >
          <Highlighter size={14} />
        </ToolbarButton>
        <ToolbarButton title="Bulleted list" onClick={() => runCommand("insertUnorderedList")}>
          <List size={14} />
        </ToolbarButton>
        <ToolbarButton title="Numbered list" onClick={() => runCommand("insertOrderedList")}>
          <ListOrdered size={14} />
        </ToolbarButton>
        <ToolbarButton
          title="Insert link"
          onClick={() => {
            const next = window.prompt("Link URL");
            if (next) runCommand("createLink", false, next);
          }}
        >
          <Link2 size={14} />
        </ToolbarButton>
        <ToolbarButton
          title="Insert image URL"
          onClick={() => {
            const next = window.prompt("Image URL");
            if (next) runCommand("insertImage", false, next);
          }}
        >
          <ImagePlus size={14} />
        </ToolbarButton>
        <label className="inline-flex cursor-pointer items-center gap-2 rounded-[var(--radius-sm)] border border-[color:var(--hairline)] px-2.5 py-1.5 text-xs text-fg-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-fg">
          <ImagePlus size={14} />
          {uploading ? "Uploading…" : "Upload image"}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void uploadImage(file);
            }}
          />
        </label>
        {variableOptions.length > 0 && (
          <select
            defaultValue=""
            onChange={(event) => {
              const key = event.target.value;
              if (!key) return;
              runCommand("insertText", false, `{{${key}}}`);
              event.target.value = "";
            }}
            className="ml-auto h-8 rounded-[var(--radius-sm)] border border-[color:var(--hairline)] bg-bg px-2 text-xs text-fg outline-none"
          >
            <option value="">Insert variable…</option>
            {variableOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
        )}
      </div>
      <div
        ref={editorRef}
        contentEditable
        suppressContentEditableWarning
        onInput={emitChange}
        onBlur={emitChange}
        data-placeholder={placeholder}
        className={cn(
          "min-h-[180px] px-4 py-3 text-sm leading-7 text-fg outline-none",
          "[&:empty:before]:pointer-events-none [&:empty:before]:text-fg-subtle [&:empty:before]:content-[attr(data-placeholder)]",
          "[&_a]:text-accent [&_a]:underline [&_img]:max-h-48 [&_img]:rounded-[var(--radius-md)]",
        )}
      />
    </div>
  );
}
