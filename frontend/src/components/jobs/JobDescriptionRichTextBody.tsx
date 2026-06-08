import { cn } from "@/lib/cn";
import { List, ListOrdered, PenLine } from "lucide-react";
import { useEffect, useState, type ClipboardEvent, type MutableRefObject } from "react";

import { markdownToHtml, normalizePastedHtml } from "./job-description-markdown";

interface JobDescriptionRichTextBodyProps {
  editorRef: MutableRefObject<HTMLDivElement | null>;
  initialMarkdown?: string;
  onInput?: () => void;
  onBlur?: () => void;
  minHeightClassName?: string;
}

type BlockFormat = "p" | "h1" | "h2" | "h3" | "ul" | "ol";

interface FormatState {
  bold: boolean;
  italic: boolean;
  block: BlockFormat;
}

function isSelectionInside(editor: HTMLDivElement | null): boolean {
  if (!editor) return false;
  const selection = document.getSelection();
  const anchorNode = selection?.anchorNode;
  if (!anchorNode) return false;
  return editor.contains(anchorNode);
}

function getFormatState(editor: HTMLDivElement | null): FormatState {
  if (!editor || !isSelectionInside(editor)) {
    return { bold: false, italic: false, block: "p" };
  }

  const selection = document.getSelection();
  let current: Node | null = selection?.anchorNode ?? null;
  let bold = false;
  let italic = false;
  let block: BlockFormat = "p";

  while (current && current !== editor) {
    if (current instanceof HTMLElement) {
      const tag = current.tagName;
      if (!bold && (tag === "STRONG" || tag === "B")) bold = true;
      if (!italic && (tag === "EM" || tag === "I")) italic = true;
      if (tag === "H1") block = "h1";
      else if (tag === "H2") block = "h2";
      else if (tag === "H3") block = "h3";
      else if (tag === "UL") block = "ul";
      else if (tag === "OL") block = "ol";
    }
    current = current.parentNode;
  }

  return { bold, italic, block };
}

function updateBlockFormat(target: BlockFormat) {
  if (target === "ul") {
    document.execCommand("insertUnorderedList");
    return;
  }
  if (target === "ol") {
    document.execCommand("insertOrderedList");
    return;
  }
  document.execCommand("formatBlock", false, target);
}

function insertHtmlAtSelection(html: string) {
  const selection = document.getSelection();
  if (!selection || selection.rangeCount === 0) return;

  const range = selection.getRangeAt(0);
  range.deleteContents();

  const template = document.createElement("template");
  template.innerHTML = html;
  const fragment = template.content;
  const lastNode = fragment.lastChild;
  range.insertNode(fragment);

  if (lastNode) {
    const nextRange = document.createRange();
    nextRange.setStartAfter(lastNode);
    nextRange.collapse(true);
    selection.removeAllRanges();
    selection.addRange(nextRange);
  }
}

export function JobDescriptionRichTextBody({
  editorRef,
  initialMarkdown = "",
  onInput,
  onBlur,
  minHeightClassName = "min-h-[320px]",
}: JobDescriptionRichTextBodyProps) {
  const [formatState, setFormatState] = useState<FormatState>({ bold: false, italic: false, block: "p" });

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    if (!editor.innerHTML.trim() && initialMarkdown.trim()) {
      editor.innerHTML = markdownToHtml(initialMarkdown);
    }
  }, [editorRef, initialMarkdown]);

  useEffect(() => {
    const syncState = () => setFormatState(getFormatState(editorRef.current));
    document.addEventListener("selectionchange", syncState);
    return () => document.removeEventListener("selectionchange", syncState);
  }, [editorRef]);

  const applyInlineFormat = (cmd: "bold" | "italic") => {
    editorRef.current?.focus();
    document.execCommand(cmd);
    setFormatState(getFormatState(editorRef.current));
  };

  const applyBlock = (block: BlockFormat) => {
    editorRef.current?.focus();
    updateBlockFormat(block);
    setFormatState(getFormatState(editorRef.current));
  };

  const handlePaste = (event: ClipboardEvent<HTMLDivElement>) => {
    event.preventDefault();
    const html = event.clipboardData.getData("text/html");
    const text = event.clipboardData.getData("text/plain");

    editorRef.current?.focus();

    if (html) {
      insertHtmlAtSelection(normalizePastedHtml(html));
      onInput?.();
      setFormatState(getFormatState(editorRef.current));
      return;
    }

    if (text) {
      insertHtmlAtSelection(markdownToHtml(text));
      onInput?.();
      setFormatState(getFormatState(editorRef.current));
    }
  };

  const toolbarButtonClass = (active = false) =>
    cn(
      "inline-flex h-8 min-w-8 items-center justify-center rounded-[var(--radius-sm)] px-2 text-[12px] font-medium transition-colors",
      active
        ? "bg-[rgba(31,58,46,0.12)] text-fg shadow-[0_0_0_1px_rgba(31,58,46,0.08)_inset]"
        : "text-fg-muted hover:bg-[color:var(--hairline)] hover:text-fg",
    );

  return (
    <div
      className={cn(
        "rounded-[var(--radius-lg)] border border-[color:var(--hairline-strong)] bg-[rgba(31,58,46,0.02)]",
        "shadow-[0_1px_0_rgba(255,255,255,0.55)_inset] transition-colors duration-[var(--duration-fast)]",
        "hover:border-[rgba(31,58,46,0.28)] focus-within:border-[rgba(31,58,46,0.38)] focus-within:bg-[rgba(31,58,46,0.04)] focus-within:shadow-[0_0_0_3px_rgba(31,58,46,0.08)]",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[color:var(--hairline)] px-4 py-3">
        <div className="min-w-0">
          <div className="inline-flex items-center gap-2 rounded-full border border-[rgba(31,58,46,0.12)] bg-bg px-2.5 py-1 text-[11px] uppercase tracking-[0.18em] text-fg-muted">
            <PenLine size={11} strokeWidth={1.75} className="text-accent" />
            Rich text editor
          </div>
          <p className="mt-2 text-sm font-medium text-fg">JD content</p>
          <p className="mt-1 text-xs leading-5 text-fg-subtle">
            Markdown-safe editor. Paste from ChatGPT or docs and keep headings, lists, and emphasis.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex items-center gap-0.5 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-1">
            <button type="button" aria-label="Bold" title="Bold" onMouseDown={(e) => { e.preventDefault(); applyInlineFormat("bold"); }} className={toolbarButtonClass(formatState.bold)}>
              B
            </button>
            <button type="button" aria-label="Italic" title="Italic" onMouseDown={(e) => { e.preventDefault(); applyInlineFormat("italic"); }} className={toolbarButtonClass(formatState.italic)}>
              <span className="italic">I</span>
            </button>
          </div>

          <div className="inline-flex items-center gap-0.5 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-1">
            {(["h1", "h2", "h3"] as const).map((block) => (
              <button
                key={block}
                type="button"
                aria-label={block.toUpperCase()}
                title={block.toUpperCase()}
                onMouseDown={(e) => {
                  e.preventDefault();
                  applyBlock(block);
                }}
                className={toolbarButtonClass(formatState.block === block)}
              >
                {block.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="inline-flex items-center gap-0.5 rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg-elevated p-1">
            <button
              type="button"
              aria-label="Bullet list"
              title="Bullet list"
              onMouseDown={(e) => {
                e.preventDefault();
                applyBlock("ul");
              }}
              className={toolbarButtonClass(formatState.block === "ul")}
            >
              <List size={14} strokeWidth={1.75} />
            </button>
            <button
              type="button"
              aria-label="Numbered list"
              title="Numbered list"
              onMouseDown={(e) => {
                e.preventDefault();
                applyBlock("ol");
              }}
              className={toolbarButtonClass(formatState.block === "ol")}
            >
              <ListOrdered size={14} strokeWidth={1.75} />
            </button>
          </div>
        </div>
      </div>

      <div className="p-4">
        <div className="rounded-[calc(var(--radius-lg)-4px)] border border-dashed border-[rgba(31,58,46,0.18)] bg-bg px-4 py-4 transition-colors duration-[var(--duration-fast)] focus-within:border-[rgba(31,58,46,0.34)] focus-within:border-solid">
          <div
            ref={editorRef}
            contentEditable
            suppressContentEditableWarning
            data-placeholder="Click here to start writing the job description… Add responsibilities, requirements, and hiring signals."
            className={cn(
              minHeightClassName,
              "text-[0.9375rem] font-sans leading-relaxed text-fg outline-none",
              "[&_h1]:font-display [&_h1]:text-[2rem] [&_h1]:font-medium [&_h1]:mt-6 [&_h1]:mb-3",
              "[&_h2]:font-display [&_h2]:text-2xl [&_h2]:font-medium [&_h2]:mt-6 [&_h2]:mb-2",
              "[&_h3]:font-display [&_h3]:text-xl [&_h3]:font-medium [&_h3]:mt-5 [&_h3]:mb-2",
              "[&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2",
              "[&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-2",
              "[&_li]:mb-1",
              "[&_b]:font-semibold [&_strong]:font-semibold",
              "[&_i]:italic [&_em]:italic",
              "empty:before:content-[attr(data-placeholder)]",
              "empty:before:text-fg-subtle empty:before:pointer-events-none empty:before:block empty:before:max-w-2xl empty:before:leading-7",
            )}
            onInput={() => {
              onInput?.();
              setFormatState(getFormatState(editorRef.current));
            }}
            onBlur={onBlur}
            onKeyUp={() => setFormatState(getFormatState(editorRef.current))}
            onMouseUp={() => setFormatState(getFormatState(editorRef.current))}
            onPaste={handlePaste}
          />
        </div>

        <div className="mt-3 flex flex-wrap gap-2 text-xs text-fg-subtle">
          {["Paste rich text", "Headings H1-H3", "Bullet or numbered lists"].map((item) => (
            <span
              key={item}
              className="rounded-full border border-[color:var(--hairline)] bg-bg px-3 py-1"
            >
              {item}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
