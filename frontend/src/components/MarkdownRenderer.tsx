import React from "react";

/* ── inline parser ─────────────────────────────────────────────────────────── */

function parseInline(text: string): React.ReactNode[] {
  // Order matters: bold before italic, code before everything
  const regex = /(\*\*.*?\*\*|\*.*?\*|`.*?`|\[.*?\]\(.*?\))/g;
  const parts = text.split(regex);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={index} className="font-semibold text-fg">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return (
        <em key={index} className="italic text-fg">
          {part.slice(1, -1)}
        </em>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={index}
          className="bg-bg-elevated border border-[color:var(--hairline)] px-1.5 py-0.5 rounded text-xs font-mono text-fg-muted"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("[") && part.includes("](") && part.endsWith(")")) {
      const match = part.match(/\[(.*?)\]\((.*?)\)/);
      if (match) {
        const [, label, url] = match;
        return (
          <a
            key={index}
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent hover:underline inline-flex items-center gap-0.5"
          >
            {label}
          </a>
        );
      }
    }
    return part;
  });
}

/* ── table helpers ─────────────────────────────────────────────────────────── */

function isTableRow(line: string): boolean {
  const t = line.trim();
  return t.startsWith("|") && t.endsWith("|") && t.length > 2;
}

function isTableSeparator(line: string): boolean {
  const t = line.trim();
  // e.g. |---|---|---| or |:---:|:---|
  return isTableRow(line) && /^[|\s:*-]+$/.test(t);
}

function parseTableCells(line: string): string[] {
  const t = line.trim();
  // Strip leading and trailing |, then split on |
  const inner = t.startsWith("|") ? t.slice(1) : t;
  const stripped = inner.endsWith("|") ? inner.slice(0, -1) : inner;
  return stripped.split("|").map((cell) => cell.trim());
}

function renderTable(rows: string[], startIdx: number): React.ReactNode {
  // Find the separator row to distinguish header from body
  let separatorIdx = -1;
  for (let i = 0; i < rows.length; i++) {
    if (isTableSeparator(rows[i])) {
      separatorIdx = i;
      break;
    }
  }

  const headerRows = separatorIdx > 0 ? rows.slice(0, separatorIdx) : [];
  const bodyRows =
    separatorIdx >= 0
      ? rows.slice(separatorIdx + 1)
      : rows;

  return (
    <div key={`table-${startIdx}`} className="overflow-x-auto mb-3">
      <table className="min-w-full text-sm font-sans border-collapse border border-[color:var(--hairline)] rounded-[var(--radius-md)]">
        {headerRows.length > 0 && (
          <thead>
            {headerRows.map((row, ri) => (
              <tr key={ri} className="bg-bg-elevated">
                {parseTableCells(row).map((cell, ci) => (
                  <th
                    key={ci}
                    className="px-3 py-2 text-left text-xs font-semibold text-fg border border-[color:var(--hairline)]"
                  >
                    {parseInline(cell)}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
        )}
        <tbody>
          {bodyRows.map((row, ri) => (
            <tr
              key={ri}
              className={ri % 2 === 0 ? "bg-bg" : "bg-bg-elevated/50"}
            >
              {parseTableCells(row).map((cell, ci) => (
                <td
                  key={ci}
                  className="px-3 py-2 text-fg border border-[color:var(--hairline)]"
                >
                  {parseInline(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── horizontal rule ───────────────────────────────────────────────────────── */

function isHorizontalRule(line: string): boolean {
  const t = line.trim();
  return /^[-*_]{3,}$/.test(t);
}

/* ── blockquote ────────────────────────────────────────────────────────────── */

function isBlockquote(line: string): boolean {
  return line.trimStart().startsWith(">");
}

function stripBlockquotePrefix(line: string): string {
  return line.trimStart().replace(/^>\s?/, "");
}

/* ── main component ────────────────────────────────────────────────────────── */

export function MarkdownRenderer({ text }: { text: string }) {
  if (!text) return null;

  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];

  let currentListType: "ul" | "ol" | null = null;
  let currentListItems: React.ReactNode[] = [];

  // Table accumulator
  let tableBuffer: string[] = [];
  let tableStartIdx = -1;

  // Blockquote accumulator
  let blockquoteBuffer: string[] = [];
  let blockquoteStartIdx = -1;

  const flushList = (key: string | number) => {
    if (currentListType === "ul") {
      elements.push(
        <ul
          key={`ul-${key}`}
          className="list-disc pl-5 mb-3 space-y-1 text-sm font-sans text-fg"
        >
          {currentListItems}
        </ul>
      );
    } else if (currentListType === "ol") {
      elements.push(
        <ol
          key={`ol-${key}`}
          className="list-decimal pl-5 mb-3 space-y-1 text-sm font-sans text-fg"
        >
          {currentListItems}
        </ol>
      );
    }
    currentListType = null;
    currentListItems = [];
  };

  const flushTable = () => {
    if (tableBuffer.length > 0) {
      elements.push(renderTable(tableBuffer, tableStartIdx));
      tableBuffer = [];
      tableStartIdx = -1;
    }
  };

  const flushBlockquote = () => {
    if (blockquoteBuffer.length > 0) {
      elements.push(
        <blockquote
          key={`bq-${blockquoteStartIdx}`}
          className="border-l-[3px] border-accent/40 pl-4 py-1 mb-3 text-sm font-sans text-fg-muted italic"
        >
          {/* Recursively render the blockquote body */}
          <MarkdownRenderer text={blockquoteBuffer.join("\n")} />
        </blockquote>
      );
      blockquoteBuffer = [];
      blockquoteStartIdx = -1;
    }
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();

    // ── table rows ──
    if (isTableRow(line)) {
      // Flush any ongoing non-table state
      if (currentListType) flushList(idx);
      if (blockquoteBuffer.length > 0) flushBlockquote();

      if (tableBuffer.length === 0) tableStartIdx = idx;
      tableBuffer.push(line);
      return;
    } else {
      flushTable();
    }

    // ── blockquotes ──
    if (isBlockquote(line)) {
      if (currentListType) flushList(idx);
      if (blockquoteStartIdx === -1) blockquoteStartIdx = idx;
      blockquoteBuffer.push(stripBlockquotePrefix(line));
      return;
    } else {
      flushBlockquote();
    }

    // ── horizontal rule ──
    if (isHorizontalRule(trimmed)) {
      if (currentListType) flushList(idx);
      elements.push(
        <hr
          key={idx}
          className="my-4 border-t border-[color:var(--hairline)]"
        />
      );
      return;
    }

    // ── headings ──
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      if (currentListType) flushList(idx);
      const level = headingMatch[1].length;
      const content = headingMatch[2];
      const headingClasses =
        level === 1
          ? "text-xl font-semibold mb-2 mt-4 text-fg font-display"
          : level === 2
          ? "text-lg font-semibold mb-2 mt-3 text-fg font-display"
          : "text-base font-semibold mb-1 mt-2 text-fg font-display";

      const Tag = `h${level}` as keyof React.JSX.IntrinsicElements;
      elements.push(
        <Tag key={idx} className={headingClasses}>
          {parseInline(content)}
        </Tag>
      );
      return;
    }

    // ── unordered list ──
    const ulMatch = line.match(/^(\s*)[-*]\s+(.*)$/);
    if (ulMatch) {
      if (currentListType !== "ul") {
        flushList(idx);
        currentListType = "ul";
      }
      const content = ulMatch[2];
      currentListItems.push(
        <li key={idx} className="leading-relaxed">
          {parseInline(content)}
        </li>
      );
      return;
    }

    // ── ordered list ──
    const olMatch = line.match(/^(\s*)\d+\.\s+(.*)$/);
    if (olMatch) {
      if (currentListType !== "ol") {
        flushList(idx);
        currentListType = "ol";
      }
      const content = olMatch[2];
      currentListItems.push(
        <li key={idx} className="leading-relaxed">
          {parseInline(content)}
        </li>
      );
      return;
    }

    // ── default: flush any list and render paragraph ──
    if (currentListType) flushList(idx);

    if (trimmed === "") return;

    elements.push(
      <p key={idx} className="text-sm font-sans text-fg leading-relaxed mb-2">
        {parseInline(line)}
      </p>
    );
  });

  // Final flush
  if (currentListType) flushList("final");
  flushTable();
  flushBlockquote();

  return <div className="space-y-0.5">{elements}</div>;
}
