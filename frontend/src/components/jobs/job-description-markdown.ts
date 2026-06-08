function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function applyInlineMarkdown(text: string): string {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*(.+?)\*(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>")
    .replace(/(^|[\s(])_(.+?)_(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");
}

function paragraphize(lines: string[]): string {
  return lines
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => `<p>${applyInlineMarkdown(line)}</p>`)
    .join("");
}

export function markdownToHtml(markdown: string): string {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks: string[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trimEnd();

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      blocks.push(`<h${level}>${applyInlineMarkdown(heading[2].trim())}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
        index += 1;
      }
      blocks.push(`<ul>${items.map((item) => `<li>${applyInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+\.\s+/, ""));
        index += 1;
      }
      blocks.push(`<ol>${items.map((item) => `<li>${applyInlineMarkdown(item)}</li>`).join("")}</ol>`);
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (index < lines.length) {
      const next = lines[index].trimEnd();
      if (!next.trim() || /^(#{1,3})\s+/.test(next) || /^[-*]\s+/.test(next.trim()) || /^\d+\.\s+/.test(next.trim())) {
        break;
      }
      paragraphLines.push(next);
      index += 1;
    }
    blocks.push(paragraphize(paragraphLines));
  }

  return blocks.join("");
}

function collectText(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) return node.textContent ?? "";
  if (!(node instanceof HTMLElement)) return "";

  if (node.tagName === "BR") return "\n";

  const text = Array.from(node.childNodes).map(collectText).join("");
  if (node.tagName === "STRONG" || node.tagName === "B") return `**${text.trim()}**`;
  if (node.tagName === "EM" || node.tagName === "I") return `*${text.trim()}*`;
  return text;
}

function blockToMarkdown(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) return (node.textContent ?? "").trim();
  if (!(node instanceof HTMLElement)) return "";

  const childrenText = Array.from(node.childNodes).map(collectText).join("").replace(/\u00a0/g, " ").trim();

  switch (node.tagName) {
    case "H1":
      return childrenText ? `# ${childrenText}` : "";
    case "H2":
      return childrenText ? `## ${childrenText}` : "";
    case "H3":
      return childrenText ? `### ${childrenText}` : "";
    case "UL":
      return Array.from(node.children)
        .map((child) => `- ${collectText(child).replace(/\u00a0/g, " ").trim()}`)
        .join("\n");
    case "OL":
      return Array.from(node.children)
        .map((child, index) => `${index + 1}. ${collectText(child).replace(/\u00a0/g, " ").trim()}`)
        .join("\n");
    case "P":
    case "DIV":
      return childrenText;
    default:
      return childrenText;
  }
}

export function htmlToMarkdown(html: string): string {
  const template = document.createElement("template");
  template.innerHTML = html;

  const blocks = Array.from(template.content.childNodes)
    .map(blockToMarkdown)
    .map((block) => block.trim())
    .filter(Boolean);

  return blocks.join("\n\n").replace(/\n{3,}/g, "\n\n").trim();
}

export function normalizePastedHtml(html: string): string {
  const template = document.createElement("template");
  template.innerHTML = html;

  template.content.querySelectorAll("script, style").forEach((node) => node.remove());

  template.content.querySelectorAll("*").forEach((element) => {
    const allowed = ["P", "BR", "STRONG", "B", "EM", "I", "H1", "H2", "H3", "UL", "OL", "LI", "DIV"];
    if (!allowed.includes(element.tagName)) {
      const parent = element.parentNode;
      if (!parent) return;
      while (element.firstChild) parent.insertBefore(element.firstChild, element);
      parent.removeChild(element);
      return;
    }

    Array.from(element.attributes).forEach((attribute) => {
      element.removeAttribute(attribute.name);
    });
  });

  return template.innerHTML;
}
