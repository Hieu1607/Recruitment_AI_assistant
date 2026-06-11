export function htmlToPlainText(html: string): string {
  if (typeof window === "undefined") {
    return html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  }
  const div = document.createElement("div");
  div.innerHTML = html;
  return (div.textContent || div.innerText || "").trim();
}

export function normalizeEditorHtml(html: string): string {
  return html.trim().replace(/<div><br><\/div>/g, "").replace(/^\s+|\s+$/g, "");
}

export type OutreachVariableOption = {
  key: string;
  label: string;
};
