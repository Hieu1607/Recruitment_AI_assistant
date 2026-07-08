import type { ContentSource } from "@/api";

export const TEMPLATE_VARIABLES = [
  { key: "candidate_name", label: "Candidate Name" },
  { key: "candidate_email", label: "Candidate Email" },
  { key: "company_name", label: "Company Name" },
  { key: "job_title", label: "Job Title" },
] as const;

/**
 * Variables a recruiter can configure a default value for on a template
 * (Configure Variables section). candidate_name / candidate_email are
 * intentionally excluded — they always auto-resolve from the candidate
 * selected in New message, never from a template default.
 */
export const TEMPLATE_DEFAULT_VARIABLES = [
  { key: "job_title", label: "Job Title" },
  { key: "company_name", label: "Company Name" },
] as const;

export type TemplateDefaultVariableKey = (typeof TEMPLATE_DEFAULT_VARIABLES)[number]["key"];

export function outreachContentSourceLabel(source: ContentSource): string {
  if (source === "ai_draft") return "AI Draft";
  if (source === "manual") return "Manual";
  return "Template";
}

/**
 * Scan text for `{{variable}}` placeholders and return the subset of
 * `allowedKeys` that actually appear. Used to keep a template's
 * `variables_used` in sync with its real content instead of assuming every
 * known variable is used.
 */
export function detectUsedVariables(text: string, allowedKeys: readonly string[]): string[] {
  const found = new Set<string>();
  for (const match of text.matchAll(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g)) {
    const key = match[1];
    if (allowedKeys.includes(key)) found.add(key);
  }
  return Array.from(found);
}

/** Of the template-default-configurable keys, which ones does this template
 * actually reference but have no configured default value for yet? */
export function missingRequiredTemplateDefaults(
  variablesUsed: readonly string[],
  defaultVariables: Record<string, string> | null | undefined,
): TemplateDefaultVariableKey[] {
  const defaults = defaultVariables ?? {};
  return TEMPLATE_DEFAULT_VARIABLES.map((item) => item.key).filter(
    (key) => variablesUsed.includes(key) && !defaults[key]?.trim(),
  );
}

/** Substitute `{{variable}}` placeholders with resolved values. Unknown/unset
 * variables resolve to an empty string, mirroring the backend's
 * render_template_string in outreach_service.py. */
export function renderTemplateString(template: string, variables: Record<string, string>): string {
  return template.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (_match, key: string) => variables[key] ?? "");
}
