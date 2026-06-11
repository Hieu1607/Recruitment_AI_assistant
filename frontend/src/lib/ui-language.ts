export type UiLanguage = "vi" | "en";

function normalizeLanguage(value: string | undefined): UiLanguage {
  const normalized = (value ?? "").trim().toLowerCase();
  if (normalized.startsWith("en")) return "en";
  return normalized.startsWith("vi") ? "vi" : "en";
}

export const uiLanguage: UiLanguage = normalizeLanguage(import.meta.env.VITE_UI_LANGUAGE);

export function isVietnameseUi() {
  return uiLanguage === "vi";
}
