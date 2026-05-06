import axios from "axios";

export type ApiErrorKind =
  | "network"
  | "validation"
  | "not_found"
  | "conflict"
  | "server"
  | "unknown";

export interface FieldError {
  field: string;    // e.g. "body.jd_text"
  message: string;  // e.g. "field required"
  type?: string;    // e.g. "value_error.missing"
}

export class ApiError extends Error {
  status: number;
  kind: ApiErrorKind;
  detail: string;
  fieldErrors: FieldError[];

  constructor(opts: {
    status: number;
    kind: ApiErrorKind;
    detail: string;
    fieldErrors?: FieldError[];
  }) {
    super(opts.detail);
    this.name = "ApiError";
    this.status = opts.status;
    this.kind = opts.kind;
    this.detail = opts.detail.slice(0, 500); // truncate for safety — ASVS L1
    this.fieldErrors = opts.fieldErrors ?? [];
  }
}

/**
 * Normalize an unknown error (Axios or otherwise) into ApiError.
 * Handles: network failure, 4xx with { detail: string }, 422 with
 * { detail: FieldError[] }, 5xx, and anything else.
 *
 * SECURITY: Do NOT log request/response bodies — they may contain PII.
 */
export function parseAxiosError(err: unknown): ApiError {
  // Non-Axios errors (unexpected JS throws, etc.)
  if (!axios.isAxiosError(err)) {
    const message =
      err instanceof Error ? err.message : "An unexpected error occurred";
    return new ApiError({ status: 0, kind: "unknown", detail: message });
  }

  // Network failure — no response received
  if (!err.response) {
    return new ApiError({
      status: 0,
      kind: "network",
      detail: "Network error — no response received",
    });
  }

  const { status, data } = err.response as {
    status: number;
    data: unknown;
  };

  // 422 or 400 with FastAPI-style array detail (field validation errors)
  if (status === 422 || status === 400) {
    // FastAPI 422: { detail: [{ loc: string[], msg: string, type: string }] }
    if (
      data !== null &&
      typeof data === "object" &&
      "detail" in data &&
      Array.isArray((data as { detail: unknown }).detail)
    ) {
      const rawErrors = (data as { detail: { loc: string[]; msg: string; type?: string }[] })
        .detail;
      const fieldErrors: FieldError[] = rawErrors.map((e) => ({
        field: Array.isArray(e.loc) ? e.loc.join(".") : String(e.loc ?? ""),
        message: e.msg ?? "Invalid value",
        type: e.type,
      }));
      const summary =
        fieldErrors.length > 0
          ? fieldErrors.map((f) => `${f.field}: ${f.message}`).join("; ")
          : "Validation error";
      return new ApiError({
        status,
        kind: "validation",
        detail: summary,
        fieldErrors,
      });
    }

    // 400/422 with plain string detail
    const detail = extractDetail(data) ?? "Bad request";
    return new ApiError({ status, kind: "validation", detail });
  }

  if (status === 404) {
    const detail = extractDetail(data) ?? "Resource not found";
    return new ApiError({ status, kind: "not_found", detail });
  }

  if (status === 409) {
    const detail = extractDetail(data) ?? "Conflict — resource already exists";
    return new ApiError({ status, kind: "conflict", detail });
  }

  if (status >= 500) {
    const detail = extractDetail(data) ?? "Server error";
    return new ApiError({ status, kind: "server", detail });
  }

  // All other status codes
  const detail = extractDetail(data) ?? `Unexpected error (${status})`;
  return new ApiError({ status, kind: "unknown", detail });
}

/**
 * Safely pull a string `detail` field out of an unknown response body.
 * Returns null if the body isn't a plain object with a string `detail`.
 */
function extractDetail(data: unknown): string | null {
  if (
    data !== null &&
    typeof data === "object" &&
    "detail" in data &&
    typeof (data as { detail: unknown }).detail === "string"
  ) {
    return (data as { detail: string }).detail;
  }
  return null;
}
