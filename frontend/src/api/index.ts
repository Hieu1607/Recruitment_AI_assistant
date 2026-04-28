// Re-export all types so consumers can do: import type { ResumeResponse } from "@/api"
export * from "./types";

// Re-export client utilities
export { client, isApiError } from "./client";
export {
  ApiError,
  parseAxiosError,
  type FieldError,
  type ApiErrorKind,
} from "./errors";
export { queryClient } from "./queryClient";

// Import endpoint modules
import { uploadApi } from "./endpoints/upload";
import { jobDescriptionsApi } from "./endpoints/jobDescriptions";
import { scoringApi } from "./endpoints/scoring";
import { chatApi } from "./endpoints/chat";
import { shortlistApi } from "./endpoints/shortlist";
import { outreachApi } from "./endpoints/outreach";
import { interviewQuestionsApi } from "./endpoints/interviewQuestions";

/**
 * Unified API namespace.
 *
 * Usage from any screen:
 *   import { api } from "@/api";
 *   const data = await api.upload.list({ limit: 50 });
 *   api.shortlist.sessions.list({ user_id: "..." });
 *   api.shortlist.collections.create({ ... });
 */
export const api = {
  upload: uploadApi,
  jobDescriptions: jobDescriptionsApi,
  scoring: scoringApi,
  chat: chatApi,
  shortlist: shortlistApi,
  outreach: outreachApi,
  interviewQuestions: interviewQuestionsApi,
};
