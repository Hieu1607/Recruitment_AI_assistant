// Re-export all types so consumers can do: import type { ResumeResponse } from "@/api"
export * from "./types";

// Re-export client utilities
export { client, isApiError } from "./client";
export {
    ApiError,
    parseAxiosError, type ApiErrorKind, type FieldError
} from "./errors";
export { queryClient } from "./queryClient";

// Import endpoint modules
import { authApi } from "./endpoints/auth";
import { chatApi } from "./endpoints/chat";
import { interviewQuestionsApi } from "./endpoints/interviewQuestions";
import { jobsApi } from "./endpoints/jobs";
import { jobDescriptionsApi } from "./endpoints/jobDescriptions";
import { outreachApi } from "./endpoints/outreach";
import { scoringApi } from "./endpoints/scoring";
import { shortlistApi } from "./endpoints/shortlist";
import { uploadApi } from "./endpoints/upload";

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
  auth: authApi,
  jobs: jobsApi,
  upload: uploadApi,
  candidates: {
    getById: (profileId: string) => uploadApi.getProfileById(profileId),
  },
  jobDescriptions: jobDescriptionsApi,
  scoring: scoringApi,
  chat: chatApi,
  shortlist: shortlistApi,
  outreach: outreachApi,
  interviewQuestions: interviewQuestionsApi,
};
