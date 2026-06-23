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
import { interviewInvitationsApi } from "./endpoints/interviewInvitations";
import { interviewPublicApi } from "./endpoints/interviewPublic";
import { interviewQuestionsApi } from "./endpoints/interviewQuestions";
import { interviewReportsApi } from "./endpoints/interviewReports";
import { interviewTemplatesApi } from "./endpoints/interviewTemplates";
import { jobsApi } from "./endpoints/jobs";
import { jobDescriptionsApi } from "./endpoints/jobDescriptions";
import { notificationsApi } from "./endpoints/notifications";
import { outreachApi } from "./endpoints/outreach";
import { publicJobsApi } from "./endpoints/publicJobs";
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
  publicJobs: publicJobsApi,
  upload: uploadApi,
  candidates: {
    getById: (profileId: string) => uploadApi.getProfileById(profileId),
  },
  jobDescriptions: jobDescriptionsApi,
  notifications: notificationsApi,
  scoring: scoringApi,
  chat: chatApi,
  shortlist: shortlistApi,
  outreach: outreachApi,
  interviewQuestions: interviewQuestionsApi,
  interviewTemplates: interviewTemplatesApi,
  interviewInvitations: interviewInvitationsApi,
  interviewPublic: interviewPublicApi,
  interviewReports: interviewReportsApi,
};
