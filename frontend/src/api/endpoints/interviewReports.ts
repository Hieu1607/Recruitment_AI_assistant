import { client } from "../client";
import type { InterviewReportResponse } from "../types";

export const interviewReportsApi = {
  async get(interviewSessionId: string): Promise<InterviewReportResponse> {
    const { data } = await client.get<InterviewReportResponse>(
      `/interview-reports/${interviewSessionId}`,
    );
    return data;
  },
};
