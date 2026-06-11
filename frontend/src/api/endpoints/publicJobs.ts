import { client } from "../client";
import type { PublicJobResponse, PublicResumeUploadResponse } from "../types";

export const publicJobsApi = {
  async get(token: string): Promise<PublicJobResponse> {
    const { data } = await client.get<PublicJobResponse>(`/public/jobs/${token}`);
    return data;
  },

  async uploadResume(
    token: string,
    body: { fullName: string; email: string; file: File },
  ): Promise<PublicResumeUploadResponse> {
    const fd = new FormData();
    fd.append("full_name", body.fullName);
    fd.append("email", body.email);
    fd.append("file", body.file);

    const { data } = await client.post<PublicResumeUploadResponse>(
      `/public/jobs/${token}/resumes`,
      fd,
      {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 300_000,
      },
    );
    return data;
  },
};
