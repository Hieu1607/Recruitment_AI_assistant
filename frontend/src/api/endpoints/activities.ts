import { client } from "../client";
import type { ActivityListResponse } from "../types";

export const activitiesApi = {
  async list(params?: { job_id?: string; limit?: number }): Promise<ActivityListResponse> {
    const { data } = await client.get<ActivityListResponse>("/activities/", { params });
    return data;
  },
};
