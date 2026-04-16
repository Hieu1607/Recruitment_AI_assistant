import apiClient from './client';
import type {
  JobDescriptionResponse,
  PaginatedResponse,
} from '../types';

export async function createJobDescription(payload: {
  jd_text: string;
  created_by_user_id: string;
  title?: string;
}): Promise<JobDescriptionResponse> {
  const { data } = await apiClient.post<JobDescriptionResponse>(
    '/job-descriptions/',
    payload,
  );
  return data;
}

export async function listJobDescriptions(
  params?: Record<string, unknown>,
): Promise<PaginatedResponse<JobDescriptionResponse>> {
  const { data } = await apiClient.get<
    PaginatedResponse<JobDescriptionResponse>
  >('/job-descriptions/', { params });
  return data;
}

export async function getJobDescription(
  id: string,
): Promise<JobDescriptionResponse> {
  const { data } = await apiClient.get<JobDescriptionResponse>(
    `/job-descriptions/${id}`,
  );
  return data;
}

export async function updateJobDescription(
  id: string,
  payload: Partial<JobDescriptionResponse>,
): Promise<JobDescriptionResponse> {
  const { data } = await apiClient.patch<JobDescriptionResponse>(
    `/job-descriptions/${id}`,
    payload,
  );
  return data;
}

export async function deleteJobDescription(
  id: string,
): Promise<{ deleted: boolean; job_description_id: string }> {
  const { data } = await apiClient.delete<{
    deleted: boolean;
    job_description_id: string;
  }>(`/job-descriptions/${id}`);
  return data;
}
