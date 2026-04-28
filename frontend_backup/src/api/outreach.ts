import apiClient from './client';
import type { OutreachResponse, PaginatedResponse } from '../types';

export async function createOutreach(payload: {
  candidate_profile_id: string;
  created_by_user_id: string;
  content_source: 'ai_draft' | 'template';
  subject: string;
  body: string;
}): Promise<OutreachResponse> {
  const { data } = await apiClient.post<OutreachResponse>(
    '/outreach/',
    payload,
  );
  return data;
}

export async function listOutreach(
  params?: Record<string, unknown>,
): Promise<PaginatedResponse<OutreachResponse>> {
  const { data } = await apiClient.get<PaginatedResponse<OutreachResponse>>(
    '/outreach/',
    { params },
  );
  return data;
}

export async function getOutreach(id: string): Promise<OutreachResponse> {
  const { data } = await apiClient.get<OutreachResponse>(`/outreach/${id}`);
  return data;
}

export async function updateOutreach(
  id: string,
  payload: Partial<Pick<OutreachResponse, 'subject' | 'body' | 'sent_status'>>,
): Promise<OutreachResponse> {
  const { data } = await apiClient.patch<OutreachResponse>(
    `/outreach/${id}`,
    payload,
  );
  return data;
}

export async function deleteOutreach(id: string): Promise<void> {
  await apiClient.delete(`/outreach/${id}`);
}
