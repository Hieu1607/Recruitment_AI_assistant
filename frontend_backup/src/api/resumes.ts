import apiClient from './client';
import type {
  BatchParseResponse,
  PaginatedResponse,
  ResumeResponse,
} from '../types';

export async function batchParseResumes(
  files: File[],
  uploadedByUserId?: string,
): Promise<BatchParseResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  if (uploadedByUserId) {
    formData.append('uploaded_by_user_id', uploadedByUserId);
  }
  const { data } = await apiClient.post<BatchParseResponse>(
    '/upload/batch-parse',
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return data;
}

export async function listResumes(
  params?: Record<string, unknown>,
): Promise<PaginatedResponse<ResumeResponse>> {
  const { data } = await apiClient.get<PaginatedResponse<ResumeResponse>>(
    '/upload/',
    { params },
  );
  return data;
}

export async function getResume(id: string): Promise<ResumeResponse> {
  const { data } = await apiClient.get<ResumeResponse>(`/upload/${id}`);
  return data;
}

export async function updateResume(
  id: string,
  payload: Partial<ResumeResponse>,
): Promise<ResumeResponse> {
  const { data } = await apiClient.patch<ResumeResponse>(
    `/upload/${id}`,
    payload,
  );
  return data;
}

export async function deleteResume(
  id: string,
  deleteFile = false,
): Promise<{ deleted: boolean; resume_id: string }> {
  const { data } = await apiClient.delete<{
    deleted: boolean;
    resume_id: string;
  }>(`/upload/${id}`, { params: { delete_file: deleteFile } });
  return data;
}
