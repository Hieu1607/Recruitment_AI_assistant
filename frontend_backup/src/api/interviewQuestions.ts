import apiClient from './client';
import type { QuestionSetResponse, PaginatedResponse } from '../types';

export async function createQuestionSet(payload: {
  candidate_profile_id: string;
  job_description_id: string;
  generated_by_user_id: string;
  question_payload: Record<string, unknown>;
}): Promise<QuestionSetResponse> {
  const { data } = await apiClient.post<QuestionSetResponse>(
    '/interview-questions/',
    payload,
  );
  return data;
}

export async function listQuestionSets(
  params?: Record<string, unknown>,
): Promise<PaginatedResponse<QuestionSetResponse>> {
  const { data } = await apiClient.get<
    PaginatedResponse<QuestionSetResponse>
  >('/interview-questions/', { params });
  return data;
}

export async function getQuestionSet(
  id: string,
): Promise<QuestionSetResponse> {
  const { data } = await apiClient.get<QuestionSetResponse>(
    `/interview-questions/${id}`,
  );
  return data;
}

export async function updateQuestionSet(
  id: string,
  questionPayload: Record<string, unknown>,
): Promise<QuestionSetResponse> {
  const { data } = await apiClient.patch<QuestionSetResponse>(
    `/interview-questions/${id}`,
    { question_payload: questionPayload },
  );
  return data;
}

export async function deleteQuestionSet(id: string): Promise<void> {
  await apiClient.delete(`/interview-questions/${id}`);
}
