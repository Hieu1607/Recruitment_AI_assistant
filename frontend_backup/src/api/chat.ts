import apiClient from './client';
import type { ChatResponse, ChatHistoryResponse } from '../types';

export async function sendMessage(
  message: string,
  sessionId?: string,
  candidateLimit?: number,
): Promise<ChatResponse> {
  const { data } = await apiClient.post<ChatResponse>('/chat/', {
    message,
    ...(sessionId && { session_id: sessionId }),
    ...(candidateLimit != null && { candidate_limit: candidateLimit }),
  });
  return data;
}

export async function getChatHistory(
  sessionId: string,
): Promise<ChatHistoryResponse> {
  const { data } = await apiClient.get<ChatHistoryResponse>(
    `/chat/${sessionId}`,
  );
  return data;
}

export async function deleteSession(
  sessionId: string,
): Promise<{ session_id: string; deleted: boolean }> {
  const { data } = await apiClient.delete<{
    session_id: string;
    deleted: boolean;
  }>(`/chat/${sessionId}`);
  return data;
}
