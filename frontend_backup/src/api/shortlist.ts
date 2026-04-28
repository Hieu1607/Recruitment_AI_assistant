import apiClient from './client';
import type {
  SessionResponse,
  TurnResponse,
  CollectionResponse,
  ShortlistItemResponse,
} from '../types';

// --- Sessions ---

export async function createSession(
  userId: string,
  title?: string,
): Promise<SessionResponse> {
  const { data } = await apiClient.post<SessionResponse>(
    '/shortlist/sessions/',
    { user_id: userId, ...(title && { session_title: title }) },
  );
  return data;
}

export async function listSessions(
  userId: string,
  params?: Record<string, unknown>,
): Promise<SessionResponse[]> {
  const { data } = await apiClient.get<SessionResponse[]>(
    '/shortlist/sessions/',
    { params: { user_id: userId, ...params } },
  );
  return data;
}

export async function getSession(id: string): Promise<SessionResponse> {
  const { data } = await apiClient.get<SessionResponse>(
    `/shortlist/sessions/${id}`,
  );
  return data;
}

export async function updateSession(
  id: string,
  title: string,
): Promise<SessionResponse> {
  const { data } = await apiClient.patch<SessionResponse>(
    `/shortlist/sessions/${id}`,
    { session_title: title },
  );
  return data;
}

export async function deleteSession(id: string): Promise<void> {
  await apiClient.delete(`/shortlist/sessions/${id}`);
}

// --- Turns ---

export async function createTurn(
  sessionId: string,
  payload: { user_question: string },
): Promise<TurnResponse> {
  const { data } = await apiClient.post<TurnResponse>(
    `/shortlist/sessions/${sessionId}/turns/`,
    payload,
  );
  return data;
}

export async function listTurns(
  sessionId: string,
  params?: Record<string, unknown>,
): Promise<TurnResponse[]> {
  const { data } = await apiClient.get<TurnResponse[]>(
    `/shortlist/sessions/${sessionId}/turns/`,
    { params },
  );
  return data;
}

export async function getTurn(id: string): Promise<TurnResponse> {
  const { data } = await apiClient.get<TurnResponse>(
    `/shortlist/turns/${id}`,
  );
  return data;
}

export async function deleteTurn(id: string): Promise<void> {
  await apiClient.delete(`/shortlist/turns/${id}`);
}

// --- Collections ---

export async function createCollection(payload: {
  name: string;
  created_by_user_id: string;
  source_query_turn_id?: string;
}): Promise<CollectionResponse> {
  const { data } = await apiClient.post<CollectionResponse>(
    '/shortlist/collections/',
    payload,
  );
  return data;
}

export async function listCollections(
  userId: string,
  params?: Record<string, unknown>,
): Promise<CollectionResponse[]> {
  const { data } = await apiClient.get<CollectionResponse[]>(
    '/shortlist/collections/',
    { params: { user_id: userId, ...params } },
  );
  return data;
}

export async function getCollection(id: string): Promise<CollectionResponse> {
  const { data } = await apiClient.get<CollectionResponse>(
    `/shortlist/collections/${id}`,
  );
  return data;
}

export async function renameCollection(
  id: string,
  name: string,
): Promise<CollectionResponse> {
  const { data } = await apiClient.patch<CollectionResponse>(
    `/shortlist/collections/${id}`,
    { name },
  );
  return data;
}

export async function deleteCollection(id: string): Promise<void> {
  await apiClient.delete(`/shortlist/collections/${id}`);
}

// --- Items ---

export async function addItem(
  collectionId: string,
  candidateProfileId: string,
): Promise<ShortlistItemResponse> {
  const { data } = await apiClient.post<ShortlistItemResponse>(
    `/shortlist/collections/${collectionId}/items/`,
    { candidate_profile_id: candidateProfileId },
  );
  return data;
}

export async function listItems(
  collectionId: string,
  params?: Record<string, unknown>,
): Promise<ShortlistItemResponse[]> {
  const { data } = await apiClient.get<ShortlistItemResponse[]>(
    `/shortlist/collections/${collectionId}/items/`,
    { params },
  );
  return data;
}

export async function removeItem(
  collectionId: string,
  candidateId: string,
): Promise<void> {
  await apiClient.delete(
    `/shortlist/collections/${collectionId}/items/${candidateId}`,
  );
}
