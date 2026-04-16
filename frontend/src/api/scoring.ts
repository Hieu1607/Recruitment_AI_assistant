import apiClient from './client';
import type { ScoreResponse } from '../types';

export async function scoreCandidates(payload: {
  job_description_id: string;
  initiated_by_user_id: string;
  score_threshold?: number;
  candidate_profile_ids?: string[];
  section_weights?: Record<string, number>;
  batch_size?: number;
}): Promise<ScoreResponse> {
  const { data } = await apiClient.post<ScoreResponse>('/score/', payload);
  return data;
}
