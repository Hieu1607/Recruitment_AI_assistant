import { useState, useCallback, useEffect } from 'react';
import {
  listJobDescriptions,
  createJobDescription,
  updateJobDescription,
  deleteJobDescription,
} from '../api/jobDescriptions';
import type { JobDescriptionResponse, PaginatedResponse } from '../types';

export function useJobDescriptions(initialLimit = 50, initialOffset = 0) {
  const [jobDescriptions, setJobDescriptions] = useState<JobDescriptionResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [limit, setLimit] = useState(initialLimit);
  const [offset, setOffset] = useState(initialOffset);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchJobDescriptions = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data: PaginatedResponse<JobDescriptionResponse> = await listJobDescriptions({
        limit,
        offset,
      });
      setJobDescriptions(data.items);
      setTotal(data.total);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to list job descriptions');
    } finally {
      setIsLoading(false);
    }
  }, [limit, offset]);

  useEffect(() => {
    fetchJobDescriptions();
  }, [fetchJobDescriptions]);

  const addJobDescription = async (payload: {
    jd_text: string;
    created_by_user_id: string;
    title?: string;
  }): Promise<JobDescriptionResponse | null> => {
    setError(null);
    try {
      const result = await createJobDescription(payload);
      await fetchJobDescriptions();
      return result;
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to create job description');
      return null;
    }
  };

  const editJobDescription = async (
    id: string,
    payload: Partial<Pick<JobDescriptionResponse, 'title' | 'jd_text' | 'is_active'>>,
  ): Promise<JobDescriptionResponse | null> => {
    setError(null);
    try {
      const result = await updateJobDescription(id, payload);
      setJobDescriptions((prev) =>
        prev.map((jd) => (jd.id === id ? result : jd)),
      );
      return result;
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to update job description');
      return null;
    }
  };

  const removeJobDescription = async (id: string): Promise<boolean> => {
    setError(null);
    try {
      const result = await deleteJobDescription(id);
      if (result.deleted) {
        setJobDescriptions((prev) => prev.filter((jd) => jd.id !== id));
        setTotal((prev) => prev - 1);
        return true;
      }
      return false;
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to delete job description');
      return false;
    }
  };

  return {
    jobDescriptions,
    total,
    limit,
    offset,
    isLoading,
    error,
    setLimit,
    setOffset,
    setError,
    addJobDescription,
    editJobDescription,
    removeJobDescription,
    refreshJobDescriptions: fetchJobDescriptions,
  };
}
