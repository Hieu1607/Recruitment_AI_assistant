import { useState, useCallback, useEffect } from 'react';
import { listResumes, batchParseResumes, deleteResume } from '../api/resumes';
import type { ResumeResponse, PaginatedResponse, BatchParseResponse } from '../types';

export function useResumes(initialLimit = 50, initialOffset = 0) {
  const [resumes, setResumes] = useState<ResumeResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [limit, setLimit] = useState(initialLimit);
  const [offset, setOffset] = useState(initialOffset);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchResumes = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data: PaginatedResponse<ResumeResponse> = await listResumes({ limit, offset });
      setResumes(data.items);
      setTotal(data.total);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to list resumes');
    } finally {
      setIsLoading(false);
    }
  }, [limit, offset]);

  useEffect(() => {
    fetchResumes();
  }, [fetchResumes]);

  const uploadBatch = async (files: File[], uploadedByUserId?: string): Promise<BatchParseResponse | null> => {
    setIsUploading(true);
    setError(null);
    try {
      const result = await batchParseResumes(files, uploadedByUserId);
      await fetchResumes(); // Refresh list after upload
      return result;
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to upload resumes');
      return null;
    } finally {
      setIsUploading(false);
    }
  };

  const removeResume = async (id: string, removeFile = false): Promise<boolean> => {
    setError(null);
    try {
      const result = await deleteResume(id, removeFile);
      if (result.deleted) {
        // remove from local list for fast UI update
        setResumes((prev) => prev.filter((r) => r.id !== id));
        setTotal((prev) => prev - 1);
        return true;
      }
      return false;
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to delete resume');
      return false;
    }
  };

  return {
    resumes,
    total,
    limit,
    offset,
    isLoading,
    isUploading,
    error,
    setLimit,
    setOffset,
    uploadBatch,
    removeResume,
    refreshResumes: fetchResumes,
    setError,
  };
}
