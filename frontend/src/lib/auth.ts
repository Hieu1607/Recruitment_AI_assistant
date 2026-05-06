import { create } from "zustand";

export interface CurrentUser {
  id: string;
  email: string;
  display_name: string;
}

const SELECTED_JOB_KEY = "recruit_ai_selected_job_id";

interface AuthState {
  user: CurrentUser | null;
  selectedJobId: string | null;
  setUser: (user: CurrentUser) => void;
  clearUser: () => void;
  setSelectedJobId: (jobId: string | null) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  selectedJobId: localStorage.getItem(SELECTED_JOB_KEY),
  setUser: (user) => set({ user }),
  clearUser: () => set({ user: null }),
  setSelectedJobId: (jobId) => {
    if (jobId) localStorage.setItem(SELECTED_JOB_KEY, jobId);
    else localStorage.removeItem(SELECTED_JOB_KEY);
    set({ selectedJobId: jobId });
  },
}));

export function useUserId(): string | null {
  return useAuthStore((s) => s.user?.id ?? null);
}

export function useSelectedJobId(): string | null {
  return useAuthStore((s) => s.selectedJobId);
}
