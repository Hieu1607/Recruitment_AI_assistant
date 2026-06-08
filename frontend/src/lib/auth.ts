import { create } from "zustand";

export interface CurrentUser {
  id: string;
  email: string;
  display_name: string;
  gmail_connected: boolean;
}

type CurrentUserUpdate = Omit<CurrentUser, "gmail_connected"> & Partial<Pick<CurrentUser, "gmail_connected">>;

const SELECTED_JOB_KEY = "recruit_ai_selected_job_id";

interface AuthState {
  user: CurrentUser | null;
  selectedJobId: string | null;
  setUser: (user: CurrentUserUpdate) => void;
  clearUser: () => void;
  setSelectedJobId: (jobId: string | null) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  selectedJobId: localStorage.getItem(SELECTED_JOB_KEY),
  setUser: (user) =>
    set((state) => ({
      user: {
        ...user,
        gmail_connected:
          typeof user.gmail_connected === "boolean"
            ? user.gmail_connected
            : (state.user?.gmail_connected ?? false),
      },
    })),
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
