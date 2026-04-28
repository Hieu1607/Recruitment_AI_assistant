import { QueryClient, QueryCache, MutationCache } from "@tanstack/react-query";
import { toast } from "sonner";
import { isApiError } from "./client";
import { ApiError } from "./errors";

/**
 * Global error handler for all TanStack Query operations.
 *
 * Rules:
 * - Only `ApiError` instances surface — unknown JS errors bubble to the React
 *   error boundary instead.
 * - Validation errors (422 / `kind === "validation"`) are skipped here so
 *   forms can render fieldErrors inline without duplicate toast notices.
 * - Network failures get a friendly generic message.
 * - All other ApiErrors show the normalized `detail` string.
 */
function notifyOnError(error: unknown): void {
  if (!isApiError(error)) return;
  if (error.kind === "validation") return;

  const message =
    error.kind === "network"
      ? "Can't reach the server. Check your connection."
      : error.detail;

  toast.error(message);
}

export const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: notifyOnError }),
  mutationCache: new MutationCache({ onError: notifyOnError }),
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        if (error instanceof ApiError) {
          // Do not retry on 4xx — only on network errors (status 0) or 5xx
          if (error.status >= 400 && error.status < 500) return false;
        }
        return failureCount < 2;
      },
    },
    mutations: {
      retry: false,
    },
  },
});
