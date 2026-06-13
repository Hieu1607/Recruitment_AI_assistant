import { useAuthStore } from "./auth";

const TOKEN_KEY = "easyhr.token";
const LOGIN_PATH = "/login";
const AUTH_CALLBACK_PATH = "/auth/callback";

let isRedirectingForExpiredSession = false;

export function storeAccessToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function getAccessToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function clearAuthenticatedSession(): void {
  localStorage.removeItem(TOKEN_KEY);
  useAuthStore.getState().clearUser();
}

export function isAuthenticatedSession(): boolean {
  return getAccessToken() !== null;
}

export function handleExpiredSession(): void {
  clearAuthenticatedSession();

  if (typeof window === "undefined" || isRedirectingForExpiredSession) {
    return;
  }

  const { pathname, search, hash } = window.location;
  if (pathname === LOGIN_PATH || pathname === AUTH_CALLBACK_PATH) {
    return;
  }

  isRedirectingForExpiredSession = true;
  const redirectTarget = `${pathname}${search}${hash}`;
  window.location.replace(`/login?redirect=${encodeURIComponent(redirectTarget)}`);
}
