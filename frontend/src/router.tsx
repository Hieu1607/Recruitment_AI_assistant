import { api } from "@/api";
import { AppShell } from "@/components/layout/AppShell";
import { useAuthStore } from "@/lib/auth";
import { routePatterns } from "@/routes";
import React from "react";
import { createBrowserRouter, redirect } from "react-router";

async function requireAuth({ request }: { request: Request }) {
  if (!api.auth.isAuthenticated()) {
    const url = new URL(request.url);
    return redirect(`/login?redirect=${encodeURIComponent(url.pathname)}`);
  }
  if (!useAuthStore.getState().user) {
    try {
      const user = await api.auth.me();
      useAuthStore.getState().setUser(user);
    } catch {
      api.auth.clearToken();
      const url = new URL(request.url);
      return redirect(`/login?redirect=${encodeURIComponent(url.pathname)}`);
    }
  }
  return null;
}

function redirectIfAuth() {
  if (api.auth.isAuthenticated()) {
    return redirect("/dashboard");
  }
  return null;
}

// Route-level lazy: each route is its own chunk
function lazy<T extends () => Promise<{ default: React.ComponentType }>>(loader: T) {
  return { lazy: async () => ({ Component: (await loader()).default }) };
}

export const router = createBrowserRouter([
  // Public routes (no shell)
  {
    path: routePatterns.landing,
    loader: redirectIfAuth,
    ...lazy(() => import("@/routes/landing")),
  },
  {
    path: routePatterns.login,
    loader: redirectIfAuth,
    ...lazy(() => import("@/routes/login")),
  },
  {
    path: routePatterns.authCallback,
    ...lazy(() => import("@/routes/auth-callback")),
  },
  // Authenticated routes (wrapped in AppShell)
  {
    Component: AppShell,
    loader: requireAuth,
    children: [
      { path: routePatterns.dashboard, ...lazy(() => import("@/routes/dashboard")) },
      { path: routePatterns.candidates, ...lazy(() => import("@/routes/candidates/list")) },
      { path: routePatterns.candidateDetail, ...lazy(() => import("@/routes/candidates/detail")) },
      { path: routePatterns.jobDescriptions, ...lazy(() => import("@/routes/job-descriptions/list")) },
      { path: routePatterns.jobDescriptionNew, ...lazy(() => import("@/routes/job-descriptions/edit")) },
      { path: routePatterns.jobDescriptionEdit, ...lazy(() => import("@/routes/job-descriptions/edit")) },
      { path: routePatterns.scoring, ...lazy(() => import("@/routes/scoring/setup")) },
      { path: routePatterns.scoringResults, ...lazy(() => import("@/routes/scoring/results")) },
      { path: routePatterns.chat, ...lazy(() => import("@/routes/chat")) },
      { path: routePatterns.chatSession, ...lazy(() => import("@/routes/chat")) },
      { path: routePatterns.shortlists, ...lazy(() => import("@/routes/shortlists/list")) },
      { path: routePatterns.shortlistCollection, ...lazy(() => import("@/routes/shortlists/collection")) },
      { path: routePatterns.outreach, ...lazy(() => import("@/routes/outreach")) },
      { path: routePatterns.interviewQuestions, ...lazy(() => import("@/routes/interview-questions/list")) },
      { path: routePatterns.interviewQuestionDetail, ...lazy(() => import("@/routes/interview-questions/detail")) },
      { path: routePatterns.settings, ...lazy(() => import("@/routes/settings")) },
      { path: "/dev/primitives", ...lazy(() => import("@/routes/dev/primitives")) },
    ],
  },
  {
    path: "*",
    ...lazy(() => import("@/routes/not-found")),
  },
]);
