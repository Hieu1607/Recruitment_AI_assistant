import { useCallback, useEffect, useState } from "react";

export const NOTIFICATION_PREFERENCES_KEY = "easyhr.notification-preferences";

export type NotificationPreferenceKey =
  | "candidate_applied"
  | "interview_completed"
  | "scoring_completed"
  | "realtime_toasts";

export type NotificationPreferences = Record<NotificationPreferenceKey, boolean>;

export const DEFAULT_NOTIFICATION_PREFERENCES: NotificationPreferences = {
  candidate_applied: true,
  interview_completed: true,
  scoring_completed: true,
  realtime_toasts: true,
};

export function readNotificationPreferences(): NotificationPreferences {
  try {
    const raw = localStorage.getItem(NOTIFICATION_PREFERENCES_KEY);
    if (!raw) return DEFAULT_NOTIFICATION_PREFERENCES;
    const parsed = JSON.parse(raw) as Partial<Record<NotificationPreferenceKey, unknown>>;
    return {
      candidate_applied:
        typeof parsed.candidate_applied === "boolean"
          ? parsed.candidate_applied
          : DEFAULT_NOTIFICATION_PREFERENCES.candidate_applied,
      interview_completed:
        typeof parsed.interview_completed === "boolean"
          ? parsed.interview_completed
          : DEFAULT_NOTIFICATION_PREFERENCES.interview_completed,
      scoring_completed:
        typeof parsed.scoring_completed === "boolean"
          ? parsed.scoring_completed
          : DEFAULT_NOTIFICATION_PREFERENCES.scoring_completed,
      realtime_toasts:
        typeof parsed.realtime_toasts === "boolean"
          ? parsed.realtime_toasts
          : DEFAULT_NOTIFICATION_PREFERENCES.realtime_toasts,
    };
  } catch {
    return DEFAULT_NOTIFICATION_PREFERENCES;
  }
}

export function writeNotificationPreferences(preferences: NotificationPreferences) {
  localStorage.setItem(NOTIFICATION_PREFERENCES_KEY, JSON.stringify(preferences));
  window.dispatchEvent(new Event("easyhr:notification-preferences-changed"));
}

export function isNotificationTypeEnabled(type: string, preferences: NotificationPreferences) {
  if (type === "candidate_applied") return preferences.candidate_applied;
  if (type === "interview_completed") return preferences.interview_completed;
  if (type === "scoring_completed") return preferences.scoring_completed;
  return true;
}

export function shouldShowRealtimeNotification(type: string, preferences: NotificationPreferences) {
  return preferences.realtime_toasts && isNotificationTypeEnabled(type, preferences);
}

export function useNotificationPreferences() {
  const [preferences, setPreferences] = useState(readNotificationPreferences);

  useEffect(() => {
    const sync = () => setPreferences(readNotificationPreferences());
    window.addEventListener("storage", sync);
    window.addEventListener("easyhr:notification-preferences-changed", sync);
    return () => {
      window.removeEventListener("storage", sync);
      window.removeEventListener("easyhr:notification-preferences-changed", sync);
    };
  }, []);

  const setPreference = useCallback(
    (key: NotificationPreferenceKey, value: boolean) => {
      const next = { ...preferences, [key]: value };
      setPreferences(next);
      writeNotificationPreferences(next);
    },
    [preferences],
  );

  return { preferences, setPreference };
}
