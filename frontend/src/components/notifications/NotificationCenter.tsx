import { api, queryClient, type NotificationResponse } from "@/api";
import { cn } from "@/lib/cn";
import {
  isNotificationTypeEnabled,
  shouldShowRealtimeNotification,
  useNotificationPreferences,
} from "@/lib/notification-preferences";
import { routes } from "@/routes";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Bell, CheckCheck, Circle, Clock, FileText, Mic, Sparkles } from "lucide-react";
import { useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router";
import { toast } from "sonner";

function relativeTime(value: string) {
  const diff = Date.now() - new Date(value).getTime();
  const minutes = Math.max(0, Math.floor(diff / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function notificationIcon(type: string) {
  if (type === "candidate_applied") return FileText;
  if (type === "interview_completed") return Mic;
  if (type === "scoring_completed") return Sparkles;
  return Bell;
}

export function NotificationCenter() {
  const navigate = useNavigate();
  const initializedRef = useRef(false);
  const seenIdsRef = useRef<Set<string>>(new Set());
  const { preferences } = useNotificationPreferences();
  const { data } = useQuery({
    queryKey: ["notifications"],
    queryFn: () => api.notifications.list({ limit: 20 }),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });

  const markReadMutation = useMutation({
    mutationFn: (id: string) => api.notifications.markRead(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["notifications"] }),
  });

  const markAllReadMutation = useMutation({
    mutationFn: () => api.notifications.markAllRead(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["notifications"] }),
  });

  const notifications = (data?.items ?? []).filter((item) =>
    isNotificationTypeEnabled(item.notification_type, preferences),
  );
  const unreadCount = notifications.filter((item) => item.read_at === null).length;

  useEffect(() => {
    if (!data) return;

    if (!initializedRef.current) {
      seenIdsRef.current = new Set(data.items.map((item) => item.id));
      initializedRef.current = true;
      return;
    }

    const fresh = data.items
      .filter(
        (item) =>
          !seenIdsRef.current.has(item.id) &&
          item.read_at === null &&
          shouldShowRealtimeNotification(item.notification_type, preferences),
      )
      .reverse();
    for (const item of fresh) {
      toast(item.title, {
        description: item.body,
        action: item.target_url
          ? {
              label: "Open",
              onClick: () => navigate(item.target_url!),
            }
          : undefined,
      });
      seenIdsRef.current.add(item.id);
    }
  }, [data, navigate, preferences]);

  const openNotification = (item: NotificationResponse) => {
    if (!item.read_at) markReadMutation.mutate(item.id);
    if (item.target_url) navigate(item.target_url);
  };

  return (
    <details className="relative group">
      <summary className="list-none cursor-pointer">
        <span
          aria-label="Notifications"
          className="relative size-9 rounded-md flex items-center justify-center text-fg-muted hover:text-fg hover:bg-[color:var(--hairline)] transition-colors"
        >
          <Bell size={16} strokeWidth={1.5} />
          {unreadCount > 0 && (
            <span className="absolute right-1.5 top-1.5 min-w-4 rounded-full bg-danger px-1 text-center font-mono text-[9px] leading-4 text-white">
              {unreadCount > 9 ? "9+" : unreadCount}
            </span>
          )}
        </span>
      </summary>

      <div
        className="absolute right-0 top-full mt-2 w-80 overflow-hidden rounded-lg bg-bg-elevated hairline shadow-lg z-50"
        role="menu"
      >
        <div className="flex items-center justify-between gap-3 px-4 py-3 hairline-b">
          <div>
            <p className="font-sans text-sm font-medium text-fg">Notifications</p>
            <p className="text-xs text-fg-muted">{unreadCount} unread</p>
          </div>
          <button
            type="button"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-fg disabled:opacity-40"
            aria-label="Mark all as read"
            disabled={unreadCount === 0 || markAllReadMutation.isPending}
            onClick={() => markAllReadMutation.mutate()}
          >
            <CheckCheck size={15} strokeWidth={1.75} />
          </button>
        </div>

        <div className="max-h-[360px] overflow-y-auto py-1">
          {notifications.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <Clock size={20} strokeWidth={1.5} className="mx-auto text-fg-subtle" />
              <p className="mt-2 text-sm text-fg-muted">No notifications yet.</p>
            </div>
          ) : (
            notifications.map((item) => {
              const Icon = notificationIcon(item.notification_type);
              const unread = item.read_at === null;
              const content = (
                <>
                  <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-[color:var(--hairline)] text-fg-muted">
                    <Icon size={15} strokeWidth={1.75} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5">
                      {unread && <Circle size={7} fill="currentColor" className="text-accent" />}
                      <span className="truncate text-sm font-medium text-fg">{item.title}</span>
                    </span>
                    <span className="mt-0.5 line-clamp-2 text-xs leading-5 text-fg-muted">{item.body}</span>
                    <span className="mt-1 block font-mono text-[10px] text-fg-subtle">
                      {relativeTime(item.created_at)}
                    </span>
                  </span>
                </>
              );

              return item.target_url ? (
                <button
                  key={item.id}
                  type="button"
                  className={cn(
                    "flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-[color:var(--hairline)]",
                    unread && "bg-[rgba(74,124,89,0.06)]",
                  )}
                  onClick={() => openNotification(item)}
                >
                  {content}
                </button>
              ) : (
                <div key={item.id} className="flex items-start gap-3 px-4 py-3">
                  {content}
                </div>
              );
            })
          )}
        </div>

        <div className="hairline-t px-4 py-2">
          <Link to={routes.settings} className="text-xs font-medium text-fg-muted hover:text-fg">
            Notification settings
          </Link>
        </div>
      </div>
    </details>
  );
}
