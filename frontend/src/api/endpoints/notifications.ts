import { client } from "../client";
import type { NotificationListResponse, NotificationResponse } from "../types";

export const notificationsApi = {
  async list(params?: { limit?: number; unread_only?: boolean }): Promise<NotificationListResponse> {
    const { data } = await client.get<NotificationListResponse>("/notifications/", { params });
    return data;
  },

  async markRead(notificationId: string): Promise<NotificationResponse> {
    const { data } = await client.post<NotificationResponse>(`/notifications/${notificationId}/read`);
    return data;
  },

  async markAllRead(): Promise<{ updated_count: number }> {
    const { data } = await client.post<{ updated_count: number }>("/notifications/read-all");
    return data;
  },
};
