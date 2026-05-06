import { client } from "../client";
import type {
  ChatRequest,
  ChatResponse,
  ChatHistoryResponse,
  ChatDeleteResponse,
} from "../types";

export const chatApi = {
  /**
   * Send a message to the recruiter chatbot and receive an answer.
   * POST /chat/
   * Omit session_id to start a new session; pass it to maintain context.
   */
  async send(body: ChatRequest): Promise<ChatResponse> {
    const { data } = await client.post<ChatResponse>("/chat/", body);
    return data;
  },

  /**
   * Retrieve message history for a chat session.
   * GET /chat/{session_id}
   */
  async getHistory(sessionId: string): Promise<ChatHistoryResponse> {
    const { data } = await client.get<ChatHistoryResponse>(
      `/chat/${sessionId}`,
    );
    return data;
  },

  /**
   * Clear and delete a chat session.
   * DELETE /chat/{session_id}
   * Returns 200 with { session_id, deleted }.
   */
  async deleteSession(sessionId: string): Promise<ChatDeleteResponse> {
    const { data } = await client.delete<ChatDeleteResponse>(
      `/chat/${sessionId}`,
    );
    return data;
  },
};
