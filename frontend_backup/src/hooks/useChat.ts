import { useState, useCallback } from 'react';
import { sendMessage, deleteSession } from '../api/chat';
import type { ChatMessage } from '../types';

export interface UseChatReturn {
  messages: ChatMessage[];
  sessionId: string | null;
  candidatesInScope: number | null;
  isLoading: boolean;
  error: string | null;
  send: (text: string) => Promise<void>;
  startNewChat: () => void;
  setError: (error: string | null) => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [candidatesInScope, setCandidatesInScope] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = useCallback(async (text: string) => {
    if (!text.trim()) return;

    const userMessage: ChatMessage = { role: 'human', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendMessage(text, sessionId ?? undefined);

      // Store the session ID from first response (or keep updating it)
      setSessionId(response.session_id);
      setCandidatesInScope(response.candidates_in_scope);

      const aiMessage: ChatMessage = { role: 'ai', content: response.answer };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err: any) {
      const status = err.response?.status;
      const detail = err.response?.data?.detail || err.message || 'Failed to send message';

      // If the stored session is stale (backend was restarted), auto-reset and retry
      if (status === 404 && sessionId !== null) {
        setSessionId(null);
        setCandidatesInScope(null);
        // Remove the optimistically-added user message and surface error to let user retry
        setMessages((prev) => prev.slice(0, -1));
        setError('Session expired (backend may have restarted). Starting a fresh session — please resend your message.');
      } else {
        setError(detail);
        // Remove the optimistically-added user message on failure
        setMessages((prev) => prev.slice(0, -1));
      }
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const startNewChat = useCallback(() => {
    // Try to clean up the session on the backend, but don't block on it
    if (sessionId) {
      deleteSession(sessionId).catch(() => {
        // Ignore — session may already be gone (backend restart)
      });
    }
    setMessages([]);
    setSessionId(null);
    setCandidatesInScope(null);
    setError(null);
  }, [sessionId]);

  return {
    messages,
    sessionId,
    candidatesInScope,
    isLoading,
    error,
    send,
    startNewChat,
    setError,
  };
}
