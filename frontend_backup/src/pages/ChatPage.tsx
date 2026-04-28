import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import { useToast } from '../components/Toast';
import { Send, Bot, User, AlertCircle, MessageSquarePlus, Users, X } from 'lucide-react';

export default function ChatPage() {
  const { toast } = useToast();
  const { messages, sessionId, candidatesInScope, isLoading, error, send, startNewChat, setError } =
    useChat();
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (error) toast(error, 'error');
  }, [error]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll to bottom whenever messages change or loading state changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async () => {
    const text = inputText.trim();
    if (!text || isLoading) return;
    setInputText('');
    await send(text);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    startNewChat();
    setInputText('');
    inputRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)] max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white shrink-0">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Recruiter Chat</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Ask questions about candidates using AI
          </p>
        </div>
        <div className="flex items-center gap-3">
          {candidatesInScope !== null && (
            <span className="flex items-center gap-1.5 text-sm text-indigo-700 bg-indigo-50 border border-indigo-200 px-3 py-1 rounded-full">
              <Users className="w-4 h-4" />
              {candidatesInScope} candidate{candidatesInScope !== 1 ? 's' : ''} in scope
            </span>
          )}
          <button
            onClick={handleNewChat}
            className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            title="Start a new chat session"
          >
            <MessageSquarePlus className="w-4 h-4" />
            New chat
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3 shrink-0">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 shrink-0" />
          <p className="text-sm text-red-700 flex-1">{error}</p>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 transition">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="p-4 bg-indigo-50 rounded-full mb-4">
              <Bot className="w-10 h-10 text-indigo-500" />
            </div>
            <h2 className="text-lg font-semibold text-gray-800 mb-2">
              How can I help you today?
            </h2>
            <p className="text-sm text-gray-500 max-w-sm">
              Ask me to find candidates, compare skills, summarise experience, or shortlist
              profiles. I have access to all uploaded resumes.
            </p>
            <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-md">
              {[
                'Who has experience with Python and machine learning?',
                'List candidates with 5+ years of experience',
                'Which candidates have a computer science degree?',
                'Compare top 3 candidates for a senior engineer role',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => {
                    setInputText(suggestion);
                    inputRef.current?.focus();
                  }}
                  className="text-left text-xs text-gray-600 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 hover:bg-gray-100 hover:border-gray-300 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-3 ${msg.role === 'human' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'ai' && (
                  <div className="shrink-0 w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center mt-1">
                    <Bot className="w-4 h-4 text-indigo-600" />
                  </div>
                )}
                <div
                  className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap break-words ${
                    msg.role === 'human'
                      ? 'bg-indigo-600 text-white rounded-br-sm'
                      : 'bg-white border border-gray-200 text-gray-800 rounded-bl-sm shadow-sm'
                  }`}
                >
                  {msg.content}
                </div>
                {msg.role === 'human' && (
                  <div className="shrink-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center mt-1">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}

            {/* AI typing indicator */}
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="shrink-0 w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center mt-1">
                  <Bot className="w-4 h-4 text-indigo-600" />
                </div>
                <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
                  <div className="flex gap-1 items-center h-5">
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Session info strip */}
      {sessionId && (
        <div className="px-6 pb-1 shrink-0">
          <p className="text-xs text-gray-400 text-center">
            Session: <span className="font-mono">{sessionId.slice(0, 8)}…</span>
            &nbsp;·&nbsp;Note: session history is lost on backend restart
          </p>
        </div>
      )}

      {/* Input area */}
      <div className="px-6 pb-6 pt-2 shrink-0 bg-white border-t border-gray-200">
        <div className="flex gap-3 items-end bg-gray-50 border border-gray-300 rounded-2xl px-4 py-3 focus-within:border-indigo-400 focus-within:ring-1 focus-within:ring-indigo-400 transition">
          <textarea
            ref={inputRef}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about candidates… (Enter to send, Shift+Enter for new line)"
            rows={1}
            disabled={isLoading}
            className="flex-1 bg-transparent text-sm text-gray-800 placeholder-gray-400 resize-none outline-none max-h-36 overflow-y-auto disabled:opacity-60"
            style={{ fieldSizing: 'content' } as React.CSSProperties}
          />
          <button
            onClick={handleSend}
            disabled={!inputText.trim() || isLoading}
            className="shrink-0 w-9 h-9 flex items-center justify-center rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            aria-label="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          AI responses are generated — verify important details before acting on them.
        </p>
      </div>
    </div>
  );
}
