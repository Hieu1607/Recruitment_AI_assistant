import React, { useEffect, useMemo, useState } from "react";

import { QueryAndCandidateLayout } from "../components/QueryAndCandidateLayout";

type Candidate = {
  id: string;
  fullName: string;
  currentJobTitle?: string | null;
  locationNormalized?: string | null;
};

type AskResult = {
  answer: string;
  matchedCount: number;
  matchedCandidateIds: string[];
  routingStrategy: string;
};

type SessionRecord = {
  id: string;
  title: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export function QueryWorkspacePage(): JSX.Element {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResult | null>(null);
  const [allCandidates, setAllCandidates] = useState<Candidate[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void bootstrap();
  }, []);

  const bootstrap = async (): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      await Promise.all([fetchCandidates(), ensureSession(), fetchSessions()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to initialize query workspace");
    } finally {
      setLoading(false);
    }
  };

  const fetchCandidates = async (): Promise<void> => {
    const response = await fetch(`${API_BASE}/v1/candidates?limit=200`, {
      headers: { "X-Role": "recruiter" },
    });
    const data = (await response.json()) as Candidate[];
    if (!response.ok) {
      throw new Error("Failed to load candidates");
    }
    setAllCandidates(data);
  };

  const fetchSessions = async (): Promise<void> => {
    const response = await fetch(`${API_BASE}/v1/query-sessions`, {
      headers: { "X-Role": "recruiter" },
    });
    const data = (await response.json()) as SessionRecord[];
    if (!response.ok) {
      return;
    }
    setSessions(data);
  };

  const ensureSession = async (): Promise<void> => {
    if (activeSessionId) {
      return;
    }
    const response = await fetch(`${API_BASE}/v1/query-sessions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Role": "recruiter",
      },
      body: JSON.stringify({ title: "Candidate filtering workspace" }),
    });
    const data = (await response.json()) as SessionRecord;
    if (!response.ok) {
      throw new Error("Failed to create query session");
    }
    setActiveSessionId(data.id);
  };

  const askQuestion = async (): Promise<void> => {
    if (!activeSessionId) {
      setError("No active query session");
      return;
    }
    if (!question.trim()) {
      setError("Please type a question");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/v1/query-sessions/${activeSessionId}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Role": "recruiter",
        },
        body: JSON.stringify({ question }),
      });
      const data = (await response.json()) as AskResult & { message?: string };
      if (!response.ok) {
        throw new Error(data.message ?? "Query failed");
      }
      setResult(data);
      if (data.matchedCandidateIds.length > 0) {
        setSelectedCandidateId(data.matchedCandidateIds[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected query error");
    } finally {
      setLoading(false);
    }
  };

  const matchedCandidates = useMemo(() => {
    if (!result) {
      return [];
    }
    const matchedSet = new Set(result.matchedCandidateIds);
    return allCandidates.filter((candidate) => matchedSet.has(candidate.id));
  }, [allCandidates, result]);

  const selectedCandidate = matchedCandidates.find((candidate) => candidate.id === selectedCandidateId) ?? null;

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <header style={headerStyle}>
        <h2 style={{ margin: 0 }}>Natural-Language Query Workspace</h2>
        <p style={{ margin: "0.45rem 0 0", color: "#475569" }}>
          Ask filtering questions, inspect matched counts, and open candidate details.
        </p>
      </header>

      {sessions.length > 0 ? (
        <div style={sessionBarStyle}>
          <strong style={{ fontSize: "0.9rem" }}>Session</strong>
          <select
            value={activeSessionId ?? ""}
            onChange={(event) => setActiveSessionId(event.target.value)}
            style={inputStyle}
          >
            {sessions.map((sessionItem) => (
              <option key={sessionItem.id} value={sessionItem.id}>
                {sessionItem.title ?? sessionItem.id}
              </option>
            ))}
          </select>
        </div>
      ) : null}

      <QueryAndCandidateLayout
        queryBox={
          <>
            <label style={{ fontWeight: 600, color: "#334155" }}>Ask a recruiter query</label>
            <textarea
              rows={5}
              style={textAreaStyle}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Example: How many educated candidates with 4 years experience in Jakarta?"
            />
            <button style={buttonStyle} disabled={loading} onClick={() => void askQuestion()}>
              Ask
            </button>
            {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
            {result ? (
              <article style={answerCardStyle}>
                <p style={{ margin: "0 0 0.4rem" }}>{result.answer}</p>
                <small style={{ color: "#64748b" }}>
                  Strategy: {result.routingStrategy} | Matched: {result.matchedCount}
                </small>
              </article>
            ) : null}
          </>
        }
        result={result}
        candidates={matchedCandidates}
        selectedCandidateId={selectedCandidateId}
        onSelectCandidate={(candidateId) => setSelectedCandidateId(candidateId)}
      />

      {selectedCandidate ? (
        <section style={candidateDetailStyle}>
          <h3 style={{ marginTop: 0 }}>Candidate details</h3>
          <p style={{ marginBottom: "0.35rem" }}>
            <strong>Name:</strong> {selectedCandidate.fullName}
          </p>
          <p style={{ margin: "0.1rem 0" }}>
            <strong>Role:</strong> {selectedCandidate.currentJobTitle ?? "N/A"}
          </p>
          <p style={{ margin: "0.1rem 0" }}>
            <strong>Location:</strong> {selectedCandidate.locationNormalized ?? "N/A"}
          </p>
        </section>
      ) : null}
    </section>
  );
}

const headerStyle: React.CSSProperties = {
  background: "linear-gradient(120deg, #eef2ff 0%, #ecfeff 100%)",
  border: "1px solid #e5e7eb",
  borderRadius: 14,
  padding: "1rem",
};

const sessionBarStyle: React.CSSProperties = {
  display: "flex",
  gap: "0.7rem",
  alignItems: "center",
};

const inputStyle: React.CSSProperties = {
  border: "1px solid #d1d5db",
  borderRadius: 8,
  padding: "0.45rem 0.6rem",
  minWidth: 300,
};

const textAreaStyle: React.CSSProperties = {
  width: "100%",
  marginTop: "0.45rem",
  border: "1px solid #d1d5db",
  borderRadius: 10,
  padding: "0.65rem",
  resize: "vertical",
};

const buttonStyle: React.CSSProperties = {
  marginTop: "0.7rem",
  border: "none",
  borderRadius: 10,
  background: "#0f766e",
  color: "#ffffff",
  padding: "0.55rem 0.95rem",
  cursor: "pointer",
};

const answerCardStyle: React.CSSProperties = {
  marginTop: "0.9rem",
  border: "1px solid #dbeafe",
  background: "#f8fafc",
  borderRadius: 10,
  padding: "0.75rem",
};

const candidateDetailStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  background: "#ffffff",
  padding: "1rem",
};
