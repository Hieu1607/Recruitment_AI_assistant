import React, { useEffect, useState } from "react";

import { ScoreBreakdownTable } from "../components/ScoreBreakdownTable";

type Candidate = {
  id: string;
  fullName: string;
};

type ScoreItem = {
  candidateId: string;
  totalScore: number;
  passedThreshold: boolean;
  rationale: string;
  componentScores: {
    criterionKey: string;
    weight: number;
    score: number;
    weightedScore: number;
    evidenceSummary?: string;
  }[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export function MatchingPage(): JSX.Element {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [jobDescriptionText, setJobDescriptionText] = useState("");
  const [scoringPromptTemplate, setScoringPromptTemplate] = useState(
    "Evaluate each candidate against the job description and score skills, education, and experience."
  );
  const [scoreThreshold, setScoreThreshold] = useState(60);
  const [results, setResults] = useState<ScoreItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void fetchCandidates();
  }, []);

  const fetchCandidates = async (): Promise<void> => {
    const response = await fetch(`${API_BASE}/v1/candidates`, { headers: { "X-Role": "recruiter" } });
    const data = (await response.json()) as Candidate[];
    if (!response.ok) {
      setError("Failed to load candidates");
      return;
    }
    setCandidates(data);
  };

  const toggleCandidate = (candidateId: string): void => {
    setSelectedIds((current) =>
      current.includes(candidateId) ? current.filter((id) => id !== candidateId) : [...current, candidateId]
    );
  };

  const runMatching = async (): Promise<void> => {
    if (selectedIds.length === 0) {
      setError("Select at least one candidate");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/v1/match-runs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Role": "recruiter",
        },
        body: JSON.stringify({
          jobDescriptionText,
          candidateIds: selectedIds,
          scoringPromptTemplate,
          scoreThreshold,
        }),
      });
      const data = (await response.json()) as { scores: ScoreItem[]; message?: string };
      if (!response.ok) {
        throw new Error(data.message ?? "Matching failed");
      }
      setResults(data.scores);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <header style={headerStyle}>
        <h2 style={{ margin: 0 }}>Batch Matching Workspace</h2>
        <p style={{ margin: "0.4rem 0 0", color: "#475569" }}>
          Provide one JD and one shared scoring prompt for a selected candidate set.
        </p>
      </header>

      <section style={cardStyle}>
        <label style={labelStyle}>Job description</label>
        <textarea
          style={textAreaStyle}
          rows={6}
          value={jobDescriptionText}
          onChange={(event) => setJobDescriptionText(event.target.value)}
          placeholder="Paste role responsibilities and requirements"
        />

        <label style={labelStyle}>Scoring prompt template</label>
        <textarea
          style={textAreaStyle}
          rows={4}
          value={scoringPromptTemplate}
          onChange={(event) => setScoringPromptTemplate(event.target.value)}
        />

        <label style={labelStyle}>Threshold ({scoreThreshold})</label>
        <input
          type="range"
          min={0}
          max={100}
          value={scoreThreshold}
          onChange={(event) => setScoreThreshold(Number(event.target.value))}
        />
      </section>

      <section style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>Candidate selection</h3>
        <div style={{ display: "grid", gap: "0.3rem", maxHeight: 220, overflow: "auto" }}>
          {candidates.map((candidate) => (
            <label key={candidate.id} style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
              <input
                type="checkbox"
                checked={selectedIds.includes(candidate.id)}
                onChange={() => toggleCandidate(candidate.id)}
              />
              <span>{candidate.fullName}</span>
            </label>
          ))}
        </div>

        <button disabled={loading} onClick={() => void runMatching()} style={buttonStyle}>
          Run match scoring
        </button>
        {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      </section>

      <section style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>Result list</h3>
        {results.length === 0 ? <p>No score list available yet.</p> : null}
        {results.map((result) => (
          <article key={result.candidateId} style={resultCardStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <strong>{result.candidateId}</strong>
              <span style={{ color: result.passedThreshold ? "#166534" : "#991b1b" }}>
                Total: {result.totalScore.toFixed(1)} ({result.passedThreshold ? "PASS" : "FAIL"})
              </span>
            </div>
            <p style={{ marginBottom: "0.6rem", color: "#475569" }}>{result.rationale}</p>
            <ScoreBreakdownTable components={result.componentScores} />
          </article>
        ))}
      </section>
    </section>
  );
}

const headerStyle: React.CSSProperties = {
  background: "linear-gradient(120deg, #f8faff 0%, #ecfeff 100%)",
  border: "1px solid #e5e7eb",
  borderRadius: 14,
  padding: "1rem",
};

const cardStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  padding: "1rem",
  display: "grid",
  gap: "0.6rem",
};

const labelStyle: React.CSSProperties = { fontWeight: 600, color: "#334155" };

const textAreaStyle: React.CSSProperties = {
  border: "1px solid #d1d5db",
  borderRadius: 10,
  padding: "0.6rem",
  resize: "vertical",
};

const resultCardStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  borderRadius: 10,
  padding: "0.8rem",
  marginBottom: "0.8rem",
};

const buttonStyle: React.CSSProperties = {
  border: "none",
  borderRadius: 10,
  background: "#0f766e",
  color: "#fff",
  padding: "0.55rem 0.9rem",
  cursor: "pointer",
};
