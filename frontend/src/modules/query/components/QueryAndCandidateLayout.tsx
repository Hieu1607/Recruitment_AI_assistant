import React from "react";

type AskResult = {
  answer: string;
  matchedCount: number;
  matchedCandidateIds: string[];
  routingStrategy: string;
};

type Candidate = {
  id: string;
  fullName: string;
  currentJobTitle?: string | null;
  locationNormalized?: string | null;
};

type Props = {
  queryBox: JSX.Element;
  result: AskResult | null;
  candidates: Candidate[];
  selectedCandidateId: string | null;
  onSelectCandidate: (candidateId: string) => void;
};

export function QueryAndCandidateLayout({
  queryBox,
  result,
  candidates,
  selectedCandidateId,
  onSelectCandidate,
}: Props): JSX.Element {
  return (
    <section style={containerStyle}>
      <div style={chatPaneStyle}>{queryBox}</div>
      <div style={candidatePaneStyle}>
        <h3 style={{ marginTop: 0 }}>Matched candidates</h3>
        {!result ? <p style={mutedTextStyle}>Ask a question to populate this list.</p> : null}
        {result ? (
          <p style={mutedTextStyle}>
            {result.matchedCount} match(es) via {result.routingStrategy}
          </p>
        ) : null}
        <div style={{ display: "grid", gap: "0.5rem", maxHeight: 420, overflowY: "auto" }}>
          {candidates.map((candidate) => (
            <button
              key={candidate.id}
              style={{
                ...candidateCardStyle,
                borderColor: selectedCandidateId === candidate.id ? "#0f766e" : "#e5e7eb",
                background: selectedCandidateId === candidate.id ? "#f0fdfa" : "#ffffff",
              }}
              onClick={() => onSelectCandidate(candidate.id)}
            >
              <strong>{candidate.fullName}</strong>
              <span style={{ color: "#64748b", fontSize: "0.85rem" }}>
                {candidate.currentJobTitle ?? "Unknown role"}
              </span>
              <span style={{ color: "#64748b", fontSize: "0.82rem" }}>
                {candidate.locationNormalized ?? "Location N/A"}
              </span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

const containerStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1.2fr 1fr",
  gap: "1rem",
};

const chatPaneStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  background: "#ffffff",
  padding: "1rem",
};

const candidatePaneStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  background: "#ffffff",
  padding: "1rem",
  display: "grid",
  alignContent: "start",
};

const candidateCardStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  borderRadius: 10,
  padding: "0.6rem 0.8rem",
  display: "grid",
  gap: "0.15rem",
  textAlign: "left",
  cursor: "pointer",
};

const mutedTextStyle: React.CSSProperties = {
  marginTop: 0,
  color: "#64748b",
  fontSize: "0.9rem",
};
