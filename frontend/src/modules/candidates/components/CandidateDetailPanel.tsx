import React from "react";

type Trace = {
  id: string;
  sourcePage: number;
  sourceBBox: Record<string, number>;
  sourceTextSnippet: string;
  mappedFieldName: string;
  confidenceScore: number | null;
};

type Candidate = {
  id: string;
  fullName: string;
  email: string | null;
  phone: string | null;
  locationNormalized: string | null;
  educated: boolean;
  everStudiedAbroad: boolean;
  profileStatus: string;
};

type Props = {
  candidate: Candidate | null;
  traces: Trace[];
};

export function CandidateDetailPanel({ candidate, traces }: Props): JSX.Element {
  if (!candidate) {
    return (
      <section style={panelStyle}>
        <h3 style={{ marginTop: 0 }}>Candidate details</h3>
        <p>Select a candidate from the list to inspect extracted source traces.</p>
      </section>
    );
  }

  return (
    <section style={panelStyle}>
      <h3 style={{ marginTop: 0 }}>{candidate.fullName}</h3>
      <p style={{ margin: "0.2rem 0" }}>
        <strong>Status:</strong> {candidate.profileStatus}
      </p>
      <p style={{ margin: "0.2rem 0" }}>
        <strong>Email:</strong> {candidate.email ?? "-"}
      </p>
      <p style={{ margin: "0.2rem 0" }}>
        <strong>Phone:</strong> {candidate.phone ?? "-"}
      </p>
      <p style={{ margin: "0.2rem 0" }}>
        <strong>Location:</strong> {candidate.locationNormalized ?? "-"}
      </p>

      <h4 style={{ marginBottom: "0.4rem" }}>Extraction traces</h4>
      {traces.length === 0 ? (
        <p>No trace records found for this candidate.</p>
      ) : (
        <div style={{ maxHeight: 420, overflow: "auto", borderTop: "1px solid #d6d6d6", paddingTop: "0.5rem" }}>
          {traces.map((trace) => (
            <article key={trace.id} style={{ marginBottom: "0.8rem", paddingBottom: "0.8rem", borderBottom: "1px solid #efefef" }}>
              <div style={{ fontSize: "0.85rem", color: "#4b5563" }}>
                Page {trace.sourcePage} | Field: {trace.mappedFieldName} | Confidence: {trace.confidenceScore ?? "n/a"}
              </div>
              <div style={{ whiteSpace: "pre-wrap", marginTop: "0.3rem" }}>{trace.sourceTextSnippet}</div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

const panelStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  padding: "1rem",
  boxShadow: "0 2px 12px rgba(15, 23, 42, 0.06)",
};
