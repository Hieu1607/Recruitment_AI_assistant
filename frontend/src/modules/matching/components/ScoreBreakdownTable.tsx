import React from "react";

type ScoreComponent = {
  criterionKey: string;
  weight: number;
  score: number;
  weightedScore: number;
  evidenceSummary?: string;
};

type Props = {
  components: ScoreComponent[];
};

export function ScoreBreakdownTable({ components }: Props): JSX.Element {
  if (components.length === 0) {
    return <p style={{ margin: 0 }}>No component score breakdown available.</p>;
  }

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.92rem" }}>
      <thead>
        <tr style={{ background: "#f8fafc" }}>
          <th style={thStyle}>Criterion</th>
          <th style={thStyle}>Weight</th>
          <th style={thStyle}>Score</th>
          <th style={thStyle}>Weighted</th>
          <th style={thStyle}>Evidence</th>
        </tr>
      </thead>
      <tbody>
        {components.map((component) => (
          <tr key={`${component.criterionKey}-${component.weight}`}>
            <td style={tdStyle}>{component.criterionKey}</td>
            <td style={tdStyle}>{component.weight.toFixed(2)}</td>
            <td style={tdStyle}>{component.score.toFixed(1)}</td>
            <td style={tdStyle}>{component.weightedScore.toFixed(1)}</td>
            <td style={tdStyle}>{component.evidenceSummary ?? "-"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const thStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  padding: "0.45rem",
  textAlign: "left",
};

const tdStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  padding: "0.45rem",
  verticalAlign: "top",
};
