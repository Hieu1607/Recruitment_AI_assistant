import React from "react";

type InterviewQuestion = {
  prompt: string;
  category?: string;
  difficulty?: string;
};

type Props = {
  questions: InterviewQuestion[];
};

export function InterviewQuestionPanel({ questions }: Props): JSX.Element {
  return (
    <section style={panelStyle}>
      <h3 style={{ marginTop: 0 }}>Interview questions</h3>
      {questions.length === 0 ? <p>No questions generated yet.</p> : null}
      <ol style={{ margin: 0, paddingLeft: "1.1rem", display: "grid", gap: "0.55rem" }}>
        {questions.map((question, index) => (
          <li key={`${question.prompt}-${index}`}>
            <p style={{ margin: 0 }}>{question.prompt}</p>
            <small style={{ color: "#64748b" }}>
              {question.category ?? "general"} | {question.difficulty ?? "n/a"}
            </small>
          </li>
        ))}
      </ol>
    </section>
  );
}

const panelStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  background: "#ffffff",
  padding: "1rem",
};
