import React, { useEffect, useState } from "react";

import { InterviewQuestionPanel } from "../components/InterviewQuestionPanel";

type Candidate = {
  id: string;
  fullName: string;
};

type Shortlist = {
  id: string;
  name: string;
  candidateIds: string[];
};

type InterviewQuestion = {
  prompt: string;
  category?: string;
  difficulty?: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export function EngagementPage(): JSX.Element {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<string[]>([]);
  const [shortlistName, setShortlistName] = useState("High Potential Batch");
  const [shortlists, setShortlists] = useState<Shortlist[]>([]);
  const [draftMessageId, setDraftMessageId] = useState<string | null>(null);
  const [subjectPreview, setSubjectPreview] = useState<string>("");
  const [sentStatus, setSentStatus] = useState<string>("not_sent");
  const [jobDescriptionId, setJobDescriptionId] = useState("");
  const [questions, setQuestions] = useState<InterviewQuestion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void initialize();
  }, []);

  const initialize = async (): Promise<void> => {
    try {
      await Promise.all([fetchCandidates(), fetchShortlists()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to initialize engagement workspace");
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
    setCandidates(data);
  };

  const fetchShortlists = async (): Promise<void> => {
    const response = await fetch(`${API_BASE}/v1/shortlists`, {
      headers: { "X-Role": "recruiter" },
    });
    const data = (await response.json()) as Shortlist[];
    if (!response.ok) {
      return;
    }
    setShortlists(data);
  };

  const toggleCandidate = (candidateId: string): void => {
    setSelectedCandidateIds((current) =>
      current.includes(candidateId) ? current.filter((id) => id !== candidateId) : [...current, candidateId]
    );
  };

  const createShortlist = async (): Promise<void> => {
    if (selectedCandidateIds.length === 0) {
      setError("Select candidates before creating a shortlist");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/v1/shortlists`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Role": "recruiter",
        },
        body: JSON.stringify({
          name: shortlistName,
          candidateIds: selectedCandidateIds,
        }),
      });
      const data = (await response.json()) as Shortlist & { message?: string };
      if (!response.ok) {
        throw new Error(data.message ?? "Failed to create shortlist");
      }
      setShortlists((current) => [data, ...current]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected shortlist error");
    } finally {
      setLoading(false);
    }
  };

  const createDraft = async (): Promise<void> => {
    if (selectedCandidateIds.length === 0) {
      setError("Select one candidate to draft outreach");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/v1/outreach/drafts`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Role": "recruiter",
        },
        body: JSON.stringify({
          candidateId: selectedCandidateIds[0],
          sourceType: "ai_draft",
          intent: "introduce role and gather availability",
        }),
      });
      const data = (await response.json()) as {
        id: string;
        subject: string;
        sentStatus: string;
        message?: string;
      };
      if (!response.ok) {
        throw new Error(data.message ?? "Failed to generate outreach draft");
      }
      setDraftMessageId(data.id);
      setSubjectPreview(data.subject);
      setSentStatus(data.sentStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected outreach draft error");
    } finally {
      setLoading(false);
    }
  };

  const approveAndSend = async (): Promise<void> => {
    if (!draftMessageId) {
      setError("Create a draft before approve-and-send");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/v1/outreach/${draftMessageId}/approve-and-send`, {
        method: "POST",
        headers: { "X-Role": "recruiter" },
      });
      const data = (await response.json()) as { sentStatus: string; message?: string };
      if (!response.ok) {
        throw new Error(data.message ?? "Failed to approve/send outreach");
      }
      setSentStatus(data.sentStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected approve/send error");
    } finally {
      setLoading(false);
    }
  };

  const generateQuestions = async (): Promise<void> => {
    if (selectedCandidateIds.length === 0) {
      setError("Select one candidate to generate questions");
      return;
    }
    if (!jobDescriptionId.trim()) {
      setError("Provide a Job Description ID");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/v1/interview-questions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Role": "recruiter",
        },
        body: JSON.stringify({
          candidateId: selectedCandidateIds[0],
          jobDescriptionId,
          questionCount: 8,
        }),
      });
      const data = (await response.json()) as {
        questions: InterviewQuestion[];
        message?: string;
      };
      if (!response.ok) {
        throw new Error(data.message ?? "Failed to generate interview questions");
      }
      setQuestions(data.questions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected interview generation error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <header style={headerStyle}>
        <h2 style={{ margin: 0 }}>Engagement Workspace</h2>
        <p style={{ margin: "0.45rem 0 0", color: "#475569" }}>
          Save shortlists, draft and send outreach, and generate interview questions.
        </p>
      </header>

      <section style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>Candidate selection</h3>
        <div style={candidateGridStyle}>
          {candidates.map((candidate) => (
            <label key={candidate.id} style={candidateCheckStyle}>
              <input
                type="checkbox"
                checked={selectedCandidateIds.includes(candidate.id)}
                onChange={() => toggleCandidate(candidate.id)}
              />
              {candidate.fullName}
            </label>
          ))}
        </div>
      </section>

      <section style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>Shortlists</h3>
        <div style={{ display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap" }}>
          <input
            style={inputStyle}
            value={shortlistName}
            onChange={(event) => setShortlistName(event.target.value)}
            placeholder="Shortlist name"
          />
          <button style={buttonStyle} disabled={loading} onClick={() => void createShortlist()}>
            Save shortlist
          </button>
        </div>
        <ul style={{ marginBottom: 0 }}>
          {shortlists.map((item) => (
            <li key={item.id}>
              {item.name} ({item.candidateIds.length} candidates)
            </li>
          ))}
        </ul>
      </section>

      <section style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>Outreach draft and send</h3>
        <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
          <button style={buttonStyle} disabled={loading} onClick={() => void createDraft()}>
            Create draft
          </button>
          <button style={buttonStyle} disabled={loading} onClick={() => void approveAndSend()}>
            Approve and send
          </button>
        </div>
        <p style={{ marginBottom: 0, color: "#334155" }}>
          Draft: {draftMessageId ?? "N/A"} | Subject: {subjectPreview || "N/A"} | Sent status: {sentStatus}
        </p>
      </section>

      <section style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>Interview questions</h3>
        <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
          <input
            style={inputStyle}
            value={jobDescriptionId}
            onChange={(event) => setJobDescriptionId(event.target.value)}
            placeholder="Job Description ID"
          />
          <button style={buttonStyle} disabled={loading} onClick={() => void generateQuestions()}>
            Generate
          </button>
        </div>
        <InterviewQuestionPanel questions={questions} />
      </section>

      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
    </section>
  );
}

const headerStyle: React.CSSProperties = {
  background: "linear-gradient(120deg, #fff7ed 0%, #f0fdfa 100%)",
  border: "1px solid #e5e7eb",
  borderRadius: 14,
  padding: "1rem",
};

const cardStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  background: "#ffffff",
  padding: "1rem",
  display: "grid",
  gap: "0.6rem",
};

const candidateGridStyle: React.CSSProperties = {
  maxHeight: 220,
  overflowY: "auto",
  display: "grid",
  gap: "0.4rem",
};

const candidateCheckStyle: React.CSSProperties = {
  display: "flex",
  gap: "0.5rem",
  alignItems: "center",
};

const inputStyle: React.CSSProperties = {
  border: "1px solid #d1d5db",
  borderRadius: 8,
  padding: "0.5rem 0.6rem",
  minWidth: 260,
};

const buttonStyle: React.CSSProperties = {
  border: "none",
  borderRadius: 10,
  background: "#0f766e",
  color: "#ffffff",
  padding: "0.55rem 0.9rem",
  cursor: "pointer",
};
