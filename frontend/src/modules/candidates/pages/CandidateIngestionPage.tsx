import React, { useEffect, useState } from "react";

import { CandidateDetailPanel } from "../components/CandidateDetailPanel";

type Candidate = {
  id: string;
  fullName: string;
  phone: string | null;
  email: string | null;
  locationNormalized: string | null;
  educated: boolean;
  everStudiedAbroad: boolean;
  profileStatus: string;
};

type Trace = {
  id: string;
  sourcePage: number;
  sourceBBox: Record<string, number>;
  sourceTextSnippet: string;
  mappedFieldName: string;
  confidenceScore: number | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export function CandidateIngestionPage(): JSX.Element {
  const [files, setFiles] = useState<FileList | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [query, setQuery] = useState("");
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  const [editPayload, setEditPayload] = useState({
    fullName: "",
    phone: "",
    email: "",
    locationNormalized: "",
    educated: false,
    everStudiedAbroad: false,
  });
  const [traces, setTraces] = useState<Trace[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void fetchCandidates();
  }, []);

  const fetchCandidates = async (): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const search = query ? `?q=${encodeURIComponent(query)}` : "";
      const response = await fetch(`${API_BASE}/v1/candidates${search}`, {
        headers: { "X-Role": "recruiter" },
      });
      const data = (await response.json()) as Candidate[];
      if (!response.ok) {
        throw new Error((data as unknown as { message?: string }).message ?? "Failed to load candidates");
      }
      setCandidates(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const fetchTraces = async (candidateId: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/v1/candidates/${candidateId}/traces`, {
      headers: { "X-Role": "recruiter" },
    });
    const data = (await response.json()) as Trace[];
    if (!response.ok) {
      throw new Error("Failed to load extraction traces");
    }
    setTraces(data);
  };

  const handleUpload = async (): Promise<void> => {
    if (!files || files.length === 0) {
      setError("Select at least one PDF file before uploading");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      Array.from(files).forEach((file) => formData.append("files", file));
      const response = await fetch(`${API_BASE}/v1/resumes/upload`, {
        method: "POST",
        headers: { "X-Role": "recruiter" },
        body: formData,
      });
      if (!response.ok) {
        const data = (await response.json()) as { message?: string };
        throw new Error(data.message ?? "Upload failed");
      }
      await fetchCandidates();
      setFiles(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveCandidate = async (): Promise<void> => {
    if (!selectedCandidate) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/v1/candidates/${selectedCandidate.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "X-Role": "recruiter",
        },
        body: JSON.stringify(editPayload),
      });
      if (!response.ok) {
        const data = (await response.json()) as { message?: string };
        throw new Error(data.message ?? "Update failed");
      }
      await fetchCandidates();
      await fetchTraces(selectedCandidate.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <header style={headerStyle}>
        <h2 style={{ margin: 0 }}>Candidate Ingestion Workspace</h2>
        <p style={{ margin: "0.4rem 0 0", color: "#475569" }}>
          Upload resume PDFs, review extracted candidates, and persist profile updates.
        </p>
      </header>

      <div style={uploadCardStyle}>
        <input type="file" accept="application/pdf" multiple onChange={(e) => setFiles(e.target.files)} />
        <button onClick={() => void handleUpload()} disabled={loading} style={buttonStyle}>
          Upload and ingest
        </button>
      </div>

      <div style={{ display: "flex", gap: "0.6rem" }}>
        <input
          style={inputStyle}
          placeholder="Search by name, email, skills"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={() => void fetchCandidates()} style={buttonStyle}>
          Search
        </button>
      </div>

      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "1rem" }}>
        <section style={listStyle}>
          <h3 style={{ marginTop: 0 }}>Candidates</h3>
          {loading ? <p>Loading...</p> : null}
          {candidates.map((candidate) => (
            <button
              key={candidate.id}
              onClick={() => {
                setSelectedCandidate(candidate);
                setEditPayload({
                  fullName: candidate.fullName,
                  phone: candidate.phone ?? "",
                  email: candidate.email ?? "",
                  locationNormalized: candidate.locationNormalized ?? "",
                  educated: candidate.educated,
                  everStudiedAbroad: candidate.everStudiedAbroad,
                });
                void fetchTraces(candidate.id);
              }}
              style={{
                ...candidateItemStyle,
                background: selectedCandidate?.id === candidate.id ? "#eef2ff" : "#fff",
              }}
            >
              <strong>{candidate.fullName}</strong>
              <span style={{ color: "#64748b" }}>{candidate.profileStatus}</span>
            </button>
          ))}

          {selectedCandidate ? (
            <div style={{ marginTop: "1rem", display: "grid", gap: "0.5rem" }}>
              <h4 style={{ marginBottom: 0 }}>Quick review update</h4>
              <input
                style={inputStyle}
                value={editPayload.fullName}
                onChange={(event) =>
                  setEditPayload((current) => ({ ...current, fullName: event.target.value }))
                }
                placeholder="Full name"
              />
              <input
                style={inputStyle}
                value={editPayload.email}
                onChange={(event) =>
                  setEditPayload((current) => ({ ...current, email: event.target.value }))
                }
                placeholder="Email"
              />
              <input
                style={inputStyle}
                value={editPayload.phone}
                onChange={(event) =>
                  setEditPayload((current) => ({ ...current, phone: event.target.value }))
                }
                placeholder="Phone"
              />
              <input
                style={inputStyle}
                value={editPayload.locationNormalized}
                onChange={(event) =>
                  setEditPayload((current) => ({ ...current, locationNormalized: event.target.value }))
                }
                placeholder="Location"
              />
              <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <input
                  type="checkbox"
                  checked={editPayload.educated}
                  onChange={(event) =>
                    setEditPayload((current) => ({ ...current, educated: event.target.checked }))
                  }
                />
                Educated
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <input
                  type="checkbox"
                  checked={editPayload.everStudiedAbroad}
                  onChange={(event) =>
                    setEditPayload((current) => ({ ...current, everStudiedAbroad: event.target.checked }))
                  }
                />
                Ever studied abroad
              </label>
              <button onClick={() => void handleSaveCandidate()} style={buttonStyle}>
                Save candidate changes
              </button>
            </div>
          ) : null}
        </section>

        <CandidateDetailPanel candidate={selectedCandidate} traces={traces} />
      </div>
    </section>
  );
}

const headerStyle: React.CSSProperties = {
  background: "linear-gradient(120deg, #fff9f1 0%, #f2fffb 100%)",
  border: "1px solid #e5e7eb",
  borderRadius: 14,
  padding: "1rem",
};

const uploadCardStyle: React.CSSProperties = {
  display: "flex",
  gap: "0.8rem",
  alignItems: "center",
  background: "#ffffff",
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  padding: "0.9rem",
};

const listStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  padding: "1rem",
  display: "grid",
  gap: "0.5rem",
  alignContent: "start",
};

const candidateItemStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  borderRadius: 10,
  textAlign: "left",
  padding: "0.6rem 0.8rem",
  display: "grid",
  gap: "0.2rem",
  cursor: "pointer",
};

const inputStyle: React.CSSProperties = {
  flex: 1,
  border: "1px solid #d1d5db",
  borderRadius: 10,
  padding: "0.55rem 0.7rem",
};

const buttonStyle: React.CSSProperties = {
  border: "none",
  borderRadius: 10,
  background: "#0f766e",
  color: "#fff",
  padding: "0.55rem 0.9rem",
  cursor: "pointer",
};
