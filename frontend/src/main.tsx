import React from "react";
import ReactDOM from "react-dom/client";
import { CandidateIngestionPage } from "./modules/candidates/pages/CandidateIngestionPage";
import { EngagementPage } from "./modules/engagement/pages/EngagementPage";
import { MatchingPage } from "./modules/matching/pages/MatchingPage";
import { QueryWorkspacePage } from "./modules/query/pages/QueryWorkspacePage";

function App(): JSX.Element {
  const [activeTab, setActiveTab] = React.useState<"ingestion" | "matching" | "query" | "engagement">(
    "ingestion"
  );

  return (
    <main style={appStyle}>
      <header style={heroStyle}>
        <h1 style={{ margin: 0, fontSize: "2rem" }}>Recruitment AI Assistant</h1>
        <p style={{ margin: "0.45rem 0 0", color: "#475569" }}>
          Resume ingestion, candidate profile review, and one-to-many JD matching.
        </p>
      </header>

      <nav style={{ display: "flex", gap: "0.6rem" }}>
        <button
          style={{ ...tabStyle, ...(activeTab === "ingestion" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("ingestion")}
        >
          Phase 3: Ingestion
        </button>
        <button
          style={{ ...tabStyle, ...(activeTab === "matching" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("matching")}
        >
          Phase 4: Matching
        </button>
        <button
          style={{ ...tabStyle, ...(activeTab === "query" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("query")}
        >
          Phase 5: Query
        </button>
        <button
          style={{ ...tabStyle, ...(activeTab === "engagement" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("engagement")}
        >
          Phase 6: Engagement
        </button>
      </nav>

      {activeTab === "ingestion" ? <CandidateIngestionPage /> : null}
      {activeTab === "matching" ? <MatchingPage /> : null}
      {activeTab === "query" ? <QueryWorkspacePage /> : null}
      {activeTab === "engagement" ? <EngagementPage /> : null}
    </main>
  );
}

const appStyle: React.CSSProperties = {
  fontFamily: "'Trebuchet MS', 'Gill Sans', sans-serif",
  padding: "1rem",
  maxWidth: 1240,
  margin: "0 auto",
  display: "grid",
  gap: "1rem",
};

const heroStyle: React.CSSProperties = {
  background: "radial-gradient(circle at 10% 10%, #fff7ed 0%, #ecfeff 60%, #f8fafc 100%)",
  border: "1px solid #e2e8f0",
  borderRadius: 16,
  padding: "1rem",
};

const tabStyle: React.CSSProperties = {
  border: "1px solid #cbd5e1",
  background: "#f8fafc",
  borderRadius: 999,
  padding: "0.45rem 0.85rem",
  cursor: "pointer",
};

const activeTabStyle: React.CSSProperties = {
  background: "#115e59",
  borderColor: "#115e59",
  color: "#fff",
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
