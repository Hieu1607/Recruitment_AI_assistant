export function ScoringIntroScene({ sceneId }: { sceneId: string }) {
  const rows = [
    { name: "Avery Chen", score: 92, active: sceneId === "shortlist" },
    { name: "Jordan Lee", score: 88, active: sceneId === "shortlist" || sceneId === "reorder" },
    { name: "Priya Raman", score: 84, active: sceneId === "shortlist" },
  ];

  return (
    <div className="intro-motion-scene intro-motion-zoom" data-scene={sceneId}>
      <div className="intro-motion-header">
        <p className="intro-motion-kicker">Scoring</p>
        <h3 className="intro-motion-title">Rank candidates by fit</h3>
      </div>
      <div className="intro-motion-panel">
        <div className="intro-motion-panel__meta">
          <span>Match analysis</span>
          <span>{sceneId === "analysis" ? "Running..." : "Top matches"}</span>
        </div>
        <div className="intro-motion-score-list">
          {rows.map((row, index) => (
            <div
              key={row.name}
              className="intro-motion-score-row"
              data-rank={index + 1}
              data-active={row.active}
            >
              <span>{row.name}</span>
              <span>{row.score}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
