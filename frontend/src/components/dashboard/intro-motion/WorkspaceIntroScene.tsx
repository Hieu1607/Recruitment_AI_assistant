export function WorkspaceIntroScene({ sceneId }: { sceneId: string }) {
  const showResumes = sceneId !== "frame";
  const showJobReady = sceneId === "job-ready" || sceneId === "handoff";
  const showScoringReady = sceneId === "handoff";

  return (
    <div className="intro-motion-scene intro-motion-zoom" data-scene={sceneId}>
      <div className="intro-motion-header">
        <p className="intro-motion-kicker">Workspace</p>
        <h3 className="intro-motion-title">One recruiting workspace</h3>
      </div>
      <div className="intro-motion-browser">
        <aside className="intro-motion-browser__sidebar">
          <span className="is-active">Candidates</span>
          <span>Scoring</span>
          <span>Chat</span>
        </aside>
        <main className="intro-motion-browser__canvas">
          <div className="intro-motion-chip-row">
            <span className={showResumes ? "is-visible" : ""}>3 resumes parsed</span>
            <span className={showJobReady ? "is-visible" : ""}>JD attached</span>
            <span className={showScoringReady ? "is-visible" : ""}>Scoring ready</span>
          </div>
          <div className="intro-motion-browser__stack">
            <div />
            <div />
            <div />
          </div>
        </main>
      </div>
    </div>
  );
}
