export function ChatIntroScene({ sceneId }: { sceneId: string }) {
  const showAnswer = sceneId === "answer" || sceneId === "followups";

  return (
    <div className="intro-motion-scene intro-motion-zoom" data-scene={sceneId}>
      <div className="intro-motion-header">
        <p className="intro-motion-kicker">AI Chat</p>
        <h3 className="intro-motion-title">Ask the candidate pool</h3>
      </div>
      <div className="intro-motion-chat">
        <div className="intro-motion-chat__prompt">
          Top backend candidates with Python and FastAPI?
        </div>
        <div className={`intro-motion-chat__answer${showAnswer ? " is-visible" : ""}`}>
          <p>Avery Chen, Jordan Lee, and Priya Raman match the strongest backend signals.</p>
        </div>
        <div className={`intro-motion-chat__followups${sceneId === "followups" ? " is-visible" : ""}`}>
          <span>Show only candidates open to remote work</span>
          <span>Draft outreach for the top 3</span>
        </div>
      </div>
    </div>
  );
}
