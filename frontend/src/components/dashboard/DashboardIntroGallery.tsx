import { IntroMotionCard, INTRO_CARD_DEFINITIONS } from "./intro-motion";
import { ChatIntroScene } from "./intro-motion/ChatIntroScene";
import { ScoringIntroScene } from "./intro-motion/ScoringIntroScene";
import { WorkspaceIntroScene } from "./intro-motion/WorkspaceIntroScene";

export function DashboardIntroGallery() {
  return (
    <section data-testid="dashboard-intro-gallery" className="grid gap-6 xl:grid-cols-3">
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.workspace} testId="intro-card-workspace">
        {(timeline) => <WorkspaceIntroScene sceneId={timeline.sceneId} />}
      </IntroMotionCard>
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.scoring} testId="intro-card-scoring">
        {(timeline) => <ScoringIntroScene sceneId={timeline.sceneId} />}
      </IntroMotionCard>
      <IntroMotionCard definition={INTRO_CARD_DEFINITIONS.chat} testId="intro-card-chat">
        {(timeline) => <ChatIntroScene sceneId={timeline.sceneId} />}
      </IntroMotionCard>
    </section>
  );
}
