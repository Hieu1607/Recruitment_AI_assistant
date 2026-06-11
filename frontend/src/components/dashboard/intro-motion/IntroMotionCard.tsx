import { useEffect, useRef, useState } from "react";

import type { IntroCardDefinition } from "./scenes";
import { useIntroMotionTimeline } from "./useIntroMotionTimeline";
import { useReducedMotion } from "./useReducedMotion";

type IntroMotionCardProps = {
  definition: IntroCardDefinition;
  testId: string;
  children: (state: {
    sceneId: string;
    motionMode: "idle" | "live" | "reduced";
    loopTick: number;
  }) => React.ReactNode;
};

export function IntroMotionCard({ definition, testId, children }: IntroMotionCardProps) {
  const reducedMotion = useReducedMotion();
  const [inView, setInView] = useState(false);
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = hostRef.current;
    if (!node || reducedMotion || typeof IntersectionObserver === "undefined") {
      if (!reducedMotion) {
        setInView(true);
      }
      return undefined;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        setInView(entry.isIntersecting);
      },
      { threshold: 0.35 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [reducedMotion]);

  const timeline = useIntroMotionTimeline(definition.scenes, {
    active: inView,
    reducedMotion,
  });

  return (
    <div
      ref={hostRef}
      data-testid={testId}
      data-card-kind={definition.kind}
      data-motion-mode={timeline.motionMode}
      data-scene-state={reducedMotion ? "final" : timeline.sceneId}
      className="intro-motion-card"
    >
      <div className="intro-motion-card__chrome">
        <div className="intro-motion-card__label">{definition.label}</div>
        {children(timeline)}
      </div>
    </div>
  );
}
