import { useEffect, useMemo, useState } from "react";

import type { IntroSceneDefinition } from "./scenes";

type IntroMotionMode = "idle" | "live" | "reduced";

export function useIntroMotionTimeline(
  scenes: IntroSceneDefinition[],
  options: { active: boolean; reducedMotion: boolean },
) {
  const { active, reducedMotion } = options;
  const finalSceneIndex = Math.max(scenes.length - 1, 0);
  const [sceneIndex, setSceneIndex] = useState(reducedMotion ? finalSceneIndex : 0);
  const [loopTick, setLoopTick] = useState(0);

  useEffect(() => {
    if (reducedMotion) {
      setSceneIndex(finalSceneIndex);
      return undefined;
    }
    if (!active || scenes.length === 0) {
      setSceneIndex(0);
      return undefined;
    }

    let cancelled = false;
    let timeoutId: number | undefined;

    const runScene = (index: number) => {
      if (cancelled) return;
      setSceneIndex(index);
      const scene = scenes[index];
      timeoutId = window.setTimeout(() => {
        if (cancelled) return;
        if (index === finalSceneIndex) {
          setLoopTick((value) => value + 1);
          runScene(0);
          return;
        }
        runScene(index + 1);
      }, scene.enterMs + scene.holdMs);
    };

    runScene(0);

    return () => {
      cancelled = true;
      if (timeoutId) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [active, finalSceneIndex, reducedMotion, scenes]);

  const motionMode = useMemo<IntroMotionMode>(() => {
    if (reducedMotion) return "reduced";
    if (active) return "live";
    return "idle";
  }, [active, reducedMotion]);

  return {
    sceneIndex,
    sceneId: reducedMotion ? "final" : scenes[sceneIndex]?.id ?? "final",
    motionMode,
    loopTick,
  };
}
