import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function ScoringSetupRoute() {
  return (
    <RoutePlaceholder
      screen="Scoring Setup"
      description="Match run configuration — select JD, choose candidates, set section weights with live donut chart, threshold slider, and batch size."
      phase="Phase 5"
      requirements={["SCORE-01", "SCORE-02", "SCORE-03", "SCORE-04", "SCORE-05", "SCORE-06"]}
    />
  );
}
