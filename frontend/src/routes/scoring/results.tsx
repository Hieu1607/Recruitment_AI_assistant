import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function ScoringResultsRoute() {
  return (
    <RoutePlaceholder
      screen="Match Results"
      description="Flagship scoring results — summary strip, sortable candidate table with serif numerals, component score bars, expand row for rationale and radar chart."
      phase="Phase 5"
      requirements={["SCORE-07", "SCORE-08", "SCORE-09", "SCORE-10", "SCORE-11", "SCORE-12"]}
    />
  );
}
