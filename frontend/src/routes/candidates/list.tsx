import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function CandidatesListRoute() {
  return (
    <RoutePlaceholder
      screen="Candidates"
      description="Resume management — table/grid views, status filters, upload modal, pagination."
      phase="Phase 3"
      requirements={[
        "CAND-01", "CAND-02", "CAND-03", "CAND-04", "CAND-05",
        "CAND-06", "CAND-07", "CAND-08", "CAND-09", "CAND-10",
        "CAND-15", "CAND-16",
      ]}
    />
  );
}
