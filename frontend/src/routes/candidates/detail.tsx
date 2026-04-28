import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function CandidateDetailRoute() {
  return (
    <RoutePlaceholder
      screen="Candidate Detail"
      description="Candidate profile hub — overview, resume PDF, scoring history, outreach history, interview questions tabs."
      phase="Phase 7"
      requirements={["CAND-11", "CAND-12", "CAND-13", "CAND-14"]}
    />
  );
}
