import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function InterviewQuestionsListRoute() {
  return (
    <RoutePlaceholder
      screen="Interview Questions"
      description="AI-generated question sets — filter by candidate and JD, generate new sets, view list of existing sets."
      phase="Phase 11"
      requirements={["INTV-01", "INTV-02"]}
    />
  );
}
