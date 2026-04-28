import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function InterviewQuestionDetailRoute() {
  return (
    <RoutePlaceholder
      screen="Question Set Detail"
      description="Interview question set — questions grouped by category, drag-reorder, recruiter notes per question, export as PDF."
      phase="Phase 11"
      requirements={["INTV-03", "INTV-04", "INTV-05", "INTV-06", "INTV-07"]}
    />
  );
}
