import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function JobDescriptionsListRoute() {
  return (
    <RoutePlaceholder
      screen="Job Descriptions"
      description="Manage job descriptions — grid card layout, is_active filter, create and score actions."
      phase="Phase 4"
      requirements={["JD-01", "JD-06"]}
    />
  );
}
