import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function DashboardRoute() {
  return (
    <RoutePlaceholder
      screen="Dashboard"
      description="Recruiter overview — metric cards, recent activity, quick actions, editorial insight."
      phase="Phase 8"
      requirements={["DASH-01", "DASH-02", "DASH-03", "DASH-04", "DASH-05", "DASH-06", "DASH-07"]}
    />
  );
}
