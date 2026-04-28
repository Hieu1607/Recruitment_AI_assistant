import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function OutreachRoute() {
  return (
    <RoutePlaceholder
      screen="Outreach"
      description="Email-client 3-column layout — folder sidebar, message list, message detail with compose and status management."
      phase="Phase 10"
      requirements={["OUT-01", "OUT-02", "OUT-03", "OUT-04", "OUT-05", "OUT-06"]}
    />
  );
}
