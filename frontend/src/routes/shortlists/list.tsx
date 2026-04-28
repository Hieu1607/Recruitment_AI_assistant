import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function ShortlistsListRoute() {
  return (
    <RoutePlaceholder
      screen="Shortlists"
      description="Collections grid and query history tabs — manage saved candidate sets and browse chat session turn history."
      phase="Phase 9"
      requirements={[
        "SHORT-01", "SHORT-02", "SHORT-03", "SHORT-04",
        "SHORT-05", "SHORT-06", "SHORT-09", "SHORT-10",
      ]}
    />
  );
}
