import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function ShortlistCollectionRoute() {
  return (
    <RoutePlaceholder
      screen="Collection Detail"
      description="Candidate collection detail — editable name, candidate table with skills and latest match score, add/remove candidates."
      phase="Phase 9"
      requirements={["SHORT-07", "SHORT-08"]}
    />
  );
}
