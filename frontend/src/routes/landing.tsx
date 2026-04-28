import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function LandingRoute() {
  return (
    <RoutePlaceholder
      screen="Landing"
      description="Public marketing page — hero, value strip, product showcase, feature deep-dives, social proof, and CTA."
      phase="Phase 12"
      requirements={["MKTG-01", "MKTG-02", "MKTG-03", "MKTG-04", "MKTG-05", "MKTG-06"]}
    />
  );
}
