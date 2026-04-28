import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function JobDescriptionEditRoute() {
  return (
    <RoutePlaceholder
      screen="Create / Edit JD"
      description="Notion-style full-page editor for job description title and body, with autosave and optional AI polish."
      phase="Phase 4"
      requirements={["JD-02", "JD-03", "JD-04", "JD-05"]}
    />
  );
}
