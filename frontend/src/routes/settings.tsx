import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function SettingsRoute() {
  return (
    <RoutePlaceholder
      screen="Settings"
      description="Platform settings — profile, workspace, API keys, notifications, danger zone tabs."
      phase="Phase 12"
      requirements={["PLAT-02"]}
    />
  );
}
