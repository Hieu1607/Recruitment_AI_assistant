import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function LoginRoute() {
  return (
    <RoutePlaceholder
      screen="Login / Sign Up"
      description="Authentication — split-screen editorial panel with form. UI-only since backend does not enforce auth yet."
      phase="Phase 12"
      requirements={["AUTH-01", "AUTH-02", "AUTH-03", "AUTH-04", "AUTH-05"]}
    />
  );
}
