import { Navigate } from "react-router";

import { routes } from "@/routes";

export default function JobDescriptionCompatibilityRoute() {
  return <Navigate to={routes.jobDescriptions} replace />;
}
