import { routes } from "@/routes";
import { Navigate } from "react-router";

export default function InterviewQuestionsListRoute() {
  return <Navigate to={routes.interviewTemplates} replace />;
}
