import { routes } from "@/routes";
import { Navigate, useParams } from "react-router";

export default function InterviewQuestionsDetailRoute() {
  const { id } = useParams<{ id: string }>();

  if (!id) {
    return <Navigate to={routes.interviewTemplates} replace />;
  }

  return <Navigate to={routes.interviewTemplateDetail(id)} replace />;
}
