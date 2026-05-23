import { PublicInterviewShell } from "@/components/interviews/PublicInterviewShell";
import { useParams } from "react-router";

export default function PublicInterviewRoute() {
  const { token } = useParams<{ token: string }>();

  return <PublicInterviewShell token={token ?? ""} />;
}
