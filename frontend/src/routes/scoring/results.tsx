import { routes } from "@/routes";
import { useEffect } from "react";
import { useNavigate } from "react-router";

export default function ScoringResultsRoute() {
  const navigate = useNavigate();

  useEffect(() => {
    navigate(routes.scoring, { replace: true });
  }, [navigate]);

  return null;
}
