import { api } from "@/api";
import { ReportView } from "@/components/interviews/ReportView";
import { EmptyState, Skeleton } from "@/components/ui";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router";

export default function InterviewReportRoute() {
  const { interviewSessionId } = useParams<{ interviewSessionId: string }>();

  const { data, isLoading, error } = useQuery({
    queryKey: ["interview-report", interviewSessionId],
    queryFn: () => api.interviewReports.get(interviewSessionId!),
    enabled: !!interviewSessionId,
  });

  if (isLoading) {
    return (
      <div className="px-8 py-8 space-y-4">
        <Skeleton className="h-12 w-64" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-56 w-full" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="px-8 py-8">
        <EmptyState
          heading="Interview report unavailable"
          body={error instanceof Error ? error.message : "The interview report could not be loaded."}
        />
      </div>
    );
  }

  return (
    <div className="px-8 py-8 min-h-full">
      <div className="mx-auto max-w-5xl">
        <ReportView report={data} />
      </div>
    </div>
  );
}
