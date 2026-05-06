import { api, type QuestionSetResponse } from "@/api";
import {
    Button,
    DataTable,
    EmptyState,
    Modal,
    ModalContent,
    ModalDescription,
    ModalFooter,
    ModalHeader,
    ModalTitle,
    type ColumnDef,
} from "@/components/ui";
import { cn } from "@/lib/cn";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Calendar, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { toast } from "sonner";

function getQuestionCount(payload: Record<string, any> | undefined | null): number {
  if (!payload || !Array.isArray(payload.categories)) return 0;
  return payload.categories.reduce((acc: number, cat: any) => {
    return acc + (Array.isArray(cat.questions) ? cat.questions.length : 0);
  }, 0);
}

export default function InterviewQuestionsListRoute() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();

  const [isGenerateOpen, setIsGenerateOpen] = useState(false);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string>("");
  const [selectedJdId, setSelectedJdId] = useState<string>("");

  const candidateId = searchParams.get("candidate") || undefined;
  const jdId = searchParams.get("jd") || undefined;

  const { data, isLoading } = useQuery({
    queryKey: ["interview-questions", { candidateId, jdId }],
    queryFn: () =>
      api.interviewQuestions.list({
        candidate_profile_id: candidateId,
        job_description_id: jdId,
        limit: 50,
      }),
  });

  const { data: candidates } = useQuery({
    queryKey: ["candidates"],
    queryFn: () => api.upload.list({ limit: 100 }),
    staleTime: 60000,
  });

  const { data: jds } = useQuery({
    queryKey: ["job-descriptions"],
    queryFn: () => api.jobDescriptions.list({ limit: 100, is_active: true }),
    staleTime: 60000,
  });

  const generateMutation = useMutation({
    mutationFn: () =>
      api.interviewQuestions.generate({
        candidate_profile_id: selectedCandidateId,
        job_description_id: selectedJdId,
      }),
    onSuccess: (newSet) => {
      queryClient.invalidateQueries({ queryKey: ["interview-questions"] });
      toast.success("Interview questions generated");
      setIsGenerateOpen(false);
      navigate(`/interview-questions/${newSet.id}`);
    },
    onError: (err: any) => {
      toast.error(err.message || "Failed to generate questions");
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.interviewQuestions.remove(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["interview-questions"] });
      toast.success("Question set deleted");
    },
    onError: (err: any) => {
      toast.error(err.message || "Failed to delete question set");
    },
  });

  const columns: ColumnDef<QuestionSetResponse>[] = [
    {
      key: "candidate",
      header: "Candidate",
      render: (row) => (
        <span className="font-sans font-medium text-fg">
          {row.candidate_full_name || "Unknown Candidate"}
        </span>
      ),
    },
    {
      key: "jd",
      header: "Job Description",
      render: (row) => (
        <span className="text-fg-muted">
          {row.job_description_title || "Unknown JD"}
        </span>
      ),
    },
    {
      key: "count",
      header: "Questions",
      render: (row) => (
        <span className="text-fg-muted font-mono text-sm">
          {getQuestionCount(row.question_payload)} questions
        </span>
      ),
    },
    {
      key: "created",
      header: "Generated",
      render: (row) => (
        <div className="flex items-center text-fg-muted text-sm">
          <Calendar className="w-4 h-4 mr-1.5 opacity-50" />
          {new Date(row.created_at).toLocaleDateString()}
        </div>
      ),
    },
    {
      key: "actions",
      header: "",
      width: 100,
      render: (row) => (
        <div className="flex justify-end gap-2" onClick={(e) => e.stopPropagation()}>
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              if (confirm("Delete this set permanently?")) {
                deleteMutation.mutate(row.id);
              }
            }}
            className="text-fg-muted hover:text-danger hover:bg-danger/10"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      ),
    },
  ];

  return (
    <div className="px-8 py-8 min-h-full">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="font-display text-[2rem] font-medium text-fg leading-tight">
              Interview Questions
            </h1>
            <p className="text-sm text-fg-muted mt-1 font-sans">
              Generated question sets for upcoming interviews
            </p>
          </div>
          <Button
            variant="primary"
            icon={<Plus size={15} strokeWidth={2} />}
            onClick={() => setIsGenerateOpen(true)}
          >
            Generate new set
          </Button>
        </div>

        <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] overflow-hidden">
          {isLoading ? (
            <div className="p-8 text-center text-fg-muted font-sans text-sm">Loading…</div>
          ) : !data?.items?.length ? (
            <EmptyState
              icon={<Calendar className="w-8 h-8" />}
              heading="No interview questions"
              body="Generate your first set of questions to prepare for an interview."
              action={{
                label: "Generate new set",
                onClick: () => setIsGenerateOpen(true)
              }}
            />
          ) : (
            <DataTable
              columns={columns}
              data={data.items}
              onRowClick={(row) => navigate(`/interview-questions/${row.id}`)}
            />
          )}
        </div>
      </div>

      <Modal open={isGenerateOpen} onOpenChange={setIsGenerateOpen}>
        <ModalContent className="sm:max-w-md">
          <ModalHeader>
            <ModalTitle>Generate Interview Questions</ModalTitle>
            <ModalDescription>
              Select a candidate and job description to generate tailored questions.
            </ModalDescription>
          </ModalHeader>
          <div className="py-4 space-y-4">
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-fg-muted uppercase tracking-wide">
                Candidate
              </label>
              <select
                className={cn(
                  "w-full h-9 px-3 text-sm font-sans rounded-[var(--radius-md)]",
                  "border border-[color:var(--hairline-strong)] bg-bg text-fg",
                  "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent outline-none",
                )}
                value={selectedCandidateId}
                onChange={(e) => setSelectedCandidateId(e.target.value)}
              >
                <option value="">Select candidate…</option>
                {candidates?.items?.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.original_file_name || "Unknown Candidate"}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-fg-muted uppercase tracking-wide">
                Job Description
              </label>
              <select
                className={cn(
                  "w-full h-9 px-3 text-sm font-sans rounded-[var(--radius-md)]",
                  "border border-[color:var(--hairline-strong)] bg-bg text-fg",
                  "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent outline-none",
                )}
                value={selectedJdId}
                onChange={(e) => setSelectedJdId(e.target.value)}
              >
                <option value="">Select job description…</option>
                {jds?.items?.map((jd) => (
                  <option key={jd.id} value={jd.id}>
                    {jd.title || "Untitled"}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <ModalFooter>
            <Button variant="ghost" onClick={() => setIsGenerateOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={() => generateMutation.mutate()}
              disabled={!selectedCandidateId || !selectedJdId || generateMutation.isPending}
            >
              {generateMutation.isPending ? "Generating…" : "Generate"}
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}
