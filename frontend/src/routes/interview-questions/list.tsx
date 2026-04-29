import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router";
import { Plus, Trash2, Calendar } from "lucide-react";
import { toast } from "sonner";
import { api, type QuestionSetResponse } from "@/api";
import {
  DataTable,
  type ColumnDef,
  Button,
  EmptyState,
  Modal,
  ModalHeader,
  ModalTitle,
  ModalDescription,
  ModalContent,
  ModalFooter,
} from "@/components/ui";

const PLACEHOLDER_USER_ID = "00000000-0000-0000-0000-000000000001";

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
      api.interviewQuestions.create({
        candidate_profile_id: selectedCandidateId,
        job_description_id: selectedJdId,
        generated_by_user_id: PLACEHOLDER_USER_ID,
        question_payload: {
          categories: [
            {
              name: "Technical",
              questions: [
                { id: "q1", text: "Generated tech question 1", difficulty: "medium" },
              ],
            },
            {
              name: "Behavioral",
              questions: [
                { id: "q2", text: "Generated behavioral question 1", difficulty: "hard" },
              ],
            },
          ],
        },
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
        <span className="font-serif font-medium text-forest-900">
          {row.candidate_full_name || "Unknown Candidate"}
        </span>
      ),
    },
    {
      key: "jd",
      header: "Job Description",
      render: (row) => (
        <span className="text-forest-700">
          {row.job_description_title || "Unknown JD"}
        </span>
      ),
    },
    {
      key: "count",
      header: "Questions",
      render: (row) => (
        <span className="text-forest-600 font-mono text-sm">
          {getQuestionCount(row.question_payload)} questions
        </span>
      ),
    },
    {
      key: "created",
      header: "Generated",
      render: (row) => (
        <div className="flex items-center text-forest-500 text-sm">
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
            className="text-forest-500 hover:text-red-600 hover:bg-red-50"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      ),
    },
  ];

  return (
    <div className="flex-1 overflow-auto bg-sand-50 p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-serif text-forest-900">Interview Questions</h1>
            <p className="text-forest-600 mt-1">
              Generated question sets for upcoming interviews.
            </p>
          </div>
          <Button onClick={() => setIsGenerateOpen(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Generate new set
          </Button>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-sand-200 overflow-hidden">
          {isLoading ? (
            <div className="p-8 text-center text-forest-500">Loading...</div>
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
              className="w-full"
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
            <div className="space-y-2">
              <label className="text-sm font-medium text-forest-900 block">
                Candidate
              </label>
              <select
                className="w-full px-3 py-2 border border-sand-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent-500 text-forest-900"
                value={selectedCandidateId}
                onChange={(e) => setSelectedCandidateId(e.target.value)}
              >
                <option value="">Select candidate...</option>
                {candidates?.items?.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.original_file_name || "Unknown Candidate"}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-forest-900 block">
                Job Description
              </label>
              <select
                className="w-full px-3 py-2 border border-sand-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent-500 text-forest-900"
                value={selectedJdId}
                onChange={(e) => setSelectedJdId(e.target.value)}
              >
                <option value="">Select job description...</option>
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
              onClick={() => generateMutation.mutate()}
              disabled={!selectedCandidateId || !selectedJdId || generateMutation.isPending}
            >
              {generateMutation.isPending ? "Generating..." : "Generate"}
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}
