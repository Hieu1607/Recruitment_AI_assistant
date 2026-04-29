import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api";
import { Button, Badge, Skeleton, EmptyState } from "@/components/ui";
import { toast } from "sonner";
import {
  ArrowLeft,
  Printer,
  Trash2,
  Plus,
  GripVertical,
  Save,
} from "lucide-react";
import {
  DndContext,
  DragOverlay,
  closestCorners,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragStartEvent,
  DragOverEvent,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

interface Question {
  id: string;
  text: string;
  difficulty: "easy" | "medium" | "hard" | string;
  notes?: string;
}

interface Category {
  name: string;
  questions: Question[];
}

interface QuestionPayload {
  categories: Category[];
}

function SortableQuestion({
  question,
  categoryId,
  onChange,
  onDelete,
}: {
  question: Question;
  categoryId: string;
  onChange: (updated: Question) => void;
  onDelete: () => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({
    id: question.id,
    data: {
      type: "Question",
      question,
      categoryId,
    },
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  if (isDragging) {
    return (
      <div
        ref={setNodeRef}
        style={style}
        className="opacity-30 border-2 border-dashed border-forest-300 rounded-xl h-32 w-full bg-sand-50"
      />
    );
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="bg-white border border-sand-200 rounded-xl p-4 shadow-sm flex gap-4 group print:border-none print:shadow-none print:p-0 print:mb-6 print:break-inside-avoid"
    >
      <div
        {...attributes}
        {...listeners}
        className="mt-1 cursor-grab opacity-0 group-hover:opacity-100 transition-opacity print:hidden"
      >
        <GripVertical className="w-5 h-5 text-sand-400" />
      </div>
      <div className="flex-1 space-y-3">
        <div className="flex items-start justify-between">
          <textarea
            value={question.text}
            onChange={(e) => onChange({ ...question, text: e.target.value })}
            className="w-full text-lg font-serif italic text-forest-900 border-none outline-none focus:ring-1 focus:ring-accent-500 rounded p-1 resize-none overflow-hidden print:p-0"
            rows={2}
            placeholder="Question text..."
          />
          <div className="flex items-center gap-2 ml-4 print:hidden">
            <Badge
              variant={
                question.difficulty === "easy"
                  ? "success"
                  : question.difficulty === "hard"
                  ? "danger"
                  : "warning"
              }
            >
              {question.difficulty}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={onDelete}
              className="opacity-0 group-hover:opacity-100 text-forest-400 hover:text-red-500 hover:bg-red-50"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
        <div>
          <textarea
            value={question.notes || ""}
            onChange={(e) => onChange({ ...question, notes: e.target.value })}
            placeholder="Expected answer or notes (optional)"
            className="w-full text-sm text-forest-600 bg-sand-50 rounded-lg p-3 border border-transparent focus:border-sand-300 focus:outline-none focus:ring-1 focus:ring-sand-300 resize-none min-h-[60px] print:bg-white print:border-sand-200 print:text-xs"
          />
        </div>
      </div>
    </div>
  );
}

export default function InterviewQuestionsDetailRoute() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: qSet, isLoading } = useQuery({
    queryKey: ["interview-questions", id],
    queryFn: () => api.interviewQuestions.get(id!),
    enabled: !!id,
  });

  const updateMutation = useMutation({
    mutationFn: (newPayload: QuestionPayload) =>
      api.interviewQuestions.update(id!, { question_payload: newPayload as any }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["interview-questions"] });
      toast.success("Question set saved");
      setHasChanges(false);
    },
    onError: (err: any) => toast.error(err.message || "Failed to save"),
  });

  const [categories, setCategories] = useState<Category[]>([]);
  const [hasChanges, setHasChanges] = useState(false);
  const [activeQuestion, setActiveQuestion] = useState<Question | null>(null);

  useEffect(() => {
    if (qSet && qSet.question_payload && !hasChanges) {
      setCategories((qSet.question_payload as unknown as QuestionPayload).categories || []);
    }
  }, [qSet, hasChanges]);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  if (isLoading) {
    return (
      <div className="p-8 space-y-4">
        <Skeleton className="h-10 w-1/3" />
        <Skeleton className="h-6 w-1/4" />
        <div className="space-y-4 pt-8">
          <Skeleton className="h-32 w-full rounded-xl" />
          <Skeleton className="h-32 w-full rounded-xl" />
        </div>
      </div>
    );
  }

  if (!qSet) {
    return (
      <div className="p-8">
        <EmptyState heading="Not found" body="The question set was not found." />
      </div>
    );
  }

  const handleDragStart = (event: DragStartEvent) => {
    const { active } = event;
    const { question } = active.data.current as { question: Question };
    setActiveQuestion(question);
  };

  const handleDragOver = (event: DragOverEvent) => {
    const { active, over } = event;
    if (!over) return;

    const activeId = active.id;
    const overId = over.id;

    if (activeId === overId) return;

    const activeContainer = active.data.current?.categoryId;
    const overContainer = over.data.current?.categoryId || over.id; // over can be an item or a container

    if (!activeContainer || !overContainer || activeContainer === overContainer) {
      return;
    }

    setCategories((prev) => {
      const activeCatIndex = prev.findIndex((c) => c.name === activeContainer);
      const overCatIndex = prev.findIndex((c) => c.name === overContainer);
      if (activeCatIndex < 0 || overCatIndex < 0) return prev;

      const activeCat = prev[activeCatIndex];
      const overCat = prev[overCatIndex];

      const activeItemIndex = activeCat.questions.findIndex((q) => q.id === activeId);
      const overItemIndex = overCat.questions.findIndex((q) => q.id === overId);

      const activeItem = activeCat.questions[activeItemIndex];

      const newPrev = [...prev];
      newPrev[activeCatIndex] = {
        ...activeCat,
        questions: activeCat.questions.filter((q) => q.id !== activeId),
      };

      const newOverItems = [...overCat.questions];
      const insertIndex = overItemIndex >= 0 ? overItemIndex : newOverItems.length;
      newOverItems.splice(insertIndex, 0, activeItem);

      newPrev[overCatIndex] = {
        ...overCat,
        questions: newOverItems,
      };

      return newPrev;
    });
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    setActiveQuestion(null);

    if (!over) return;

    const activeId = active.id;
    const overId = over.id;

    if (activeId === overId) return;

    const activeContainer = active.data.current?.categoryId;
    const overContainer = over.data.current?.categoryId || over.id;

    if (activeContainer === overContainer) {
      setCategories((prev) => {
        const catIndex = prev.findIndex((c) => c.name === activeContainer);
        if (catIndex < 0) return prev;

        const cat = prev[catIndex];
        const oldIndex = cat.questions.findIndex((q) => q.id === activeId);
        const newIndex = cat.questions.findIndex((q) => q.id === overId);

        const newPrev = [...prev];
        newPrev[catIndex] = {
          ...cat,
          questions: arrayMove(cat.questions, oldIndex, newIndex),
        };
        setHasChanges(true);
        return newPrev;
      });
    } else {
      setHasChanges(true);
    }
  };

  const handleUpdateQuestion = (catName: string, updatedQuestion: Question) => {
    setCategories((prev) =>
      prev.map((cat) => {
        if (cat.name !== catName) return cat;
        return {
          ...cat,
          questions: cat.questions.map((q) =>
            q.id === updatedQuestion.id ? updatedQuestion : q
          ),
        };
      })
    );
    setHasChanges(true);
  };

  const handleDeleteQuestion = (catName: string, qId: string) => {
    setCategories((prev) =>
      prev.map((cat) => {
        if (cat.name !== catName) return cat;
        return {
          ...cat,
          questions: cat.questions.filter((q) => q.id !== qId),
        };
      })
    );
    setHasChanges(true);
  };

  const handleAddQuestion = (catName: string) => {
    const newQuestion: Question = {
      id: `q-${Date.now()}`,
      text: "",
      difficulty: "medium",
      notes: "",
    };
    setCategories((prev) =>
      prev.map((cat) => {
        if (cat.name !== catName) return cat;
        return { ...cat, questions: [...cat.questions, newQuestion] };
      })
    );
    setHasChanges(true);
  };

  const handleSave = () => {
    updateMutation.mutate({ categories });
  };

  return (
    <div className="flex-1 overflow-auto bg-sand-50 pb-20 print:bg-white print:p-0">
      <div className="max-w-4xl mx-auto p-8 space-y-8 print:p-0 print:space-y-4">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between print:hidden">
          <div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/interview-questions")}
              className="-ml-3 mb-2 text-forest-500"
            >
              <ArrowLeft className="w-4 h-4 mr-1" /> Back to list
            </Button>
            <h1 className="text-3xl font-serif text-forest-900">
              Interview for {qSet.candidate_full_name || "Unknown"}
            </h1>
            <p className="text-forest-600 mt-1">
              Job Description: {qSet.job_description_title || "Untitled"}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="secondary" onClick={() => window.print()}>
              <Printer className="w-4 h-4 mr-2" />
              Print
            </Button>
            {hasChanges && (
              <Button onClick={handleSave} disabled={updateMutation.isPending}>
                <Save className="w-4 h-4 mr-2" />
                {updateMutation.isPending ? "Saving..." : "Save changes"}
              </Button>
            )}
          </div>
        </div>

        {/* Print Header */}
        <div className="hidden print:block mb-8">
          <h1 className="text-2xl font-serif text-forest-900 border-b border-forest-900 pb-2 mb-2">
            Interview Guide: {qSet.candidate_full_name || "Unknown"}
          </h1>
          <p className="text-sm font-sans text-forest-600">
            Role: {qSet.job_description_title || "Untitled"}
          </p>
        </div>

        {/* DND Context */}
        <DndContext
          sensors={sensors}
          collisionDetection={closestCorners}
          onDragStart={handleDragStart}
          onDragOver={handleDragOver}
          onDragEnd={handleDragEnd}
        >
          <div className="space-y-10 print:space-y-8">
            {categories.map((cat) => (
              <div key={cat.name} className="space-y-4 break-inside-avoid">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-serif text-forest-800">
                    {cat.name}
                  </h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleAddQuestion(cat.name)}
                    className="text-forest-500 print:hidden"
                  >
                    <Plus className="w-4 h-4 mr-1" /> Add
                  </Button>
                </div>
                
                <SortableContext
                  id={cat.name}
                  items={cat.questions.map((q) => q.id)}
                  strategy={verticalListSortingStrategy}
                >
                  <div className="space-y-3">
                    {cat.questions.map((q) => (
                      <SortableQuestion
                        key={q.id}
                        question={q}
                        categoryId={cat.name}
                        onChange={(updated) => handleUpdateQuestion(cat.name, updated)}
                        onDelete={() => handleDeleteQuestion(cat.name, q.id)}
                      />
                    ))}
                    {cat.questions.length === 0 && (
                      <div className="p-8 border-2 border-dashed border-sand-200 rounded-xl text-center text-forest-400 print:hidden">
                        Drop questions here
                      </div>
                    )}
                  </div>
                </SortableContext>
              </div>
            ))}
          </div>

          <DragOverlay>
            {activeQuestion ? (
              <div className="bg-white border border-sand-200 rounded-xl p-4 shadow-xl opacity-90">
                <div className="text-lg font-serif italic text-forest-900">
                  {activeQuestion.text}
                </div>
              </div>
            ) : null}
          </DragOverlay>
        </DndContext>
      </div>
    </div>
  );
}
