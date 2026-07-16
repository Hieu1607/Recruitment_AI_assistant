import type {
  CandidateEvaluationComponentScore,
  CandidateEvaluationResponse,
} from "@/api";
import type { RadarDataPoint } from "@/components/ui/score-visualization";

export const SECTION_LABELS: Record<string, string> = {
  skills: "Skills",
  experience: "Experience",
  education: "Education",
  projects: "Projects",
  summary: "Summary",
  languages: "Languages",
  achievements: "Achievements",
  certifications: "Certifications",
  publications: "Publications",
  other: "Other",
};

const SECTION_ORDER = [
  "skills",
  "experience",
  "education",
  "projects",
  "summary",
  "languages",
  "achievements",
  "certifications",
  "publications",
  "other",
];

function criterionSortKey(component: CandidateEvaluationComponentScore) {
  const sectionIndex = SECTION_ORDER.indexOf(component.section ?? "");
  const normalizedSectionIndex = sectionIndex === -1 ? SECTION_ORDER.length : sectionIndex;
  return `${String(normalizedSectionIndex).padStart(2, "0")}:${criterionIdentity(component)}`;
}

function normalizeCriterionText(value: string) {
  return value
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function criterionIdentity(component: CandidateEvaluationComponentScore) {
  const requirement = normalizeCriterionText(component.requirementText ?? "");
  const fallbackKey = normalizeCriterionText(component.criterionKey);
  return `${component.section ?? "other"}:${requirement || fallbackKey}`;
}

export function formatCriterionLabel(component: CandidateEvaluationComponentScore) {
  if (component.requirementText?.trim()) return component.requirementText.trim();
  return component.criterionKey
    .split(".")
    .join(" ")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function completedEvaluations(evaluations: CandidateEvaluationResponse[]) {
  return evaluations.filter((evaluation) => evaluation.status === "completed");
}

export function getSectionAverage(
  components: CandidateEvaluationComponentScore[],
  section: string,
) {
  const scores = components
    .filter((component) => component.section === section)
    .map((component) => component.scorePercent);
  if (scores.length === 0) return null;
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

export function buildCriterionRadarData(
  evaluation: CandidateEvaluationResponse,
  jobEvaluations: CandidateEvaluationResponse[],
): RadarDataPoint[] {
  const validPeers = completedEvaluations(jobEvaluations);
  const currentCriteria = new Map<string, CandidateEvaluationComponentScore>();
  evaluation.componentScores.forEach((component) => {
    const identity = criterionIdentity(component);
    if (!currentCriteria.has(identity)) currentCriteria.set(identity, component);
  });
  const orderedCriteria = [...currentCriteria.values()].sort((left, right) =>
    criterionSortKey(left).localeCompare(criterionSortKey(right)),
  );

  return orderedCriteria.map((component, index) => {
    const identity = criterionIdentity(component);
    const peerScores = validPeers
      .map((peer) => peer.componentScores.find(
        (candidateComponent) => criterionIdentity(candidateComponent) === identity,
      ))
      .filter((peerComponent): peerComponent is CandidateEvaluationComponentScore => Boolean(peerComponent))
      .map((peerComponent) => peerComponent.scorePercent);
    const comparisonValue = peerScores.length > 0
      ? Number((peerScores.reduce((sum, score) => sum + score, 0) / peerScores.length).toFixed(1))
      : null;

    return {
      subject: `C${index + 1}`,
      detail: formatCriterionLabel(component),
      value: Number(component.scorePercent.toFixed(1)),
      comparisonValue,
      fullMark: 100,
    };
  });
}
