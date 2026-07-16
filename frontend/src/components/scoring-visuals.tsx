import type { CandidateEvaluationResponse } from "@/api";
import { buildCriterionRadarData } from "@/components/scoring-utils";
import { ScoreRadar } from "@/components/ui/score-visualization";

export function CriteriaRadar({
  evaluation,
  jobEvaluations,
  size = 360,
}: {
  evaluation: CandidateEvaluationResponse;
  jobEvaluations: CandidateEvaluationResponse[];
  size?: number;
}) {
  const data = buildCriterionRadarData(evaluation, jobEvaluations);
  if (data.length < 3) {
    return (
      <p className="text-sm leading-6 text-fg-muted">
        At least three scored criteria are needed to draw the radar chart.
      </p>
    );
  }

  return (
    <div>
      <ScoreRadar
        data={data}
        size={size}
        primaryLabel="Candidate"
        comparisonLabel="Job average"
      />
      <details open className="mx-auto mt-3 max-w-md rounded-[var(--radius-md)] border border-[color:var(--hairline)] bg-bg px-3 py-2">
        <summary className="cursor-pointer text-xs font-medium text-fg-muted">
          View criteria labels ({data.length})
        </summary>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {data.map((point) => (
            <div key={point.subject} className="flex items-start gap-2 text-xs leading-5">
              <span className="shrink-0 font-mono font-medium text-accent">{point.subject}</span>
              <span className="text-fg-muted">{point.detail}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
