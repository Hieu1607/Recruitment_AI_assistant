import React, { useState } from 'react';
import { ChevronDown, ChevronRight, CheckCircle2, XCircle } from 'lucide-react';
import type { CandidateScore } from '../types';

interface Props {
  scores: CandidateScore[];
}

export default function ScoreResultsTable({ scores }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (candidateId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(candidateId)) {
        next.delete(candidateId);
      } else {
        next.add(candidateId);
      }
      return next;
    });
  };

  if (scores.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500 text-sm">
        No candidate scores returned.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm text-gray-600">
        <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="px-4 py-3 font-medium w-8"></th>
            <th className="px-4 py-3 font-medium">Candidate ID</th>
            <th className="px-4 py-3 font-medium">Total Score</th>
            <th className="px-4 py-3 font-medium">Result</th>
            <th className="px-4 py-3 font-medium">Rationale</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {scores.map((candidate) => (
            <React.Fragment key={candidate.candidateId}>
              <tr
                className="hover:bg-gray-50 transition-colors cursor-pointer"
                onClick={() => toggle(candidate.candidateId)}
              >
                <td className="px-4 py-3 text-gray-400">
                  {expanded.has(candidate.candidateId) ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </td>
                <td className="px-4 py-3 font-mono text-xs text-gray-700">
                  <span className="truncate max-w-[180px] block" title={candidate.candidateId}>
                    {candidate.candidateId}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center space-x-2">
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          candidate.totalScore >= 70
                            ? 'bg-green-500'
                            : candidate.totalScore >= 50
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${Math.min(100, candidate.totalScore)}%` }}
                      />
                    </div>
                    <span className="font-semibold text-gray-900 text-xs">
                      {candidate.totalScore.toFixed(1)}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  {candidate.passedThreshold ? (
                    <span className="flex items-center text-xs font-medium text-green-700 bg-green-100 px-2.5 py-1 rounded-full w-fit">
                      <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                      Pass
                    </span>
                  ) : (
                    <span className="flex items-center text-xs font-medium text-red-700 bg-red-100 px-2.5 py-1 rounded-full w-fit">
                      <XCircle className="w-3.5 h-3.5 mr-1" />
                      Fail
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-600 max-w-xs">
                  <span className="line-clamp-2 text-xs">{candidate.rationale}</span>
                </td>
              </tr>

              {expanded.has(candidate.candidateId) && (
                <tr className="bg-indigo-50/40">
                  <td colSpan={5} className="px-8 py-4">
                    <p className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-3">
                      Component Scores
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {candidate.componentScores.map((cs) => (
                        <div
                          key={cs.criterionKey}
                          className="bg-white rounded-lg border border-gray-200 p-3"
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-xs font-semibold text-gray-700 capitalize">
                              {cs.criterionKey.replace(/_/g, ' ')}
                            </span>
                            <span className="text-xs text-gray-500">
                              weight: {(cs.weight * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex items-center space-x-2 mb-1">
                            <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                              <div
                                className={`h-1.5 rounded-full ${
                                  cs.score >= 70
                                    ? 'bg-green-500'
                                    : cs.score >= 50
                                    ? 'bg-yellow-500'
                                    : 'bg-red-400'
                                }`}
                                style={{ width: `${Math.min(100, cs.score)}%` }}
                              />
                            </div>
                            <span className="text-xs font-medium text-gray-700 w-10 text-right">
                              {cs.score.toFixed(1)}
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 line-clamp-3">
                            {cs.evidenceSummary}
                          </p>
                        </div>
                      ))}
                    </div>
                    <div className="mt-3 p-3 bg-white rounded-lg border border-gray-200">
                      <p className="text-xs font-semibold text-gray-700 mb-1">Rationale</p>
                      <p className="text-xs text-gray-600">{candidate.rationale}</p>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
