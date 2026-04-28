import React, { useState, useEffect } from 'react';
import { AlertCircle, X, ChevronRight, BarChart2, Loader2 } from 'lucide-react';
import { listJobDescriptions } from '../api/jobDescriptions';
import { scoreCandidates } from '../api/scoring';
import ScoreResultsTable from '../components/ScoreResultsTable';
import { useToast } from '../components/Toast';
import type { JobDescriptionResponse, ScoreResponse } from '../types';

const MOCK_USER_ID = '123e4567-e89b-12d3-a456-426614174000';

const DEFAULT_WEIGHTS: Record<string, number> = {
  skills: 0.3,
  experience: 0.3,
  projects: 0.2,
  education: 0.1,
  summary: 0.1,
};

type Step = 1 | 2 | 3;

export default function ScoringPage() {
  const { toast } = useToast();

  // Step navigation
  const [step, setStep] = useState<Step>(1);

  // Step 1 — Job Description
  const [jdList, setJdList] = useState<JobDescriptionResponse[]>([]);
  const [jdLoading, setJdLoading] = useState(false);
  const [selectedJdId, setSelectedJdId] = useState('');

  // Step 2 — Candidate IDs (optional)
  const [candidateIdsInput, setCandidateIdsInput] = useState('');

  // Step 3 — Scoring config
  const [threshold, setThreshold] = useState(50);
  const [weights, setWeights] = useState<Record<string, number>>(DEFAULT_WEIGHTS);
  const [batchSize, setBatchSize] = useState(10);

  // Results
  const [isScoring, setIsScoring] = useState(false);
  const [results, setResults] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load JDs on mount
  useEffect(() => {
    setJdLoading(true);
    listJobDescriptions({ limit: 100, offset: 0 })
      .then((res) => setJdList(res.items))
      .catch((err) => {
        const msg = err.response?.data?.detail || err.message || 'Failed to load job descriptions';
        setError(msg);
        toast(msg, 'error');
      })
      .finally(() => setJdLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Parse candidate IDs from textarea (one per line or comma-separated)
  const parsedCandidateIds = candidateIdsInput
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const selectedJd = jdList.find((jd) => jd.id === selectedJdId);

  const handleWeightChange = (key: string, value: number) => {
    setWeights((prev) => ({ ...prev, [key]: value }));
  };

  const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
  const allWeightsZero = Object.values(weights).every((w) => w === 0);

  const handleSubmit = async () => {
    if (!selectedJdId) {
      setError('Please select a job description.');
      return;
    }
    if (allWeightsZero) {
      setError('At least one section weight must be greater than 0.');
      return;
    }

    setError(null);
    setIsScoring(true);
    setResults(null);

    try {
      const payload: Parameters<typeof scoreCandidates>[0] = {
        job_description_id: selectedJdId,
        initiated_by_user_id: MOCK_USER_ID,
        score_threshold: threshold,
        section_weights: weights,
        batch_size: batchSize,
      };
      if (parsedCandidateIds.length > 0) {
        payload.candidate_profile_ids = parsedCandidateIds;
      }
      const data = await scoreCandidates(payload);
      setResults(data);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Scoring failed. Please try again.';
      setError(msg);
      toast(msg, 'error');
    } finally {
      setIsScoring(false);
    }
  };

  const resetForm = () => {
    setStep(1);
    setSelectedJdId('');
    setCandidateIdsInput('');
    setThreshold(50);
    setWeights(DEFAULT_WEIGHTS);
    setBatchSize(10);
    setResults(null);
    setError(null);
  };

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Candidate Scoring</h1>
        <p className="text-gray-500 text-sm mt-1">
          Score candidates against a job description using AI-powered evaluation.
        </p>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 shrink-0" />
          <div className="flex-1">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <div className="text-sm text-red-700 mt-1">{error}</div>
          </div>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700 transition">
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Step Progress Indicator */}
      {!results && (
        <div className="flex items-center mb-8 space-x-2">
          {([1, 2, 3] as Step[]).map((s, idx) => (
            <React.Fragment key={s}>
              <button
                onClick={() => !isScoring && setStep(s)}
                disabled={isScoring}
                className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-semibold transition ${
                  step === s
                    ? 'bg-indigo-600 text-white'
                    : step > s
                    ? 'bg-green-100 text-green-700 border border-green-300'
                    : 'bg-gray-100 text-gray-400 border border-gray-200'
                }`}
              >
                {s}
              </button>
              <span
                className={`text-sm font-medium ${
                  step === s ? 'text-indigo-700' : 'text-gray-400'
                }`}
              >
                {s === 1 ? 'Job Description' : s === 2 ? 'Candidates' : 'Configuration'}
              </span>
              {idx < 2 && <ChevronRight className="w-4 h-4 text-gray-300 flex-shrink-0" />}
            </React.Fragment>
          ))}
        </div>
      )}

      {/* ─── Results View ─── */}
      {results && !isScoring && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Scoring Results</h2>
              <p className="text-sm text-gray-500 mt-0.5">
                Run ID: <span className="font-mono text-xs">{results.match_run_id}</span>
              </p>
            </div>
            <button
              onClick={resetForm}
              className="px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
            >
              Score Again
            </button>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
            {[
              { label: 'Total Candidates', value: results.total_candidates },
              { label: 'Passed', value: results.total_passed_candidates },
              { label: 'Failed', value: results.total_candidates - results.total_passed_candidates },
              { label: 'Batches Run', value: results.batches },
            ].map((card) => (
              <div key={card.label} className="bg-white rounded-xl border border-gray-200 p-4 text-center shadow-sm">
                <p className="text-2xl font-bold text-gray-900">{card.value}</p>
                <p className="text-xs text-gray-500 mt-1">{card.label}</p>
              </div>
            ))}
          </div>

          {/* Results Table */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center">
              <BarChart2 className="w-4 h-4 text-indigo-600 mr-2" />
              <h3 className="text-sm font-semibold text-gray-900">
                Candidate Scores ({results.scores.length})
              </h3>
            </div>
            <ScoreResultsTable scores={results.scores} />
          </div>
        </div>
      )}

      {/* ─── Scoring in Progress ─── */}
      {isScoring && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-12 flex flex-col items-center justify-center">
          <Loader2 className="w-12 h-12 text-indigo-600 animate-spin mb-4" />
          <p className="text-base font-semibold text-gray-900">Scoring candidates...</p>
          <p className="text-sm text-yellow-700 mt-2 bg-yellow-50 px-4 py-2 rounded-full font-medium">
            ⚠️ Scoring may take a while for large candidate sets
          </p>
        </div>
      )}

      {/* ─── Step Forms ─── */}
      {!isScoring && !results && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm">

          {/* Step 1 — Select Job Description */}
          {step === 1 && (
            <div className="p-6">
              <h2 className="text-base font-semibold text-gray-900 mb-4">
                Step 1: Select a Job Description
              </h2>
              {jdLoading ? (
                <p className="text-sm text-gray-500">Loading job descriptions...</p>
              ) : jdList.length === 0 ? (
                <p className="text-sm text-gray-500">
                  No job descriptions found. Create one first.
                </p>
              ) : (
                <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                  {jdList.map((jd) => (
                    <label
                      key={jd.id}
                      className={`flex items-start p-4 rounded-lg border cursor-pointer transition ${
                        selectedJdId === jd.id
                          ? 'border-indigo-500 bg-indigo-50'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <input
                        type="radio"
                        name="jd"
                        value={jd.id}
                        checked={selectedJdId === jd.id}
                        onChange={() => setSelectedJdId(jd.id)}
                        className="mt-0.5 mr-3 accent-indigo-600"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {jd.title || <span className="text-gray-400 italic">Untitled</span>}
                          </p>
                          <span
                            className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                              jd.is_active
                                ? 'bg-green-100 text-green-700'
                                : 'bg-gray-100 text-gray-500'
                            }`}
                          >
                            {jd.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">
                          {jd.jd_text}
                        </p>
                      </div>
                    </label>
                  ))}
                </div>
              )}
              <div className="mt-6 flex justify-end">
                <button
                  onClick={() => {
                    if (!selectedJdId) {
                      setError('Please select a job description.');
                      return;
                    }
                    setError(null);
                    setStep(2);
                  }}
                  disabled={!selectedJdId}
                  className="px-5 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
                >
                  Next: Select Candidates
                </button>
              </div>
            </div>
          )}

          {/* Step 2 — Select Candidates */}
          {step === 2 && (
            <div className="p-6">
              <h2 className="text-base font-semibold text-gray-900 mb-1">
                Step 2: Select Candidates <span className="font-normal text-gray-400">(optional)</span>
              </h2>
              <p className="text-sm text-gray-500 mb-4">
                Leave blank to score all candidates in the system. Enter candidate profile IDs
                (one per line or comma-separated) to restrict scoring to a subset.
              </p>

              {selectedJd && (
                <div className="mb-4 p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                  <p className="text-xs text-indigo-700 font-medium">Selected JD:</p>
                  <p className="text-sm font-semibold text-indigo-900 mt-0.5">
                    {selectedJd.title || 'Untitled'}
                  </p>
                </div>
              )}

              <textarea
                rows={6}
                value={candidateIdsInput}
                onChange={(e) => setCandidateIdsInput(e.target.value)}
                placeholder={
                  'e.g.\n3fa85f64-5717-4562-b3fc-2c963f66afa6\n8d4e5b21-1234-4f6a-9abc-de1234567890'
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-y"
              />
              {parsedCandidateIds.length > 0 && (
                <p className="text-xs text-gray-500 mt-1">
                  {parsedCandidateIds.length} candidate ID{parsedCandidateIds.length !== 1 ? 's' : ''} entered
                </p>
              )}

              <div className="mt-6 flex justify-between">
                <button
                  onClick={() => setStep(1)}
                  className="px-5 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
                >
                  Back
                </button>
                <button
                  onClick={() => setStep(3)}
                  className="px-5 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
                >
                  Next: Configure Scoring
                </button>
              </div>
            </div>
          )}

          {/* Step 3 — Configuration */}
          {step === 3 && (
            <div className="p-6">
              <h2 className="text-base font-semibold text-gray-900 mb-4">
                Step 3: Configure Scoring
              </h2>

              {/* Selected JD + candidate summary */}
              <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200 text-sm space-y-1">
                <p className="text-gray-700">
                  <span className="font-medium">JD:</span>{' '}
                  {selectedJd?.title || 'Untitled'}
                </p>
                <p className="text-gray-700">
                  <span className="font-medium">Candidates:</span>{' '}
                  {parsedCandidateIds.length > 0
                    ? `${parsedCandidateIds.length} selected`
                    : 'All candidates'}
                </p>
              </div>

              {/* Score Threshold */}
              <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Score Threshold
                  </label>
                  <span className="text-sm font-semibold text-indigo-600">{threshold}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full accent-indigo-600"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>0</span>
                  <span>50</span>
                  <span>100</span>
                </div>
              </div>

              {/* Section Weights */}
              <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-gray-700">Section Weights</label>
                  <span
                    className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                      allWeightsZero
                        ? 'bg-red-100 text-red-700'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    Total: {(totalWeight * 100).toFixed(0)}%
                  </span>
                </div>
                {allWeightsZero && (
                  <p className="text-xs text-red-600 mb-2">
                    At least one section weight must be greater than 0.
                  </p>
                )}
                <div className="space-y-4">
                  {Object.entries(weights).map(([key, val]) => (
                    <div key={key}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm capitalize text-gray-700">{key}</span>
                        <span className="text-sm font-medium text-gray-900">
                          {(val * 100).toFixed(0)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={val}
                        onChange={(e) => handleWeightChange(key, Number(e.target.value))}
                        className="w-full accent-indigo-600"
                      />
                    </div>
                  ))}
                </div>
              </div>

              {/* Batch Size */}
              <div className="mb-6">
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Batch Size
                </label>
                <div className="flex items-center space-x-3">
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={batchSize}
                    onChange={(e) =>
                      setBatchSize(Math.min(50, Math.max(1, Number(e.target.value))))
                    }
                    className="w-24 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                  <span className="text-xs text-gray-400">
                    Candidates processed per LLM batch (1–50)
                  </span>
                </div>
              </div>

              <div className="flex justify-between">
                <button
                  onClick={() => setStep(2)}
                  className="px-5 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={allWeightsZero || !selectedJdId}
                  className="px-6 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center"
                >
                  <BarChart2 className="w-4 h-4 mr-2" />
                  Run Scoring
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
