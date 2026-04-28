import React, { useState, useEffect, useCallback } from 'react';
import {
  HelpCircle,
  Plus,
  Pencil,
  Trash2,
  AlertCircle,
  X,
  Check,
  Eye,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import type { QuestionSetResponse, PaginatedResponse } from '../types';
import {
  createQuestionSet,
  listQuestionSets,
  updateQuestionSet,
  deleteQuestionSet,
} from '../api/interviewQuestions';
import { useToast } from '../components/Toast';

const MOCK_USER_ID = '123e4567-e89b-12d3-a456-426614174000';
const PAGE_LIMIT = 20;

// ─── JSON payload renderer ────────────────────────────────────────────────────

function JsonPayloadView({ payload }: { payload: Record<string, unknown> }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (key: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });

  const entries = Object.entries(payload);
  if (entries.length === 0) {
    return <p className="text-xs text-gray-400 italic">Empty payload</p>;
  }

  return (
    <div className="space-y-2">
      {entries.map(([key, value]) => {
        const isObject = typeof value === 'object' && value !== null;
        const isArray = Array.isArray(value);

        if (isObject || isArray) {
          const isExp = expanded.has(key);
          return (
            <div key={key} className="border border-gray-200 rounded-lg overflow-hidden">
              <button
                type="button"
                onClick={() => toggle(key)}
                className="w-full flex items-center px-3 py-2 bg-gray-50 hover:bg-gray-100 transition text-left"
              >
                {isExp ? (
                  <ChevronDown className="w-3.5 h-3.5 text-gray-400 mr-2 shrink-0" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 text-gray-400 mr-2 shrink-0" />
                )}
                <span className="text-xs font-semibold text-gray-700">{key}</span>
                <span className="ml-2 text-xs text-gray-400">
                  {isArray
                    ? `[${(value as unknown[]).length} item${(value as unknown[]).length !== 1 ? 's' : ''}]`
                    : '{object}'}
                </span>
              </button>
              {isExp && (
                <div className="px-4 py-3 bg-white">
                  <pre className="text-xs text-gray-700 whitespace-pre-wrap break-all font-mono">
                    {JSON.stringify(value, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          );
        }

        return (
          <div key={key} className="flex gap-3 items-start">
            <span className="text-xs font-semibold text-gray-600 w-32 shrink-0 pt-0.5">
              {key}
            </span>
            <span className="text-xs text-gray-800 break-all">{String(value)}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── View Modal ───────────────────────────────────────────────────────────────

function ViewModal({
  qs,
  onClose,
  onEdit,
}: {
  qs: QuestionSetResponse;
  onClose: () => void;
  onEdit: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl mx-4 max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 sticky top-0 bg-white rounded-t-xl">
          <h2 className="text-lg font-semibold text-gray-900 truncate pr-4">
            Question Set
            {qs.candidate_full_name && (
              <span className="ml-2 text-base font-normal text-gray-500">
                — {qs.candidate_full_name}
              </span>
            )}
          </h2>
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={onEdit}
              className="flex items-center px-3 py-1.5 text-sm text-indigo-700 bg-indigo-50 hover:bg-indigo-100 rounded-lg transition"
            >
              <Pencil className="w-3.5 h-3.5 mr-1.5" />
              Edit
            </button>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition p-1"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="overflow-y-auto flex-1 p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                Candidate
              </p>
              <p className="font-mono text-xs text-gray-700">
                {qs.candidate_full_name || qs.candidate_profile_id}
              </p>
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                Job Description
              </p>
              <p className="text-xs text-gray-700">
                {qs.job_description_title || (
                  <span className="font-mono">{qs.job_description_id}</span>
                )}
              </p>
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                Created
              </p>
              <p className="text-xs text-gray-700">
                {new Date(qs.created_at).toLocaleString()}
              </p>
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                ID
              </p>
              <p className="font-mono text-xs text-gray-500 truncate">{qs.id}</p>
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              Question Payload
            </p>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <JsonPayloadView payload={qs.question_payload} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Create / Edit Form ───────────────────────────────────────────────────────

interface FormState {
  candidate_profile_id: string;
  job_description_id: string;
  payloadJson: string;
}

function QuestionSetForm({
  initial,
  isEdit,
  onSubmit,
  onClose,
}: {
  initial: FormState | null;
  isEdit: boolean;
  onSubmit: (data: FormState) => Promise<void>;
  onClose: () => void;
}) {
  const [form, setForm] = useState<FormState>(
    initial ?? {
      candidate_profile_id: '',
      job_description_id: '',
      payloadJson: '{}',
    },
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jsonError, setJsonError] = useState<string | null>(null);

  const set = (patch: Partial<FormState>) =>
    setForm((prev) => ({ ...prev, ...patch }));

  const validateJson = (raw: string): boolean => {
    try {
      JSON.parse(raw);
      setJsonError(null);
      return true;
    } catch {
      setJsonError('Invalid JSON — please fix before saving.');
      return false;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!isEdit) {
      if (!form.candidate_profile_id.trim()) {
        setError('Candidate profile ID is required.');
        return;
      }
      if (!form.job_description_id.trim()) {
        setError('Job description ID is required.');
        return;
      }
    }

    if (!validateJson(form.payloadJson)) return;

    setSaving(true);
    setError(null);
    try {
      await onSubmit(form);
    } catch (err: any) {
      setError(
        err.response?.data?.detail || err.message || 'Failed to save question set',
      );
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-xl mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 sticky top-0 bg-white rounded-t-xl">
          <h2 className="text-lg font-semibold text-gray-900">
            {isEdit ? 'Edit Question Payload' : 'Generate Question Set'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 flex items-start">
              <AlertCircle className="w-4 h-4 mr-2 mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          {!isEdit && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Candidate Profile ID <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={form.candidate_profile_id}
                  onChange={(e) => set({ candidate_profile_id: e.target.value })}
                  placeholder="UUID of the candidate profile"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent font-mono"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Job Description ID <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={form.job_description_id}
                  onChange={(e) => set({ job_description_id: e.target.value })}
                  placeholder="UUID of the job description"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent font-mono"
                />
              </div>
            </>
          )}

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-sm font-medium text-gray-700">
                Question Payload (JSON) <span className="text-red-500">*</span>
              </label>
              {isEdit && (
                <span className="text-xs text-amber-600 bg-amber-50 px-2 py-0.5 rounded font-medium">
                  Replaces entire payload
                </span>
              )}
            </div>
            <textarea
              rows={10}
              value={form.payloadJson}
              onChange={(e) => {
                set({ payloadJson: e.target.value });
                setJsonError(null);
              }}
              onBlur={(e) => validateJson(e.target.value)}
              placeholder={'{\n  "technical": [...],\n  "behavioral": [...]\n}'}
              className={`w-full px-3 py-2 border rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-y ${
                jsonError ? 'border-red-400 bg-red-50' : 'border-gray-300'
              }`}
            />
            {jsonError && (
              <p className="text-xs text-red-600 mt-1 flex items-center">
                <AlertCircle className="w-3.5 h-3.5 mr-1" />
                {jsonError}
              </p>
            )}
            <p className="text-xs text-gray-400 mt-1">
              Free-form JSON — structure is not validated by the backend.
            </p>
          </div>

          <div className="flex justify-end space-x-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving || !!jsonError}
              className="px-4 py-2 text-sm text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center"
            >
              {saving ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Saving...
                </>
              ) : (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  {isEdit ? 'Save Changes' : 'Create'}
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function InterviewQuestionsPage() {
  const { toast } = useToast();
  const [items, setItems] = useState<QuestionSetResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [candidateFilter, setCandidateFilter] = useState('');
  const [jdFilter, setJdFilter] = useState('');

  // Modals
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [viewingItem, setViewingItem] = useState<QuestionSetResponse | null>(null);
  const [editingItem, setEditingItem] = useState<QuestionSetResponse | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  // ── Fetch ──────────────────────────────────────────────────────────────────

  const fetchItems = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const params: Record<string, unknown> = { limit: PAGE_LIMIT, offset };
      if (candidateFilter.trim()) params.candidate_profile_id = candidateFilter.trim();
      if (jdFilter.trim()) params.job_description_id = jdFilter.trim();
      const data: PaginatedResponse<QuestionSetResponse> = await listQuestionSets(params);
      setItems(data.items);
      setTotal(data.total);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to load question sets';
      setError(msg);
      toast(msg, 'error');
    } finally {
      setIsLoading(false);
    }
  }, [offset, candidateFilter, jdFilter]);

  useEffect(() => {
    fetchItems();
  }, [fetchItems]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setOffset(0);
  }, [candidateFilter, jdFilter]);

  // ── Actions ────────────────────────────────────────────────────────────────

  const handleCreate = async (data: FormState) => {
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(data.payloadJson);
    } catch {
      throw new Error('Invalid JSON payload');
    }
    await createQuestionSet({
      candidate_profile_id: data.candidate_profile_id,
      job_description_id: data.job_description_id,
      generated_by_user_id: MOCK_USER_ID,
      question_payload: payload,
    });
    setShowCreateForm(false);
    await fetchItems();
  };

  const handleEdit = async (data: FormState) => {
    if (!editingItem) return;
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(data.payloadJson);
    } catch {
      throw new Error('Invalid JSON payload');
    }
    const updated = await updateQuestionSet(editingItem.id, payload);
    setEditingItem(null);
    setItems((prev) => prev.map((i) => (i.id === editingItem.id ? updated : i)));
  };

  const handleDelete = async (id: string) => {
    setDeleting(id);
    setError(null);
    try {
      await deleteQuestionSet(id); // 204 — no body
      setDeleteConfirm(null);
      setItems((prev) => prev.filter((i) => i.id !== id));
      setTotal((prev) => Math.max(0, prev - 1));
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to delete question set';
      setError(msg);
      toast(msg, 'error');
    } finally {
      setDeleting(null);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  const filterInput = (
    value: string,
    setter: (v: string) => void,
    placeholder: string,
  ) => (
    <input
      type="text"
      value={value}
      onChange={(e) => setter(e.target.value)}
      placeholder={placeholder}
      className="px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent font-mono w-72"
    />
  );

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Interview Questions</h1>
          <p className="text-gray-500 text-sm mt-1">
            Generate and manage AI-created interview question sets per candidate and job description.
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition"
        >
          <Plus className="w-4 h-4 mr-2" />
          Generate Question Set
        </button>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 shrink-0" />
          <div className="flex-1 text-sm text-red-700">{error}</div>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-600 transition ml-3"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-4 items-center">
        <span className="text-sm text-gray-600 font-medium">Filter:</span>
        {filterInput(
          candidateFilter,
          setCandidateFilter,
          'Candidate profile ID...',
        )}
        {filterInput(
          jdFilter,
          setJdFilter,
          'Job description ID...',
        )}
        {(candidateFilter || jdFilter) && (
          <button
            onClick={() => {
              setCandidateFilter('');
              setJdFilter('');
            }}
            className="text-xs text-gray-500 hover:text-gray-800 flex items-center gap-1 transition"
          >
            <X className="w-3.5 h-3.5" />
            Clear filters
          </button>
        )}
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <h2 className="text-lg font-medium text-gray-900">
            Question Sets
            {total > 0 && (
              <span className="ml-2 text-sm text-gray-500 font-normal">({total} total)</span>
            )}
          </h2>
        </div>

        {isLoading ? (
          <div className="p-8 text-center text-gray-500 text-sm">Loading question sets...</div>
        ) : items.length === 0 ? (
          <div className="p-12 text-center">
            <HelpCircle className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500 text-sm">No question sets found.</p>
            <button
              onClick={() => setShowCreateForm(true)}
              className="mt-3 text-indigo-600 hover:text-indigo-700 text-sm font-medium transition"
            >
              Generate your first question set
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-gray-600">
              <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 font-medium">Candidate</th>
                  <th className="px-6 py-3 font-medium">Job Description</th>
                  <th className="px-6 py-3 font-medium">Payload Keys</th>
                  <th className="px-6 py-3 font-medium">Created</th>
                  <th className="px-6 py-3 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {items.map((qs) => {
                  const payloadKeys = Object.keys(qs.question_payload);
                  return (
                    <tr key={qs.id} className="hover:bg-gray-50 transition-colors">
                      {/* Candidate */}
                      <td className="px-6 py-4">
                        <span
                          className="font-mono text-xs text-gray-700 truncate block max-w-[160px]"
                          title={qs.candidate_full_name || qs.candidate_profile_id}
                        >
                          {qs.candidate_full_name || qs.candidate_profile_id}
                        </span>
                      </td>

                      {/* Job Description */}
                      <td className="px-6 py-4">
                        <span
                          className="text-xs text-gray-700 truncate block max-w-[200px]"
                          title={qs.job_description_title || qs.job_description_id}
                        >
                          {qs.job_description_title || (
                            <span className="font-mono">{qs.job_description_id}</span>
                          )}
                        </span>
                      </td>

                      {/* Payload Keys */}
                      <td className="px-6 py-4">
                        {payloadKeys.length === 0 ? (
                          <span className="text-xs text-gray-400 italic">empty</span>
                        ) : (
                          <div className="flex flex-wrap gap-1">
                            {payloadKeys.slice(0, 4).map((k) => (
                              <span
                                key={k}
                                className="px-2 py-0.5 rounded text-xs bg-indigo-50 text-indigo-700 font-medium"
                              >
                                {k}
                              </span>
                            ))}
                            {payloadKeys.length > 4 && (
                              <span className="text-xs text-gray-400">
                                +{payloadKeys.length - 4} more
                              </span>
                            )}
                          </div>
                        )}
                      </td>

                      {/* Created */}
                      <td className="px-6 py-4 text-gray-500 whitespace-nowrap text-xs">
                        {new Date(qs.created_at).toLocaleDateString()}
                      </td>

                      {/* Actions */}
                      <td className="px-6 py-4 text-right">
                        {deleteConfirm === qs.id ? (
                          <div className="flex items-center justify-end space-x-2">
                            <span className="text-xs text-red-600 font-medium">Delete?</span>
                            <button
                              onClick={() => handleDelete(qs.id)}
                              disabled={deleting === qs.id}
                              className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition disabled:opacity-50"
                            >
                              {deleting === qs.id ? '...' : 'Yes'}
                            </button>
                            <button
                              onClick={() => setDeleteConfirm(null)}
                              className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1"
                            >
                              Cancel
                            </button>
                          </div>
                        ) : (
                          <div className="flex items-center justify-end space-x-1">
                            <button
                              onClick={() => setViewingItem(qs)}
                              className="text-gray-400 hover:text-indigo-600 transition p-1 rounded-full hover:bg-indigo-50"
                              title="View"
                            >
                              <Eye className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => setEditingItem(qs)}
                              className="text-gray-400 hover:text-indigo-600 transition p-1 rounded-full hover:bg-indigo-50"
                              title="Edit payload"
                            >
                              <Pencil className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => setDeleteConfirm(qs.id)}
                              className="text-gray-400 hover:text-red-600 transition p-1 rounded-full hover:bg-red-50"
                              title="Delete"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {/* Pagination */}
            {total > PAGE_LIMIT && (
              <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex items-center justify-between">
                <span className="text-sm text-gray-500">
                  Showing {offset + 1} to {Math.min(offset + PAGE_LIMIT, total)} of {total} entries
                </span>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setOffset(Math.max(0, offset - PAGE_LIMIT))}
                    disabled={offset === 0}
                    className="px-3 py-1 text-sm border border-gray-300 rounded bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => setOffset(offset + PAGE_LIMIT)}
                    disabled={offset + PAGE_LIMIT >= total}
                    className="px-3 py-1 text-sm border border-gray-300 rounded bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Modals */}
      {showCreateForm && (
        <QuestionSetForm
          initial={null}
          isEdit={false}
          onSubmit={handleCreate}
          onClose={() => setShowCreateForm(false)}
        />
      )}

      {editingItem && (
        <QuestionSetForm
          initial={{
            candidate_profile_id: editingItem.candidate_profile_id,
            job_description_id: editingItem.job_description_id,
            payloadJson: JSON.stringify(editingItem.question_payload, null, 2),
          }}
          isEdit={true}
          onSubmit={handleEdit}
          onClose={() => setEditingItem(null)}
        />
      )}

      {viewingItem && (
        <ViewModal
          qs={viewingItem}
          onClose={() => setViewingItem(null)}
          onEdit={() => {
            setEditingItem(viewingItem);
            setViewingItem(null);
          }}
        />
      )}
    </div>
  );
}
