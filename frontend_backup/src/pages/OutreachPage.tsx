import React, { useState, useCallback, useEffect } from 'react';
import {
  Mail,
  Plus,
  Pencil,
  Trash2,
  AlertCircle,
  X,
  Check,
  Send,
  Filter,
} from 'lucide-react';
import type { OutreachResponse } from '../types';
import {
  createOutreach,
  listOutreach,
  updateOutreach,
  deleteOutreach,
} from '../api/outreach';
import { useToast } from '../components/Toast';

const MOCK_USER_ID = '123e4567-e89b-12d3-a456-426614174000';
const PAGE_LIMIT = 50;

// ─── Status badge ────────────────────────────────────────────────────────────

const STATUS_STYLES: Record<OutreachResponse['sent_status'], string> = {
  not_sent: 'bg-yellow-100 text-yellow-800',
  sent: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
};

const STATUS_LABELS: Record<OutreachResponse['sent_status'], string> = {
  not_sent: 'Not Sent',
  sent: 'Sent',
  failed: 'Failed',
};

function StatusBadge({ status }: { status: OutreachResponse['sent_status'] }) {
  return (
    <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${STATUS_STYLES[status]}`}>
      {STATUS_LABELS[status]}
    </span>
  );
}

// ─── Form ────────────────────────────────────────────────────────────────────

interface OutreachFormData {
  candidate_profile_id: string;
  content_source: 'ai_draft' | 'template';
  subject: string;
  body: string;
}

const emptyForm: OutreachFormData = {
  candidate_profile_id: '',
  content_source: 'ai_draft',
  subject: '',
  body: '',
};

interface OutreachFormProps {
  initial: OutreachFormData | null;
  isEdit: boolean;
  onSubmit: (data: OutreachFormData) => Promise<void>;
  onClose: () => void;
}

function OutreachForm({ initial, isEdit, onSubmit, onClose }: OutreachFormProps) {
  const [form, setForm] = useState<OutreachFormData>(initial ?? emptyForm);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = (patch: Partial<OutreachFormData>) =>
    setForm((prev) => ({ ...prev, ...patch }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.candidate_profile_id.trim()) {
      setError('Candidate profile ID is required.');
      return;
    }
    if (!form.subject.trim()) {
      setError('Subject is required.');
      return;
    }
    if (!form.body.trim()) {
      setError('Body is required.');
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await onSubmit(form);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to save message');
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 sticky top-0 bg-white">
          <h2 className="text-lg font-semibold text-gray-900">
            {isEdit ? 'Edit Message' : 'New Outreach Message'}
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
          )}

          {!isEdit && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Content Source <span className="text-red-500">*</span>
              </label>
              <div className="flex rounded-lg border border-gray-300 overflow-hidden text-sm">
                {(['ai_draft', 'template'] as const).map((src) => (
                  <button
                    key={src}
                    type="button"
                    onClick={() => set({ content_source: src })}
                    className={`flex-1 px-4 py-2 transition ${
                      form.content_source === src
                        ? 'bg-indigo-600 text-white font-medium'
                        : 'bg-white text-gray-600 hover:bg-gray-50'
                    } ${src === 'template' ? 'border-l border-gray-300' : ''}`}
                  >
                    {src === 'ai_draft' ? 'AI Draft' : 'Template'}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Subject <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={form.subject}
              onChange={(e) => set({ subject: e.target.value })}
              placeholder="Email subject line"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Body <span className="text-red-500">*</span>
            </label>
            <textarea
              rows={8}
              value={form.body}
              onChange={(e) => set({ body: e.target.value })}
              placeholder="Write the outreach message body here..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-y"
            />
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
              disabled={saving}
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

// ─── View modal ───────────────────────────────────────────────────────────────

function OutreachViewModal({
  message,
  onClose,
}: {
  message: OutreachResponse;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 sticky top-0 bg-white">
          <h2 className="text-lg font-semibold text-gray-900 truncate pr-4">
            {message.subject}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition shrink-0"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6 space-y-4 text-sm">
          <div className="flex flex-wrap gap-2 items-center">
            <StatusBadge status={message.sent_status} />
            <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-700">
              {message.content_source === 'ai_draft' ? 'AI Draft' : 'Template'}
            </span>
          </div>

          <div className="space-y-1">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Candidate
            </p>
            <p className="font-mono text-gray-800">
              {message.candidate_full_name || message.candidate_profile_id}
            </p>
          </div>

          {message.sent_at && (
            <div className="space-y-1">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Sent at
              </p>
              <p className="text-gray-700">
                {new Date(message.sent_at).toLocaleString()}
              </p>
            </div>
          )}

          <div className="space-y-1">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Body
            </p>
            <div className="bg-gray-50 rounded-lg p-4 whitespace-pre-wrap text-gray-800 leading-relaxed">
              {message.body}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

type SentStatusFilter = 'all' | OutreachResponse['sent_status'];

export default function OutreachPage() {
  const { toast } = useToast();
  const [messages, setMessages] = useState<OutreachResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<SentStatusFilter>('all');
  const [candidateFilter, setCandidateFilter] = useState('');

  // Modals
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingMessage, setEditingMessage] = useState<OutreachResponse | null>(null);
  const [viewingMessage, setViewingMessage] = useState<OutreachResponse | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  // Per-row action loading
  const [markingSent, setMarkingSent] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  // ── Fetch ──────────────────────────────────────────────────────────────────

  const fetchMessages = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const params: Record<string, unknown> = { limit: PAGE_LIMIT, offset };
      if (statusFilter !== 'all') params.sent_status = statusFilter;
      if (candidateFilter.trim()) params.candidate_profile_id = candidateFilter.trim();
      const data = await listOutreach(params);
      setMessages(data.items);
      setTotal(data.total);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to load outreach messages';
      setError(msg);
      toast(msg, 'error');
    } finally {
      setIsLoading(false);
    }
  }, [offset, statusFilter, candidateFilter]);

  useEffect(() => {
    fetchMessages();
  }, [fetchMessages]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setOffset(0);
  }, [statusFilter, candidateFilter]);

  // ── Actions ────────────────────────────────────────────────────────────────

  const handleCreate = async (data: OutreachFormData) => {
    await createOutreach({
      ...data,
      created_by_user_id: MOCK_USER_ID,
    });
    setShowCreateForm(false);
    await fetchMessages();
  };

  const handleEdit = async (data: OutreachFormData) => {
    if (!editingMessage) return;
    await updateOutreach(editingMessage.id, {
      subject: data.subject,
      body: data.body,
    });
    setEditingMessage(null);
    await fetchMessages();
  };

  const handleMarkSent = async (id: string) => {
    setMarkingSent(id);
    setError(null);
    try {
      const updated = await updateOutreach(id, { sent_status: 'sent' });
      setMessages((prev) => prev.map((m) => (m.id === id ? updated : m)));
      toast('Message marked as sent.', 'success');
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to mark as sent';
      setError(msg);
      toast(msg, 'error');
    } finally {
      setMarkingSent(null);
    }
  };

  const handleDelete = async (id: string) => {
    setDeleting(id);
    setError(null);
    try {
      await deleteOutreach(id); // 204 — no body to parse
      setDeleteConfirm(null);
      setMessages((prev) => prev.filter((m) => m.id !== id));
      setTotal((prev) => Math.max(0, prev - 1));
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to delete message';
      setError(msg);
      toast(msg, 'error');
    } finally {
      setDeleting(null);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Outreach</h1>
          <p className="text-gray-500 text-sm mt-1">
            Create and manage outreach messages to candidates.
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Message
        </button>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 shrink-0" />
          <div className="flex-1 text-sm text-red-700">{error}</div>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 transition ml-3">
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-4">
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-gray-400" />
          <span className="text-sm text-gray-600 font-medium">Filter:</span>
        </div>

        <div className="flex rounded-lg border border-gray-200 overflow-hidden text-sm">
          {(['all', 'not_sent', 'sent', 'failed'] as SentStatusFilter[]).map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`px-3 py-1.5 transition border-l first:border-l-0 border-gray-200 ${
                statusFilter === s
                  ? 'bg-indigo-600 text-white font-medium'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              {s === 'all' ? 'All' : STATUS_LABELS[s as OutreachResponse['sent_status']]}
            </button>
          ))}
        </div>

        <input
          type="text"
          value={candidateFilter}
          onChange={(e) => setCandidateFilter(e.target.value)}
          placeholder="Filter by candidate ID..."
          className="px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent font-mono w-72"
        />
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
          <h2 className="text-lg font-medium text-gray-900">
            Messages
            {total > 0 && (
              <span className="ml-2 text-sm text-gray-500 font-normal">({total} total)</span>
            )}
          </h2>
        </div>

        {isLoading ? (
          <div className="p-8 text-center text-gray-500 text-sm">Loading messages...</div>
        ) : messages.length === 0 ? (
          <div className="p-12 text-center">
            <Mail className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500 text-sm">No outreach messages found.</p>
            <button
              onClick={() => setShowCreateForm(true)}
              className="mt-3 text-indigo-600 hover:text-indigo-700 text-sm font-medium transition"
            >
              Create your first message
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-gray-600">
              <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 font-medium">Candidate</th>
                  <th className="px-6 py-3 font-medium">Subject</th>
                  <th className="px-6 py-3 font-medium">Source</th>
                  <th className="px-6 py-3 font-medium">Status</th>
                  <th className="px-6 py-3 font-medium">Sent At</th>
                  <th className="px-6 py-3 font-medium">Created</th>
                  <th className="px-6 py-3 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {messages.map((msg) => (
                  <tr key={msg.id} className="hover:bg-gray-50 transition-colors">
                    {/* Candidate */}
                    <td className="px-6 py-4">
                      <span
                        className="font-mono text-xs text-gray-700 truncate block max-w-[160px]"
                        title={msg.candidate_full_name || msg.candidate_profile_id}
                      >
                        {msg.candidate_full_name || msg.candidate_profile_id}
                      </span>
                    </td>

                    {/* Subject */}
                    <td className="px-6 py-4">
                      <button
                        onClick={() => setViewingMessage(msg)}
                        className="text-indigo-600 hover:text-indigo-800 hover:underline transition truncate max-w-[200px] block text-left"
                        title={msg.subject}
                      >
                        {msg.subject}
                      </button>
                    </td>

                    {/* Content source */}
                    <td className="px-6 py-4">
                      <span className="px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-700">
                        {msg.content_source === 'ai_draft' ? 'AI Draft' : 'Template'}
                      </span>
                    </td>

                    {/* Status */}
                    <td className="px-6 py-4">
                      <StatusBadge status={msg.sent_status} />
                    </td>

                    {/* Sent at */}
                    <td className="px-6 py-4 text-gray-500 text-xs whitespace-nowrap">
                      {msg.sent_at
                        ? new Date(msg.sent_at).toLocaleString()
                        : <span className="text-gray-300">—</span>}
                    </td>

                    {/* Created at */}
                    <td className="px-6 py-4 text-gray-500 whitespace-nowrap text-xs">
                      {new Date(msg.created_at).toLocaleDateString()}
                    </td>

                    {/* Actions */}
                    <td className="px-6 py-4 text-right">
                      {deleteConfirm === msg.id ? (
                        <div className="flex items-center justify-end space-x-2">
                          <span className="text-xs text-red-600 font-medium">Delete?</span>
                          <button
                            onClick={() => handleDelete(msg.id)}
                            disabled={deleting === msg.id}
                            className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition disabled:opacity-50"
                          >
                            {deleting === msg.id ? '...' : 'Yes'}
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
                          {/* Mark as sent — only for not_sent messages */}
                          {msg.sent_status === 'not_sent' && (
                            <button
                              onClick={() => handleMarkSent(msg.id)}
                              disabled={markingSent === msg.id}
                              className="text-gray-400 hover:text-green-600 transition p-1 rounded-full hover:bg-green-50 disabled:opacity-50"
                              title="Mark as sent"
                            >
                              {markingSent === msg.id ? (
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-600" />
                              ) : (
                                <Send className="w-4 h-4" />
                              )}
                            </button>
                          )}

                          {/* Edit — only for not_sent messages */}
                          {msg.sent_status === 'not_sent' && (
                            <button
                              onClick={() => setEditingMessage(msg)}
                              className="text-gray-400 hover:text-indigo-600 transition p-1 rounded-full hover:bg-indigo-50"
                              title="Edit"
                            >
                              <Pencil className="w-4 h-4" />
                            </button>
                          )}

                          {/* Delete */}
                          <button
                            onClick={() => setDeleteConfirm(msg.id)}
                            className="text-gray-400 hover:text-red-600 transition p-1 rounded-full hover:bg-red-50"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
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
        <OutreachForm
          initial={null}
          isEdit={false}
          onSubmit={handleCreate}
          onClose={() => setShowCreateForm(false)}
        />
      )}

      {editingMessage && (
        <OutreachForm
          initial={{
            candidate_profile_id: editingMessage.candidate_profile_id,
            content_source: editingMessage.content_source,
            subject: editingMessage.subject,
            body: editingMessage.body,
          }}
          isEdit={true}
          onSubmit={handleEdit}
          onClose={() => setEditingMessage(null)}
        />
      )}

      {viewingMessage && (
        <OutreachViewModal
          message={viewingMessage}
          onClose={() => setViewingMessage(null)}
        />
      )}
    </div>
  );
}
