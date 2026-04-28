import React, { useState, useEffect } from 'react';
import { useJobDescriptions } from '../hooks/useJobDescriptions';
import { useToast } from '../components/Toast';
import {
  FileText,
  Plus,
  Pencil,
  Trash2,
  AlertCircle,
  X,
  Check,
  ToggleLeft,
  ToggleRight,
} from 'lucide-react';

const MOCK_USER_ID = '123e4567-e89b-12d3-a456-426614174000';

interface JDFormData {
  title: string;
  jd_text: string;
}

const emptyForm: JDFormData = { title: '', jd_text: '' };

export default function JobDescriptionsPage() {
  const { toast } = useToast();
  const {
    jobDescriptions,
    isLoading,
    error,
    addJobDescription,
    editJobDescription,
    removeJobDescription,
    setError,
    offset,
    limit,
    setOffset,
    total,
  } = useJobDescriptions();

  useEffect(() => {
    if (error) toast(error, 'error');
  }, [error]); // eslint-disable-line react-hooks/exhaustive-deps

  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [formData, setFormData] = useState<JDFormData>(emptyForm);
  const [isSaving, setIsSaving] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const openCreateForm = () => {
    setEditingId(null);
    setFormData(emptyForm);
    setFormError(null);
    setShowForm(true);
  };

  const openEditForm = (jd: { id: string; title: string | null; jd_text: string }) => {
    setEditingId(jd.id);
    setFormData({ title: jd.title ?? '', jd_text: jd.jd_text });
    setFormError(null);
    setShowForm(true);
  };

  const closeForm = () => {
    setShowForm(false);
    setEditingId(null);
    setFormData(emptyForm);
    setFormError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.jd_text.trim()) {
      setFormError('Job description text is required.');
      return;
    }

    setIsSaving(true);
    setFormError(null);

    if (editingId) {
      const payload: Record<string, unknown> = {};
      payload.jd_text = formData.jd_text;
      payload.title = formData.title || null;
      const result = await editJobDescription(editingId, payload as any);
      if (result) closeForm();
    } else {
      const result = await addJobDescription({
        jd_text: formData.jd_text,
        created_by_user_id: MOCK_USER_ID,
        title: formData.title || undefined,
      });
      if (result) closeForm();
    }
    setIsSaving(false);
  };

  const handleToggleActive = async (jd: { id: string; is_active: boolean }) => {
    await editJobDescription(jd.id, { is_active: !jd.is_active });
  };

  const handleDelete = async (id: string) => {
    await removeJobDescription(id);
    setDeleteConfirm(null);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Job Descriptions</h1>
          <p className="text-gray-500 text-sm mt-1">
            Create and manage job descriptions for candidate scoring.
          </p>
        </div>
        <button
          onClick={openCreateForm}
          className="flex items-center px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition"
        >
          <Plus className="w-4 h-4 mr-2" />
          New JD
        </button>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 shrink-0" />
          <div className="flex-1">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <div className="text-sm text-red-700 mt-1">{error}</div>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-red-500 hover:text-red-700 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Create / Edit Form Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">
                {editingId ? 'Edit Job Description' : 'New Job Description'}
              </h2>
              <button
                onClick={closeForm}
                className="text-gray-400 hover:text-gray-600 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleSubmit} className="p-6 space-y-4">
              {formError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                  {formError}
                </div>
              )}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Title <span className="text-gray-400">(optional, max 255 chars)</span>
                </label>
                <input
                  type="text"
                  maxLength={255}
                  value={formData.title}
                  onChange={(e) => setFormData((prev) => ({ ...prev, title: e.target.value }))}
                  placeholder="e.g. Senior Frontend Engineer"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description <span className="text-red-500">*</span>
                </label>
                <textarea
                  rows={8}
                  value={formData.jd_text}
                  onChange={(e) => setFormData((prev) => ({ ...prev, jd_text: e.target.value }))}
                  placeholder="Paste the full job description text here..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-y"
                />
              </div>
              <div className="flex justify-end space-x-3 pt-2">
                <button
                  type="button"
                  onClick={closeForm}
                  className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSaving}
                  className="px-4 py-2 text-sm text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center"
                >
                  {isSaving ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Check className="w-4 h-4 mr-2" />
                      {editingId ? 'Save Changes' : 'Create'}
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Job Descriptions List */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex justify-between items-center">
          <h2 className="text-lg font-medium text-gray-900">
            Job Descriptions
            {total > 0 && (
              <span className="ml-2 text-sm text-gray-500 font-normal">({total} total)</span>
            )}
          </h2>
        </div>

        {isLoading ? (
          <div className="p-8 text-center text-gray-500 text-sm">Loading job descriptions...</div>
        ) : jobDescriptions.length === 0 ? (
          <div className="p-12 text-center">
            <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500 text-sm">No job descriptions yet.</p>
            <button
              onClick={openCreateForm}
              className="mt-3 text-indigo-600 hover:text-indigo-700 text-sm font-medium transition"
            >
              Create your first job description
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-gray-600">
              <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 font-medium">Title</th>
                  <th className="px-6 py-3 font-medium">Description</th>
                  <th className="px-6 py-3 font-medium">Status</th>
                  <th className="px-6 py-3 font-medium">Created</th>
                  <th className="px-6 py-3 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {jobDescriptions.map((jd) => (
                  <tr key={jd.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 font-medium text-gray-900">
                      <span className="truncate max-w-[200px] block" title={jd.title ?? 'Untitled'}>
                        {jd.title || <span className="text-gray-400 italic">Untitled</span>}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-600">
                      <span
                        className="truncate max-w-[300px] block"
                        title={jd.jd_text}
                      >
                        {jd.jd_text.length > 100
                          ? jd.jd_text.substring(0, 100) + '...'
                          : jd.jd_text}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                          jd.is_active
                            ? 'bg-green-100 text-green-800'
                            : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {jd.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-500 whitespace-nowrap">
                      {new Date(jd.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 text-right">
                      {deleteConfirm === jd.id ? (
                        <div className="flex items-center justify-end space-x-2">
                          <span className="text-xs text-red-600 font-medium">Confirm?</span>
                          <button
                            onClick={() => handleDelete(jd.id)}
                            className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition"
                          >
                            Delete
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
                            onClick={() => handleToggleActive(jd)}
                            className="text-gray-400 hover:text-indigo-600 transition p-1 rounded-full hover:bg-indigo-50"
                            title={jd.is_active ? 'Deactivate' : 'Activate'}
                          >
                            {jd.is_active ? (
                              <ToggleRight className="w-4 h-4 text-green-600" />
                            ) : (
                              <ToggleLeft className="w-4 h-4" />
                            )}
                          </button>
                          <button
                            onClick={() => openEditForm(jd)}
                            className="text-gray-400 hover:text-indigo-600 transition p-1 rounded-full hover:bg-indigo-50"
                            title="Edit"
                          >
                            <Pencil className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => setDeleteConfirm(jd.id)}
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

            {/* Pagination Controls */}
            {total > limit && (
              <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex items-center justify-between">
                <span className="text-sm text-gray-500">
                  Showing {offset + 1} to {Math.min(offset + limit, total)} of {total} entries
                </span>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setOffset(Math.max(0, offset - limit))}
                    disabled={offset === 0}
                    className="px-3 py-1 text-sm border border-gray-300 rounded bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => setOffset(offset + limit)}
                    disabled={offset + limit >= total}
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
    </div>
  );
}
