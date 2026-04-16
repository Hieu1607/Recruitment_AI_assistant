import React, { useState, useRef } from 'react';
import { useResumes } from '../hooks/useResumes';
import { Upload, FileText, Trash2, AlertCircle, CheckCircle2, Clock, X } from 'lucide-react';

export default function ResumesPage() {
  const {
    resumes,
    isLoading,
    isUploading,
    error,
    uploadBatch,
    removeResume,
    setError,
    offset,
    limit,
    setOffset,
    total,
  } = useResumes();

  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const validateAndUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    const nonPdfs = fileArray.filter((f) => f.type !== 'application/pdf');

    if (nonPdfs.length > 0) {
      setError('Only PDF files are allowed.');
      return;
    }

    // Harcoded user for now, as auth is planned later
    const MOCK_USER_ID = '123e4567-e89b-12d3-a456-426614174000';
    await uploadBatch(fileArray, MOCK_USER_ID);
    if (fileInputRef.current) {
        fileInputRef.current.value = '';
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError(null);
    await validateAndUpload(e.dataTransfer.files);
  };

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    setError(null);
    await validateAndUpload(e.target.files);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'processed':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case 'processing':
      case 'uploaded':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <FileText className="w-5 h-5 text-gray-400" />;
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Resumes</h1>
          <p className="text-gray-500 text-sm mt-1">
            Upload and manage candidate resumes for parsing.
          </p>
        </div>
      </div>

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

      {/* Upload Area */}
      <div
        className={`mb-8 p-10 border-2 border-dashed rounded-xl text-center transition-colors ${
          dragActive
            ? 'border-indigo-500 bg-indigo-50'
            : isUploading
            ? 'border-gray-200 bg-gray-50 opacity-70'
            : 'border-gray-300 hover:border-gray-400 bg-white'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="application/pdf"
          onChange={handleChange}
          className="hidden"
          id="file-upload"
          disabled={isUploading}
        />
        <label
          htmlFor="file-upload"
          className={`flex flex-col items-center justify-center cursor-pointer ${
            isUploading ? 'cursor-not-allowed' : ''
          }`}
        >
          {isUploading ? (
            <div className="flex flex-col items-center">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600 mb-4"></div>
              <p className="text-sm font-medium text-gray-900">Processing Resumes...</p>
              <p className="text-xs text-yellow-600 mt-2 font-medium bg-yellow-50 px-3 py-1 rounded-full">
                ⚠️ This may take 30+ seconds for large batches
              </p>
            </div>
          ) : (
            <>
              <div className="p-3 bg-indigo-50 text-indigo-600 rounded-full mb-4">
                <Upload className="w-6 h-6" />
              </div>
              <p className="text-sm font-medium text-gray-900">
                Click to upload or drag and drop
              </p>
              <p className="text-xs text-gray-500 mt-1">PDF files only (max 10MB each)</p>
            </>
          )}
        </label>
      </div>

      {/* Resumes List */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex justify-between items-center">
          <h2 className="text-lg font-medium text-gray-900">
            Uploaded Resumes
            {total > 0 && <span className="ml-2 text-sm text-gray-500 font-normal">({total} total)</span>}
          </h2>
        </div>

        {isLoading ? (
          <div className="p-8 text-center text-gray-500 text-sm">Loading resumes...</div>
        ) : resumes.length === 0 ? (
          <div className="p-12 text-center">
            <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500 text-sm">No resumes uploaded yet.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-gray-600">
              <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 font-medium">File Name</th>
                  <th className="px-6 py-3 font-medium">Status</th>
                  <th className="px-6 py-3 font-medium">Uploaded Date</th>
                  <th className="px-6 py-3 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {resumes.map((resume) => (
                  <tr key={resume.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 font-medium text-gray-900">
                      <div className="flex items-center">
                         {getStatusIcon(resume.upload_status)}
                        <span className="ml-3 truncate max-w-[200px] sm:max-w-[300px]" title={resume.original_file_name}>
                          {resume.original_file_name}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-2.5 py-1 rounded-full text-xs font-medium capitalize ${
                          resume.upload_status === 'processed'
                            ? 'bg-green-100 text-green-800'
                            : resume.upload_status === 'failed'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}
                      >
                        {resume.upload_status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-500 whitespace-nowrap">
                      {resume.uploaded_at ? new Date(resume.uploaded_at).toLocaleDateString() : 'N/A'}
                    </td>
                    <td className="px-6 py-4 text-right">
                      {deleteConfirm === resume.id ? (
                        <div className="flex items-center justify-end space-x-2">
                            <span className="text-xs text-red-600 font-medium mr-2">Delete file too?</span>
                            <button
                                onClick={() => {
                                    removeResume(resume.id, true);
                                    setDeleteConfirm(null);
                                }}
                                className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition"
                            >
                                Yes
                            </button>
                            <button
                                onClick={() => {
                                    removeResume(resume.id, false);
                                    setDeleteConfirm(null);
                                }}
                                className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-2 py-1 rounded transition"
                            >
                                Record Only
                            </button>
                            <button
                                onClick={() => setDeleteConfirm(null)}
                                className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1"
                            >
                                Cancel
                            </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setDeleteConfirm(resume.id)}
                          className="text-gray-400 hover:text-red-600 transition p-1 rounded-full hover:bg-red-50"
                          title="Delete Resume"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
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
