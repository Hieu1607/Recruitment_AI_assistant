import React, { useState, useEffect, useCallback } from 'react';
import {
  FolderOpen,
  Plus,
  Pencil,
  Trash2,
  AlertCircle,
  X,
  Check,
  ChevronDown,
  ChevronRight,
  Users,
  MessageSquare,
  UserMinus,
} from 'lucide-react';
import { useToast } from '../components/Toast';
import type {
  CollectionResponse,
  ShortlistItemResponse,
  SessionResponse,
  TurnResponse,
} from '../types';
import {
  listCollections,
  createCollection,
  renameCollection,
  deleteCollection,
  listItems,
  removeItem,
  listSessions,
  listTurns,
  deleteSession,
} from '../api/shortlist';

const MOCK_USER_ID = '123e4567-e89b-12d3-a456-426614174000';

// ─── Collections Panel ──────────────────────────────────────────────────────

function CollectionsSidebar({
  collections,
  selectedId,
  onSelect,
  onNew,
  onRename,
  onDelete,
  isLoading,
  error,
  onDismissError,
}: {
  collections: CollectionResponse[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (c: CollectionResponse) => void;
  onDelete: (id: string) => void;
  isLoading: boolean;
  error: string | null;
  onDismissError: () => void;
}) {
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  return (
    <div className="w-64 shrink-0 border-r border-gray-200 flex flex-col bg-gray-50 min-h-0">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-700">Collections</h2>
        <button
          onClick={onNew}
          className="p-1 text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 rounded transition"
          title="New collection"
        >
          <Plus className="w-4 h-4" />
        </button>
      </div>

      {error && (
        <div className="mx-3 mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700 flex items-start">
          <AlertCircle className="w-3 h-3 mr-1.5 mt-0.5 shrink-0" />
          <span className="flex-1">{error}</span>
          <button onClick={onDismissError}>
            <X className="w-3 h-3 text-red-400 hover:text-red-600" />
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto py-2">
        {isLoading ? (
          <div className="px-4 py-3 text-xs text-gray-400">Loading...</div>
        ) : collections.length === 0 ? (
          <div className="px-4 py-6 text-center">
            <FolderOpen className="w-8 h-8 text-gray-300 mx-auto mb-2" />
            <p className="text-xs text-gray-400">No collections yet</p>
          </div>
        ) : (
          collections.map((c) => (
            <div
              key={c.id}
              className={`group flex items-center px-3 py-2 cursor-pointer rounded mx-2 mb-0.5 transition ${
                selectedId === c.id
                  ? 'bg-indigo-100 text-indigo-800'
                  : 'hover:bg-gray-100 text-gray-700'
              }`}
              onClick={() => onSelect(c.id)}
            >
              <FolderOpen
                className={`w-4 h-4 mr-2 shrink-0 ${
                  selectedId === c.id ? 'text-indigo-600' : 'text-gray-400'
                }`}
              />
              <span className="flex-1 text-sm truncate" title={c.name}>
                {c.name}
              </span>
              <span className="text-xs text-gray-400 mr-1">{c.item_count}</span>

              {deleteConfirm === c.id ? (
                <div
                  className="flex items-center space-x-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    onClick={() => {
                      onDelete(c.id);
                      setDeleteConfirm(null);
                    }}
                    className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-1.5 py-0.5 rounded"
                  >
                    Del
                  </button>
                  <button
                    onClick={() => setDeleteConfirm(null)}
                    className="text-xs text-gray-500 hover:text-gray-700"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ) : (
                <div
                  className="hidden group-hover:flex items-center space-x-0.5"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    onClick={() => onRename(c)}
                    className="p-0.5 text-gray-400 hover:text-indigo-600 rounded"
                    title="Rename"
                  >
                    <Pencil className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => setDeleteConfirm(c.id)}
                    className="p-0.5 text-gray-400 hover:text-red-600 rounded"
                    title="Delete"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ─── Collection Items View ───────────────────────────────────────────────────

function CollectionItemsView({
  collection,
  items,
  isLoading,
  error,
  onRemoveItem,
}: {
  collection: CollectionResponse;
  items: ShortlistItemResponse[];
  isLoading: boolean;
  error: string | null;
  onRemoveItem: (itemId: string, candidateId: string) => void;
}) {
  const [removeConfirm, setRemoveConfirm] = useState<string | null>(null);

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900">{collection.name}</h2>
        <p className="text-xs text-gray-500 mt-0.5">
          {collection.item_count} candidate{collection.item_count !== 1 ? 's' : ''}
          {collection.source_query_turn_id && (
            <span className="ml-2 text-indigo-500">
              · sourced from query turn
            </span>
          )}
        </p>
      </div>

      {error && (
        <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 flex items-center">
          <AlertCircle className="w-4 h-4 mr-2 shrink-0" />
          {error}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <div className="text-sm text-gray-400">Loading candidates...</div>
        ) : items.length === 0 ? (
          <div className="py-12 text-center">
            <Users className="w-10 h-10 text-gray-300 mx-auto mb-3" />
            <p className="text-sm text-gray-400">No candidates in this collection.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {items.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200 hover:border-gray-300 transition"
              >
                <div>
                  <p className="text-sm font-medium text-gray-800 font-mono">
                    {item.candidate_profile_id}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Added {new Date(item.added_at).toLocaleString()}
                  </p>
                </div>

                {removeConfirm === item.id ? (
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-red-600 font-medium">Remove?</span>
                    <button
                      onClick={() => {
                        onRemoveItem(item.id, item.candidate_profile_id);
                        setRemoveConfirm(null);
                      }}
                      className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition"
                    >
                      Remove
                    </button>
                    <button
                      onClick={() => setRemoveConfirm(null)}
                      className="text-xs text-gray-500 hover:text-gray-700"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setRemoveConfirm(item.id)}
                    className="text-gray-400 hover:text-red-600 transition p-1 rounded-full hover:bg-red-50"
                    title="Remove from collection"
                  >
                    <UserMinus className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Session History Panel ───────────────────────────────────────────────────

function SessionHistoryPanel({
  sessions,
  isLoading,
  error,
  onDeleteSession,
}: {
  sessions: SessionResponse[];
  isLoading: boolean;
  error: string | null;
  onDeleteSession: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [turns, setTurns] = useState<Record<string, TurnResponse[]>>({});
  const [turnsLoading, setTurnsLoading] = useState<Set<string>>(new Set());
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const toggleSession = async (sessionId: string) => {
    if (expanded.has(sessionId)) {
      setExpanded((prev) => {
        const s = new Set(prev);
        s.delete(sessionId);
        return s;
      });
      return;
    }

    setExpanded((prev) => new Set(prev).add(sessionId));

    if (!turns[sessionId]) {
      setTurnsLoading((prev) => new Set(prev).add(sessionId));
      try {
        const data = await listTurns(sessionId);
        setTurns((prev) => ({ ...prev, [sessionId]: data }));
      } catch {
        // silently fail — turns panel will be empty
      } finally {
        setTurnsLoading((prev) => {
          const s = new Set(prev);
          s.delete(sessionId);
          return s;
        });
      }
    }
  };

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900">Query Session History</h2>
        <p className="text-xs text-gray-500 mt-0.5">
          Expand a session to view its turns
        </p>
      </div>

      {error && (
        <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 flex items-center">
          <AlertCircle className="w-4 h-4 mr-2 shrink-0" />
          {error}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <div className="text-sm text-gray-400">Loading sessions...</div>
        ) : sessions.length === 0 ? (
          <div className="py-12 text-center">
            <MessageSquare className="w-10 h-10 text-gray-300 mx-auto mb-3" />
            <p className="text-sm text-gray-400">No query sessions yet.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {sessions.map((session) => (
              <div
                key={session.id}
                className="bg-white rounded-lg border border-gray-200 overflow-hidden"
              >
                <div
                  className="flex items-center px-4 py-3 cursor-pointer hover:bg-gray-50 transition"
                  onClick={() => toggleSession(session.id)}
                >
                  {expanded.has(session.id) ? (
                    <ChevronDown className="w-4 h-4 text-gray-400 mr-2 shrink-0" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-gray-400 mr-2 shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800 truncate">
                      {session.session_title || (
                        <span className="text-gray-400 italic">Untitled session</span>
                      )}
                    </p>
                    <p className="text-xs text-gray-400 mt-0.5">
                      {session.turn_count} turn{session.turn_count !== 1 ? 's' : ''} ·{' '}
                      {new Date(session.updated_at).toLocaleString()}
                    </p>
                  </div>

                  {deleteConfirm === session.id ? (
                    <div
                      className="flex items-center space-x-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <span className="text-xs text-red-600 font-medium">Delete?</span>
                      <button
                        onClick={() => {
                          onDeleteSession(session.id);
                          setDeleteConfirm(null);
                        }}
                        className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-0.5 rounded"
                      >
                        Yes
                      </button>
                      <button
                        onClick={() => setDeleteConfirm(null)}
                        className="text-xs text-gray-400 hover:text-gray-600"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteConfirm(session.id);
                      }}
                      className="text-gray-300 hover:text-red-500 transition p-1 rounded-full hover:bg-red-50"
                      title="Delete session"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>

                {expanded.has(session.id) && (
                  <div className="border-t border-gray-100 bg-gray-50 px-4 py-3 space-y-3">
                    {turnsLoading.has(session.id) ? (
                      <p className="text-xs text-gray-400">Loading turns...</p>
                    ) : !turns[session.id] || turns[session.id].length === 0 ? (
                      <p className="text-xs text-gray-400">No turns recorded.</p>
                    ) : (
                      turns[session.id].map((turn, idx) => (
                        <div
                          key={turn.id}
                          className="space-y-1.5"
                        >
                          <div className="flex items-start space-x-2">
                            <span className="text-xs font-semibold text-gray-500 w-4 shrink-0">
                              {idx + 1}.
                            </span>
                            <div className="flex-1 space-y-1">
                              <div className="bg-indigo-50 rounded px-2.5 py-1.5">
                                <p className="text-xs font-medium text-indigo-700 mb-0.5">
                                  Question
                                </p>
                                <p className="text-xs text-indigo-900">{turn.user_question}</p>
                              </div>
                              <div className="bg-white border border-gray-200 rounded px-2.5 py-1.5">
                                <p className="text-xs font-medium text-gray-500 mb-0.5">Answer</p>
                                <p className="text-xs text-gray-700 whitespace-pre-wrap">
                                  {turn.answer_text}
                                </p>
                                {turn.matched_count !== null && (
                                  <p className="text-xs text-gray-400 mt-1">
                                    {turn.matched_count} candidate
                                    {turn.matched_count !== 1 ? 's' : ''} matched
                                  </p>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

type Tab = 'collections' | 'sessions';

export default function ShortlistPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<Tab>('collections');

  // Collections state
  const [collections, setCollections] = useState<CollectionResponse[]>([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [collectionsError, setCollectionsError] = useState<string | null>(null);

  // Selected collection + items
  const [selectedCollectionId, setSelectedCollectionId] = useState<string | null>(null);
  const [items, setItems] = useState<ShortlistItemResponse[]>([]);
  const [itemsLoading, setItemsLoading] = useState(false);
  const [itemsError, setItemsError] = useState<string | null>(null);

  // Sessions state
  const [sessions, setSessions] = useState<SessionResponse[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState<string | null>(null);

  // New / rename collection modal
  const [showCollectionModal, setShowCollectionModal] = useState(false);
  const [renamingCollection, setRenamingCollection] = useState<CollectionResponse | null>(null);
  const [collectionNameInput, setCollectionNameInput] = useState('');
  const [collectionModalError, setCollectionModalError] = useState<string | null>(null);
  const [collectionModalSaving, setCollectionModalSaving] = useState(false);

  // ── Fetchers ────────────────────────────────────────────────────────────────

  const fetchCollections = useCallback(async () => {
    setCollectionsLoading(true);
    setCollectionsError(null);
    try {
      const data = await listCollections(MOCK_USER_ID);
      setCollections(data);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to load collections';
      setCollectionsError(msg);
      toast(msg, 'error');
    } finally {
      setCollectionsLoading(false);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchItems = useCallback(async (collectionId: string) => {
    setItemsLoading(true);
    setItemsError(null);
    try {
      const data = await listItems(collectionId, { limit: 500 });
      setItems(data);
    } catch (err: any) {
      setItemsError(
        err.response?.data?.detail || err.message || 'Failed to load items',
      );
    } finally {
      setItemsLoading(false);
    }
  }, []);

  const fetchSessions = useCallback(async () => {
    setSessionsLoading(true);
    setSessionsError(null);
    try {
      const data = await listSessions(MOCK_USER_ID);
      setSessions(data);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to load sessions';
      setSessionsError(msg);
      toast(msg, 'error');
    } finally {
      setSessionsLoading(false);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    fetchCollections();
  }, [fetchCollections]);

  useEffect(() => {
    if (activeTab === 'sessions') fetchSessions();
  }, [activeTab, fetchSessions]);

  useEffect(() => {
    if (selectedCollectionId) fetchItems(selectedCollectionId);
    else setItems([]);
  }, [selectedCollectionId, fetchItems]);

  // ── Collection actions ───────────────────────────────────────────────────────

  const openNewCollection = () => {
    setRenamingCollection(null);
    setCollectionNameInput('');
    setCollectionModalError(null);
    setShowCollectionModal(true);
  };

  const openRenameCollection = (c: CollectionResponse) => {
    setRenamingCollection(c);
    setCollectionNameInput(c.name);
    setCollectionModalError(null);
    setShowCollectionModal(true);
  };

  const closeCollectionModal = () => {
    setShowCollectionModal(false);
    setRenamingCollection(null);
    setCollectionNameInput('');
    setCollectionModalError(null);
  };

  const handleCollectionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const name = collectionNameInput.trim();
    if (!name) {
      setCollectionModalError('Name is required.');
      return;
    }
    setCollectionModalSaving(true);
    setCollectionModalError(null);
    try {
      if (renamingCollection) {
        const updated = await renameCollection(renamingCollection.id, name);
        setCollections((prev) =>
          prev.map((c) => (c.id === renamingCollection.id ? updated : c)),
        );
      } else {
        await createCollection({ name, created_by_user_id: MOCK_USER_ID });
        await fetchCollections();
      }
      closeCollectionModal();
    } catch (err: any) {
      const status = err.response?.status;
      if (status === 409) {
        const msg = 'A collection with this name already exists. Please choose a different name.';
        setCollectionModalError(msg);
        toast(msg, 'warning');
      } else {
        const msg = err.response?.data?.detail || err.message || 'Failed to save collection';
        setCollectionModalError(msg);
        toast(msg, 'error');
      }
    } finally {
      setCollectionModalSaving(false);
    }
  };

  const handleDeleteCollection = async (id: string) => {
    try {
      await deleteCollection(id);
      setCollections((prev) => prev.filter((c) => c.id !== id));
      if (selectedCollectionId === id) setSelectedCollectionId(null);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to delete collection';
      setCollectionsError(msg);
      toast(msg, 'error');
    }
  };

  // ── Item actions ─────────────────────────────────────────────────────────────

  const handleRemoveItem = async (_itemId: string, candidateId: string) => {
    if (!selectedCollectionId) return;
    try {
      await removeItem(selectedCollectionId, candidateId);
      setItems((prev) => prev.filter((i) => i.candidate_profile_id !== candidateId));
      setCollections((prev) =>
        prev.map((c) =>
          c.id === selectedCollectionId
            ? { ...c, item_count: Math.max(0, c.item_count - 1) }
            : c,
        ),
      );
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to remove item';
      setItemsError(msg);
      toast(msg, 'error');
    }
  };

  // ── Session actions ──────────────────────────────────────────────────────────

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to delete session';
      setSessionsError(msg);
      toast(msg, 'error');
    }
  };

  const selectedCollection = collections.find((c) => c.id === selectedCollectionId) ?? null;

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full max-h-screen">
      {/* Page Header */}
      <div className="px-6 py-4 border-b border-gray-200 bg-white flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Shortlists</h1>
          <p className="text-gray-500 text-sm mt-0.5">
            Saved candidate collections and recruiter query history.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex rounded-lg border border-gray-200 overflow-hidden text-sm">
          <button
            onClick={() => setActiveTab('collections')}
            className={`px-4 py-2 transition ${
              activeTab === 'collections'
                ? 'bg-indigo-600 text-white font-medium'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            Collections
          </button>
          <button
            onClick={() => setActiveTab('sessions')}
            className={`px-4 py-2 transition border-l border-gray-200 ${
              activeTab === 'sessions'
                ? 'bg-indigo-600 text-white font-medium'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            Session History
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {activeTab === 'collections' ? (
          <>
            <CollectionsSidebar
              collections={collections}
              selectedId={selectedCollectionId}
              onSelect={setSelectedCollectionId}
              onNew={openNewCollection}
              onRename={openRenameCollection}
              onDelete={handleDeleteCollection}
              isLoading={collectionsLoading}
              error={collectionsError}
              onDismissError={() => setCollectionsError(null)}
            />

            <div className="flex-1 flex flex-col min-h-0">
              {selectedCollection ? (
                <CollectionItemsView
                  collection={selectedCollection}
                  items={items}
                  isLoading={itemsLoading}
                  error={itemsError}
                  onRemoveItem={handleRemoveItem}
                />
              ) : (
                <div className="flex-1 flex items-center justify-center">
                  <div className="text-center">
                    <FolderOpen className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                    <p className="text-gray-400 text-sm">
                      Select a collection to view its candidates
                    </p>
                    <button
                      onClick={openNewCollection}
                      className="mt-4 flex items-center mx-auto px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition"
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      New Collection
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <SessionHistoryPanel
            sessions={sessions}
            isLoading={sessionsLoading}
            error={sessionsError}
            onDeleteSession={handleDeleteSession}
          />
        )}
      </div>

      {/* New / Rename Collection Modal */}
      {showCollectionModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">
                {renamingCollection ? 'Rename Collection' : 'New Collection'}
              </h2>
              <button
                onClick={closeCollectionModal}
                className="text-gray-400 hover:text-gray-600 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCollectionSubmit} className="p-6 space-y-4">
              {collectionModalError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 flex items-start">
                  <AlertCircle className="w-4 h-4 mr-2 mt-0.5 shrink-0" />
                  {collectionModalError}
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Collection name <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={collectionNameInput}
                  onChange={(e) => setCollectionNameInput(e.target.value)}
                  placeholder="e.g. Senior Engineers — Q2"
                  autoFocus
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-400 mt-1">
                  Names must be unique across your collections.
                </p>
              </div>

              <div className="flex justify-end space-x-3 pt-1">
                <button
                  type="button"
                  onClick={closeCollectionModal}
                  className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={collectionModalSaving}
                  className="px-4 py-2 text-sm text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center"
                >
                  {collectionModalSaving ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Check className="w-4 h-4 mr-2" />
                      {renamingCollection ? 'Rename' : 'Create'}
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
