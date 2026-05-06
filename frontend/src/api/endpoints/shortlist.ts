import { client } from "../client";
import type {
  SessionResponse,
  SessionListResponse,
  SessionCreateRequest,
  SessionUpdateRequest,
  TurnResponse,
  TurnListResponse,
  TurnCreateRequest,
  CollectionResponse,
  CollectionListResponse,
  CollectionCreateRequest,
  CollectionUpdateRequest,
  ShortlistItemResponse,
  ShortlistItemListResponse,
  ShortlistItemCreateRequest,
} from "../types";

/**
 * Query Sessions — persist recruiter chat history.
 * Base path: /shortlist/sessions/
 */
const sessions = {
  /**
   * List sessions for a user (ordered by most recently updated).
   * GET /shortlist/sessions/
   */
  async list(params: {
    user_id: string;
    limit?: number;
    offset?: number;
  }): Promise<SessionListResponse> {
    const { data } = await client.get<SessionListResponse>(
      "/shortlist/sessions/",
      { params },
    );
    return data;
  },

  /**
   * Get a single query session with its turn count.
   * GET /shortlist/sessions/{session_id}
   */
  async get(sessionId: string): Promise<SessionResponse> {
    const { data } = await client.get<SessionResponse>(
      `/shortlist/sessions/${sessionId}`,
    );
    return data;
  },

  /**
   * Create a new query session.
   * POST /shortlist/sessions/
   */
  async create(body: SessionCreateRequest): Promise<SessionResponse> {
    const { data } = await client.post<SessionResponse>(
      "/shortlist/sessions/",
      body,
    );
    return data;
  },

  /**
   * Update the session title.
   * PATCH /shortlist/sessions/{session_id}
   */
  async update(
    sessionId: string,
    body: SessionUpdateRequest,
  ): Promise<SessionResponse> {
    const { data } = await client.patch<SessionResponse>(
      `/shortlist/sessions/${sessionId}`,
      body,
    );
    return data;
  },

  /**
   * Delete a session and all its turns (cascade).
   * DELETE /shortlist/sessions/{session_id} → 204 No Content
   */
  async remove(sessionId: string): Promise<void> {
    await client.delete(`/shortlist/sessions/${sessionId}`);
  },
};

/**
 * Query Turns — individual question/answer pairs within a session.
 * Base path: /shortlist/sessions/{session_id}/turns and /shortlist/turns/{turn_id}
 */
const turns = {
  /**
   * List turns in a session (chronological order).
   * GET /shortlist/sessions/{session_id}/turns
   */
  async listForSession(
    sessionId: string,
    params?: { limit?: number; offset?: number },
  ): Promise<TurnListResponse> {
    const { data } = await client.get<TurnListResponse>(
      `/shortlist/sessions/${sessionId}/turns`,
      { params },
    );
    return data;
  },

  /**
   * Get a single turn by ID.
   * GET /shortlist/turns/{turn_id}
   */
  async get(turnId: string): Promise<TurnResponse> {
    const { data } = await client.get<TurnResponse>(
      `/shortlist/turns/${turnId}`,
    );
    return data;
  },

  /**
   * Add a turn to a session.
   * POST /shortlist/sessions/{session_id}/turns
   */
  async create(
    sessionId: string,
    body: TurnCreateRequest,
  ): Promise<TurnResponse> {
    const { data } = await client.post<TurnResponse>(
      `/shortlist/sessions/${sessionId}/turns`,
      body,
    );
    return data;
  },

  /**
   * Delete a turn.
   * DELETE /shortlist/turns/{turn_id} → 204 No Content
   */
  async remove(turnId: string): Promise<void> {
    await client.delete(`/shortlist/turns/${turnId}`);
  },
};

/**
 * Shortlist Collections — named candidate sets.
 * Base path: /shortlist/collections/
 */
const collections = {
  /**
   * List collections for a user (ordered by most recently created).
   * GET /shortlist/collections/
   */
  async list(params: {
    user_id: string;
    limit?: number;
    offset?: number;
  }): Promise<CollectionListResponse> {
    const { data } = await client.get<CollectionListResponse>(
      "/shortlist/collections/",
      { params },
    );
    return data;
  },

  /**
   * Get a single collection with item count.
   * GET /shortlist/collections/{collection_id}
   */
  async get(collectionId: string): Promise<CollectionResponse> {
    const { data } = await client.get<CollectionResponse>(
      `/shortlist/collections/${collectionId}`,
    );
    return data;
  },

  /**
   * Create a named collection.
   * POST /shortlist/collections/
   */
  async create(body: CollectionCreateRequest): Promise<CollectionResponse> {
    const { data } = await client.post<CollectionResponse>(
      "/shortlist/collections/",
      body,
    );
    return data;
  },

  /**
   * Rename a collection.
   * PATCH /shortlist/collections/{collection_id}
   */
  async update(
    collectionId: string,
    body: CollectionUpdateRequest,
  ): Promise<CollectionResponse> {
    const { data } = await client.patch<CollectionResponse>(
      `/shortlist/collections/${collectionId}`,
      body,
    );
    return data;
  },

  /**
   * Delete a collection and all its items (cascade).
   * DELETE /shortlist/collections/{collection_id} → 204 No Content
   */
  async remove(collectionId: string): Promise<void> {
    await client.delete(`/shortlist/collections/${collectionId}`);
  },
};

/**
 * Shortlist Items — candidates within a collection.
 * Base path: /shortlist/collections/{collection_id}/items
 */
const items = {
  /**
   * List items in a collection (ordered by added date, oldest first).
   * GET /shortlist/collections/{collection_id}/items
   */
  async listForCollection(
    collectionId: string,
    params?: { limit?: number; offset?: number },
  ): Promise<ShortlistItemListResponse> {
    const { data } = await client.get<ShortlistItemListResponse>(
      `/shortlist/collections/${collectionId}/items`,
      { params },
    );
    return data;
  },

  /**
   * Add a candidate to a collection.
   * POST /shortlist/collections/{collection_id}/items
   */
  async add(
    collectionId: string,
    body: ShortlistItemCreateRequest,
  ): Promise<ShortlistItemResponse> {
    const { data } = await client.post<ShortlistItemResponse>(
      `/shortlist/collections/${collectionId}/items`,
      body,
    );
    return data;
  },

  /**
   * Remove a candidate from a collection.
   * DELETE /shortlist/collections/{collection_id}/items/{candidate_id} → 204 No Content
   */
  async remove(collectionId: string, candidateId: string): Promise<void> {
    await client.delete(
      `/shortlist/collections/${collectionId}/items/${candidateId}`,
    );
  },
};

export const shortlistApi = {
  sessions,
  turns,
  collections,
  items,
};
