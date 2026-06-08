# Backend API Reference for Frontend Developers

Base URL: `http://localhost:8000`

All API endpoints are mounted under `/api/v1`. The root endpoint `GET /` returns a welcome message and can be used as a health check.

Interactive docs (Swagger UI) are available at `http://localhost:8000/docs` when the backend is running.

---

## Table of Contents

- [CORS](#cors)
- [Authentication](#authentication)
- [Pagination](#pagination)
- [Resume Upload & Management](#1-resume-upload--management----apiv1upload)
- [Job Descriptions](#2-job-descriptions----apiv1job-descriptions)
- [Candidate Scoring](#3-candidate-scoring----apiv1score)
- [Chat / Recruiter Chatbot](#4-chat--recruiter-chatbot----apiv1chat)
- [Shortlists (Sessions, Turns, Collections, Items)](#5-shortlists----apiv1shortlist)
- [Outreach Messages](#6-outreach-messages----apiv1outreach)
- [Interview Questions](#7-interview-questions----apiv1interview-questions)
- [Data Shapes Quick Reference](#data-shapes-quick-reference)
- [Enums](#enums)
- [Error Handling](#error-handling)
- [Important Notes](#important-notes)

---

## CORS

The backend allows cross-origin requests only from origins listed in the `BACKEND_CORS_ORIGINS` environment variable. For local development, make sure `http://localhost:5173` (the Vite dev server) is included. When configured, all HTTP methods and headers are allowed, and credentials are supported.

## Authentication

JWT infrastructure exists on the backend (`SECRET_KEY`, `HS256`, 30-minute expiry), and auth-sensitive routes now rely on `Bearer <token>` headers. Google OAuth uses progressive consent:

- `GET /api/v1/auth/google/login` starts basic Google sign-in with `openid email profile`.
- `GET /api/v1/auth/google/connect-gmail` is an authenticated endpoint that returns a Google authorize URL for the Gmail send consent flow.
- Google callback handling branches by signed OAuth state so Gmail consent can return directly to Outreach without affecting the normal login callback flow.

## Candidate Email Sending

Candidate email uses Google OAuth and Gmail API. Recruiters sign in with Google and grant `gmail.send`. The backend stores encrypted OAuth tokens in `oauth_identities`, sends mail from Celery tasks, and updates `interview_invitations.sent_at` or `outreach_messages.sent_status` only after Gmail API accepts the message.

## Pagination

All list endpoints accept these query parameters:

| Param    | Type | Default | Range   | Description        |
|----------|------|---------|---------|--------------------|
| `limit`  | int  | 50      | 1 - 200 | Max records to return |
| `offset` | int  | 0       | >= 0    | Records to skip    |

List responses always include a `total` field (count of returned items) and an `items` array.

---

## 1. Resume Upload & Management -- `/api/v1/upload`

### POST `/api/v1/upload/batch-parse`

Upload and parse one or more resume PDFs. Only `.pdf` files are accepted.

**Request** -- `Content-Type: multipart/form-data`

| Field                | Type          | Required | Description                  |
|----------------------|---------------|----------|------------------------------|
| `files`              | File[] (binary) | Yes    | One or more PDF files        |
| `uploaded_by_user_id`| string (UUID) | No       | UUID of the uploading user   |

**Response** -- `200 OK`

```json
{
  "total_files": 2,
  "processed_files": 2,
  "failed_files": 0,
  "items": [
    {
      "file_name": "resume.pdf",
      "resume_document_id": "uuid",
      "candidate_profile_id": "uuid",
      "status": "processed"
    }
  ]
}
```

**Errors**

| Status | Condition                                    |
|--------|----------------------------------------------|
| 400    | No files uploaded                            |
| 400    | Non-PDF file included                        |
| 400    | `uploaded_by_user_id` is not a valid UUID    |

> **Note:** Processing is synchronous -- the endpoint blocks until the LLM finishes parsing every PDF. For large batches this can take a while.

---

### GET `/api/v1/upload/`

List resume documents with optional filters.

**Query Parameters**

| Param                | Type        | Required | Description                                         |
|----------------------|-------------|----------|-----------------------------------------------------|
| `upload_status`      | string      | No       | Filter: `uploaded`, `processing`, `processed`, `failed` |
| `uploaded_by_user_id`| UUID        | No       | Filter by uploader                                  |
| `limit`              | int         | No       | Default 50, max 200                                 |
| `offset`             | int         | No       | Default 0                                           |

**Response** -- `200 OK`

```json
{
  "total": 5,
  "items": [
    {
      "id": "uuid",
      "original_file_name": "resume.pdf",
      "storage_uri": "/app/pdfs/abc_resume.pdf",
      "upload_status": "processed",
      "duplicate_group_key": null,
      "uploaded_by_user_id": "uuid",
      "uploaded_at": "2026-04-16T10:00:00+00:00",
      "processed_at": "2026-04-16T10:05:00+00:00",
      "retention_expires_at": "2027-04-16T10:00:00+00:00"
    }
  ]
}
```

---

### GET `/api/v1/upload/{resume_id}`

Get a single resume document by UUID.

**Response** -- `200 OK` -- Same shape as a single item in the list response.

| Status | Condition       |
|--------|-----------------|
| 404    | Resume not found |

---

### PATCH `/api/v1/upload/{resume_id}`

Update a resume document. All fields are optional.

**Request Body** -- `application/json`

```json
{
  "original_file_name": "new_name.pdf",
  "upload_status": "processed"
}
```

| Field               | Type   | Required | Constraints                                      |
|---------------------|--------|----------|--------------------------------------------------|
| `original_file_name`| string | No       | Min 1 character                                  |
| `upload_status`     | string | No       | `uploaded`, `processing`, `processed`, or `failed` |

**Response** -- `200 OK` -- Returns the updated resume object.

| Status | Condition                         |
|--------|-----------------------------------|
| 404    | Resume not found                  |
| 422    | Invalid status or empty filename  |

---

### DELETE `/api/v1/upload/{resume_id}`

Delete a resume document and its cascade relations (CandidateProfile, ExtractionTrace).

**Query Parameters**

| Param        | Type | Default | Description                          |
|--------------|------|---------|--------------------------------------|
| `delete_file`| bool | false   | Also delete the physical PDF on disk |

**Response** -- `200 OK`

```json
{
  "deleted": true,
  "resume_id": "uuid"
}
```

| Status | Condition       |
|--------|-----------------|
| 404    | Resume not found |

---

## 2. Job Descriptions -- `/api/v1/job-descriptions`

### POST `/api/v1/job-descriptions/`

Create a new job description.

**Request Body** -- `application/json`

```json
{
  "title": "Senior Software Engineer",
  "jd_text": "We are looking for...",
  "created_by_user_id": "uuid"
}
```

| Field               | Type        | Required | Constraints      |
|---------------------|-------------|----------|------------------|
| `title`             | string      | No       | Max 255 chars    |
| `jd_text`           | string      | Yes      | Min 1 char       |
| `created_by_user_id`| UUID        | Yes      |                  |

**Response** -- `201 Created`

```json
{
  "id": "uuid",
  "title": "Senior Software Engineer",
  "jd_text": "We are looking for...",
  "created_by_user_id": "uuid",
  "created_at": "2026-04-16T10:00:00+00:00",
  "is_active": true
}
```

| Status | Condition                        |
|--------|----------------------------------|
| 422    | Empty `jd_text` or invalid UUID  |

---

### GET `/api/v1/job-descriptions/`

List job descriptions.

**Query Parameters**

| Param      | Type | Required | Description             |
|------------|------|----------|-------------------------|
| `is_active`| bool | No       | Filter by active status |
| `limit`    | int  | No       | Default 50, max 200     |
| `offset`   | int  | No       | Default 0               |

**Response** -- `200 OK`

```json
{
  "total": 3,
  "items": [
    {
      "id": "uuid",
      "title": "Senior Software Engineer",
      "jd_text": "We are looking for...",
      "created_by_user_id": "uuid",
      "created_at": "2026-04-16T10:00:00+00:00",
      "is_active": true
    }
  ]
}
```

---

### GET `/api/v1/job-descriptions/{jd_id}`

Get a single job description by UUID.

**Response** -- `200 OK` -- Same shape as a single item in the list response.

| Status | Condition              |
|--------|------------------------|
| 404    | Job description not found |

---

### PATCH `/api/v1/job-descriptions/{jd_id}`

Partially update a job description. Only included fields are changed.

**Request Body** -- `application/json`

```json
{
  "title": "New Title",
  "jd_text": "Updated description...",
  "is_active": false
}
```

| Field      | Type   | Required | Constraints   |
|------------|--------|----------|---------------|
| `title`    | string | No       | Max 255 chars |
| `jd_text`  | string | No       | Min 1 char    |
| `is_active`| bool   | No       |               |

**Response** -- `200 OK` -- Returns the updated object.

| Status | Condition                       |
|--------|---------------------------------|
| 404    | Job description not found       |
| 422    | Empty `jd_text` provided        |

---

### DELETE `/api/v1/job-descriptions/{jd_id}`

Delete a job description.

**Response** -- `200 OK`

```json
{
  "deleted": true,
  "job_description_id": "uuid"
}
```

| Status | Condition              |
|--------|------------------------|
| 404    | Job description not found |

---

## 3. Candidate Scoring -- `/api/v1/score`

### POST `/api/v1/score/`

Score candidates against a job description using the LLM.

**Request Body** -- `application/json`

```json
{
  "job_description_id": "uuid",
  "initiated_by_user_id": "uuid",
  "score_threshold": 50.0,
  "candidate_profile_ids": ["uuid1", "uuid2"],
  "section_weights": {
    "skills": 35,
    "experience": 35,
    "projects": 15,
    "education": 10,
    "summary": 5
  },
  "batch_size": 10
}
```

| Field                   | Type       | Required | Default | Constraints                                    |
|-------------------------|------------|----------|---------|------------------------------------------------|
| `job_description_id`    | UUID       | Yes      |         |                                                |
| `initiated_by_user_id`  | UUID       | Yes      |         |                                                |
| `score_threshold`       | float      | No       | 50.0    | 0 - 100                                       |
| `candidate_profile_ids` | UUID[]     | No       | null    | Omit to score all candidates in the database   |
| `section_weights`       | object     | No       | null    | See section weights below                      |
| `batch_size`            | int        | No       | 10      | 1 - 50                                        |

**Section Weights**

All fields are optional floats (>= 0). Only fields you explicitly set are included in scoring; omitted fields default to weight 0. Weights are normalized to sum to 1.0 before being sent to the LLM.

Omit `section_weights` entirely to use system defaults:

| Section      | Default Weight |
|--------------|---------------|
| `skills`     | 35            |
| `experience` | 35            |
| `projects`   | 15            |
| `education`  | 10            |
| `summary`    | 5             |

Available sections: `skills`, `experience`, `education`, `projects`, `summary`, `languages`, `achievements`, `certifications`, `publications`, `other`.

**Response** -- `200 OK`

```json
{
  "match_run_id": "uuid",
  "job_description_id": "uuid",
  "total_candidates": 10,
  "total_passed_candidates": 7,
  "batches": 1,
  "scores": [
    {
      "candidateId": "uuid",
      "totalScore": 78.5,
      "passedThreshold": true,
      "rationale": "Strong match with relevant experience...",
      "componentScores": [
        {
          "criterionKey": "skills",
          "weight": 35,
          "score": 85,
          "weightedScore": 29.75,
          "evidenceSummary": "Proficient in required technologies"
        }
      ]
    }
  ]
}
```

| Field in each score     | Type    | Description                                        |
|-------------------------|---------|----------------------------------------------------|
| `candidateId`           | string  | UUID of the candidate profile                      |
| `totalScore`            | float   | Weighted total score, 0 - 100                      |
| `passedThreshold`       | bool    | Whether `totalScore >= score_threshold`             |
| `rationale`             | string  | LLM-generated explanation                          |
| `componentScores`       | array   | Per-section breakdown                              |
| `componentScores[].criterionKey`   | string | Section name (e.g. `skills`)          |
| `componentScores[].weight`         | float  | Normalized weight used                |
| `componentScores[].score`          | float  | Raw section score (0 - 100)           |
| `componentScores[].weightedScore`  | float  | `weight * score / 100`                |
| `componentScores[].evidenceSummary`| string | LLM evidence for this section         |

| Status | Condition                                              |
|--------|--------------------------------------------------------|
| 404    | Job description not found or no candidates found       |
| 422    | All section weights are 0 (at least one must be > 0)   |

> **Note:** This endpoint is synchronous and calls the LLM for each batch. Large candidate sets will take proportionally longer.

---

## 4. Chat / Recruiter Chatbot -- `/api/v1/chat`

The chatbot uses a LangGraph pipeline to answer questions about the candidate pool. It supports DSL-based filtering (structured queries) and LLM-powered free-form analysis. Conversation history is kept per session (last 5 messages).

### POST `/api/v1/chat/`

Send a message and receive an answer.

**Request Body** -- `application/json`

```json
{
  "message": "Show me candidates with Python skills",
  "session_id": "uuid",
  "candidate_limit": 500
}
```

| Field             | Type   | Required | Default | Constraints                        |
|-------------------|--------|----------|---------|------------------------------------|
| `message`         | string | Yes      |         | Min 1 character                    |
| `session_id`      | string | No       | null    | Omit to start a new session        |
| `candidate_limit` | int    | No       | 500     | 1 - 2000                          |

**Response** -- `200 OK`

```json
{
  "session_id": "uuid",
  "answer": "Found 5 candidates with Python skills...",
  "candidates_in_scope": 5
}
```

| Field                | Type   | Description                                           |
|----------------------|--------|-------------------------------------------------------|
| `session_id`         | string | Use this in subsequent requests to maintain context   |
| `answer`             | string | The chatbot's response text                           |
| `candidates_in_scope`| int   | Number of candidates matching the current filter      |

**Frontend flow:**
1. First message: omit `session_id`. The backend generates one and returns it.
2. Subsequent messages: pass the returned `session_id` to maintain conversation context.
3. The chatbot remembers the last 5 messages per session.

| Status | Condition             |
|--------|-----------------------|
| 500    | Graph execution error |

---

### GET `/api/v1/chat/{session_id}`

Retrieve the message history for a session.

**Response** -- `200 OK`

```json
{
  "session_id": "uuid",
  "messages": [
    { "role": "human", "content": "Show me candidates with Python skills" },
    { "role": "ai", "content": "Found 5 candidates..." }
  ]
}
```

| Status | Condition         |
|--------|-------------------|
| 404    | Session not found |

---

### DELETE `/api/v1/chat/{session_id}`

Clear and delete a chat session.

**Response** -- `200 OK`

```json
{
  "session_id": "uuid",
  "deleted": true
}
```

| Status | Condition         |
|--------|-------------------|
| 404    | Session not found |

---

## 5. Shortlists -- `/api/v1/shortlist`

Manages query sessions, conversation turns, shortlist collections, and collection items. This is the persistent layer for recruiter query history and saved candidate sets.

### 5a. Query Sessions

#### POST `/api/v1/shortlist/sessions/`

Create a new query session.

**Request Body** -- `application/json`

```json
{
  "user_id": "uuid",
  "session_title": "Python developers search"
}
```

| Field           | Type   | Required | Constraints   |
|-----------------|--------|----------|---------------|
| `user_id`       | UUID   | Yes      |               |
| `session_title` | string | No       | Max 255 chars |

**Response** -- `201 Created`

```json
{
  "id": "uuid",
  "user_id": "uuid",
  "session_title": "Python developers search",
  "turn_count": 0,
  "created_at": "2026-04-16T10:00:00+00:00",
  "updated_at": "2026-04-16T10:00:00+00:00"
}
```

---

#### GET `/api/v1/shortlist/sessions/`

List sessions for a user. Ordered by most recently updated.

**Query Parameters**

| Param    | Type | Required | Description               |
|----------|------|----------|---------------------------|
| `user_id`| UUID | Yes      | Filter by user UUID       |
| `limit`  | int  | No       | Default 50, max 200       |
| `offset` | int  | No       | Default 0                 |

**Response** -- `200 OK` -- Array of `SessionResponse` objects.

---

#### GET `/api/v1/shortlist/sessions/{session_id}`

Get a single session with its turn count.

| Status | Condition         |
|--------|-------------------|
| 404    | Session not found |

---

#### PATCH `/api/v1/shortlist/sessions/{session_id}`

Update the session title.

**Request Body** -- `application/json`

```json
{
  "session_title": "Updated title"
}
```

| Status | Condition         |
|--------|-------------------|
| 404    | Session not found |

---

#### DELETE `/api/v1/shortlist/sessions/{session_id}`

Delete a session and all its turns (cascade).

**Response** -- `204 No Content`

| Status | Condition         |
|--------|-------------------|
| 404    | Session not found |

---

### 5b. Query Turns

#### POST `/api/v1/shortlist/sessions/{session_id}/turns`

Add a turn (question + answer pair) to a session.

**Request Body** -- `application/json`

```json
{
  "user_question": "Find candidates with 5+ years of Python experience",
  "answer_text": "Found 12 candidates matching...",
  "matched_candidate_ids": ["uuid1", "uuid2"],
  "matched_count": 12,
  "tool_trace_masked": { "dsl_filter": "experience_years >= 5" }
}
```

| Field                   | Type        | Required | Description                         |
|-------------------------|-------------|----------|-------------------------------------|
| `user_question`         | string      | Yes      | Min 1 char                         |
| `answer_text`           | string      | Yes      | Min 1 char                         |
| `matched_candidate_ids` | string[]    | No       | UUIDs of matched candidates        |
| `matched_count`         | int         | No       | >= 0                               |
| `tool_trace_masked`     | object      | No       | Debug/trace info (free-form JSON)  |

**Response** -- `201 Created`

```json
{
  "id": "uuid",
  "query_session_id": "uuid",
  "user_question": "Find candidates with 5+ years of Python experience",
  "answer_text": "Found 12 candidates matching...",
  "matched_candidate_ids": ["uuid1", "uuid2"],
  "matched_count": 12,
  "tool_trace_masked": { "dsl_filter": "experience_years >= 5" },
  "created_at": "2026-04-16T10:05:00+00:00"
}
```

| Status | Condition         |
|--------|-------------------|
| 404    | Session not found |

---

#### GET `/api/v1/shortlist/sessions/{session_id}/turns`

List turns in a session, ordered chronologically (oldest first).

**Query Parameters** -- `limit` (default 50, max 200), `offset` (default 0).

| Status | Condition         |
|--------|-------------------|
| 404    | Session not found |

---

#### GET `/api/v1/shortlist/turns/{turn_id}`

Get a single turn.

| Status | Condition      |
|--------|----------------|
| 404    | Turn not found |

---

#### DELETE `/api/v1/shortlist/turns/{turn_id}`

Delete a turn.

**Response** -- `204 No Content`

| Status | Condition      |
|--------|----------------|
| 404    | Turn not found |

---

### 5c. Shortlist Collections

#### POST `/api/v1/shortlist/collections/`

Create a named collection of candidates.

**Request Body** -- `application/json`

```json
{
  "created_by_user_id": "uuid",
  "name": "Top Python candidates",
  "source_query_turn_id": "uuid"
}
```

| Field                 | Type   | Required | Constraints                        |
|-----------------------|--------|----------|------------------------------------|
| `created_by_user_id`  | UUID   | Yes      |                                    |
| `name`                | string | Yes      | Min 1 char, max 255 chars         |
| `source_query_turn_id`| UUID   | No       | Link to the turn that sourced this |

**Response** -- `201 Created`

```json
{
  "id": "uuid",
  "name": "Top Python candidates",
  "created_by_user_id": "uuid",
  "source_query_turn_id": "uuid",
  "item_count": 0,
  "created_at": "2026-04-16T10:00:00+00:00"
}
```

| Status | Condition                                            |
|--------|------------------------------------------------------|
| 409    | Collection with the same name already exists for user |

---

#### GET `/api/v1/shortlist/collections/`

List collections for a user. Ordered by most recently created.

**Query Parameters**

| Param    | Type | Required | Description              |
|----------|------|----------|--------------------------|
| `user_id`| UUID | Yes      | Filter by creator UUID   |
| `limit`  | int  | No       | Default 50, max 200      |
| `offset` | int  | No       | Default 0                |

**Response** -- `200 OK` -- Array of `CollectionResponse` objects.

---

#### GET `/api/v1/shortlist/collections/{collection_id}`

Get a single collection with its item count.

| Status | Condition            |
|--------|----------------------|
| 404    | Collection not found |

---

#### PATCH `/api/v1/shortlist/collections/{collection_id}`

Rename a collection.

**Request Body** -- `application/json`

```json
{
  "name": "Renamed collection"
}
```

| Field  | Type   | Required | Constraints              |
|--------|--------|----------|--------------------------|
| `name` | string | Yes      | Min 1 char, max 255 chars |

| Status | Condition                                            |
|--------|------------------------------------------------------|
| 404    | Collection not found                                 |
| 409    | Collection with the same name already exists for user |

---

#### DELETE `/api/v1/shortlist/collections/{collection_id}`

Delete a collection and all its items (cascade).

**Response** -- `204 No Content`

| Status | Condition            |
|--------|----------------------|
| 404    | Collection not found |

---

### 5d. Shortlist Items

#### POST `/api/v1/shortlist/collections/{collection_id}/items`

Add a candidate to a collection.

**Request Body** -- `application/json`

```json
{
  "candidate_profile_id": "uuid"
}
```

**Response** -- `201 Created`

```json
{
  "id": "uuid",
  "shortlist_collection_id": "uuid",
  "candidate_profile_id": "uuid",
  "added_at": "2026-04-16T10:10:00+00:00"
}
```

| Status | Condition                                    |
|--------|----------------------------------------------|
| 404    | Collection not found                         |
| 409    | Candidate already exists in this collection  |

---

#### GET `/api/v1/shortlist/collections/{collection_id}/items`

List items in a collection, ordered by added date (oldest first).

**Query Parameters** -- `limit` (default 100, max 500), `offset` (default 0).

| Status | Condition            |
|--------|----------------------|
| 404    | Collection not found |

---

#### DELETE `/api/v1/shortlist/collections/{collection_id}/items/{candidate_id}`

Remove a candidate from a collection.

**Response** -- `204 No Content`

| Status | Condition                               |
|--------|-----------------------------------------|
| 404    | Candidate not found in this collection  |

---

## 6. Outreach Messages -- `/api/v1/outreach`

Manages outreach/email messages to candidates.

### POST `/api/v1/outreach/`

Create an outreach message.

**Request Body** -- `application/json`

```json
{
  "candidate_profile_id": "uuid",
  "created_by_user_id": "uuid",
  "content_source": "ai_draft",
  "subject": "Opportunity at our company",
  "body": "Dear candidate, we would like to..."
}
```

| Field                  | Type   | Required | Constraints                      |
|------------------------|--------|----------|----------------------------------|
| `candidate_profile_id` | UUID   | Yes      |                                  |
| `created_by_user_id`   | UUID   | Yes      |                                  |
| `content_source`       | string | Yes      | `ai_draft` or `template`        |
| `subject`              | string | Yes      | Min 1 char, max 255 chars       |
| `body`                 | string | Yes      | Min 1 char                      |

**Response** -- `201 Created`

```json
{
  "id": "uuid",
  "candidate_profile_id": "uuid",
  "candidate_full_name": "John Doe",
  "created_by_user_id": "uuid",
  "content_source": "ai_draft",
  "subject": "Opportunity at our company",
  "body": "Dear candidate, we would like to...",
  "sent_status": "not_sent",
  "sent_at": null,
  "created_at": "2026-04-16T10:00:00+00:00"
}
```

| Status | Condition              |
|--------|------------------------|
| 404    | Candidate not found    |

---

### GET `/api/v1/outreach/`

List outreach messages with optional filters.

**Query Parameters**

| Param                  | Type   | Required | Description                                 |
|------------------------|--------|----------|---------------------------------------------|
| `created_by_user_id`   | UUID   | No       | Filter by creator                           |
| `candidate_profile_id` | UUID   | No       | Filter by candidate                         |
| `sent_status`          | string | No       | Filter: `not_sent`, `sent`, `failed`        |
| `limit`                | int    | No       | Default 50, max 200                         |
| `offset`               | int    | No       | Default 0                                   |

**Response** -- `200 OK`

```json
{
  "total": 3,
  "items": [ /* OutreachResponse objects */ ]
}
```

> **Note:** The list endpoint returns `total` as the real count across all pages (not just the returned page), unlike other list endpoints.

---

### GET `/api/v1/outreach/{message_id}`

Get a single outreach message.

| Status | Condition         |
|--------|-------------------|
| 404    | Message not found |

---

### PATCH `/api/v1/outreach/{message_id}`

Update subject, body, or sent status.

**Request Body** -- `application/json`

```json
{
  "subject": "Updated subject",
  "body": "Updated body...",
  "sent_status": "sent"
}
```

| Field        | Type   | Required | Constraints                       |
|--------------|--------|----------|-----------------------------------|
| `subject`    | string | No       | Min 1 char, max 255 chars        |
| `body`       | string | No       | Min 1 char                       |
| `sent_status`| string | No       | `not_sent`, `sent`, or `failed`  |

When `sent_status` is set to `sent`, the backend automatically sets `sent_at` to the current UTC timestamp (if not already set).

| Status | Condition         |
|--------|-------------------|
| 404    | Message not found |

---

### DELETE `/api/v1/outreach/{message_id}`

Delete an outreach message.

**Response** -- `204 No Content`

| Status | Condition         |
|--------|-------------------|
| 404    | Message not found |

---

### POST `/api/v1/outreach/{message_id}/send`

Queue an outreach message for background Gmail delivery.

The sender must have a Google identity with a refresh token and the `gmail.send` scope. If Gmail has not been connected yet, or consent was revoked, the backend returns a reconnectable error instead of queueing work.

**Response** -- `202 Accepted`

Returns the current `OutreachResponse` body when the send is accepted for background processing.

| Status | Condition |
|--------|-----------|
| 202 | Send accepted and queued |
| 404 | Message not found or not owned by current user |
| 409 | `gmail_not_connected` |

---

## 7. Interview Questions -- `/api/v1/interview-questions`

Manages AI-generated interview question sets, each tied to a specific candidate and job description.

### POST `/api/v1/interview-questions/`

Create an interview question set.

**Request Body** -- `application/json`

```json
{
  "candidate_profile_id": "uuid",
  "job_description_id": "uuid",
  "generated_by_user_id": "uuid",
  "question_payload": {
    "questions": [
      {
        "question": "Describe your experience with distributed systems",
        "category": "technical",
        "difficulty": "senior"
      }
    ]
  }
}
```

| Field                  | Type   | Required | Description                              |
|------------------------|--------|----------|------------------------------------------|
| `candidate_profile_id` | UUID   | Yes      |                                          |
| `job_description_id`   | UUID   | Yes      |                                          |
| `generated_by_user_id` | UUID   | Yes      |                                          |
| `question_payload`     | object | Yes      | Free-form JSON with the question content |

**Response** -- `201 Created`

```json
{
  "id": "uuid",
  "candidate_profile_id": "uuid",
  "candidate_full_name": "John Doe",
  "job_description_id": "uuid",
  "job_description_title": "Senior Software Engineer",
  "generated_by_user_id": "uuid",
  "question_payload": { /* same as input */ },
  "created_at": "2026-04-16T10:00:00+00:00"
}
```

| Status | Condition                                     |
|--------|-----------------------------------------------|
| 404    | Candidate or job description not found        |

---

### GET `/api/v1/interview-questions/`

List interview question sets with optional filters.

**Query Parameters**

| Param                  | Type | Required | Description              |
|------------------------|------|----------|--------------------------|
| `generated_by_user_id` | UUID | No       | Filter by creator        |
| `candidate_profile_id` | UUID | No       | Filter by candidate      |
| `job_description_id`   | UUID | No       | Filter by job description|
| `limit`                | int  | No       | Default 50, max 200      |
| `offset`               | int  | No       | Default 0                |

**Response** -- `200 OK`

```json
{
  "total": 2,
  "items": [ /* QuestionSetResponse objects */ ]
}
```

---

### GET `/api/v1/interview-questions/{question_set_id}`

Get a single question set.

| Status | Condition              |
|--------|------------------------|
| 404    | Question set not found |

---

### PATCH `/api/v1/interview-questions/{question_set_id}`

Replace the question payload.

**Request Body** -- `application/json`

```json
{
  "question_payload": { /* updated content */ }
}
```

| Field              | Type   | Required | Description            |
|--------------------|--------|----------|------------------------|
| `question_payload` | object | Yes      | Replaces entire payload |

| Status | Condition              |
|--------|------------------------|
| 404    | Question set not found |

---

### DELETE `/api/v1/interview-questions/{question_set_id}`

Delete a question set.

**Response** -- `204 No Content`

| Status | Condition              |
|--------|------------------------|
| 404    | Question set not found |

---

## Data Shapes Quick Reference

### ResumeResponse

```typescript
interface ResumeResponse {
  id: string                        // UUID
  original_file_name: string
  storage_uri: string               // server-side file path
  upload_status: string             // "uploaded" | "processing" | "processed" | "failed"
  duplicate_group_key: string | null
  uploaded_by_user_id: string       // UUID
  uploaded_at: string | null        // ISO 8601 datetime
  processed_at: string | null       // ISO 8601 datetime
  retention_expires_at: string | null // ISO 8601 datetime
}
```

### JobDescriptionResponse

```typescript
interface JobDescriptionResponse {
  id: string              // UUID
  title: string | null
  jd_text: string
  created_by_user_id: string  // UUID
  created_at: string      // ISO 8601 datetime
  is_active: boolean
}
```

### ScoreResponse

```typescript
interface ScoreResponse {
  match_run_id: string        // UUID
  job_description_id: string  // UUID
  total_candidates: number
  total_passed_candidates: number
  batches: number
  scores: CandidateScore[]
}

interface CandidateScore {
  candidateId: string       // UUID
  totalScore: number        // 0 - 100
  passedThreshold: boolean
  rationale: string
  componentScores: ComponentScore[]
}

interface ComponentScore {
  criterionKey: string
  weight: number
  score: number
  weightedScore: number
  evidenceSummary: string
}
```

### ChatResponse

```typescript
interface ChatResponse {
  session_id: string
  answer: string
  candidates_in_scope: number
}
```

### SessionResponse

```typescript
interface SessionResponse {
  id: string              // UUID
  user_id: string         // UUID
  session_title: string | null
  turn_count: number
  created_at: string      // ISO 8601 datetime
  updated_at: string      // ISO 8601 datetime
}
```

### TurnResponse

```typescript
interface TurnResponse {
  id: string                          // UUID
  query_session_id: string            // UUID
  user_question: string
  answer_text: string
  matched_candidate_ids: string[] | null  // UUIDs
  matched_count: number | null
  tool_trace_masked: Record<string, any> | null
  created_at: string                  // ISO 8601 datetime
}
```

### CollectionResponse

```typescript
interface CollectionResponse {
  id: string                       // UUID
  name: string
  created_by_user_id: string       // UUID
  source_query_turn_id: string | null  // UUID
  item_count: number
  created_at: string               // ISO 8601 datetime
}
```

### ShortlistItemResponse

```typescript
interface ShortlistItemResponse {
  id: string                       // UUID
  shortlist_collection_id: string  // UUID
  candidate_profile_id: string     // UUID
  added_at: string                 // ISO 8601 datetime
}
```

### OutreachResponse

```typescript
interface OutreachResponse {
  id: string                       // UUID
  candidate_profile_id: string     // UUID
  candidate_full_name: string | null
  created_by_user_id: string       // UUID
  content_source: "ai_draft" | "template"
  subject: string
  body: string
  sent_status: "not_sent" | "sent" | "failed"
  sent_at: string | null           // ISO 8601 datetime
  created_at: string               // ISO 8601 datetime
}
```

### QuestionSetResponse

```typescript
interface QuestionSetResponse {
  id: string                       // UUID
  candidate_profile_id: string     // UUID
  candidate_full_name: string | null
  job_description_id: string       // UUID
  job_description_title: string | null
  generated_by_user_id: string     // UUID
  question_payload: Record<string, any>
  created_at: string               // ISO 8601 datetime
}
```

---

## Enums

| Enum            | Values                                         | Used In                    |
|-----------------|-------------------------------------------------|----------------------------|
| UploadStatus    | `uploaded`, `processing`, `processed`, `failed` | ResumeDocument             |
| ProfileStatus   | `draft`, `reviewed`, `approved`, `archived`     | CandidateProfile           |
| MatchRunStatus  | `running`, `completed`, `failed`                | MatchRun                   |
| ContentSource   | `ai_draft`, `template`                          | OutreachMessage            |
| SentStatus      | `not_sent`, `sent`, `failed`                    | OutreachMessage            |
| UserStatus      | `active`, `suspended`                           | UserAccount                |
| RoleName        | `admin`, `recruiter`, `viewer`                  | RoleAssignment             |

---

## Error Handling

All error responses follow this shape:

```json
{
  "detail": "Human-readable error message"
}
```

Common HTTP status codes:

| Status | Meaning                                                  |
|--------|----------------------------------------------------------|
| 200    | Success                                                  |
| 201    | Created (POST endpoints for job-descriptions, sessions, turns, collections, items, outreach, interview-questions) |
| 204    | No content (DELETE on sessions, turns, collections, items, outreach, interview-questions) |
| 400    | Bad request (invalid input, wrong file type)             |
| 404    | Resource not found                                       |
| 409    | Conflict (duplicate collection name, candidate already in collection) |
| 422    | Validation error (invalid field values, constraint violations) |
| 500    | Server error (LLM failure, unexpected exceptions)        |

For `422` validation errors from FastAPI/Pydantic, the response body may include more detail:

```json
{
  "detail": [
    {
      "loc": ["body", "jd_text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Important Notes

1. **All datetimes are UTC** in ISO 8601 format with timezone offset (e.g. `2026-04-16T10:00:00+00:00`).

2. **All IDs are UUIDs** represented as strings in responses.

3. **Chat sessions are in-memory only.** They are lost when the backend restarts. Consider caching the `session_id` and handling "session not found" gracefully on the frontend by starting a new session.

4. **Resume parsing is synchronous.** The `POST /upload/batch-parse` endpoint blocks until all PDFs are processed by the LLM. Show a loading state in the UI -- large batches can take 30+ seconds.

5. **Scoring is also synchronous.** The `POST /score/` endpoint blocks while the LLM evaluates candidates in batches. Display appropriate loading feedback.

6. **PDF-only uploads.** The backend rejects any file that does not end with `.pdf`. Validate on the frontend before uploading to give faster feedback.

7. **The `scores` array uses camelCase keys** (`candidateId`, `totalScore`, `passedThreshold`, `componentScores`, `criterionKey`, `weightedScore`, `evidenceSummary`) while the rest of the API uses snake_case. This is because the scores are generated by the LLM and passed through as-is.

8. **Shortlist collection names are unique per user.** Creating or renaming a collection to a name that already exists for that user returns `409 Conflict`.

9. **DELETE endpoints on shortlist, outreach, and interview-questions return `204 No Content`** with an empty body, unlike the older upload/job-description DELETE endpoints which return `200` with a JSON body. Handle both patterns on the frontend.

10. **Outreach `sent_at` is auto-set.** When you PATCH `sent_status` to `sent`, the backend fills in `sent_at` automatically. No need to send it from the frontend.

11. **Interview question `question_payload` is free-form JSON.** The backend does not validate its internal structure -- the frontend and LLM determine the schema.

12. **Swagger UI** is available at `/docs` for interactive testing during development.
