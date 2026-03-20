# API Reference With Examples

This file summarizes implemented backend APIs and provides practical request/response examples.

## Base URL

- Local: `http://localhost:8000`

## Auth and Role Header

All protected endpoints use role-based access through header `X-Role`.

Common values:
- `admin`
- `recruiter`
- `viewer`

Example:

```bash
curl -H "X-Role: recruiter" http://localhost:8000/v1/candidates
```

## Health and Metrics

### GET `/health`

Liveness probe.

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok"
}
```

### GET `/metrics`

Returns in-memory metrics snapshot.

```bash
curl http://localhost:8000/metrics
```

## Resumes

### POST `/v1/resumes/upload`

Upload one or multiple PDF resumes.

```bash
curl -X POST http://localhost:8000/v1/resumes/upload \
  -H "X-Role: recruiter" \
  -F "files=@./samples/cv1.pdf" \
  -F "files=@./samples/cv2.pdf"
```

Example response (`202`):

```json
{
  "jobId": "50468583-c59a-40b0-90db-cf57a8ee9ac0",
  "acceptedCount": 2
}
```

## Candidates

### GET `/v1/candidates`

List candidates, optionally filtered with `q` and `limit`.

```bash
curl "http://localhost:8000/v1/candidates?q=python%20jakarta&limit=20" \
  -H "X-Role: viewer"
```

Example response (`200`):

```json
[
  {
    "id": "0922fd51-e0a8-46cf-9f95-4c0402642cb0",
    "fullName": "Jane Candidate",
    "phone": "+62 812 0000 0001",
    "email": "jane@example.com",
    "locationNormalized": "Jakarta, Indonesia",
    "educated": true,
    "everStudiedAbroad": false,
    "profileStatus": "reviewed"
  }
]
```

### PATCH `/v1/candidates/{candidateId}`

Update reviewed fields in candidate profile.

```bash
curl -X PATCH http://localhost:8000/v1/candidates/0922fd51-e0a8-46cf-9f95-4c0402642cb0 \
  -H "Content-Type: application/json" \
  -H "X-Role: recruiter" \
  -d '{
    "fullName": "Jane Candidate",
    "currentJobTitle": "Senior Backend Engineer",
    "educated": true,
    "everStudiedAbroad": false
  }'
```

### GET `/v1/candidates/{candidateId}/traces`

Get extraction trace blocks for auditability.

```bash
curl "http://localhost:8000/v1/candidates/0922fd51-e0a8-46cf-9f95-4c0402642cb0/traces?limit=50" \
  -H "X-Role: viewer"
```

## Matching

### POST `/v1/match-runs`

Run batch matching for one JD and many candidate profiles.

```bash
curl -X POST http://localhost:8000/v1/match-runs \
  -H "Content-Type: application/json" \
  -H "X-Role: recruiter" \
  -d '{
    "jobDescriptionText": "We are looking for a backend engineer with Python and FastAPI expertise.",
    "candidateIds": [
      "11111111-1111-1111-1111-111111111111",
      "22222222-2222-2222-2222-222222222222"
    ],
    "scoringPromptTemplate": "Score each candidate with weighted criteria for skills, experience, and education.",
    "scoreThreshold": 75
  }'
```

Example response (`200`):

```json
{
  "matchRunId": "d91c57f5-ce4d-46f4-ba61-a95b4dfab66f",
  "scoreThreshold": 75,
  "scores": [
    {
      "candidateId": "11111111-1111-1111-1111-111111111111",
      "totalScore": 84,
      "passedThreshold": true,
      "rationale": "Strong backend and API design track record.",
      "componentScores": [
        {
          "criterionKey": "skills",
          "weight": 0.4,
          "score": 90,
          "weightedScore": 36,
          "evidenceSummary": "Strong Python/FastAPI skills"
        }
      ]
    }
  ]
}
```

## Query Sessions

### POST `/v1/query-sessions`

Create/reopen query session.

```bash
curl -X POST http://localhost:8000/v1/query-sessions \
  -H "Content-Type: application/json" \
  -H "X-Role: viewer" \
  -d '{"title":"Jakarta data engineer shortlist"}'
```

### GET `/v1/query-sessions`

List query sessions for user.

```bash
curl "http://localhost:8000/v1/query-sessions?limit=20" \
  -H "X-Role: viewer"
```

### POST `/v1/query-sessions/{sessionId}/ask`

Ask natural-language question over candidate data.

```bash
curl -X POST http://localhost:8000/v1/query-sessions/a9c16de7-a148-4faa-8f84-a99446d93b0a/ask \
  -H "Content-Type: application/json" \
  -H "X-Role: viewer" \
  -d '{"question":"Show candidates with 5+ years Python experience in Jakarta."}'
```

Example response (`200`):

```json
{
  "answer": "I found 3 matching candidates with 5+ years of Python experience in Jakarta.",
  "matchedCount": 3,
  "matchedCandidateIds": [
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
    "33333333-3333-3333-3333-333333333333"
  ],
  "routingStrategy": "hybrid",
  "queryTurnId": "0f34d4fe-9b44-4f2d-9e19-d7bc20c3749b"
}
```

## Shortlists

### POST `/v1/shortlists`

Create shortlist from candidate IDs.

```bash
curl -X POST http://localhost:8000/v1/shortlists \
  -H "Content-Type: application/json" \
  -H "X-Role: recruiter" \
  -d '{
    "name":"Backend finalists",
    "candidateIds":[
      "11111111-1111-1111-1111-111111111111",
      "22222222-2222-2222-2222-222222222222"
    ],
    "sourceQueryTurnId":"0f34d4fe-9b44-4f2d-9e19-d7bc20c3749b"
  }'
```

### GET `/v1/shortlists`

List user shortlists.

```bash
curl "http://localhost:8000/v1/shortlists?limit=100" \
  -H "X-Role: viewer"
```

## Outreach

### POST `/v1/outreach/drafts`

Create outreach draft for candidate.

```bash
curl -X POST http://localhost:8000/v1/outreach/drafts \
  -H "Content-Type: application/json" \
  -H "X-Role: recruiter" \
  -d '{
    "candidateId":"11111111-1111-1111-1111-111111111111",
    "sourceType":"shortlist",
    "templateId":"template_backend_intro",
    "intent":"invite_for_screening"
  }'
```

### POST `/v1/outreach/{messageId}/approve-and-send`

Approve and send prepared outreach message.

```bash
curl -X POST http://localhost:8000/v1/outreach/b9302ef4-fef8-4e25-a65d-76fda005f21b/approve-and-send \
  -H "X-Role: recruiter"
```

## Interview Questions

### POST `/v1/interview-questions`

Generate interview question set.

```bash
curl -X POST http://localhost:8000/v1/interview-questions \
  -H "Content-Type: application/json" \
  -H "X-Role: recruiter" \
  -d '{
    "candidateId":"11111111-1111-1111-1111-111111111111",
    "jobDescriptionId":"6d301ac5-8f7f-4d7e-98c8-53e8a347fcb4",
    "questionCount":10
  }'
```

Example response (`201`):

```json
{
  "id": "a7f6a160-feb3-49dc-9be9-09d7b27f9a9c",
  "candidateId": "11111111-1111-1111-1111-111111111111",
  "jobDescriptionId": "6d301ac5-8f7f-4d7e-98c8-53e8a347fcb4",
  "questions": [
    {
      "prompt": "Design a robust FastAPI endpoint for high-throughput ingestion. What trade-offs would you make?",
      "category": "system-design",
      "difficulty": "medium"
    }
  ]
}
```

## Standard Error Shape

Most handled errors follow:

```json
{
  "error": "forbidden",
  "message": "You do not have permission to perform this action"
}
```

Common statuses:
- `400` bad request
- `403` forbidden
- `404` not found
- `422` validation error
- `500` internal error

## Source of Truth

- Contract: `specs/001-recruitment-ai-assistant/contracts/recruitment-api.yaml`
- Runtime routes: `backend/src/api/routes/*.py`
