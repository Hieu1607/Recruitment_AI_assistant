# Feature Specification: Recruitment AI Assistant Website

**Feature Branch**: `001-recruitment-ai-assistant`  
**Created**: 2026-03-18  
**Status**: Draft  
**Input**: User description: "Xay dung website tro ly AI tuyen dung: parse CV PDF thanh du lieu chuan, match CV voi JD, hoi dap/loc CV, hien thi CV, luu shortlist, gui email ung vien, tao cau hoi phong van tu CV va JD"

## Clarifications

### Session 2026-03-18

- Q: What should be the default candidate data retention period? -> A: Retain data for 12 months, then auto-delete or anonymize candidate records.
- Q: What access model should be used for recruiter users? -> A: Role-based access with Admin, Recruiter, and Viewer roles.
- Q: What level of logging should be used for CV and chat processing? -> A: Log metadata and masked snippets only, without full PII in logs.
- Q: Should outreach emails be auto-sent by AI or require human approval? -> A: Every email send requires recruiter approval; AI may draft content.
- Q: How should CV-JD compatibility be presented and filtered? -> A: Use a 0-100 score with a configurable minimum pass threshold.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Parse CV and build candidate profile (Priority: P1)

As a recruiter, I upload resume PDF files and receive normalized candidate profiles that can be saved and searched, so I no longer need to manually copy candidate data.

**Why this priority**: This is the foundation for every later feature. Without structured profiles, matching, filtering, and outreach cannot happen.

**Independent Test**: Upload a batch of resumes, review extracted profiles, and confirm profiles can be stored with all required fields.

**Acceptance Scenarios**:

1. **Given** a recruiter uploads one or more resume PDFs, **When** extraction is completed, **Then** the system returns one structured candidate profile per valid resume with required fields mapped.
2. **Given** a resume does not contain a field, **When** profile generation completes, **Then** that field is stored as `null` or `false` according to the schema rules.
3. **Given** duplicate resumes for the same candidate are uploaded, **When** profile processing runs, **Then** the recruiter can identify and avoid saving unintended duplicates.

---

### User Story 2 - Match candidates against a job description (Priority: P1)

As a recruiter, I provide a job description and receive compatibility results for each candidate, so I can rank and prioritize applicants quickly.

**Why this priority**: Candidate-job fit scoring is a primary business value for screening decisions.

**Independent Test**: Select a candidate set, submit a job description, and verify each candidate receives a comparable compatibility result and explanation.

**Acceptance Scenarios**:

1. **Given** a saved candidate pool and a job description, **When** matching is requested, **Then** the system returns a compatibility result for every selected candidate.
2. **Given** compatibility results are available, **When** the recruiter sorts by relevance, **Then** candidates are listed in descending fit order.
3. **Given** a candidate has missing data, **When** matching runs, **Then** the candidate still receives a result with explicit uncertainty notes instead of being silently dropped.

---

### User Story 3 - Ask questions and filter candidates in natural language (Priority: P1)

As a recruiter, I ask natural-language questions about the candidate pool (counts, conditions, and combinations) and receive exact filtered results.

**Why this priority**: Conversational analytics and filtering is the most complex and highest-value workflow requested.

**Independent Test**: Ask representative questions (counts, skills + years, education + location + certification combinations) and confirm returned numbers and candidate lists match stored data.

**Acceptance Scenarios**:

1. **Given** candidate profiles exist, **When** the recruiter asks "How many graduated from Bach Khoa Ha Noi?", **Then** the assistant returns a count with the matched candidate set.
2. **Given** candidate profiles with skills and experience exist, **When** the recruiter asks "How many know Python and have over 3 years experience?", **Then** the assistant applies all conditions and returns only qualified candidates.
3. **Given** multi-condition queries are asked, **When** the recruiter requests "People from Bac Ninh, graduated from Su Pham university, and have SMO certification", **Then** the assistant returns the exact intersection set and can open each matching CV detail.

---

### User Story 4 - Take follow-up actions from filtered results (Priority: P2)

As a recruiter, after obtaining filtered candidates, I review CV details, save shortlist results, draft/send candidate emails, and generate interview questions from CV + JD.

**Why this priority**: Follow-up actions convert analysis into practical recruiting outcomes.

**Independent Test**: From a filtered result, open candidate CV details, save shortlist, generate an outreach email, and generate interview questions for one candidate-job pair.

**Acceptance Scenarios**:

1. **Given** a filtered candidate set, **When** the recruiter opens a candidate, **Then** the interface shows that candidate's CV details alongside the conversation context.
2. **Given** a filtered set, **When** the recruiter chooses to save the result, **Then** the system stores the shortlist with a retrievable label and timestamp.
3. **Given** a candidate and message intent, **When** the recruiter requests outreach support, **Then** the assistant provides either AI-generated email content or applies a recruiter-provided template before sending.
4. **Given** one candidate CV and one job description, **When** the recruiter requests interview questions, **Then** the assistant generates a relevant interview question set.

### Edge Cases

- Resume file is password-protected, corrupted, image-only, or unreadable.
- Resume language differs from the job description language.
- Candidate profile has conflicting values across resume sections (for example, two different phone numbers or overlapping roles with different dates).
- Uploaded resumes are duplicated across batches with minor formatting differences.
- Natural-language query includes ambiguous terms (for example, school nicknames, location aliases, or shorthand certifications).
- Query combines many conditions and returns zero matches.
- Recruiter requests outreach for candidates without valid email addresses.
- Recruiter attempts to save a shortlist with a name that already exists.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow recruiters to upload one or more resume PDFs in a single workflow.
- **FR-002**: System MUST extract resume information and produce one structured candidate profile per valid resume.
- **FR-003**: System MUST support the minimum profile fields: full_name, phone, email, location, contact, current_job_title, educated, ever_studied_abroad, major, cpa, education, experience, experience_years, skills, languages, projects, summary, achievements, publications, certifications, references, and other.
- **FR-004**: System MUST normalize location values into one of these forms: province/city only, province/city + country, or country only.
- **FR-005**: System MUST store missing optional text fields as `null` and missing boolean evidence fields as `false` unless explicitly supported by resume evidence.
- **FR-006**: System MUST preserve source traceability so recruiters can review how key extracted values map back to CV content.
- **FR-007**: System MUST enable recruiters to review and edit extracted profile fields before final save.
- **FR-008**: System MUST save approved candidate profiles into a searchable candidate database.
- **FR-009**: System MUST allow recruiters to submit a job description and choose the candidate set to evaluate.
- **FR-010**: System MUST send one batch request to the configured LLM provider containing the selected candidate set, one job description, and one shared scoring prompt template for that run.
- **FR-030**: System MUST represent compatibility using a total numeric score from 0 to 100.
- **FR-031**: System MUST allow recruiters to configure a minimum pass threshold and filter candidates by that threshold.
- **FR-032**: System MUST return scoring output as a list where each list item contains candidate reference, weighted component score list (for example: skills, education, experience), total score, and scoring rationale.
- **FR-011**: System MUST allow sorting and filtering by compatibility result.
- **FR-012**: System MUST support natural-language questions over candidate data, including count questions, condition-based filters, and multi-condition intersections.
- **FR-013**: System MUST return both a direct answer (for example, count) and the underlying matched candidate set when applicable.
- **FR-014**: System MUST allow recruiters to open and inspect each matched candidate CV detail from within the Q&A workflow.
- **FR-015**: System MUST provide a side-by-side experience where conversation and candidate detail view are available in the same working context.
- **FR-016**: System MUST allow recruiters to save filtered candidate results as reusable shortlist records.
- **FR-017**: System MUST support recruiter-initiated candidate outreach via email using either AI-generated content or recruiter-provided templates.
- **FR-029**: System MUST require explicit recruiter approval before every outbound email is sent, including emails drafted by AI.
- **FR-018**: System MUST log outreach actions with candidate reference, message subject, and sent status.
- **FR-019**: System MUST generate interview question sets when provided one candidate CV and one job description.
- **FR-020**: System MUST keep an auditable history of recruiter queries, filter outputs, and saved shortlist actions.
- **FR-021**: System MUST provide clear error messages and recovery guidance when extraction, matching, filtering, saving, or outreach actions fail.
- **FR-022**: System MUST enforce a default retention policy that auto-deletes or anonymizes candidate personal data after 12 months.
- **FR-023**: Design and implementation MUST prefer the simplest solution that satisfies these scenarios and constraints.
- **FR-024**: Requirements MUST define behavior in plain language and avoid unnecessary framework-specific detail.
- **FR-025**: System MUST enforce role-based access control with at least three roles: Admin, Recruiter, and Viewer.
- **FR-026**: System MUST restrict sensitive actions by role, including profile edits, shortlist saves, and outreach sending.
- **FR-027**: System MUST store operational logs using metadata and masked snippets only, and MUST NOT store full raw CV text or full personally identifiable information in logs.
- **FR-028**: System MUST keep logs sufficient for audit and troubleshooting while preserving candidate privacy.

### Key Entities *(include if feature involves data)*

- **Candidate Profile**: Structured representation of one candidate with all required fields, normalized values, validation status, and source traceability.
- **Resume Document**: Uploaded CV artifact with metadata (upload time, file name, parse status, language, duplicate indicators).
- **Job Description**: Role requirement input used for compatibility evaluation and interview-question generation.
- **Compatibility Result**: Candidate-to-job fit output returned as list items containing candidate reference, weighted component score list, total 0-100 score, rationale summary, confidence indicators, and threshold pass/fail status.
- **Query Session**: Conversational context containing recruiter questions, interpreted filters, answers, and matched candidate sets.
- **Shortlist Record**: Saved snapshot of filtered candidates with name, creator, timestamp, and retrieval metadata.
- **Outreach Message**: Email draft or sent message associated with candidate, message content source (AI or template), and delivery status.
- **Interview Question Set**: Generated interview questions tied to a specific candidate profile and job description pair.

### Assumptions

- The primary user roles are Admin, Recruiter, and Viewer with different permission scopes.
- Resumes may be in mixed formats and quality; extraction quality is expected to vary, and recruiter review is part of the normal flow.
- Matching and conversational filtering operate only on profiles already saved in the system.
- Outreach emails are sent only when a valid recipient email exists and the recruiter explicitly triggers the action.
- The system is expected to support Vietnamese recruiting contexts, including Vietnamese school and location names.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Recruiters can process and save structured profiles for at least 90% of readable uploaded resumes in a standard batch without manual data entry for required fields.
- **SC-002**: For a selected candidate pool, 100% of candidates receive a compatibility result when a job description is submitted.
- **SC-003**: At least 95% of tested natural-language filter questions return correct counts and candidate sets against a validated reference dataset.
- **SC-004**: Recruiters can complete the end-to-end workflow (upload resumes, run one filter query, open a CV, save shortlist) within 10 minutes for a 20-candidate batch.
- **SC-005**: At least 90% of pilot recruiters report that the assistant reduces initial screening effort compared with their prior manual process.
- **SC-006**: Outreach support and interview-question generation are successfully completed for at least 95% of requested candidate-job pairs with sufficient profile data.
