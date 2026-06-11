# Job Public Application Link & QR Plan

## Goal
Mỗi `Job` có một link public và QR code để ứng viên nộp resume mà không cần đăng nhập. Trang public phải yêu cầu ứng viên nhập `full_name` và `email` trước khi upload PDF để làm fallback khi resume parse kém hoặc không parse được. HR có thể cấu hình lời nhắn riêng cho từng job, lời nhắn này hiển thị trên trang nộp resume.

## Product Scope
- HR xem, copy và tải QR/link ứng tuyển trong màn hình chi tiết hoặc form edit của từng job.
- HR chỉnh được lời nhắn gửi ứng viên theo từng job.
- Ứng viên mở link public, thấy tiêu đề job, lời nhắn của HR, form nhập tên/email và upload PDF.
- Public upload tạo resume/candidate vào đúng job, không yêu cầu auth, không expose `job_id` thật trong URL.
- Nếu PDF parse thiếu hoặc lỗi trường tên/email, hệ thống dùng thông tin ứng viên nhập làm fallback.

## Assumptions
- Public URL dùng token ngẫu nhiên lưu trên `jobs`, ví dụ `/apply/{public_apply_token}`.
- QR code được render từ public URL ở frontend, không lưu QR image trong database.
- Public upload vẫn ghi resume vào job của HR; `uploaded_by_user_id` có thể dùng `jobs.owner_user_id` để tương thích schema hiện tại.
- Chỉ nhận PDF ở giai đoạn đầu, theo upload flow hiện có.

## Data Model

### `jobs` additions
- `public_apply_token VARCHAR(64) UNIQUE NOT NULL`
- `public_apply_enabled BOOLEAN NOT NULL DEFAULT TRUE`
- `candidate_message TEXT NULL`
- `public_apply_created_at TIMESTAMPTZ NOT NULL DEFAULT now()`
- `public_apply_disabled_at TIMESTAMPTZ NULL`

### Candidate fallback fields
Nên lưu dữ liệu ứng viên tự nhập tách biệt với dữ liệu parse từ PDF để HR có thể kiểm tra chất lượng nguồn:
- Add to `candidate_profiles`:
  - `submitted_full_name VARCHAR(255) NULL`
  - `submitted_email VARCHAR(320) NULL`
- Parsing rule:
  - `full_name = parsed_full_name` if usable, else `submitted_full_name`.
  - `email = parsed_email` if usable, else `submitted_email`.
  - Always persist submitted values for traceability.

## Backend Plan

### Sprint 1 - Schema & Token Generation
- Add migration for `jobs` public apply columns and candidate submitted fields.
- Backfill `public_apply_token` for existing jobs using cryptographically random URL-safe tokens.
- Add model fields to `Job` and `CandidateProfile`.
- Ensure new jobs auto-generate `public_apply_token`.
- Add service helper:
  - `generate_public_apply_token() -> str`
  - `build_public_apply_url(token: str) -> str`
  - `resolve_public_job_by_token(db, token) -> Job`

### Sprint 2 - Authenticated Job Settings APIs
- Extend `JobResponse` with:
  - `candidate_message`
  - `public_apply_enabled`
  - `public_apply_url`
- Extend `JobCreateRequest` and `JobUpdateRequest` with optional `candidate_message`.
- Add endpoint to rotate token:
  - `POST /api/v1/jobs/{job_id}/application-link/rotate`
  - Auth required, owner-only.
  - Invalidates old public link immediately.
- Optional endpoint if frontend wants a dedicated payload:
  - `GET /api/v1/jobs/{job_id}/application-link`
  - Returns `{ public_apply_url, public_apply_enabled, candidate_message }`.

### Sprint 3 - Public Candidate APIs
- Add public router mounted under `/api/v1/public`.
- `GET /api/v1/public/jobs/{token}`
  - No auth.
  - Returns only safe fields: `job_title`, `candidate_message`, `public_apply_enabled`.
  - Return `404` for unknown token, `410` or `403` for disabled link.
- `POST /api/v1/public/jobs/{token}/resumes`
  - No auth.
  - Multipart fields:
    - `full_name`: required, 1-255 chars.
    - `email`: required, valid email, max 320 chars.
    - `file`: required PDF.
  - Validate enabled public link before storing file.
  - Store resume under the resolved job.
  - Pass `submitted_full_name` and `submitted_email` into parsing/update flow.
  - Return minimal result to candidate: `{ submitted: true, candidate_profile_id?: string }`.

### Sprint 4 - Resume Parse Integration
- Update `parse_pdf_to_sections` or wrap it for public submissions so fallback values are applied after parse.
- If parse succeeds but omits name/email, patch `CandidateProfile` using submitted values.
- If parse fails before creating `CandidateProfile`, create a minimal candidate profile using submitted name/email and mark resume status as `failed` or `uploaded` according to current retry strategy.
- Add extraction trace payload fields to record that candidate-provided fallback was used.

## Frontend Plan

### Sprint 5 - HR Job UI
- Add an "Application Link" section to each job edit/detail screen.
- Show:
  - Public apply URL with copy button.
  - QR code generated from the public URL.
  - Download QR action if using canvas/SVG QR component.
  - Toggle for `public_apply_enabled`.
  - Textarea for `candidate_message`.
  - Rotate link button with confirmation.
- Update `JobResponse` TypeScript type and jobs API client.

### Sprint 6 - Public Apply Page
- Add public route outside `AppShell` and without auth loader:
  - `/apply/:token`
- On load, call `GET /api/v1/public/jobs/{token}`.
- Render:
  - Job title.
  - HR candidate message.
  - Required fields: full name, email, PDF upload.
  - Clear success state after upload.
- Validation:
  - Email format client-side and server-side.
  - PDF-only, max file size aligned with backend limit.
  - Disable submit while uploading.
- Error states:
  - Invalid link.
  - Disabled/expired link.
  - Upload failed.

## Security & Abuse Controls
- Use random token, not `job_id`, slug, or sequential ID.
- Public endpoints must never return owner user data, internal job IDs, resume storage paths, or candidate lists.
- Add upload size limit and PDF MIME/extension checks.
- Consider per-token rate limiting before production exposure.
- Rotating token must invalidate old QR/link.
- Keep CORS policy explicit for frontend origin.

## Acceptance Gates
- HR can create a job and immediately copy a public apply link.
- QR code resolves to the same public apply page as the copied link.
- Candidate can submit resume without auth using name, email and PDF.
- Candidate submission appears under the correct job in HR candidate/resume views.
- If PDF parse returns empty name/email, candidate profile uses submitted name/email.
- HR message saved on the job appears on the public page.
- Disabling or rotating the link blocks old links.
- Public endpoints do not expose private job, HR, candidate, or storage data.

## Test Plan
- Backend migration test: existing jobs receive unique public tokens.
- Backend API tests:
  - Owner can read/update application settings.
  - Non-owner cannot access another job's application settings.
  - Public job lookup works for enabled token.
  - Public job lookup fails for disabled/unknown token.
  - Public upload requires name, valid email and PDF.
  - Public upload applies submitted name/email fallback when parse output is poor.
- Frontend tests:
  - Public route does not trigger auth redirect.
  - HR job screen renders link, QR and message editor.
  - Candidate form blocks invalid email and non-PDF files.
  - Successful upload shows confirmation.

## Open Decisions
- Whether public links should support expiry dates beyond manual disable/rotate.
- Whether HR needs a custom short slug in addition to random token.
- Whether failed parse submissions should create candidate profiles immediately or remain as failed resumes awaiting manual review.
