# AI Recruitment Platform — Frontend Screens Specification

> Document dùng cho Claude Design. Mô tả chi tiết từng screen, layout chung, component patterns, và visual direction.
> Backend: FastAPI, base `http://localhost:8000/api/v1`. Frontend: Vite (port 5173).

---

## 0. Design Direction (Aesthetic Brief)

Đây là một **AI-powered Recruitment Platform** dành cho recruiter/HR chuyên nghiệp. Target users là những người làm tuyển dụng cấp trung đến cao — họ cần một công cụ trông **đáng tin cậy, cao cấp, hiện đại**, không phải một SaaS generic.

### Tone & Feel
- **Editorial-meets-enterprise**: như giao điểm giữa Linear, Notion, Arc Browser, và một tạp chí in ấn cao cấp (Monocle, Kinfolk).
- **Confident minimalism**: nhiều không gian trắng, typography mạnh, màu tiết chế với 1 accent duy nhất. **Không** dùng gradient tím-xanh generic.
- **Trustworthy & professional**: dữ liệu ứng viên là nhạy cảm — UI phải truyền tải sự chính xác, bảo mật, và nghiêm túc.
- **Subtle AI personality**: nhấn mạnh AI ở những nơi có ý nghĩa (scoring, chat, generation) nhưng không lạm dụng sparkle/magic emoji.

### Visual System (gợi ý — Designer tự do điều chỉnh)
- **Typography**
  - Display/Heading: một serif có character (gợi ý: *Fraunces*, *PP Editorial New*, *Instrument Serif*, hoặc *GT Sectra*) — tạo cảm giác editorial, cao cấp.
  - Body/UI: một sans-serif sạch nhưng có hint of personality (gợi ý: *Geist*, *Söhne*, *General Sans*, *PP Neue Montreal*). **Tránh** Inter, Roboto, Arial.
  - Mono (cho UUID, code, technical data): *JetBrains Mono*, *IBM Plex Mono*, hoặc *Geist Mono*.
- **Palette**
  - Background: off-white ấm `#FAFAF7` hoặc bone `#F5F3EE` (không pure white) cho theme sáng; rich charcoal `#1A1A1A` hoặc deep ink `#0F1012` cho theme tối. Designer chọn 1 theme chính, nhưng hệ thống phải dễ switch dark/light.
  - Foreground: `#111` / `#0A0A0A` cho text chính.
  - Accent: **một** màu duy nhất có gu — gợi ý *deep forest green* `#1F3A2E`, *burnt sienna* `#B8532A`, *electric cobalt* `#2A4DDB`, hoặc *muted coral* `#D4604A`. Tuyệt đối không dùng purple gradient.
  - Semantic: success/passed (muted green), warning (amber), danger (brick red) — nhưng bão hoà thấp, không neon.
- **Spacing & Grid**
  - 8px base grid. Card padding 24-32px.
  - Sidebar fixed 240-260px. Content max-width 1280-1440px với padding lớn 2 bên.
- **Motion**
  - Page transitions: fade + subtle Y translate (8-12px), 240ms ease-out.
  - Staggered reveal cho list items (50-80ms delay giữa items).
  - Loading states: skeleton với subtle shimmer, KHÔNG spinner generic.
  - Hover states có character — ví dụ underline animate từ trái, card lift với shadow mềm.
- **Details tạo dấu ấn**
  - Serif numerals cho metrics lớn (tabular-nums để không nhảy khi update).
  - Subtle grain/noise texture trên background (opacity 2-3%).
  - Hairline borders `1px solid rgba(0,0,0,0.06)` thay vì shadow nặng.
  - Empty states được minh hoạ đàng hoàng (không phải stock illustration) — có thể dùng editorial typography + một SVG hình học tối giản.

---

## 1. Shared Layout (áp dụng cho tất cả screens TRỪ Landing & Login)

Tất cả screens authenticated dùng **chung một layout shell**. Designer thiết kế shell này cẩn thận vì nó xuất hiện ở mọi nơi.

### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│  TOP BAR (56-64px)                                          │
│  [Logo]    [Breadcrumb / Page context]      [Search] [User] │
├──────────┬──────────────────────────────────────────────────┤
│          │                                                  │
│          │                                                  │
│ SIDEBAR  │           MAIN CONTENT AREA                      │
│ (240px)  │           (với max-width và padding rộng)        │
│          │                                                  │
│          │                                                  │
│          │                                                  │
└──────────┴──────────────────────────────────────────────────┘
```

### Top Bar
- Logo bên trái (wordmark nhẹ nhàng + ký hiệu hình học nhỏ).
- Breadcrumb ở giữa: ví dụ `Candidates / Resume parsing / Batch #A7F2`.
- Bên phải: global search (`⌘K` style, mở command palette), notifications icon (subtle dot khi có), avatar user với dropdown (profile, settings, sign out).
- Nền top bar hoà vào background, chỉ có 1 hairline border bottom.

### Sidebar (Primary Navigation)
Các mục nav theo thứ tự logic của workflow recruiter:

1. **Dashboard** — overview
2. **Candidates** — quản lý resume & candidate profiles
3. **Job Descriptions** — tạo và quản lý JD
4. **Scoring** — chạy matching candidates vs JD
5. **AI Chat** — recruiter chatbot
6. **Shortlists** — collections và query history
7. **Outreach** — email messages
8. **Interview Prep** — question sets

Style:
- Mỗi item có icon line-style tinh tế (gợi ý: Lucide hoặc Phosphor thin variant) + label.
- Active state: background subtle + accent color bar bên trái (2-3px).
- Collapsed state (icon-only) khi hover expand.
- Phần dưới cùng: "Upload resume" CTA button (primary action luôn available) + workspace switcher nhỏ.

### Content Area Pattern
Mỗi page tuân thủ anatomy:
1. **Page Header** — Serif title lớn (32-40px) + subtitle một dòng + action buttons bên phải.
2. **Filters/Toolbar** (nếu cần) — inline, hairline-bordered.
3. **Main content** — table, grid, form, hoặc detail view.
4. **Pagination/Footer** — sticky nếu list dài.

---

## 2. Screen List (chi tiết từng màn)

### 🎨 Screen 01 — Landing Page (Public, UNIQUE LAYOUT)

**Purpose**: Giới thiệu sản phẩm cho recruiter khách. Đây là screen DUY NHẤT phá layout chung — hãy làm nó **đáng nhớ**.

**Sections**:
1. **Hero** — Editorial-style. Serif heading rất lớn (72-96px) kiểu: *"Hire like it's 2030."* hoặc *"Resumes, read at the speed of thought."* Kèm một paragraph subheading. CTA "Get started" (primary) và "Watch demo" (ghost).
2. **Product value strip** — 3-4 cột nhỏ với icon + tagline ngắn: "Parse 500 CVs in minutes", "AI scoring against any JD", "Chat with your candidate pool", "Generate interview questions in seconds".
3. **Showcase section** — mockup của Scoring screen hoặc Chat screen, với browser frame tinh tế. Có thể dùng parallax/scroll-reveal.
4. **Feature deep-dives** — 3-4 blocks alternating left/right layout, mỗi block mô tả 1 module với screenshot lớn.
5. **Social proof** — logo bar hoặc testimonial dạng editorial quote (serif lớn, italicized).
6. **CTA cuối** — Big, bold, simple.
7. **Footer** — minimal, editorial.

**Visual freedom**: Có thể dùng large serif numerals, marquee/ticker subtle, hero animation với text reveal staggered.

---

### 🔐 Screen 02 — Login / Sign Up (Public, UNIQUE LAYOUT)

**Purpose**: Authentication. Mặc dù backend chưa enforce auth, frontend vẫn chuẩn bị UI.

**Layout**: Split screen 2 cột (hoặc centered card tuỳ Designer).
- **Bên trái (60%)**: Editorial panel — background màu accent hoặc ảnh/pattern tinh tế, với một quote lớn hoặc tagline của sản phẩm. Logo ở góc trên.
- **Bên phải (40%)**: Form area — background off-white. Form centered vertically:
  - Heading serif: "Welcome back" hoặc "Create your account"
  - Subtext
  - Email input + Password input (floating label hoặc minimal underline style)
  - Primary button "Sign in"
  - Divider "or"
  - SSO buttons (Google, Microsoft) — outline style
  - Link "Forgot password?" và toggle "Sign up / Sign in"

**States**: Loading (button skeleton), error (inline error text màu danger, shake animation nhẹ), success redirect.

---

### 📊 Screen 03 — Dashboard (uses Shared Layout)

**Purpose**: Overview tổng quan. Landing page sau khi login.

**Content**:
1. **Greeting header** — "Good morning, [Name]" (serif) + ngày hôm nay nhỏ bên dưới.
2. **Metric cards row** (4 cards):
   - Total candidates (với serif numeral lớn)
   - Resumes processed today
   - Active job descriptions
   - Pending outreach messages
   Mỗi card có sparkline nhỏ và % change so với kỳ trước.
3. **Recent activity timeline** (2 cột):
   - Cột trái (2/3): Activity feed — resume uploads, score runs, chat sessions, outreach sent. Mỗi item có icon, text mô tả, timestamp (relative: "2 hours ago").
   - Cột phải (1/3): "Quick actions" card với primary buttons lớn: Upload resumes, Create JD, Start scoring, Open chat.
4. **Top shortlist collections** — grid cards hiển thị 3-4 collections gần nhất với item count.

**Empty state**: Nếu chưa có data, show onboarding cards — "Step 1: Upload your first resumes", "Step 2: Create a job description", v.v. với checkmark progress.

---

### 📄 Screen 04 — Candidates (Resume Management)

**Purpose**: Quản lý tất cả resume đã upload. Endpoint: `GET /upload/`.

**Layout**:
- **Header**: Title "Candidates" + subtitle "Manage parsed resumes and profiles" + button "Upload resumes" (primary, mở modal).
- **Toolbar**:
  - Search input (tìm theo tên file hoặc tên ứng viên).
  - Filter chips: Upload status (`uploaded`, `processing`, `processed`, `failed`), Uploaded by, Date range.
  - View toggle: Table / Card grid.
  - Sort dropdown.
- **Main**: Table view mặc định với các columns:
  - Checkbox (bulk select)
  - Candidate name (parsed từ profile, link đến detail)
  - Original filename (mono font, muted)
  - Upload status (badge với dot indicator màu)
  - Uploaded by
  - Uploaded at (relative time, hover xem exact)
  - Retention expires
  - Actions menu (View, Edit, Delete)
- **Pagination**: Bottom, shows "Showing 1-50 of 234" + prev/next + limit selector (50/100/200).

**Row hover**: Subtle background tint + reveal action buttons.

**Status badges** (quan trọng — consistent across app):
- `uploaded` — neutral grey
- `processing` — amber với pulsing dot
- `processed` — green muted với checkmark
- `failed` — brick red với warning icon

---

### 📤 Screen 05 — Upload Resumes Modal/Dialog

**Purpose**: Batch upload PDFs. Endpoint: `POST /upload/batch-parse`. Lưu ý: processing SYNC, có thể mất 30s+.

**Design**: Large modal centered (không fullscreen).

**States**:
1. **Idle**: Dropzone lớn với hairline dashed border, icon PDF + text "Drop PDFs here or click to browse". Note nhỏ: "Only .pdf files, up to N files per batch".
2. **Files selected**: List các file đã chọn, mỗi file 1 row với filename, size, remove button. Total count ở header. Primary button "Parse N resumes".
3. **Processing (SYNC, dài)**: Progress bar indeterminate + editorial message rotating: "Reading resumes...", "Extracting skills...", "Building profiles...". Hiển thị số file đã xong / total. **Quan trọng**: disable close button, warning "Processing is synchronous, please don't close this window".
4. **Complete**: Summary — "2 of 2 resumes processed successfully" với green check. List kết quả mỗi file (success/failed + error reason). Buttons "View candidates" và "Upload more".

**Error handling**: Invalid file type, size too big, UUID invalid — inline error với màu danger.

---

### 👤 Screen 06 — Candidate Detail

**Purpose**: Xem profile đã parsed của một ứng viên. (Backend trả về CandidateProfile qua cascade — giả định frontend có endpoint để fetch full profile; nếu chưa có, dùng data từ list).

**Layout**: 3-column hoặc tab-based.

**Header area**:
- Candidate full name (serif, 36-48px)
- Subtitle: current role + years of experience
- Action buttons bên phải: "Score against JD", "Generate interview questions", "Draft outreach", "Add to shortlist"

**Tabs** (hoặc sections nếu không tab):
1. **Overview** — Summary, key skills (chip list), experience timeline (vertical), education.
2. **Resume PDF** — Embedded PDF viewer bên trái, parsed data bên phải (side-by-side để recruiter verify).
3. **Scoring history** — Các match runs từng chạy với candidate này, sorted by date.
4. **Outreach history** — Messages đã gửi.
5. **Interview questions** — Sets đã generate.

**Visual detail**: Skills hiển thị dạng chips với hairline border. Experience dạng timeline với dots và kết nối hairline giữa các job.

---

### 📋 Screen 07 — Job Descriptions List

**Purpose**: Quản lý JDs. Endpoint: `GET /job-descriptions/`.

**Layout**:
- Header: "Job Descriptions" + "Create JD" button (primary).
- Filter: `is_active` toggle.
- **Grid view** (2-3 cards/row) — mỗi card là một JD:
  - Title serif
  - `is_active` badge góc trên phải
  - JD text preview (3 lines, fade out)
  - Meta footer: created_at relative, created_by avatar
  - Hover: lift shadow mềm, reveal "View" và "Score candidates" buttons
- Alternatively table view nếu Designer prefer consistency với Candidates screen.

---

### ✍️ Screen 08 — Create / Edit Job Description

**Purpose**: Form tạo/edit JD. Endpoints: `POST`/`PATCH /job-descriptions/`.

**Layout**: Full-page form (không modal vì JD có thể dài).
- Header: "New job description" hoặc "Editing [Title]" + Save/Cancel buttons (sticky top).
- Form:
  - **Title input** — large serif input, no border, placeholder "Untitled position" (Notion-style).
  - **JD text** — large textarea với rich-text minimal (bold, italic, bullet list, heading 2). Min height 400px. Character counter nhẹ ở góc.
  - **Settings panel bên phải** (sticky): `is_active` toggle, created_by (auto), created_at (readonly).
- **AI assist CTA** (optional): button "Polish with AI" cho phép regenerate/refine JD.

**Save behaviour**: Autosave draft (indicator nhỏ "Saved 2s ago"), hoặc explicit save.

---

### 🎯 Screen 09 — Scoring / Match Run

**Purpose**: Chạy scoring candidates vs JD. Endpoint: `POST /score/`. Đây là feature FLAGSHIP — design phải đặc biệt đẹp.

**Flow**: 3 steps.

#### 9a. Setup (Step 1)
- Header: "Score candidates" + stepper 1/3.
- Left panel: 
  - Select JD (dropdown hoặc search + preview JD content).
  - Select candidates: radio "All candidates" hoặc "Choose specific" (mở multi-select với search, hiển thị count).
- Right panel:
  - **Section weights editor** — đẹp và tương tác:
    - Hiển thị 5 sections mặc định (skills, experience, projects, education, summary) với slider hoặc direct input.
    - Có thể thêm sections khác (languages, achievements, v.v.) qua "+ Add section" button.
    - Live donut chart hiển thị phân bổ weights (normalized to 100%).
    - "Reset to default" link.
  - **Threshold slider**: 0-100, default 50. Visual hint: "Candidates scoring ≥ 50 will be marked as passed".
  - **Batch size**: 1-50, default 10. Mô tả: "Smaller batches for better accuracy, larger for speed".
- Footer sticky: "Start scoring" button (primary) với estimated time ("~2 min for 50 candidates").

#### 9b. Processing (Step 2)
- Full-screen takeover hoặc large card.
- Big centered animation: editorial-style, ví dụ một SVG hình học quay chậm (không spinner generic).
- Text rotating: "Analyzing [Candidate name]...", "Evaluating skills...", "Batch 1 of 3 complete".
- Progress bar với % và ETA.
- Small panel bên dưới: live log (optional) hiển thị candidates đã xong.

#### 9c. Results (Step 3) — **MÀN QUAN TRỌNG**
- Header: "Match results" + meta ("Scored 50 candidates against [JD Title] · 35 passed · Run ID #A7F2").
- **Summary strip** (4 stats):
  - Total candidates (serif numeral)
  - Passed threshold
  - Average score
  - Highest score
- **Main table** (sortable, default by totalScore desc):
  - Rank (#1, #2, ... với serif numerals)
  - Candidate name + avatar
  - Total score — BIG, serif numeral, 0-100 với màu gradient từ muted red (low) đến accent (high). Tabular nums.
  - Passed threshold — badge (green check / grey cross).
  - Mini bar chart cho component scores (5 bars nhỏ, hover xem chi tiết).
  - Rationale preview (1 line, "Read more" để expand).
  - Actions: "View details", "Add to shortlist", "Draft outreach"
- **Row expand**: Click vào row mở panel chi tiết (accordion hoặc side drawer):
  - Full rationale text (editorial paragraph).
  - Component scores table: criterion, weight %, score, weighted score, evidence summary (italic serif quote).
  - Radar chart visualization của component scores.
- **Bulk actions bar** (sticky khi có selection): "Add N to shortlist", "Export", "Draft outreach for N".

---

### 💬 Screen 10 — AI Chat (Recruiter Chatbot)

**Purpose**: Natural language query về candidate pool. Endpoint: `POST /chat/`.

**Layout**: 2-panel chat interface.

**Left sidebar (30%)** — Sessions list:
- "+ New chat" button on top.
- List of chat sessions (title — auto-generated from first message — + turn count + updated_at relative).
- Active session highlighted với accent bar.
- Search sessions input.

**Main panel (70%)** — Chat area:
- Header: session title (editable inline), meta "X turns · last updated Y ago", delete button.
- **Messages area**:
  - User messages: right-aligned, background subtle, serif nhẹ, max-width 70%.
  - AI messages: left-aligned, không bubble (prose style, editorial), full-width. Có avatar/logo AI nhỏ.
  - Khi AI response có kết quả `candidates_in_scope > 0`: hiển thị **inline candidate cards** dưới message — horizontal scrollable hoặc grid 2 cột. Mỗi card: avatar, name, top 3 skills, "View" link. Text nhỏ: "Found N candidates matching your query".
  - Copy button khi hover message.
- **Empty state**: Editorial hero "Ask anything about your candidates" + gợi ý prompts (chips): "Who has 5+ years of Python?", "Show me candidates with AWS and Kubernetes", "Senior engineers in San Francisco".
- **Input area** (sticky bottom):
  - Textarea auto-grow với placeholder "Message the recruiter assistant...".
  - Send button (primary) hoặc Enter.
  - Setting icon: adjust `candidate_limit` (default 500, max 2000).
  - Character / token hint.

**Important**: Sessions in-memory, dễ bị mất khi backend restart. Handle gracefully: khi session_id không tìm được, auto-start new session và show toast "Session expired, started new conversation".

---

### 🗂️ Screen 11 — Shortlists (Collections & History)

**Purpose**: Quản lý shortlist collections và query session history. Endpoints: `/shortlist/sessions`, `/shortlist/collections`.

**Layout**: Tabs ở top — "Collections" và "Query History".

#### 11a. Collections tab
- Header: "Shortlist collections" + "New collection" button.
- **Grid of collection cards** (3 cols):
  - Collection name (serif, bold)
  - Item count badge
  - Created at relative
  - Source query indicator (nếu có `source_query_turn_id`) — icon nhỏ + tooltip "Created from query: [question preview]"
  - Hover: reveal "View", "Rename", "Delete" actions.
- Empty state: "No collections yet. Start by scoring candidates or querying the AI."

#### 11b. Query History tab
- **Sessions list** (left, 30%):
  - Mỗi session: title (editable), turn_count, updated_at.
  - Click chọn session.
- **Turns timeline** (right, 70%):
  - Vertical timeline của từng turn trong session.
  - Mỗi turn: user question (serif italic quote style), answer text (editorial prose), matched_count badge ("12 candidates"), toggle "Show matched candidates" (reveals list).
  - Action cho mỗi turn: "Create collection from this turn" button.

---

### 📁 Screen 12 — Collection Detail

**Purpose**: Xem chi tiết 1 collection, quản lý items. Endpoints: `/shortlist/collections/{id}/items`.

**Layout**:
- Header: Collection name (editable inline, serif) + meta (N candidates · created X ago · from query "...").
- Actions: "Add candidates", "Export", "Draft outreach to all", "Delete collection".
- **Table** của candidates trong collection:
  - Avatar + name
  - Top skills (chips)
  - Latest match score (nếu có)
  - Added at (relative)
  - Actions: View profile, Remove from collection.
- Empty state: "This collection is empty" + "Add candidates" CTA.

---

### ✉️ Screen 13 — Outreach Messages

**Purpose**: Quản lý và compose outreach emails. Endpoints: `/outreach/`.

**Layout**: Email-client style, 3-column.

**Left sidebar (20%)** — Folders:
- All messages (N)
- Not sent (N)
- Sent (N)
- Failed (N)

**Middle column (35%)** — Message list:
- Mỗi row: candidate_full_name (bold), subject (1 line truncate), body preview (1 line, muted), sent_status badge, timestamp (relative).
- Status dot indicator bên trái.
- Click chọn message → hiện ở panel phải.

**Right panel (45%)** — Message detail:
- Header: To: [candidate name + email nếu có], From: current user, sent/not-sent badge.
- Subject (large).
- Body (prose, editorial rendering).
- Meta: content_source (`ai_draft` or `template`), created_at, sent_at.
- Actions: "Edit" (nếu not_sent), "Mark as sent", "Mark as failed", "Delete".

**Compose modal** (triggered từ candidate detail hoặc "+ New message"):
- To: candidate selector (search).
- Content source toggle: AI draft / Template.
- Nếu AI draft: button "Generate" gọi LLM (nếu backend support, nếu không thì free text).
- Subject input.
- Body textarea (rich text tối thiểu).
- Save as draft / Send (Note: backend chỉ mark as sent, không thực gửi email).

---

### 🎙️ Screen 14 — Interview Questions

**Purpose**: Quản lý AI-generated interview question sets. Endpoints: `/interview-questions/`.

**Layout**:
- Header: "Interview question sets" + "Generate new set" button.
- **Filter bar**: by candidate, by job description, by creator.
- **List view** (cards hoặc table):
  - Mỗi item: Candidate name + JD title + created_at + question count (derived từ payload).
  - Click mở detail.

#### 14a. Generate modal
- Select candidate (search).
- Select JD (search).
- Button "Generate questions" → calls backend (assume backend endpoint generates; nếu không, frontend tự call LLM qua separate endpoint).

#### 14b. Question Set Detail page
- Header: "Interview for [Candidate] — [JD Title]" (serif), meta.
- **Questions grouped by category** (technical, behavioral, culture-fit, etc.):
  - Mỗi question là 1 card:
    - Question text (serif, quote-style lớn).
    - Tags: category badge, difficulty badge (junior/senior/etc.).
    - Notes area (editable textarea) — recruiter note cho câu này.
  - Reorder drag handle.
  - Edit / Delete question.
- Actions: "Add question", "Regenerate", "Export as PDF", "Print".

**Visual**: Cho đây là screen mà recruiter mang vào phỏng vấn in ra, design phải readable, editorial, đáng tin cậy. Serif heavy.

---

### ⚙️ Screen 15 — Settings (tùy chọn, nếu có thời gian)

**Layout**: Standard settings với tabs bên trái.
- Profile
- Workspace
- API keys
- Notifications
- Danger zone

---

## 3. Shared Components (Designer cần thiết kế cẩn thận)

Các component này lặp đi lặp lại, cần consistent:

1. **Button variants**:
   - Primary (accent background, solid)
   - Secondary (hairline border, transparent bg)
   - Ghost (no border, hover reveals bg)
   - Danger (brick red)
   - Icon button (square, subtle hover)
   - Size: sm / md / lg.

2. **Status badges**: dot + label, variants cho mỗi enum (UploadStatus, ProfileStatus, MatchRunStatus, SentStatus).

3. **Data table**: Row hover, sort indicators, sticky header, empty state, loading skeleton, bulk select.

4. **Modal/Dialog**: Center card, backdrop subtle blur, escape to close.

5. **Toast notifications**: Top-right, auto-dismiss 4s, variants (success, error, info).

6. **Empty states**: Editorial — một dòng heading serif + một SVG tối giản + CTA.

7. **Loading states**: Skeleton shimmer (không spinner). Inline small loader cho button states.

8. **Command palette (⌘K)**: Global search — candidates, JDs, collections, actions. Fuzzy match, recent items.

9. **Avatar**: Initial-based với subtle background colors (deterministic từ name), hoặc photo nếu có.

10. **Tooltip**: Subtle, serif nhỏ, hairline border.

11. **Score visualization**: Reusable — bar, donut, hoặc radar. Cần coi đẹp ở nhiều kích cỡ.

---

## 4. Technical Notes cho Designer

- **Responsive**: Desktop-first (primary target ≥1280px), nhưng phải work được trên 1024px và tablet. Mobile là secondary — có thể simplified layout.
- **Dark mode**: Designer nên provide cả 2 themes. Toggle ở user menu.
- **Loading states**: Upload batch-parse và scoring là SYNC — có thể mất 30s+. UX phải handle tốt: disable close, rotating messages, ETA.
- **Error states**: Khớp với HTTP codes (400, 404, 409, 422, 500). Mỗi error có inline UI tương ứng.
- **Tabular nums**: Bắt buộc cho mọi số (score, count, ID) để layout không jump.
- **UUID display**: Chỉ hiển thị khi cần (match_run_id, session_id) — truncate `#A7F2...3B91` + copy button.
- **Datetimes**: UTC từ backend → hiển thị theo local timezone, format relative ("2 hours ago") với tooltip exact.

---

## 5. Priority cho Designer

Nếu phải pick, đây là thứ tự ưu tiên thiết kế (impact cao nhất trước):

1. **Shared Layout shell** (sidebar + top bar) — ảnh hưởng mọi screen.
2. **Scoring Results (Screen 09c)** — flagship, wow factor.
3. **AI Chat (Screen 10)** — differentiator chính.
4. **Candidates list (Screen 04)** — screen dùng nhiều nhất.
5. **Candidate Detail (Screen 06)** — hub của workflow.
6. **Dashboard (Screen 03)** — first impression sau login.
7. **Landing (Screen 01)** — marketing impact.
8. Còn lại theo thứ tự trong doc.

---

*End of spec. Designer có toàn quyền reinterpret và elevate các mô tả trên — mục tiêu cuối là một sản phẩm có gu, chuyên nghiệp, và đáng nhớ. Không phải một SaaS template khác.*
