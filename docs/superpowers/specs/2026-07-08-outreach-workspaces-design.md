# Outreach Workspaces Design

## Goal

Tach `Outreach` thanh hai workspace ro rang:

- `Messages` de tao, chinh sua, gui outreach draft cho candidate
- `Templates` de tao va quan ly email template tai su dung

Chi workspace `Templates` moi co AI draft. Workspace `Messages` khong co AI.

## Problem

Man hinh `frontend/src/routes/outreach.tsx` hien dang gom ca:

- danh sach outreach messages
- modal tao message
- chon template de do noi dung vao message
- luu nhanh template moi ngay trong modal tao message

Dieu nay lam luong `new message` va `new template` bi tron, user khong co workspace rieng de quan ly template, va khong ro AI draft thuoc phan nao.

## User Flow

### 1. Templates workspace

Route: `/outreach/templates`

User vao day de:

- xem danh sach template theo job dang chon
- tao template moi
- sua template da co

Luong `New template`:

1. User nhap `Template name`
2. User nhap `AI brief`
3. User bam `Generate once`
4. Backend sinh `subject + body`
5. Frontend do ket qua vao editor
6. User sua tay
7. User bam `Save template`

AI chi sinh nhap mot lan. Khong co `Regenerate`.

### 2. Messages workspace

Route: `/outreach`

User vao day de:

- xem folder draft/sent/failed
- loc theo candidate
- doc va sua outreach message
- gui email

Luong `New message`:

1. User chon candidate
2. User chon `Start blank` hoac chon template co san
3. Neu chon template, frontend chen `subject/body` cua template vao editor
4. User sua noi dung neu can
5. User luu message draft

Workspace nay khong co:

- AI draft
- save as template
- toggle `AI Draft` / `Template`

## Information Architecture

## Shared outreach navigation

`Outreach` van la mot khu vuc chung trong sidebar chinh. Ben trong khu vuc nay co sub-navigation gom:

- `Messages`
- `Templates`

Hai route chia se cung visual language de nguoi dung cam thay dang o mot module, nhung luong thao tac tach biet.

## Messages workspace layout

Giu shell hien tai vi no hop voi message inbox:

- cot trai: folders + candidate filter + `New message`
- cot giua: message list
- cot phai: message detail editor

Chi don gian hoa compose modal.

## Templates workspace layout

Dung layout quan ly template thay vi inbox:

- phan dau trang: tieu de + mo ta + `New template`
- subnav workspace giong interviews hub
- danh sach template dang table/list
- create/edit dung modal lon hoac detail editor rieng

Ban dau uu tien:

- list templates
- create template
- edit template

Khong can them preview theo candidate trong vong dau.

## Data Model And API

## Content source

Enum `ContentSource` can them gia tri `manual`.

Y nghia:

- `template`: message duoc tao tu template hoac ban than template record
- `manual`: message duoc viet trang tu dau
- `ai_draft`: khong con duoc dung cho message outreach trong flow moi, nhung co the giu lai de tranh vo enum cu va de dung cho compatibility neu can

Workspace `Messages` se dung:

- `template` khi tao message tu template
- `manual` khi tao message blank

Workspace `Templates` van luu `OutreachTemplate.content_source = template`.

## New generate endpoint

Them endpoint moi:

- `POST /api/v1/outreach/templates/generate-draft`

Request:

- `job_id`
- `brief`
- `variables_allowed`

Response:

- `subject`
- `body_text`
- `body_html`
- `variables_used`

Endpoint nay khong tao DB record. No chi sinh ban nhap de frontend do vao editor.

## Error handling

- `404` neu `job_id` khong ton tai
- `422` neu `brief` rong
- `502` neu goi LLM that bai hoac parse ket qua that bai

Frontend giu nguyen noi dung dang sua khi generate that bai.

## Prompting

Prompt sinh outreach template can:

- viet theo giong email recruiter chuyen nghiep
- sinh ca subject va body
- khuyen khich dung placeholder variables nhu `{{candidate_name}}`, `{{job_title}}`, `{{company_name}}`
- khong tu y chen bien ngoai whitelist
- tra ket qua JSON on dinh de backend parse

Prompt nay nen duoc dat trong `backend/src/prompts/build_prompts.py` de cung pattern voi interview/scoring.

## Frontend Behavior

## New message modal

Modal moi can ngan gon va ro:

- candidate selector
- source selector:
  - `Start blank`
  - `Use template`
- template picker chi hien khi chon template
- subject input
- outreach rich editor

Neu user doi template, frontend thay noi dung bang noi dung template vua chon. Khong co logic merge phuc tap.

## New template modal

Modal moi can co 3 khoi:

1. `Template name`
2. `AI brief`
3. `Email editor`

Nut `Generate once` nam trong khoi `AI brief`, khong tron vao action save.

Editor van dung `OutreachRichEditor` va danh sach chips variables hien co.

## Editing existing templates

User co the mo template cu, sua `name`, `subject`, `body`, va `variables_used`, roi luu lai.

Khong them lich su version trong pham vi nay.

## Routing

Can bo sung route moi cho templates, uu tien theo pattern da co o interviews:

- `routes.outreach = "/outreach"`
- `routes.outreachTemplates = "/outreach/templates"`

Sidebar chinh van tro den `Messages` workspace (`/outreach`).
Trong module outreach, subnav cho phep chuyen sang `Templates`.

## Testing

## Backend

Them test cho:

- generate draft thanh cong
- brief rong -> `422`
- job khong ton tai -> `404`
- LLM loi -> `502`
- create message blank -> `content_source=manual`
- create message tu template -> `content_source=template`

## Frontend

Them test cho:

- route `/outreach` chi hien message workspace
- route `/outreach/templates` chi hien template workspace
- modal `New message` khong hien AI controls
- modal `New template` co `AI brief` va `Generate once`
- chon template trong `New message` se chen noi dung vao editor

## Non-goals

- khong them regenerate nhieu lan
- khong them preview render placeholder theo candidate
- khong them analytics hoac template versioning
- khong doi luong Gmail onboarding ngoai phan can thiet de route split van hoat dong
