# Hướng dẫn cấu hình Google Cloud Platform cho OAuth2 "Sign in with Google"

> **Đối tượng:** Bạn (người quản trị dự án)
> **Mục tiêu:** Tạo OAuth2 Client ID + Secret trên GCP để backend dùng đăng nhập Google.
> **Thời gian ước tính:** 15–25 phút (lần đầu).
> **Kết quả cuối cùng:** 3 giá trị cần điền vào `backend/.env`:
> - `GOOGLE_CLIENT_ID`
> - `GOOGLE_CLIENT_SECRET`
> - `GOOGLE_REDIRECT_URI`

---

## Tổng quan luồng OAuth2 (để hiểu đang cấu hình cho cái gì)

```
User → Frontend (http://localhost:5173)
         │ click "Sign in with Google"
         ▼
     Backend /api/v1/auth/google/login
         │ 302 redirect
         ▼
     Google consent screen (accounts.google.com)
         │ user đồng ý
         ▼
     Backend /api/v1/auth/google/callback?code=...   ← REDIRECT_URI
         │ backend đổi code → id_token → tạo/tìm user → phát JWT app
         ▼
     Frontend /auth/callback?token=<app_jwt>
         │ lưu token, gọi /auth/me
         ▼
     /dashboard
```

Ngoài login cơ bản, repo hiện còn có progressive consent cho Gmail:

- Login ban đầu chỉ dùng `openid email profile`.
- Khi user mở Outreach và chưa nối Gmail, frontend gọi backend để lấy Google authorize URL cho flow `connect_gmail`.
- Sau khi chấp nhận hoặc từ chối Gmail consent, user quay lại đúng route Outreach thay vì `/auth/callback`.

**Quan trọng:** `REDIRECT_URI` ở GCP phải trùng **tuyệt đối** với URL mà backend gọi về Google (khớp scheme, host, port, path, không trailing slash).

---

## Checklist tổng

- [ ] **Bước 1.** Tạo (hoặc chọn) GCP Project
- [ ] **Bước 2.** Bật Google Identity / People API
- [ ] **Bước 3.** Cấu hình OAuth consent screen (External, Testing)
- [ ] **Bước 4.** Thêm Test users (email Gmail của bạn + tester)
- [ ] **Bước 5.** Tạo OAuth 2.0 Client ID (Web application)
- [ ] **Bước 6.** Khai báo Authorized JavaScript origins
- [ ] **Bước 7.** Khai báo Authorized redirect URIs
- [ ] **Bước 8.** Copy Client ID + Client Secret vào `.env`
- [ ] **Bước 9.** Smoke test bằng OAuth Playground (tùy chọn, khuyên làm)
- [ ] **Bước 10.** (Production) Xác minh domain + publishing status

---

## Bước 1 — Tạo GCP Project

1. Truy cập <https://console.cloud.google.com/>.
2. Ở thanh trên cùng, bấm dropdown project → **New Project**.
3. Name: `recruitment-ai-assistant` (hoặc tên dễ nhớ). Organization: để mặc định.
4. Bấm **Create**. Đợi 10–20 giây.
5. Quay lại dropdown project, **chọn project vừa tạo** — kiểm tra tên project xuất hiện ở header.

> **Lỗi hay gặp:** Quên select project → các bước sau tạo nhầm vào project khác. **Luôn kiểm tra tên project ở header** trước khi làm bước tiếp.

---

## Bước 2 — Bật API cần thiết

OAuth2 với scope `openid email profile` **không bắt buộc phải enable API riêng**, nhưng nên bật để tránh lỗi mơ hồ về sau:

1. Menu trái → **APIs & Services → Library**.
2. Search `People API` → bấm vào → **Enable**.
3. (Tùy chọn, nếu sau này muốn đọc Gmail/Calendar) Search `Gmail API`, `Google Calendar API` → Enable khi cần.

> Với scope đăng nhập cơ bản (`openid email profile`), People API là đủ và an toàn.

---

## Bước 3 — Cấu hình OAuth consent screen

Đây là màn hình user thấy khi Google hỏi "App này muốn truy cập thông tin của bạn". **Bắt buộc cấu hình trước khi tạo Client ID.**

1. Menu trái → **APIs & Services → OAuth consent screen**.
2. User Type:
   - **External** — chọn cái này (cho phép mọi tài khoản Google đăng nhập).
   - *Internal* chỉ dùng khi bạn có Google Workspace tổ chức và chỉ cho nhân viên.
3. Bấm **Create**.

### Tab "App information"

| Field                              | Giá trị                                                  |
|------------------------------------|----------------------------------------------------------|
| App name                           | `RecruitAI` (user sẽ thấy tên này)                       |
| User support email                 | Email của bạn                                            |
| App logo                           | (tùy chọn — bỏ qua khi dev)                              |
| Application home page              | `http://localhost:5173` (dev); đổi sau khi deploy        |
| Application privacy policy link    | Bỏ trống khi Testing; bắt buộc khi publish production    |
| Application terms of service link  | Bỏ trống khi Testing                                     |
| Authorized domains                 | Bỏ trống khi Testing với `localhost`                     |
| Developer contact information      | Email của bạn                                            |

Bấm **Save and Continue**.

### Tab "Scopes"

1. Bấm **Add or Remove Scopes**.
2. Tick các scope sau (non-sensitive):
   - `openid`
   - `.../auth/userinfo.email`
   - `.../auth/userinfo.profile`
3. Bấm **Update** → **Save and Continue**.

> **Không thêm scope sensitive/restricted** (Gmail read, Drive, v.v.) nếu chưa cần — Google sẽ yêu cầu verification kéo dài.

### Tab "Test users"

1. Bấm **Add Users**.
2. Nhập email Gmail của bạn + tất cả tester sẽ dùng để login thử.
3. Bấm **Add** → **Save and Continue**.

> **Khi app ở trạng thái Testing:** chỉ các email test users này mới đăng nhập được. Email khác sẽ gặp lỗi `403: access_denied`. Bạn có thể thêm tối đa 100 test users.

### Tab "Summary" → bấm **Back to Dashboard**.

Xác nhận **Publishing status = Testing**.

---

## Bước 4 — Tạo OAuth 2.0 Client ID

1. Menu trái → **APIs & Services → Credentials**.
2. Bấm **+ Create Credentials → OAuth client ID**.
3. **Application type: Web application**. (KHÔNG chọn Desktop/Android/iOS.)
4. Name: `RecruitAI Backend Dev` (chỉ bạn thấy, đặt gì cũng được).

### Authorized JavaScript origins

Thêm **từng dòng** (bấm + ADD URI cho mỗi origin):

```
http://localhost:5173
http://localhost:8000
```

> - **Không** có trailing slash (`/`).
> - **Không** có đường dẫn (chỉ scheme://host:port).
> - Port khớp chính xác: 5173 = Vite frontend, 8000 = FastAPI backend.

### Authorized redirect URIs

Thêm **từng dòng**:

```
http://localhost:8000/api/v1/auth/google/callback
```

> **Đây là field dễ sai nhất. Kiểm tra 3 lần:**
> - scheme `http` (dev dùng http vì localhost; prod phải `https`)
> - host `localhost` (không phải `127.0.0.1` — nếu bạn test bằng 127.0.0.1 thì thêm cả dòng `http://127.0.0.1:8000/...`)
> - path **đúng chính xác** `/api/v1/auth/google/callback`, không dư ký tự, không trailing slash
> - Sai 1 ký tự → Google báo `redirect_uri_mismatch`

5. Bấm **Create**.
6. Popup hiện ra **Client ID** + **Client Secret** → **copy ngay** (Secret có thể xem lại nhưng nên copy liền cho nhanh).

---

## Bước 5 — Dán vào `.env`

Mở `backend/.env` và thêm:

```dotenv
# Google OAuth
GOOGLE_CLIENT_ID=123456789012-abcdefghijklmnop.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/google/callback

# Frontend URL mà backend sẽ redirect về sau khi login xong
FRONTEND_BASE_URL=http://localhost:5173
```

Cũng cập nhật `.env.example` (không dán giá trị thật, chỉ placeholder).

> **Bảo mật:**
> - **Không commit** `.env` (đã có trong `.gitignore` — xác nhận lại).
> - Client Secret chỉ dùng phía backend. **Không bao giờ** đưa ra frontend/browser.
> - Nếu Secret lỡ lộ ra git → quay lại Credentials, **Reset Secret**, cập nhật `.env`.

---

## Bước 6 — Smoke test với OAuth Playground (khuyên làm trước khi code)

Mục đích: xác nhận Client ID/Secret đúng, trước khi tốn thời gian debug code.

1. Mở <https://developers.google.com/oauthplayground/>.
2. Góc trên phải → bấm ⚙ gear icon → **Use your own OAuth credentials** (tick).
3. Dán Client ID + Secret vào.
4. Tạm thời thêm redirect URI `https://developers.google.com/oauthplayground` vào Authorized redirect URIs trong GCP Credentials (bước 4).
5. Left panel: chọn scope `https://www.googleapis.com/auth/userinfo.email` + `userinfo.profile` + `openid`.
6. Bấm **Authorize APIs** → login bằng 1 test user → **Allow**.
7. Bấm **Exchange authorization code for tokens**.
8. Nếu nhận được `access_token` + `id_token` → **config OK**. ✅
9. **Sau khi test xong: xóa dòng `https://developers.google.com/oauthplayground`** khỏi Authorized redirect URIs (không để production credential nhận redirect từ playground).

---

## Bước 7 — Khi deploy lên production

Bạn cần làm lại **Bước 4** với domain thật, HOẶC edit Client ID hiện tại thêm entry mới:

**Authorized JavaScript origins:**
```
https://recruitai.yourdomain.com
```

**Authorized redirect URIs:**
```
https://api.recruitai.yourdomain.com/api/v1/auth/google/callback
```

**Publishing status:** khi sẵn sàng cho public → OAuth consent screen → **Publish App**.
- Với scope non-sensitive (`openid email profile`), **không cần Google verification**, có thể publish ngay.
- Nếu thêm sensitive scope (Gmail, Drive…) → phải submit verification (mất vài ngày đến vài tuần).

---

## Troubleshooting — các lỗi hay gặp

| Lỗi                                              | Nguyên nhân                                                                 | Fix                                                                  |
|--------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------|
| `Error 400: redirect_uri_mismatch`               | Redirect URI backend gửi khác với list ở GCP                                | So sánh **y hệt** từng ký tự. Kiểm tra trailing `/`, http vs https.  |
| `Error 403: access_denied` + "app is being tested" | Email đăng nhập không nằm trong Test users                                  | Thêm email vào Test users (Bước 3).                                  |
| `Error 401: invalid_client`                      | Client Secret sai hoặc đã reset                                             | Kiểm tra `.env`, copy lại Secret từ GCP.                             |
| `This app isn't verified` warning                | App ở Testing mode (bình thường)                                            | Bấm Advanced → Go to RecruitAI (unsafe). Prod: publish + verify.     |
| `idpiframe_initialization_failed`                | Authorized JavaScript origin sai/thiếu                                      | Thêm `http://localhost:5173` vào Authorized JavaScript origins.      |
| Nhận được code nhưng exchange thất bại           | Sai `grant_type` hoặc redirect_uri lúc exchange ≠ lúc authorize             | Khi exchange, redirect_uri phải **giống hệt** lúc authorize.         |
| `invalid_grant` khi refresh                      | Code đã dùng rồi / hết hạn (code chỉ sống ~30 giây, dùng 1 lần)             | Không retry cùng 1 code. Login lại từ đầu.                           |

---

## Checklist trước khi giao cho coding agent

- [ ] Copy được Client ID, Client Secret, Redirect URI
- [ ] Đã test OAuth Playground thành công
- [ ] Đã thêm email tester vào Test users
- [ ] Đã điền vào `backend/.env`
- [ ] Đã xác nhận `.env` nằm trong `.gitignore`
- [ ] (Tùy chọn) Đã lưu Client ID/Secret vào password manager

Khi cả 6 mục này ✅ → bắt đầu chạy `docs/GOOGLE_OAUTH_CODING_PLAN.md`.
