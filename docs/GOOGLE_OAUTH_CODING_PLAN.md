# Implementation Plan — Google OAuth2 "Sign in with Google"

> **Audience:** Coding agent (Cascade/Claude/etc.)
> **Goal:** Thêm Google OAuth2 login song song với email/password hiện có. User mới đăng nhập Google lần đầu → auto-provision `UserAccount` + gán `RoleName.RECRUITER`. User đã tồn tại cùng email → link Google identity vào account cũ.
> **Constraint:** Không phá email/password flow hiện tại. Không đổi schema JWT hiện có (`sub`, `email`, `display_name`).

---

## Prerequisites (BLOCKER — xác nhận trước khi code)

- [ ] `backend/.env` đã có: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `FRONTEND_BASE_URL`
- [ ] User đã chạy xong `docs/GOOGLE_OAUTH_GCP_SETUP.md` và smoke test trên OAuth Playground OK
- [ ] `GOOGLE_REDIRECT_URI` = `http://localhost:8000/api/v1/auth/google/callback` (dev)
- [ ] Postgres đang chạy (`docker compose up -d db` hoặc tương đương), backend start được
- [ ] Frontend `.env` có `VITE_API_BASE_URL=http://localhost:8000/api/v1`

**Nếu bất kỳ mục nào fail → STOP, báo user, không code tiếp.**

---

## Thiết kế tổng (đọc hết trước khi chạy Sprint 1)

### Luồng được chọn: Backend-mediated Authorization Code flow (KHÔNG PKCE public client)

Lý do:
- Backend giữ `CLIENT_SECRET` an toàn.
- Dễ verify `id_token` bằng thư viện chuẩn.
- Không đụng localStorage token từ JS của Google → attack surface thấp.
- Tương thích 100% với `Authorization: Bearer <app_jwt>` pattern hiện tại.

### Sequence

```
1. Browser   GET  /api/v1/auth/google/login?redirect=/dashboard
2. Backend   → generate state (random 32 bytes, sign bằng SECRET_KEY với HMAC),
               nhúng redirect vào state
             → 302 Location: https://accounts.google.com/o/oauth2/v2/auth?...
3. Google    user consent → 302 GOOGLE_REDIRECT_URI?code=<code>&state=<state>
4. Backend   GET  /api/v1/auth/google/callback
             → verify state (HMAC + TTL 10 min)
             → POST https://oauth2.googleapis.com/token (exchange code)
             → verify id_token (aud, iss, exp, signature via Google JWKs)
             → upsert UserAccount by email
             → tạo app JWT (existing create_access_token)
             → 302 {FRONTEND_BASE_URL}/auth/callback?token=<app_jwt>&redirect=<path>
5. Frontend  /auth/callback route → lấy token từ query → storeToken → me() → navigate(redirect)
```

### Library choice: **Authlib** (`authlib>=1.3.0`)

- Lý do: chính thức recommended bởi cộng đồng FastAPI, xử lý OIDC id_token verification đúng chuẩn (aud/iss/exp/nonce/signature), tránh viết tay `requests.post` + parse JWT.
- Tránh dùng `google-auth-oauthlib` (thiên về desktop/CLI) và tránh tự code `httpx.post` (dễ quên verify id_token).

### Schema DB — thêm 1 bảng mới (không đụng `user_accounts`)

```
oauth_identities
  id                UUID PK
  user_id           UUID FK → user_accounts.id ON DELETE CASCADE
  provider          VARCHAR(32)  -- "google"
  provider_subject  VARCHAR(255) -- Google "sub" claim (stable, never reused)
  email             VARCHAR(320) -- Google email tại thời điểm link (snapshot, không unique)
  created_at        TIMESTAMPTZ
  UNIQUE (provider, provider_subject)
  INDEX  (user_id)
```

Lý do tách bảng:
- Một user có thể link nhiều provider (Google, GitHub…) trong tương lai.
- `provider_subject` (Google's `sub`) là **identity thật**, email có thể đổi. Không dùng email làm khóa OAuth identity.
- `user_accounts.password_hash` đã `Optional` → OAuth-only user hợp lệ.

### Email linking rule (quan trọng, tránh account takeover)

Khi callback có Google account với email E:

1. Tìm `oauth_identities(provider='google', provider_subject=google_sub)`:
   - Nếu có → login user đó. **Done.**
2. Nếu chưa có, tìm `user_accounts(email=E)`:
   - **Nếu tồn tại + `email_verified=true` từ Google** → tạo `oauth_identity` link vào user đó (auto-link). Login.
   - **Nếu tồn tại + email CHƯA verified từ Google** → reject với `400: google_email_not_verified`. (Tránh ai đó tạo Google account với email của nạn nhân mà chưa verify.)
3. Nếu không tồn tại user → tạo `UserAccount` mới (password_hash=NULL, status=ACTIVE, display_name từ Google `name`), gán role `RECRUITER`, tạo `oauth_identity`, login.

---

## Sprint Plan

**5 sprints**, mỗi sprint phải chạy qua **Checkpoint gate** trước khi sang sprint tiếp. Nếu checkpoint fail → dừng, fix, chạy lại checkpoint.

**Atomic commit rule:** mỗi sprint = 1 commit. Commit message: `feat(auth): google oauth sprint N - <tên>`.

---

### Sprint 1 — Config & Database migration ✅ DONE (commit: 3a0f807)

**Goal:** Có config object + bảng `oauth_identities` + test fixtures. Chưa có endpoint.

#### Tasks

1.1. `backend/requirements.txt` — thêm:
   ```
   authlib>=1.3.0
   httpx>=0.25.0
   itsdangerous>=2.1.0
   ```
   Chạy `pip install -r backend/requirements.txt` trong venv.

1.2. `backend/src/core/config.py` — thêm fields:
   ```python
   GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
   GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
   GOOGLE_REDIRECT_URI: str = os.getenv(
       "GOOGLE_REDIRECT_URI",
       "http://localhost:8000/api/v1/auth/google/callback",
   )
   FRONTEND_BASE_URL: str = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")
   OAUTH_STATE_TTL_SECONDS: int = 600  # 10 phút
   ```

1.3. `backend/src/models/oauth_identity.py` — tạo file mới:
   - Class `OAuthIdentity(Base)` với các cột như schema trên.
   - Dùng `UUID(as_uuid=True)` primary key, `server_default=func.now()`.
   - Relationship `user: Mapped[UserAccount] = relationship(back_populates="oauth_identities")`.

1.4. `backend/src/models/user_account.py` — thêm back-reference:
   ```python
   oauth_identities: Mapped[list["OAuthIdentity"]] = relationship(
       back_populates="user", cascade="all, delete-orphan"
   )
   ```

1.5. `backend/src/models/__init__.py` — import `OAuthIdentity` để Alembic thấy.

1.6. Tạo migration:
   ```bash
   cd backend
   alembic revision --autogenerate -m "add oauth_identities table"
   ```
   **Đọc kỹ file generated** — nếu autogenerate tạo DROP/ALTER nhầm bảng khác → xóa dòng đó. Chỉ giữ `op.create_table("oauth_identities", ...)` + index + unique constraint.

1.7. Apply migration: `alembic upgrade head`.

#### ✅ Checkpoint 1 (PHẢI pass hết trước Sprint 2)

Chạy các lệnh sau, mọi lệnh đều phải pass:

- [x] **CP1.1** `alembic current` → hiện revision mới nhất vừa tạo.
- [x] **CP1.2** Kết nối psql (hoặc `\d oauth_identities` qua docker):
  ```sql
  SELECT column_name, data_type, is_nullable
  FROM information_schema.columns WHERE table_name = 'oauth_identities';
  ```
  Phải có đủ 6 cột: `id, user_id, provider, provider_subject, email, created_at`.
- [x] **CP1.3** Unique constraint check:
  ```sql
  SELECT conname FROM pg_constraint WHERE conrelid = 'oauth_identities'::regclass;
  ```
  Phải có unique `(provider, provider_subject)`.
- [x] **CP1.4** Downgrade/upgrade round-trip OK:
  ```
  alembic downgrade -1 && alembic upgrade head
  ```
  Không lỗi, bảng vẫn đúng schema.
- [x] **CP1.5** Backend start được: `uvicorn src.main:app --reload` → GET `/` trả `{"message": ...}`.
- [x] **CP1.6** `from src.models import OAuthIdentity` không raise trong Python shell.

**Nếu fail bất kỳ:** rollback migration, fix, lặp lại. **KHÔNG** chuyển Sprint 2.

**Commit:** `feat(auth): google oauth sprint 1 - config and oauth_identities table`

---

### Sprint 2 — OAuth service layer (pure logic, không endpoint)

**Goal:** Service module xử lý state signing, token exchange, id_token verification, user upsert. Có unit test. Chưa expose HTTP.

#### Tasks

2.1. `backend/src/services/google_oauth.py` — tạo mới:

Các function bắt buộc (không phải class, giữ đơn giản):

```python
# Signatures — agent code theo đúng:

def build_authorize_url(redirect_path: str) -> tuple[str, str]:
    """Returns (authorize_url, state). redirect_path is the frontend path to go to after login (e.g. "/dashboard")."""

def verify_state(state: str) -> str:
    """Returns the original redirect_path. Raises ValueError if invalid/expired."""

async def exchange_code_for_tokens(code: str) -> dict:
    """POSTs to https://oauth2.googleapis.com/token. Returns Google token response dict."""

def verify_id_token(id_token: str) -> dict:
    """Verifies signature against Google JWKs, checks aud == CLIENT_ID, iss, exp. Returns claims dict with sub, email, email_verified, name, picture."""

def upsert_user_from_google(db: Session, claims: dict) -> UserAccount:
    """Implements email linking rule. Raises ValueError('google_email_not_verified') when appropriate."""
```

Implementation notes:
- `build_authorize_url`: dùng `itsdangerous.URLSafeTimedSerializer(settings.SECRET_KEY, salt="google-oauth-state")` để ký state. Payload = `{"redirect": redirect_path, "nonce": secrets.token_urlsafe(16)}`.
- `verify_state`: gọi `serializer.loads(state, max_age=settings.OAUTH_STATE_TTL_SECONDS)`.
- `exchange_code_for_tokens`: dùng `httpx.AsyncClient()`, body form-encoded (`grant_type=authorization_code`, `code`, `client_id`, `client_secret`, `redirect_uri` — **redirect_uri phải = `settings.GOOGLE_REDIRECT_URI` y hệt lúc authorize**).
- `verify_id_token`: dùng `authlib.jose.jwt.decode` + JWKs từ `https://www.googleapis.com/oauth2/v3/certs`. Cache JWKs in-memory với TTL 1h (tránh hit Google mỗi request). Validate `aud`, `iss in ("https://accounts.google.com", "accounts.google.com")`, `exp`.
- `upsert_user_from_google`:
  1. `db.execute(select(OAuthIdentity).where(provider='google', provider_subject=claims['sub']))`.
  2. Nếu match → return `identity.user`.
  3. Else: `db.execute(select(UserAccount).where(email == claims['email']))`.
  4. Nếu match: require `claims['email_verified'] is True`, else raise. Tạo `OAuthIdentity` link, `db.commit()`, return user.
  5. Else: tạo `UserAccount(email=claims['email'], display_name=claims.get('name') or claims['email'].split('@')[0], password_hash=None, status=ACTIVE)` + `RoleAssignment(RECRUITER)` + `OAuthIdentity`. Commit. Return.

2.2. `backend/tests/test_google_oauth_service.py` — tạo mới. Tối thiểu 6 test:
   - `test_build_then_verify_state_roundtrip`
   - `test_verify_state_rejects_tampered`
   - `test_verify_state_rejects_expired` (monkeypatch TTL to 0 hoặc freeze time)
   - `test_upsert_creates_new_user_when_not_exists`
   - `test_upsert_links_existing_user_when_email_verified`
   - `test_upsert_rejects_when_email_not_verified_and_user_exists`

   Dùng in-memory SQLite hoặc test Postgres fixture (xem file test hiện có nếu có). Mock `verify_id_token` để không gọi Google thật.

#### ✅ Checkpoint 2

- [ ] **CP2.1** `pytest backend/tests/test_google_oauth_service.py -v` — tất cả test pass.
- [ ] **CP2.2** Không có test nào dùng network thật (verify bằng `pytest --disable-socket` nếu đã cài, hoặc grep `accounts.google.com`/`oauth2.googleapis.com` trong test → chỉ được xuất hiện trong mock/patch).
- [ ] **CP2.3** Tampered state test: đổi 1 ký tự trong state → `verify_state` raise.
- [ ] **CP2.4** Import từ REPL: `from src.services.google_oauth import build_authorize_url; build_authorize_url("/dashboard")` — trả tuple `(url, state)`, url chứa `accounts.google.com`.

**Commit:** `feat(auth): google oauth sprint 2 - service layer with tests`

---

### Sprint 3 — FastAPI endpoints

**Goal:** Wire service vào 2 endpoint mới. Manual test với Google thật.

#### Tasks

3.1. `backend/src/api/v1/endpoints/auth.py` — thêm vào cuối file:

```python
from fastapi import Request
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode
from src.services import google_oauth

@router.get("/google/login")
def google_login(redirect: str = "/dashboard"):
    # Whitelist redirect — chỉ cho phép path tương đối, tránh open redirect
    if not redirect.startswith("/") or redirect.startswith("//"):
        redirect = "/dashboard"
    url, _ = google_oauth.build_authorize_url(redirect)
    return RedirectResponse(url, status_code=302)

@router.get("/google/callback")
async def google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: Session = Depends(get_db),
):
    frontend = settings.FRONTEND_BASE_URL.rstrip("/")

    if error:
        return RedirectResponse(f"{frontend}/login?error={error}", status_code=302)
    if not code or not state:
        return RedirectResponse(f"{frontend}/login?error=missing_params", status_code=302)

    try:
        redirect_path = google_oauth.verify_state(state)
    except Exception:
        return RedirectResponse(f"{frontend}/login?error=invalid_state", status_code=302)

    try:
        tokens = await google_oauth.exchange_code_for_tokens(code)
        claims = google_oauth.verify_id_token(tokens["id_token"])
        user = google_oauth.upsert_user_from_google(db, claims)
    except ValueError as e:
        return RedirectResponse(f"{frontend}/login?error={e}", status_code=302)
    except Exception:
        return RedirectResponse(f"{frontend}/login?error=oauth_failed", status_code=302)

    app_token = create_access_token(
        subject=str(user.id), email=user.email, display_name=user.display_name
    )
    qs = urlencode({"token": app_token, "redirect": redirect_path})
    return RedirectResponse(f"{frontend}/auth/callback?{qs}", status_code=302)
```

3.2. Import `settings` trong file nếu chưa có (`from src.core.config import settings`).

3.3. **KHÔNG** sửa `api.py` — `auth.router` đã được include rồi, endpoint mới tự vào.

3.4. Đảm bảo `BACKEND_CORS_ORIGINS` trong `.env` có `http://localhost:5173`. (Callback redirect về frontend KHÔNG qua CORS vì là 302 trực tiếp trên browser, nhưng `/auth/me` sau đó cần CORS.)

#### ✅ Checkpoint 3

- [ ] **CP3.1** `uvicorn` start không lỗi. OpenAPI `/docs` hiện 2 endpoint mới: `GET /api/v1/auth/google/login`, `GET /api/v1/auth/google/callback`.
- [ ] **CP3.2** `curl -i "http://localhost:8000/api/v1/auth/google/login"` → HTTP 302, `Location:` chứa `accounts.google.com/o/oauth2/v2/auth`, có query `client_id`, `redirect_uri`, `scope=openid+email+profile` (hoặc encoded), `state=`, `response_type=code`.
- [ ] **CP3.3** **Manual E2E test with real Google:**
  1. Mở trình duyệt thường (không incognito nếu chưa login Google).
  2. Truy cập `http://localhost:8000/api/v1/auth/google/login?redirect=/dashboard`.
  3. Hiện consent screen Google → login bằng test user email.
  4. Google redirect về `http://localhost:8000/api/v1/auth/google/callback?code=...`.
  5. Backend redirect về `http://localhost:5173/auth/callback?token=<jwt>&redirect=/dashboard`.
  6. (Frontend chưa có route này — 404 là OK ở checkpoint này, miễn URL có `token=ey...`.)
- [ ] **CP3.4** Check DB:
  ```sql
  SELECT email, display_name, password_hash FROM user_accounts ORDER BY created_at DESC LIMIT 1;
  SELECT provider, provider_subject, email FROM oauth_identities ORDER BY created_at DESC LIMIT 1;
  ```
  User mới được tạo với `password_hash = NULL`, identity record có `provider='google'`.
- [ ] **CP3.5** Login lần 2 cùng tài khoản Google đó → **không** tạo user mới (COUNT không đổi), chỉ redirect với token.
- [ ] **CP3.6** Test open-redirect guard: gọi `?redirect=//evil.com` → server thay bằng `/dashboard`. Kiểm URL cuối cùng.
- [ ] **CP3.7** Test tampered state: lấy callback URL thật, đổi 1 ký tự trong `state` → redirect về `/login?error=invalid_state`.

**Nếu CP3.3 fail với `redirect_uri_mismatch`:** so sánh chính xác `settings.GOOGLE_REDIRECT_URI` trong backend với Authorized redirect URIs trong GCP. **KHÔNG** tiếp tục.

**Commit:** `feat(auth): google oauth sprint 3 - login and callback endpoints`

---

### Sprint 4 — Frontend integration

**Goal:** Nút "Sign in with Google" trên `/login`, route `/auth/callback` xử lý token, đăng nhập xong vào dashboard.

#### Tasks

4.1. `frontend/src/api/endpoints/auth.ts` — thêm helper:
   ```typescript
   const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

   export const authApi = {
     // ... existing methods

     getGoogleLoginUrl(redirect: string = "/dashboard"): string {
       const qs = new URLSearchParams({ redirect });
       return `${API_BASE}/auth/google/login?${qs.toString()}`;
     },
   };
   ```

4.2. `frontend/src/routes/login.tsx` — thêm nút Google phía trên form (hoặc dưới, tùy design):
   ```tsx
   <a
     href={api.auth.getGoogleLoginUrl(searchParams.get("redirect") ?? "/dashboard")}
     className="w-full h-12 flex items-center justify-center gap-2 border border-sand-200 rounded-xl hover:bg-sand-50 transition"
   >
     {/* Google G icon SVG inline — không cần install thêm library */}
     <svg width="18" height="18" viewBox="0 0 48 48">{/* ... official Google G paths ... */}</svg>
     Sign in with Google
   </a>
   <div className="flex items-center gap-3 text-xs text-forest-400">
     <div className="flex-1 h-px bg-sand-200" /> OR <div className="flex-1 h-px bg-sand-200" />
   </div>
   ```
   Hiển thị error từ query param: đọc `searchParams.get("error")` → `toast.error(mapErrorToMessage(error))` trong `useEffect`.

4.3. `frontend/src/routes/auth-callback.tsx` — tạo mới:
   ```tsx
   import { useEffect } from "react";
   import { useNavigate, useSearchParams } from "react-router";
   import { api } from "@/api";
   import { useAuthStore } from "@/lib/auth";
   import { toast } from "sonner";

   export default function AuthCallbackRoute() {
     const [sp] = useSearchParams();
     const navigate = useNavigate();

     useEffect(() => {
       const token = sp.get("token");
       const redirect = sp.get("redirect") ?? "/dashboard";
       const error = sp.get("error");

       if (error || !token) {
         toast.error("Google sign-in failed.");
         navigate("/login", { replace: true });
         return;
       }

       api.auth.storeToken(token);
       api.auth.me()
         .then((user) => {
           useAuthStore.getState().setUser(user);
           navigate(redirect, { replace: true });
         })
         .catch(() => {
           api.auth.clearToken();
           toast.error("Could not load profile.");
           navigate("/login", { replace: true });
         });
     }, []); // eslint-disable-line react-hooks/exhaustive-deps

     return (
       <div className="min-h-screen flex items-center justify-center">
         <p className="text-forest-600">Signing you in…</p>
       </div>
     );
   }
   ```

4.4. `frontend/src/router.tsx` — thêm route `/auth/callback`:
   - Route này **public** (không guard), vì user chưa có token lúc vào.
   - Lazy import giống các route khác.

4.5. `frontend/src/routes/index.ts` — export route mới.

#### ✅ Checkpoint 4

- [ ] **CP4.1** `npm run typecheck` pass. `npm run lint` pass (0 warnings).
- [ ] **CP4.2** `npm run dev` — mở `/login`, thấy nút "Sign in with Google" có Google G icon.
- [ ] **CP4.3** **E2E happy path:**
  1. Click "Sign in with Google" → tab Google consent.
  2. Chọn test user → Allow.
  3. Redirect qua `/auth/callback?token=...` → thấy "Signing you in…" 1 giây.
  4. Landing tại `/dashboard`, header hiện display_name từ Google.
  5. `localStorage.getItem("recruitai.token")` có JWT.
  6. Decode JWT (jwt.io): `sub` là UUID, `email` là Gmail, `display_name` đúng.
- [ ] **CP4.4** **E2E with redirect param:** truy cập `/login?redirect=/candidates` → click Google → sau login phải vào `/candidates`, không phải `/dashboard`.
- [ ] **CP4.5** **Protected route:** refresh `/dashboard` sau login → vẫn ở đó (không bounce về `/login`). Existing guard dùng token trong localStorage — verify vẫn hoạt động.
- [ ] **CP4.6** **Error path:** truy cập thủ công `http://localhost:5173/login?error=google_email_not_verified` → toast lỗi hiện.
- [ ] **CP4.7** **Logout:** click logout → `clearToken()` + `clearUser()` → bounce `/login`. Re-login Google → cùng user (COUNT user accounts không đổi).
- [ ] **CP4.8** **Co-existence:** existing email/password login vẫn chạy (test bằng seed user cũ nếu có).

**Commit:** `feat(auth): google oauth sprint 4 - frontend integration`

---

### Sprint 5 — Hardening & Docs

**Goal:** Siết security, thêm E2E test tự động, update docs.

#### Tasks

5.1. **State in signed cookie (optional upgrade):** hiện tại state tự-contained trong URL. Nếu muốn chặt hơn, set httpOnly cookie `oauth_state` với giá trị state lúc `/login`, callback so sánh. Nếu thời gian hạn chế — skip, chỉ note vào `docs/BACKEND.md`.

5.2. **Rate limit callback:** thêm lớp middleware chỉ cho callback hoặc dùng `slowapi` giới hạn `/auth/google/*` ở 10 req/min/IP. Nếu chưa có slowapi — skip, note TODO.

5.3. **Audit log:** thêm log `INFO` khi:
   - User mới tạo qua Google (`user_id`, `email` — không log token, không log claims đầy đủ).
   - Link Google identity vào user cũ.
   - Callback error (error code, không log code/token).

5.4. **Tests:**
   - `backend/tests/test_auth_endpoints.py` — thêm test `/auth/google/login` trả 302 + Location đúng, `/auth/google/callback?error=access_denied` redirect về frontend login với error param.
   - Mock `exchange_code_for_tokens` + `verify_id_token` để test callback happy path + email_not_verified path.

5.5. **Docs:**
   - `docs/BACKEND.md` — thêm section "Google OAuth2" với sequence diagram + env vars.
   - `README.md` hoặc `QUICKSTART.md` — 1 đoạn ngắn: cách setup OAuth (link sang `docs/GOOGLE_OAUTH_GCP_SETUP.md`).
   - `.env.example` — thêm 4 biến OAuth (placeholder).

5.6. **Security review — agent tự check:**
   - [ ] Không log Client Secret, id_token, access_token, app JWT ở bất kỳ đâu (grep: `logger.*token`, `print.*token`, `print.*secret`).
   - [ ] `redirect` param whitelist chỉ cho path tương đối.
   - [ ] State TTL ≤ 10 phút, HMAC-signed.
   - [ ] id_token verify: aud == CLIENT_ID, iss chuẩn, exp không bỏ qua.
   - [ ] Không auto-link nếu `email_verified=false`.
   - [ ] Error redirect không echo user input vào HTML (đang dùng query string, frontend chỉ đọc `sp.get("error")` và map qua bảng hằng số).

#### ✅ Checkpoint 5 (Final)

- [ ] **CP5.1** `pytest backend/tests/ -v` — tất cả pass, coverage auth flow ≥ 80%.
- [ ] **CP5.2** `npm run build` frontend pass.
- [ ] **CP5.3** 4 env vars mới có trong `.env.example` (giá trị placeholder).
- [ ] **CP5.4** Full happy path E2E manual lại lần cuối (user mới + user đã tồn tại + đã link).
- [ ] **CP5.5** Grep security: `rg -i "secret|token|password" backend/src --type py | rg -i "print|log"` — review từng dòng, không dòng nào in ra giá trị nhạy cảm.
- [ ] **CP5.6** `.env` KHÔNG được commit (`git status` không thấy `.env`).
- [ ] **CP5.7** `docs/BACKEND.md` đã update.

**Commit:** `feat(auth): google oauth sprint 5 - hardening and docs`

**Final PR title:** `feat(auth): Sign in with Google (OAuth2 authorization code flow)`

---

## Rollback plan

Nếu cần revert sau khi merge:

1. `git revert <merge_commit>` — revert code.
2. `alembic downgrade -1` — xóa bảng `oauth_identities`. User đã tạo qua Google sẽ còn trong `user_accounts` với `password_hash=NULL` → không login được bằng password. 2 option:
   - Giữ lại, reset password qua admin tool.
   - `DELETE FROM user_accounts WHERE password_hash IS NULL;` (chỉ làm nếu chắc chắn không có legacy user nào password_hash NULL — kiểm trước).

---

## Common pitfalls — reference nhanh

| Pitfall                                                    | Phòng tránh                                                     |
|------------------------------------------------------------|------------------------------------------------------------------|
| `redirect_uri_mismatch`                                    | `settings.GOOGLE_REDIRECT_URI` phải y hệt GCP, từng ký tự.       |
| id_token không verify signature (chỉ decode)               | Dùng Authlib JWKs, **không** `jwt.get_unverified_claims`.        |
| Email takeover: attacker tạo Google với email nạn nhân     | Chỉ auto-link khi `email_verified=true`.                         |
| State replay / CSRF                                        | HMAC sign state, TTL 10 phút, kiểm trước exchange.               |
| Open redirect qua `redirect` param                         | Whitelist: phải start `/` và không start `//`.                   |
| App JWT trong URL bị log ở server/proxy                    | Acceptable vì chỉ 1 hop tới frontend; frontend đọc xong nên `history.replaceState` xóa query (optional polish). |
| Cache JWKs quá lâu                                         | TTL 1h; bắt KeyError → refresh JWKs 1 lần rồi retry.             |
| `.env` commit nhầm                                         | `git diff --cached` trước mọi commit; `.env` trong `.gitignore`. |
| CORS chặn `/auth/me` sau callback                          | `BACKEND_CORS_ORIGINS` phải có frontend origin.                  |
| Test đụng network thật                                     | Luôn mock `httpx` + `verify_id_token` trong unit test.           |

---

## Sequence compact (dán vào comment PR)

```
FE /login  ──click──► BE /auth/google/login
                        │ 302
                        ▼
                    Google consent
                        │ 302
                        ▼
              BE /auth/google/callback?code&state
                        │ verify state → exchange → verify id_token → upsert user → issue app JWT
                        ▼ 302
              FE /auth/callback?token&redirect
                        │ storeToken → /auth/me → setUser
                        ▼
              FE /dashboard (or redirect target)
```
