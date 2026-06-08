# Gmail Progressive Consent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Google login from Gmail permission so users sign in with basic scopes first, then grant `gmail.send` only from the mail-sending area via onboarding.

**Architecture:** Keep one Google OAuth callback but encode flow intent in signed state. Backend exposes a `gmail_connected` capability on the authenticated profile, and the frontend gates the Outreach route with an onboarding state that launches a separate Gmail consent flow.

**Tech Stack:** FastAPI, SQLAlchemy, React, React Router, TanStack Query, pytest, existing Google OAuth helpers.

---

## File Structure

- Modify: `backend/src/services/google_oauth.py`
  - Split authorize URL creation by flow type and enrich signed state with redirect path and flow type.
- Modify: `backend/src/api/v1/endpoints/auth.py`
  - Keep `/google/login` for basic scopes, add `/google/connect-gmail`, and branch callback behavior by flow type.
- Modify: `backend/src/models/oauth_identity.py`
  - Add a small capability helper or keep model unchanged and compute capability elsewhere.
- Modify: `backend/src/api/v1/endpoints/auth.py` response models
  - Add `gmail_connected` to authenticated profile payload.
- Modify: `backend/tests/test_google_oauth_service.py`
  - Cover flow-specific scopes and signed state parsing.
- Modify: `backend/tests/test_auth_endpoints.py`
  - Cover new endpoint, split callback behavior, and denial redirect.
- Modify: `frontend/src/api/endpoints/auth.ts`
  - Add `gmail_connected` to `UserProfile` and add `getGoogleConnectGmailUrl`.
- Modify: `frontend/src/routes/auth-callback.tsx`
  - Handle Gmail connect success and denial redirects without treating them as login failures.
- Modify: `frontend/src/routes/outreach.tsx`
  - Gate mail UI on `gmail_connected`; render onboarding state and CTA when missing.
- Modify: `frontend/src/lib/auth.ts` or the current auth store file
  - Ensure stored user profile includes `gmail_connected`.
- Create or modify tests closest to current frontend test conventions if present.

---

### Task 1: Split OAuth helper into explicit flow types

**Files:**
- Modify: `backend/src/services/google_oauth.py`
- Test: `backend/tests/test_google_oauth_service.py`

- [ ] **Step 1: Write the failing tests for flow-specific authorize URLs and state payload**

```python
def test_build_login_authorize_url_uses_basic_scopes():
    url, state = google_oauth.build_authorize_url(
        redirect_path="/dashboard",
        flow="login",
    )

    assert "openid email profile" in url
    assert "gmail.send" not in url

    payload = google_oauth.verify_state(state)
    assert payload["redirect"] == "/dashboard"
    assert payload["flow"] == "login"


def test_build_connect_gmail_authorize_url_uses_gmail_scope():
    url, state = google_oauth.build_authorize_url(
        redirect_path="/outreach",
        flow="connect_gmail",
    )

    assert "openid email profile https://www.googleapis.com/auth/gmail.send" in url

    payload = google_oauth.verify_state(state)
    assert payload["redirect"] == "/outreach"
    assert payload["flow"] == "connect_gmail"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_google_oauth_service.py -k "authorize_url or verify_state" -v`
Expected: FAIL because `build_authorize_url` does not accept `flow` and `verify_state` returns only a string.

- [ ] **Step 3: Write the minimal implementation**

```python
GOOGLE_BASIC_SCOPES = "openid email profile"
GOOGLE_GMAIL_SCOPES = "openid email profile https://www.googleapis.com/auth/gmail.send"


def _get_scopes_for_flow(flow: str) -> str:
    if flow == "connect_gmail":
        return GOOGLE_GMAIL_SCOPES
    return GOOGLE_BASIC_SCOPES


def build_authorize_url(redirect_path: str, flow: str = "login") -> tuple[str, str]:
    payload = {
        "redirect": redirect_path,
        "flow": flow,
        "nonce": secrets.token_urlsafe(16),
    }
    state = _get_serializer().dumps(payload)

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": _get_scopes_for_flow(flow),
        "state": state,
        "access_type": settings.GOOGLE_OAUTH_ACCESS_TYPE,
        "prompt": settings.GOOGLE_OAUTH_PROMPT,
        "include_granted_scopes": "true",
    }
    query = urlencode(params)
    return f"{GOOGLE_AUTH_URL}?{query}", state


def verify_state(state: str) -> dict:
    payload = _get_serializer().loads(state, max_age=settings.OAUTH_STATE_TTL_SECONDS)
    return {
        "redirect": payload["redirect"],
        "flow": payload.get("flow", "login"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_google_oauth_service.py -k "authorize_url or verify_state" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/google_oauth.py backend/tests/test_google_oauth_service.py
git commit -m "refactor(auth): split google oauth scopes by flow"
```

### Task 2: Add Gmail capability detection to auth profile

**Files:**
- Modify: `backend/src/api/v1/endpoints/auth.py`
- Modify: `backend/src/models/oauth_identity.py`
- Test: `backend/tests/test_auth_account_endpoints.py`

- [ ] **Step 1: Write the failing tests for `gmail_connected` on `/auth/me`**

```python
def test_get_me_returns_gmail_connected_false_without_google_refresh_token(client, auth_header, db_session, user):
    response = client.get("/api/v1/auth/me", headers=auth_header(user))

    assert response.status_code == 200
    assert response.json()["gmail_connected"] is False


def test_get_me_returns_gmail_connected_true_with_refresh_token_and_scope(client, auth_header, db_session, user):
    identity = OAuthIdentity(
        user_id=user.id,
        provider="google",
        provider_subject="google-sub-1",
        email=user.email,
        refresh_token_encrypted="encrypted-refresh",
        scope="openid email profile https://www.googleapis.com/auth/gmail.send",
    )
    db_session.add(identity)
    db_session.commit()

    response = client.get("/api/v1/auth/me", headers=auth_header(user))

    assert response.status_code == 200
    assert response.json()["gmail_connected"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_auth_account_endpoints.py -k "gmail_connected or get_me" -v`
Expected: FAIL because `UserResponse` does not include `gmail_connected`.

- [ ] **Step 3: Write the minimal implementation**

```python
class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    gmail_connected: bool


def _is_gmail_connected(current_user: UserAccount) -> bool:
    for identity in current_user.oauth_identities:
        if identity.provider != "google":
            continue
        if not identity.refresh_token_encrypted:
            continue
        if not identity.scope or "https://www.googleapis.com/auth/gmail.send" not in identity.scope:
            continue
        return True
    return False


@router.get("/me", response_model=UserResponse)
def get_me(current_user: UserAccount = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
        gmail_connected=_is_gmail_connected(current_user),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_auth_account_endpoints.py -k "gmail_connected or get_me" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/v1/endpoints/auth.py backend/src/models/oauth_identity.py backend/tests/test_auth_account_endpoints.py
git commit -m "feat(auth): expose gmail connection status"
```

### Task 3: Add `/auth/google/connect-gmail` and split callback behavior

**Files:**
- Modify: `backend/src/api/v1/endpoints/auth.py`
- Test: `backend/tests/test_auth_endpoints.py`

- [ ] **Step 1: Write the failing tests for the Gmail connect endpoint and callback routing**

```python
def test_google_connect_gmail_returns_302_to_google_with_gmail_scope(client):
    resp = client.get("/api/v1/auth/google/connect-gmail?redirect=/outreach")
    assert resp.status_code == 302
    assert "gmail.send" in resp.headers["location"]


def test_google_callback_denied_connect_gmail_returns_to_outreach(client):
    state = _build_valid_state("/outreach", flow="connect_gmail")
    resp = client.get(
        f"/api/v1/auth/google/callback?error=access_denied&state={state}",
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "/outreach" in resp.headers["location"]
    assert "error=gmail_consent_denied" in resp.headers["location"]


def test_google_callback_success_connect_gmail_returns_to_outreach(client):
    state = _build_valid_state("/outreach", flow="connect_gmail")
    with (
        patch("src.api.v1.endpoints.auth.google_oauth.exchange_code_for_tokens", new_callable=AsyncMock, return_value={"id_token": "fake", "refresh_token": "refresh"}),
        patch("src.api.v1.endpoints.auth.google_oauth.verify_id_token", return_value={"sub": "google-sub-999", "email": "test@example.com", "email_verified": True, "name": "Test User"}),
        patch("src.api.v1.endpoints.auth.google_oauth.upsert_user_from_google", return_value=fake_user),
    ):
        resp = client.get(f"/api/v1/auth/google/callback?code=real_code&state={state}")

    assert resp.status_code == 302
    assert "/outreach" in resp.headers["location"]
    assert "gmail_connected=1" in resp.headers["location"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_auth_endpoints.py -k "connect_gmail or callback" -v`
Expected: FAIL because `/google/connect-gmail` does not exist and callback always redirects through `/auth/callback`.

- [ ] **Step 3: Write the minimal implementation**

```python
@router.get("/google/connect-gmail")
def google_connect_gmail(
    redirect: str = "/outreach",
    current_user: UserAccount = Depends(get_current_user),
) -> RedirectResponse:
    frontend = settings.FRONTEND_BASE_URL.rstrip("/")
    if not redirect.startswith("/") or redirect.startswith("//"):
        redirect = "/outreach"
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        return RedirectResponse(f"{frontend}{redirect}?error=oauth_not_configured", status_code=302)
    url, _ = google_oauth.build_authorize_url(redirect, flow="connect_gmail")
    return RedirectResponse(url, status_code=302)


state_payload = google_oauth.verify_state(state)
redirect_path = state_payload["redirect"]
flow = state_payload["flow"]

if error:
    if flow == "connect_gmail":
        return RedirectResponse(f"{frontend}{redirect_path}?error=gmail_consent_denied", status_code=302)
    return RedirectResponse(f"{frontend}/login?error={error}", status_code=302)

if flow == "connect_gmail":
    qs = urlencode({"gmail_connected": "1"})
    return RedirectResponse(f"{frontend}{redirect_path}?{qs}", status_code=302)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_auth_endpoints.py -k "connect_gmail or callback" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/v1/endpoints/auth.py backend/tests/test_auth_endpoints.py
git commit -m "feat(auth): add gmail consent oauth flow"
```

### Task 4: Make the frontend auth API aware of Gmail connect

**Files:**
- Modify: `frontend/src/api/endpoints/auth.ts`
- Modify: `frontend/src/lib/auth.ts`

- [ ] **Step 1: Write the failing type-level and usage expectations**

```ts
export interface UserProfile {
  id: string;
  email: string;
  display_name: string;
  gmail_connected: boolean;
}

authApi.getGoogleConnectGmailUrl("/outreach");
```

- [ ] **Step 2: Run the frontend typecheck to verify it fails**

Run: `npm run build`
Expected: FAIL in places that still assume `UserProfile` has no `gmail_connected`, and because `getGoogleConnectGmailUrl` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```ts
export interface UserProfile {
  id: string;
  email: string;
  display_name: string;
  gmail_connected: boolean;
}

getGoogleConnectGmailUrl(redirect: string = "/outreach"): string {
  const base = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
  const qs = new URLSearchParams({ redirect });
  return `${base}/auth/google/connect-gmail?${qs.toString()}`;
}
```

- [ ] **Step 4: Run the frontend typecheck to verify it passes**

Run: `npm run build`
Expected: PASS or at least no type errors related to auth API changes

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/endpoints/auth.ts frontend/src/lib/auth.ts
git commit -m "feat(frontend): add gmail consent auth api helpers"
```

### Task 5: Update auth callback handling for Gmail consent redirects

**Files:**
- Modify: `frontend/src/routes/auth-callback.tsx`

- [ ] **Step 1: Write the failing behavior expectations for callback branching**

```ts
// Pseudocode for test behavior
// - token+redirect => store token and navigate as before
// - error=gmail_consent_denied with redirect=/outreach => navigate to /outreach?error=gmail_consent_denied
// - gmail_connected=1 with redirect=/outreach => refresh profile and navigate to /outreach?gmail_connected=1
```

- [ ] **Step 2: Run the relevant frontend tests or build to verify the current behavior is wrong**

Run: `npm run build`
Expected: Current route still treats every callback error as login failure.

- [ ] **Step 3: Write the minimal implementation**

```ts
useEffect(() => {
  const token = sp.get("token");
  const redirect = sp.get("redirect") ?? "/dashboard";
  const error = sp.get("error");
  const gmailConnected = sp.get("gmail_connected");

  if (error === "gmail_consent_denied") {
    navigate(`${redirect}?error=${error}`, { replace: true });
    return;
  }

  if (gmailConnected === "1" && api.auth.isAuthenticated()) {
    api.auth.me().then((user) => {
      useAuthStore.getState().setUser(user);
      navigate(`${redirect}?gmail_connected=1`, { replace: true });
    });
    return;
  }

  if (error || !token) {
    toast.error("Google sign-in failed. Please try again.");
    navigate("/login", { replace: true });
    return;
  }
```

- [ ] **Step 4: Run the frontend checks to verify it passes**

Run: `npm run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/routes/auth-callback.tsx
git commit -m "feat(frontend): branch auth callback for gmail consent"
```

### Task 6: Gate the Outreach route with onboarding

**Files:**
- Modify: `frontend/src/routes/outreach.tsx`

- [ ] **Step 1: Write the failing onboarding expectations**

```ts
// Behavior to implement:
// - when current user has gmail_connected === false, show onboarding card instead of message detail/send UI
// - CTA opens api.auth.getGoogleConnectGmailUrl("/outreach")
// - when search param error=gmail_consent_denied exists, show a warning toast or inline message
// - when gmail_connected === true, render the current route unchanged
```

- [ ] **Step 2: Run frontend checks to verify the current route has no gating**

Run: `npm run build`
Expected: PASS, but manual inspection confirms the route always renders the mail UI. This is the expected pre-change baseline.

- [ ] **Step 3: Write the minimal implementation**

```tsx
function GmailConnectOnboarding() {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-lg rounded-[var(--radius-xl)] border border-[color:var(--hairline)] bg-bg-card p-6">
        <p className="font-display text-xl text-fg mb-2">Send email from your Google account</p>
        <p className="text-sm text-fg-muted mb-4">
          Connect Gmail only when you are ready to send outreach. The app asks only for permission to send email on your behalf.
        </p>
        <a href={api.auth.getGoogleConnectGmailUrl("/outreach")}>
          <Button variant="primary">Connect Google to send email</Button>
        </a>
      </div>
    </div>
  );
}

const currentUser = useAuthStore((state) => state.user);
const gmailConnected = currentUser?.gmail_connected ?? false;

if (!gmailConnected) {
  return <GmailConnectOnboarding />;
}
```

- [ ] **Step 4: Run the frontend checks to verify it passes**

Run: `npm run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/routes/outreach.tsx
git commit -m "feat(outreach): gate sending ui behind gmail consent"
```

### Task 7: Show denial feedback and refresh profile after connect

**Files:**
- Modify: `frontend/src/routes/outreach.tsx`
- Modify: `frontend/src/lib/auth.ts`

- [ ] **Step 1: Write the failing reconnect feedback expectations**

```ts
// Behavior:
// - /outreach?error=gmail_consent_denied shows a non-fatal message
// - /outreach?gmail_connected=1 refreshes user state if needed and removes the temporary query params
```

- [ ] **Step 2: Run frontend checks to verify this behavior is missing**

Run: `npm run build`
Expected: PASS, but the route still lacks denial and success feedback handling.

- [ ] **Step 3: Write the minimal implementation**

```tsx
const [params, setParams] = useSearchParams();

useEffect(() => {
  if (params.get("error") === "gmail_consent_denied") {
    toast.error("You have not granted Gmail send permission yet.");
  }
  if (params.get("gmail_connected") === "1") {
    toast.success("Google Mail has been connected.");
  }
}, [params]);

useEffect(() => {
  if (!params.get("error") && !params.get("gmail_connected")) {
    return;
  }
  setParams((prev) => {
    const next = new URLSearchParams(prev);
    next.delete("error");
    next.delete("gmail_connected");
    return next;
  });
}, [params, setParams]);
```

- [ ] **Step 4: Run the frontend checks to verify it passes**

Run: `npm run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/routes/outreach.tsx frontend/src/lib/auth.ts
git commit -m "feat(outreach): handle gmail consent success and denial states"
```

### Task 8: Protect backend mail send surfaces with capability-aware errors

**Files:**
- Modify: `backend/src/api/v1/endpoints/outreach.py`
- Modify: `backend/worker/tasks.py`
- Test: `backend/tests/test_gmail_service.py`
- Test: `backend/tests/test_outreach_endpoints.py`

- [ ] **Step 1: Write the failing tests for missing Gmail capability**

```python
def test_send_outreach_returns_clear_error_when_google_not_connected(client, auth_header, seeded_data):
    response = client.post(
        f"/api/v1/outreach/{seeded_data.message_id}/send",
        headers=auth_header(seeded_data.owner),
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "gmail_not_connected"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_outreach_endpoints.py -k "gmail_not_connected or send" -v`
Expected: FAIL because the endpoint does not yet surface a distinct reconnect error.

- [ ] **Step 3: Write the minimal implementation**

```python
identity = db.execute(
    select(OAuthIdentity).where(
        OAuthIdentity.user_id == current_user.id,
        OAuthIdentity.provider == "google",
    )
).scalar_one_or_none()

if identity is None or not identity.refresh_token_encrypted or not identity.scope or "https://www.googleapis.com/auth/gmail.send" not in identity.scope:
    raise HTTPException(status_code=409, detail="gmail_not_connected")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_outreach_endpoints.py -k "gmail_not_connected or send" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/v1/endpoints/outreach.py backend/worker/tasks.py backend/tests/test_outreach_endpoints.py backend/tests/test_gmail_service.py
git commit -m "feat(mail): return reconnectable gmail capability errors"
```

### Task 9: Run full verification and update docs if behavior changed

**Files:**
- Modify: `docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md`
- Modify: `docs/GOOGLE_OAUTH_GCP_SETUP.md`
- Modify: `docs/BACKEND.md`

- [ ] **Step 1: Add doc notes describing progressive consent**

```md
- Google login now requests only `openid email profile`.
- Gmail send permission is requested later from the Outreach flow.
- Users who deny Gmail consent return to the same mail tab and can reconnect later.
```

- [ ] **Step 2: Run backend verification**

Run: `pytest backend/tests/test_google_oauth_service.py backend/tests/test_auth_endpoints.py backend/tests/test_auth_account_endpoints.py backend/tests/test_outreach_endpoints.py -v`
Expected: PASS

- [ ] **Step 3: Run frontend verification**

Run: `npm run build`
Expected: PASS

- [ ] **Step 4: Manual smoke test**

Run:

```bash
docker compose up --build
```

Expected:
- Login with Google does not show Gmail permission
- Open `/outreach` and see onboarding when `gmail_connected=false`
- Click `Connect Google to send email` and see Gmail consent
- Deny consent and return to `/outreach` with onboarding still visible
- Approve consent and return to `/outreach` with normal sending UI

- [ ] **Step 5: Commit**

```bash
git add docs/GOOGLE_OAUTH_GMAIL_API_SETUP.md docs/GOOGLE_OAUTH_GCP_SETUP.md docs/BACKEND.md
git commit -m "docs(auth): document progressive gmail consent flow"
```
