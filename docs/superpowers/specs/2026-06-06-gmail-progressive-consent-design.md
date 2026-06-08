# Gmail Progressive Consent Design

## Goal

Keep Google sign-in low-friction by requesting only basic identity scopes during login, then request Gmail send permission only when the recruiter enters the email-sending area and explicitly chooses to connect Gmail.

## Problem

The current OAuth login flow always requests:

- `openid`
- `email`
- `profile`
- `https://www.googleapis.com/auth/gmail.send`

This causes Gmail send permission to appear during first login, which creates unnecessary trust friction for users who only want to sign in and explore the product. It also couples authentication with an optional feature capability.

## Decision

Split Google OAuth into two user-facing flows:

1. `Login flow`
   Request only `openid email profile`.

2. `Gmail connect flow`
   Request `https://www.googleapis.com/auth/gmail.send` only after the recruiter enters a mail-sending area and clicks a clear consent CTA.

The first mail-related visit will show an onboarding state instead of the normal sending UI. If the recruiter grants permission, future visits go directly to the normal UI. If the recruiter denies permission, the app returns to the same tab and keeps showing onboarding plus a non-fatal message explaining that Gmail permission was not granted.

## User Experience

### Login

- User clicks `Sign in with Google`.
- App requests only basic Google identity scopes.
- User lands in the app without any Gmail send permission prompt.

### First visit to mail area

- When the recruiter opens the mail-sending tab for the first time, the app checks Gmail capability status.
- If Gmail is not connected, the normal mail UI is replaced with a lightweight onboarding card.
- The onboarding card explains that the app can send emails from the recruiter's own Google account and that Gmail permission is required for this feature.
- The card contains one primary CTA: `Connect Google to send email`.

### Gmail connect

- Clicking the CTA starts a separate OAuth flow for Gmail send permission.
- After success, the recruiter returns to the same tab and sees the normal sending UI.
- On future visits, onboarding is skipped.

### Denied consent

- If the recruiter denies Google consent, redirect back to the same tab.
- Show onboarding again.
- Show a dismissible message such as `You have not granted Gmail send permission yet.`
- Treat denial as a product state, not a system error.

## Backend Design

### OAuth endpoints

Keep the current login endpoint but narrow its purpose:

- `GET /api/v1/auth/google/login`
  - Purpose: authentication only
  - Scopes: `openid email profile`

Add a new endpoint:

- `GET /api/v1/auth/google/connect-gmail`
  - Purpose: grant Gmail sending capability to an already signed-in recruiter
  - Scopes: `openid email profile https://www.googleapis.com/auth/gmail.send`
  - Parameters should preserve the frontend return location, for example the current tab route.

The callback must distinguish which flow initiated the request:

- `login`
- `connect_gmail`

This flow type should be encoded in signed OAuth state so callback handling remains stateless and tamper-resistant.

### OAuth service changes

The OAuth helper should no longer depend on one global scope string for every use case. Instead it should support flow-specific authorize URL creation:

- basic login scope set
- Gmail connect scope set

The state payload should include:

- frontend redirect path
- flow type
- nonce

### Token persistence and capability detection

The existing `oauth_identities` record remains the source of truth.

Gmail capability is considered connected only when the Google identity has:

- a stored refresh token
- stored scope containing `gmail.send`

This avoids incorrectly treating any Google login as Gmail-ready.

### Callback behavior

For `login`:

- upsert or link user as today
- create app JWT
- redirect to standard frontend auth callback

For `connect_gmail`:

- require an authenticated app user context or otherwise safely associate the Google identity with the correct existing user
- update token fields and scope on the existing Google identity
- redirect back to the mail tab route with a success flag

For consent denial:

- redirect back to the initiating mail tab route with a non-fatal error code such as `gmail_consent_denied`

### API surface for frontend capability checks

Frontend needs a simple way to know whether Gmail is connected. Expose this via either:

- an added field on the existing auth/profile response, recommended
- or a dedicated mail capability endpoint

Recommended field:

- `gmail_connected: boolean`

Optional future-friendly fields:

- `google_connected: boolean`
- `gmail_scope_granted: boolean`

For this change, only `gmail_connected` is required.

## Frontend Design

### Mail tab gating

Any screen that requires real Gmail sending should gate on `gmail_connected`.

If `gmail_connected` is `false`:

- render onboarding card
- hide or replace the normal send composer/actions

If `gmail_connected` is `true`:

- render the current mail UI unchanged

### Onboarding card content

The onboarding state should explain:

- email will be sent from the recruiter's Google account
- the app requests permission only for sending mail
- permission can be revoked later from Google account settings

Primary CTA:

- `Connect Google to send email`

Secondary behavior:

- if the user dismisses the message and stays on the tab, keep the onboarding visible because the feature is still unavailable

### Redirect return behavior

The Gmail connect CTA must preserve the current tab location so the callback can send the user back to the exact sending context. This is required for:

- outreach tab
- interview invitation flow
- any future sending surface

## Error Handling

### Expected business states

These are not system failures:

- user has not connected Gmail yet
- user denied Gmail consent
- user revoked Gmail permission later
- stored Gmail token no longer yields a valid access token

All of these should map back to a recoverable reconnect or onboarding path.

### System failures

These remain true failures:

- invalid OAuth state
- token exchange failure
- identity mismatch
- missing Google OAuth configuration

These should continue to use explicit error handling and logs.

## Security Considerations

- Request least privilege during login.
- Request Gmail permission only in feature context.
- Keep signed OAuth state with flow type and redirect path.
- Do not log Google access tokens, refresh tokens, app JWTs, or sensitive mail content.
- Continue storing Google tokens encrypted with `GOOGLE_TOKEN_ENCRYPTION_KEY`.
- Treat Gmail capability as revocable and reconnectable.

## Testing Strategy

### Backend tests

- Login authorize URL does not include `gmail.send`.
- Gmail connect authorize URL includes `gmail.send`.
- OAuth state roundtrip preserves flow type and redirect path.
- Callback handles `login` and `connect_gmail` differently.
- Denied Gmail consent returns user to the initiating tab with non-fatal error.
- Capability detection returns `gmail_connected=false` without refresh token or without `gmail.send` scope.
- Capability detection returns `gmail_connected=true` only when both conditions are satisfied.

### Frontend tests

- Login button still points to the basic login flow.
- Mail tab shows onboarding when `gmail_connected=false`.
- Onboarding CTA starts Gmail connect flow.
- Consent denial returns to onboarding with the correct message.
- Successful consent returns to normal mail UI.

## Rollout Notes

- Existing users who previously connected Google with Gmail scope should continue working.
- Existing users who logged in before this change without Gmail scope should see onboarding the first time they enter a sending area.
- If legacy records contain Google identity but no usable refresh token, treat them as not connected and require reconnect.

## Non-Goals

- Changing the current visual design system beyond the new onboarding state
- Supporting multiple mail providers in this change
- Redesigning the entire auth architecture

## Open Choice Resolved

The approved product behavior is:

- first entry into a mail-sending tab shows onboarding when Gmail is not connected
- clicking the onboarding CTA starts Gmail consent
- denying consent returns to the same tab
- the same onboarding remains visible with a clear explanatory message
