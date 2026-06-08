# Hướng dẫn cấu hình Google OAuth + Gmail API để gửi email thật

> **Đối tượng:** Người quản trị dự án hoặc người vận hành local/dev.
>
> **Mục tiêu:** Cho phép app gửi email cho ứng viên từ chính Gmail của recruiter đang đăng nhập.
>
> **Kết quả cuối cùng:** Recruiter đăng nhập Google, cấp quyền `gmail.send`, backend lưu refresh token đã mã hóa, worker gửi email qua Gmail API.

---

## Khi nào cần dùng hướng dẫn này

Repo hiện đã có Google OAuth cho đăng nhập. Hướng dẫn này là phần mở rộng để xin thêm quyền gửi mail qua Gmail API.

Luồng hiện tại dùng progressive consent:

- Đăng nhập Google chỉ xin scope cơ bản `openid email profile`.
- Quyền `gmail.send` chỉ được xin sau, từ màn hình Outreach khi recruiter thật sự muốn gửi mail.
- Nếu user từ chối Gmail consent, app quay lại đúng màn hình Outreach để có thể thử lại sau.

Bạn cần làm hướng dẫn này nếu muốn:

- Gửi link phỏng vấn cho ứng viên từ Gmail thật của recruiter.
- Gửi outreach email từ màn hình Outreach thay vì chỉ lưu nháp hoặc đánh dấu thủ công.
- Tránh dùng SMTP password hoặc app password trong server.

Nếu bạn chỉ muốn đăng nhập Google, dùng `docs/GOOGLE_OAUTH_GCP_SETUP.md` là đủ.

---

## Quyết định kỹ thuật

Sử dụng Gmail API với scope tối thiểu:

```text
https://www.googleapis.com/auth/gmail.send
```

Không dùng scope rộng `https://mail.google.com/` vì app chỉ cần gửi email. Google cũng khuyến nghị dùng scope tối thiểu khi chỉ cần gửi thư.

Luồng OAuth cần `offline` access để backend nhận `refresh_token`. Refresh token cho phép Celery worker gửi email sau khi request web đã kết thúc.

---

## Bước 1: Bật Gmail API trong Google Cloud

1. Mở [Google Cloud Console](https://console.cloud.google.com/).
2. Chọn đúng project đang dùng cho OAuth login hiện tại.
3. Vào **APIs & Services** -> **Library**.
4. Tìm `Gmail API`.
5. Chọn **Gmail API** -> **Enable**.

Nếu chưa tạo OAuth Client ID cho repo, làm trước theo `docs/GOOGLE_OAUTH_GCP_SETUP.md`.

---

## Bước 2: Thêm Gmail send scope vào OAuth consent screen

1. Vào **APIs & Services** -> **OAuth consent screen**.
2. Mở phần **Data Access** hoặc **Scopes** tùy giao diện Google Cloud hiện tại.
3. Chọn **Add or remove scopes**.
4. Thêm scope:

```text
https://www.googleapis.com/auth/gmail.send
```

5. Lưu thay đổi.

Lưu ý quan trọng:

- `gmail.send` là sensitive scope. App ở trạng thái Testing vẫn dùng được với test users.
- Nếu publish app cho người dùng ngoài test users, Google có thể yêu cầu OAuth app verification.
- Chỉ request đúng scope app cần. Không thêm Gmail read/modify/full mail scopes nếu không dùng.

---

## Bước 3: Thêm test users

Khi OAuth app ở trạng thái Testing:

1. Vào **OAuth consent screen**.
2. Mở phần **Test users**.
3. Thêm Gmail của recruiter sẽ dùng để gửi email.
4. Thêm Gmail của người test nếu cần.

Nếu email không nằm trong test users, Google có thể chặn consent với lỗi `access_denied`.

---

## Bước 4: Cập nhật Authorized redirect URI

Trong **APIs & Services** -> **Credentials** -> OAuth Client ID đang dùng:

Authorized JavaScript origins cho local dev:

```text
http://localhost:5173
http://localhost:8000
```

Authorized redirect URI cho local dev:

```text
http://localhost:8000/api/v1/auth/google/callback
```

URI này phải khớp tuyệt đối với `GOOGLE_REDIRECT_URI` trong `.env`: đúng scheme, host, port, path, không thêm trailing slash.

---

## Bước 5: Tạo token encryption key

Backend cần mã hóa Google access token và refresh token trước khi lưu DB.

Sau khi implementation plan được thực thi, chạy lệnh này từ project root hoặc trong backend venv:

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Kết quả sẽ giống dạng:

```text
QImj5xapN7XQv2JZyT4E4M9bFLVwTyJ8z4W7OZq9hys=
```

Giữ giá trị này bí mật như password. Nếu mất key này, các token đã mã hóa trong DB sẽ không giải mã được.

---

## Bước 6: Cấu hình `.env` ở root repo

Repo hiện dùng root `.env` qua `docker-compose.yml`, không phải `backend/.env`.

Thêm hoặc cập nhật:

```dotenv
# Google OAuth login
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/google/callback
FRONTEND_BASE_URL=http://localhost:5173

# Google OAuth + Gmail API
# Login ban đầu chỉ dùng scope cơ bản; Gmail send được xin sau từ Outreach.
GOOGLE_OAUTH_SCOPES=openid email profile https://www.googleapis.com/auth/gmail.send
GOOGLE_OAUTH_ACCESS_TYPE=offline
GOOGLE_OAUTH_PROMPT=consent
GOOGLE_TOKEN_ENCRYPTION_KEY=your-fernet-key-from-step-5
GMAIL_SEND_ENABLED=true
GMAIL_SEND_TIMEOUT_SECONDS=20
```

Giải thích nhanh:

- `GOOGLE_OAUTH_SCOPES`: thêm quyền gửi mail bên cạnh login cơ bản.
- `GOOGLE_OAUTH_ACCESS_TYPE=offline`: yêu cầu refresh token.
- `GOOGLE_OAUTH_PROMPT=consent`: buộc hiện consent để Google trả refresh token, nhất là khi user đã từng đăng nhập trước đó.
- `GOOGLE_TOKEN_ENCRYPTION_KEY`: key mã hóa token trong DB.
- `GMAIL_SEND_ENABLED=true`: bật gửi thật. Để `false` nếu chỉ muốn test code mà không gửi email.

Không commit `.env`.

---

## Bước 7: Restart stack

Chạy:

```powershell
docker compose up --build
```

Nếu đã chạy sẵn:

```powershell
docker compose down
docker compose up --build
```

Sau khi migration mới có trong code, backend startup sẽ chạy Alembic theo command hiện có trong `docker-compose.yml`.

---

## Bước 8: Reconnect Google account

Recruiter cần đăng nhập Google lại sau khi scope Gmail được thêm.

Quy trình:

1. Logout khỏi app.
2. Mở `http://localhost:5173/login`.
3. Chọn Sign in with Google.
4. Consent screen phải hiển thị quyền gửi email.
5. Chọn Allow.
6. App redirect về dashboard.

Nếu Google không hiện consent screen hoặc backend không nhận refresh token, thử:

1. Vào [Google Account Third-party access](https://myaccount.google.com/connections).
2. Gỡ quyền của app hiện tại.
3. Đăng nhập lại qua app.

---

## Bước 9: Smoke test gửi email

Chuẩn bị một candidate có email test thật, ví dụ email phụ của bạn.

Test interview invitation:

1. Mở job workspace.
2. Chọn candidate có email.
3. Tạo hoặc chọn interview template.
4. Bấm gửi interview invitation.
5. Kiểm tra worker logs:

```powershell
docker compose logs -f worker
```

6. Kiểm tra hộp thư ứng viên nhận email từ Gmail recruiter.

Test Outreach:

1. Mở `http://localhost:5173/outreach`.
2. Nếu account chưa nối Gmail send, app hiển thị onboarding thay vì mail UI.
3. Bấm `Connect Gmail`, chấp nhận consent, rồi quay lại đúng route Outreach.
4. Tạo outreach draft cho candidate có email.
5. Mở draft và bấm `Send email`.
6. Kiểm tra status chuyển sang `sent`.
7. Kiểm tra inbox của candidate.

---

## Troubleshooting

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `redirect_uri_mismatch` | Redirect URI trong GCP khác `.env` | So sánh từng ký tự với `http://localhost:8000/api/v1/auth/google/callback`. |
| Google không trả `refresh_token` | User đã consent trước đó hoặc thiếu `access_type=offline` | Đặt `GOOGLE_OAUTH_PROMPT=consent`, gỡ app trong Google Account, login lại. |
| `access_denied` khi consent | User chưa nằm trong Test users | Thêm Gmail đó vào OAuth consent screen test users. |
| Worker báo Gmail sending disabled | `GMAIL_SEND_ENABLED=false` hoặc env chưa vào worker | Set `GMAIL_SEND_ENABLED=true`, restart `docker compose up --build`. |
| Worker báo missing refresh token | Account Google được link trước khi xin Gmail scope | Reconnect Google account theo Bước 8. |
| Worker báo decrypt token failed | Sai hoặc đổi `GOOGLE_TOKEN_ENCRYPTION_KEY` sau khi đã lưu token | Khôi phục key cũ hoặc xóa token cũ và reconnect Google. |
| Candidate không nhận email | Email ứng viên sai, spam folder, hoặc Gmail API lỗi | Kiểm tra candidate email, worker logs, Gmail Sent folder của recruiter. |
| App cảnh báo unverified | App request sensitive scope nhưng chưa verify | Dev/testing vẫn dùng với test users; production cần OAuth verification. |

---

## Production notes

Trước khi dùng production:

- Dùng HTTPS cho frontend và backend.
- Thêm production origins và redirect URIs trong OAuth Client ID.
- Cấu hình OAuth consent screen với domain thật, privacy policy, terms nếu publish.
- Xem xét Google OAuth verification vì `gmail.send` là sensitive scope.
- Lưu `GOOGLE_TOKEN_ENCRYPTION_KEY` trong secret manager, không lưu trong file thường.
- Giới hạn log: không log access token, refresh token, client secret, app JWT, hoặc nội dung email nhạy cảm.
- Có quy trình revoke: recruiter có thể gỡ quyền app trong Google Account và admin có thể xóa token trong DB.

---

## Tài liệu tham khảo chính thức

- [Gmail API server-side authorization](https://developers.google.com/workspace/gmail/api/auth/web-server)
- [Google OAuth 2.0 for Web Server Applications](https://developers.google.com/identity/protocols/oauth2/web-server)
- [Gmail API scopes](https://developers.google.com/workspace/gmail/api/auth/scopes)
- [Gmail users.messages.send](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/send)
- [Google OAuth app verification](https://support.google.com/cloud/answer/13463073)
- [Requesting minimum OAuth scopes](https://support.google.com/cloud/answer/13807380)
