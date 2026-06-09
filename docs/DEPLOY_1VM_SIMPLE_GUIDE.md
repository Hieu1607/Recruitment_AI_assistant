# Hướng dẫn deploy đồ án trên 1 VM

> **Đối tượng:** Người mới deploy lần đầu, dùng 1 VM duy nhất.
>
> **Mục tiêu:** Chạy app trên 1 VM theo cách đơn giản, dễ debug:
> - `nginx` cài native trên VM để cầm `80/443` và HTTPS
> - `Docker Compose` chạy backend, worker, database, redis, minio
> - frontend build ra file tĩnh rồi để `nginx` serve
>
> **Vì sao chọn cách này:** Ít lớp hơn, dễ hiểu hơn `nginx` trong Docker, và dễ gắn HTTPS bằng `certbot`.

---

## Kiến trúc cuối cùng

```text
Browser
  |
  | HTTPS (443)
  v
nginx trên VM
  |-- /        -> frontend/dist
  |
  |-- /api/    -> http://127.0.0.1:8000
  |
  |-- /docs    -> http://127.0.0.1:8000/docs
  v
Docker Compose
  |- backend
  |- worker
  |- db
  |- redis
  |- minio
```

Ý chính:

- `nginx` là cửa chính cho user.
- Backend không cần public trực tiếp ra internet.
- Frontend là file tĩnh nên để `nginx` serve là đủ.

---

## Khi nào nên dùng hướng dẫn này

Hãy dùng guide này nếu bạn có các điều kiện sau:

- Chỉ deploy lên **1 VM**.
- Muốn setup **dễ làm, dễ demo, dễ sửa**.
- Chưa muốn dùng Load Balancer hoặc hạ tầng nhiều máy.

Hướng dẫn này **không cố tối ưu cho production nhiều VM**. Với production thật, nên dùng Load Balancer và SSL managed ở lớp ngoài.

---

## Tổng checklist

- [ ] Tạo VM Ubuntu trên GCP
- [ ] Trỏ domain về IP của VM
- [ ] Mở firewall `22`, `80`, `443`
- [ ] Cài Docker, Docker Compose plugin, nginx, certbot
- [ ] Clone repo và chuẩn bị file `.env`
- [ ] Chỉnh env production cho domain thật
- [ ] Build frontend ra `frontend/dist`
- [ ] Chạy Docker Compose cho backend stack
- [ ] Cấu hình nginx serve frontend và proxy `/api`
- [ ] Bật HTTPS bằng certbot
- [ ] Smoke test login, API, OAuth nếu có dùng Google

---

## Bước 1 - Tạo VM

Tạo một VM Ubuntu trên GCP, ví dụ:

- OS: `Ubuntu 22.04 LTS`
- Machine type: `e2-standard-2` hoặc tương đương
- Disk: `20GB+`

Giải thích ngắn:

- Ubuntu là lựa chọn dễ tìm tài liệu.
- 1 VM là đủ cho đồ án.
- Không cần chia nhiều máy ở giai đoạn này.

---

## Bước 2 - Trỏ domain về VM

Trong DNS của domain, tạo bản ghi:

- `A` record cho domain chính, ví dụ `recruitai.yourdomain.com`
- trỏ về **public IP** của VM

Giải thích ngắn:

- HTTPS cert chỉ cấp được khi domain thật sự trỏ về VM.
- Nếu chưa có domain, bạn vẫn có thể chạy HTTP để test trước, rồi thêm HTTPS sau.

---

## Bước 3 - Mở firewall

Chỉ nên mở các cổng sau từ internet:

- `22` cho SSH
- `80` cho HTTP
- `443` cho HTTPS

Không nên mở công khai:

- `5432` Postgres
- `6379` Redis
- `9000`, `9001` MinIO
- `8000` backend

Giải thích ngắn:

- User chỉ nên đi qua `nginx`.
- Các service nội bộ cứ để container giao tiếp với nhau là đủ.

---

## Bước 4 - SSH vào VM và cài package cần thiết

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin nginx certbot python3-certbot-nginx git
sudo systemctl enable --now docker
sudo systemctl enable --now nginx
sudo usermod -aG docker $USER
```

Sau đó đăng xuất SSH rồi vào lại để group `docker` có hiệu lực.

Giải thích ngắn:

- `docker.io` và Compose plugin để chạy stack app.
- `nginx` để serve frontend và reverse proxy API.
- `certbot` để lấy và gia hạn chứng chỉ HTTPS.

---

## Bước 5 - Clone repo

Ví dụ đặt code ở `/opt/recruitai`:

```bash
sudo mkdir -p /opt
sudo chown $USER:$USER /opt
cd /opt
git clone <YOUR_REPO_URL> recruitai
cd /opt/recruitai
```

Giải thích ngắn:

- Đặt app ở một đường dẫn cố định sẽ dễ cấu hình `nginx` hơn.

---

## Bước 6 - Chuẩn bị file `.env`

Nếu chưa có `.env`, tạo từ `.env.example` hoặc tự tạo theo giá trị bạn đang dùng.

Các biến production quan trọng cần chỉnh:

```env
FRONTEND_BASE_URL=https://recruitai.yourdomain.com
GOOGLE_REDIRECT_URI=https://recruitai.yourdomain.com/api/v1/auth/google/callback
BACKEND_CORS_ORIGINS=["https://recruitai.yourdomain.com"]
```

Với frontend, khi build production nên dùng:

```env
VITE_API_BASE_URL=/api/v1
```

Giải thích ngắn:

- `FRONTEND_BASE_URL` để backend tạo link quay về frontend đúng domain thật.
- `GOOGLE_REDIRECT_URI` phải khớp tuyệt đối nếu bạn dùng Google OAuth.
- `BACKEND_CORS_ORIGINS` cần đúng domain frontend thật.
- `VITE_API_BASE_URL=/api/v1` giúp frontend gọi API cùng domain, không cần hard-code host khác.

---

## Bước 7 - Build frontend ra file tĩnh

Repo hiện có service `frontend` trong `docker-compose.yml`. Bạn có thể tận dụng chính container này để build ra thư mục `dist` trên host:

```bash
cd /opt/recruitai
docker compose run --rm -e VITE_API_BASE_URL=/api/v1 frontend npm run build
```

Sau khi chạy xong, kiểm tra:

```bash
ls /opt/recruitai/frontend/dist
```

Giải thích ngắn:

- Ta chỉ dùng container frontend như một môi trường build.
- Sau khi build xong, **không cần chạy frontend container** ở mode production của hướng dẫn này.
- `nginx` native sẽ serve trực tiếp thư mục `frontend/dist`.

---

## Bước 8 - Chạy backend stack bằng Docker Compose

Ở hướng dẫn này, bạn chỉ cần chạy các service phía sau:

```bash
cd /opt/recruitai
docker compose up -d db redis minio backend worker
```

Kiểm tra container:

```bash
docker compose ps
```

Xem log backend:

```bash
docker compose logs -f backend
```

Giải thích ngắn:

- Frontend đã thành file tĩnh nên không cần `docker compose up frontend`.
- Backend vẫn nên chạy bằng Docker để giữ môi trường giống local/dev của repo.

---

## Bước 9 - Tạo cấu hình nginx

Tạo file:

`/etc/nginx/sites-available/recruitai`

với nội dung:

```nginx
server {
    listen 80;
    server_name recruitai.yourdomain.com;

    root /opt/recruitai/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/recruitai /etc/nginx/sites-enabled/recruitai
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

Giải thích ngắn:

- `root` trỏ tới frontend đã build.
- `try_files ... /index.html` là phần quan trọng để SPA route hoạt động.
- `/api/` được chuyển tiếp vào backend container đang publish ở `127.0.0.1:8000`.

---

## Bước 10 - Bật HTTPS bằng certbot

Sau khi truy cập HTTP được rồi, chạy:

```bash
sudo certbot --nginx -d recruitai.yourdomain.com
```

Chọn:

- nhập email
- đồng ý điều khoản
- chọn redirect HTTP sang HTTPS nếu được hỏi

Kiểm tra gia hạn:

```bash
sudo certbot renew --dry-run
```

Giải thích ngắn:

- Certbot sẽ lấy cert và tự sửa config `nginx`.
- `renew --dry-run` để chắc rằng cơ chế gia hạn hoạt động.

---

## Bước 11 - Smoke test

Kiểm tra các URL:

- `https://recruitai.yourdomain.com/`
- `https://recruitai.yourdomain.com/docs`
- `https://recruitai.yourdomain.com/api/v1/auth/me`

Lưu ý:

- `auth/me` nếu chưa login có thể trả `401`, như vậy vẫn là backend đang chạy đúng.

Nếu dùng Google OAuth, kiểm tra lại trong Google Cloud:

- Authorized JavaScript origins
- Authorized redirect URIs

Ví dụ production:

```text
https://recruitai.yourdomain.com
https://recruitai.yourdomain.com/api/v1/auth/google/callback
```

Giải thích ngắn:

- Phần dễ sai nhất khi lên domain thật là OAuth redirect URI và CORS.

---

## Cách cập nhật khi sửa code

### Nếu sửa frontend

Build lại:

```bash
cd /opt/recruitai
docker compose run --rm -e VITE_API_BASE_URL=/api/v1 frontend npm run build
sudo systemctl reload nginx
```

### Nếu sửa backend

Build và chạy lại backend stack:

```bash
cd /opt/recruitai
docker compose up -d --build backend worker
```

Giải thích ngắn:

- Frontend là file tĩnh, nên chỉ cần build lại `dist`.
- Backend là container, nên rebuild container khi code hoặc dependency đổi.

---

## Vì sao guide này không dùng nginx container

Vì mục tiêu ở đây là **đơn giản cho đồ án 1 VM**:

- `nginx` native dễ cài cert hơn
- `certbot` hoạt động tự nhiên hơn trên host
- ít lớp proxy hơn nên dễ debug hơn

`nginx` container không sai. Nó chỉ không phải lựa chọn dễ nhất cho người mới trong bối cảnh này.

---

## Lỗi hay gặp

### 1. Domain mở không ra HTTPS

Nguyên nhân thường gặp:

- DNS chưa trỏ đúng IP VM
- chưa mở firewall `80/443`
- `certbot` chưa chạy thành công

### 2. Frontend mở được nhưng gọi API lỗi

Kiểm tra:

- backend container có chạy không
- `location /api/` trong `nginx` có đúng không
- `VITE_API_BASE_URL` lúc build có phải `/api/v1` không

### 3. Google login báo `redirect_uri_mismatch`

Kiểm tra:

- `GOOGLE_REDIRECT_URI` trong `.env`
- redirect URI trong Google Cloud Console

Hai giá trị này phải khớp **từng ký tự**.

### 4. Route như `/dashboard` mở trực tiếp bị 404

Nguyên nhân:

- thiếu `try_files $uri $uri/ /index.html;`

Đây là cấu hình bắt buộc với SPA dùng client-side routing.

---

## Kết luận

Với đồ án 1 VM, kiến trúc dễ làm và hợp lý nhất là:

- `nginx` native trên VM cầm `80/443` và HTTPS
- `Docker Compose` chạy backend stack
- frontend build ra file tĩnh và để `nginx` serve

Hướng này đủ sạch để demo, đủ đơn giản để sửa lỗi nhanh, và chưa cần thêm Load Balancer hay `nginx` container.
