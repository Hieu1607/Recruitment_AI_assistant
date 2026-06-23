# Hướng dẫn deploy chi tiết trên 1 VM bằng địa chỉ IP

> **Mục tiêu của tài liệu này:** giúp bạn đưa app lên **1 VM Ubuntu** theo cách dễ hiểu nhất, dùng **địa chỉ IP trước**. Khi bạn mua domain xong, có thể bổ sung HTTPS sau.
>
> **Kiến trúc được chọn trong guide này:**
> - `nginx` cài **native trên VM**
> - `Docker Compose` chạy `frontend`, `backend`, `worker`, `db`, `redis`, `minio`
> - frontend production được build thành static assets bên trong container và được `nginx` trên VM reverse proxy
>
> **Vì sao chọn cách này:** đây là cách ít lớp nhất cho đồ án 1 VM, dễ debug hơn `nginx` trong Docker, và sau này thêm HTTPS bằng `certbot` cũng dễ hơn.

---

## 1. Bạn sẽ có gì sau khi làm xong

Sau khi hoàn thành guide này, luồng request sẽ là:

```text
Trình duyệt
   |
   | HTTP qua IP của VM, ví dụ http://34.123.45.67
   v
nginx trên VM
   |-- /        -> frontend production container (port 5173)
   |-- /api/    -> backend FastAPI trong Docker
   |-- /docs    -> Swagger docs của backend
   v
Docker Compose
   |- frontend
   |- backend
   |- worker
   |- db
   |- redis
   |- minio
```

Điểm cần hiểu:

- User chỉ truy cập **1 địa chỉ duy nhất**: IP của VM.
- `nginx` là cửa chính.
- Backend vẫn chạy trong Docker, nhưng người dùng không đi thẳng vào backend.
- Frontend không chạy bằng Vite dev server ở production; nó được build ra file tĩnh.

---

## 2. Guide này dành cho ai

Hãy dùng tài liệu này nếu:

- bạn đang làm **đồ án** hoặc demo nhỏ;
- chỉ có **1 VM**;
- muốn deploy trước bằng **IP**;
- muốn hiểu rõ từng bước thay vì copy-paste mù.

Guide này **chưa bật HTTPS**, vì HTTPS chuẩn cần domain. Khi bạn có domain, tôi sẽ bổ sung phần:

- cấu hình DNS;
- cập nhật env từ IP sang domain;
- chạy `certbot`;
- ép redirect từ HTTP sang HTTPS.

---

## 3. Những gì bạn cần chuẩn bị

Trước khi bắt đầu, bạn cần có:

- 1 VM Ubuntu trên GCP
- quyền SSH vào VM
- source code repo này
- biết cách sửa file bằng `nano` hoặc `vim`

Khuyến nghị VM:

- OS: `Ubuntu 22.04 LTS`
- Machine type: `e2-standard-2` hoặc tương đương
- Disk: `20 GB` trở lên

Lý do:

- Ubuntu phổ biến, dễ tìm cách sửa lỗi.
- App có nhiều service, nên máy quá nhỏ sẽ chậm hoặc thiếu RAM.

---

## 4. Biến ví dụ dùng trong tài liệu

Để dễ đọc, tôi sẽ dùng các giá trị ví dụ sau:

- Public IP của VM: `34.123.45.67`
- Đường dẫn app trên VM: `/opt/easyhr`

Khi bạn làm thật, hãy thay:

- `34.123.45.67` bằng IP thật của VM
- repo URL bằng repo thật của bạn

---

## 5. Bước 1 - Tạo VM trên GCP

Trong Google Cloud Console, tạo một VM Ubuntu.

Bạn có thể chọn:

- Name: `easyhr-vm`
- Region: gần bạn, ví dụ Singapore
- Machine type: `e2-standard-2`
- Boot disk: `Ubuntu 22.04 LTS`

### Việc bạn đang làm ở bước này là gì

Bạn đang tạo ra chiếc máy chủ sẽ chạy toàn bộ hệ thống.

### Tại sao phải làm bước này

Vì mọi thứ sau đó, từ Docker đến nginx, đều sẽ cài trên máy này.

---

## 6. Bước 2 - Mở firewall trên GCP

Với hướng IP trước, bạn chỉ cần mở các cổng:

- `22` cho SSH
- `80` cho HTTP

Tạm thời **chưa cần mở `443`**, vì chưa dùng HTTPS.

### Việc bạn đang làm ở bước này là gì

Bạn đang cho phép internet đi vào đúng những cổng cần thiết.

### Tại sao phải làm bước này

Nếu không mở `80`, trình duyệt bên ngoài sẽ không vào được `nginx`.

### Cổng nào không nên mở công khai

Không nên public các cổng sau:

- `5432` Postgres
- `6379` Redis
- `8000` backend
- `9000`, `9001` MinIO

Lý do:

- Đây là các service nội bộ.
- User cuối chỉ nên đi qua `nginx`.

---

## 7. Bước 3 - SSH vào VM

Từ máy của bạn, SSH vào VM:

```bash
ssh <your-username>@34.123.45.67
```

Ví dụ:

```bash
ssh admin@34.123.45.67
```

### Việc bạn đang làm ở bước này là gì

Bạn đang mở terminal trực tiếp trên máy chủ.

### Tại sao phải làm bước này

Toàn bộ phần cài Docker, cài nginx, clone repo, chạy app đều làm bên trong VM.

---

## 8. Bước 4 - Cập nhật hệ điều hành và cài các gói cần thiết

Sau khi SSH vào VM, chạy:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin nginx git
```

### Ý nghĩa của từng lệnh

`sudo apt update`

- cập nhật danh sách package mới nhất từ Ubuntu repository;
- không cài gì cả, chỉ refresh danh sách.

`sudo apt install -y docker.io docker-compose-plugin nginx git`

- cài Docker Engine;
- cài Docker Compose plugin để dùng `docker compose`;
- cài `nginx` để làm web server/reverse proxy;
- cài `git` để clone repo.

### Bật Docker và nginx tự khởi động cùng máy

Chạy tiếp:

```bash
sudo systemctl enable --now docker
sudo systemctl enable --now nginx
```

### Ý nghĩa

- `enable`: bật service để máy reboot xong nó tự chạy lại;
- `--now`: bật luôn ngay bây giờ, không cần restart máy.

### Cho user hiện tại dùng Docker mà không cần `sudo`

```bash
sudo usermod -aG docker $USER
```

Sau đó thoát SSH rồi SSH lại:

```bash
exit
ssh <your-username>@34.123.45.67
```

### Tại sao phải SSH lại

Vì group `docker` chỉ có hiệu lực ở phiên đăng nhập mới.

### Kiểm tra Docker đã chạy chưa

```bash
docker --version
docker compose version
docker ps
```

Nếu `docker ps` không báo lỗi permission thì ổn.

---

## 9. Bước 5 - Tạo thư mục chứa source code

Sau khi SSH lại vào VM, chạy:

```bash
sudo mkdir -p /opt
sudo chown $USER:$USER /opt
cd /opt
```

### Ý nghĩa

- `mkdir -p /opt`: tạo thư mục `/opt` nếu chưa có;
- `chown $USER:$USER /opt`: giao quyền thư mục này cho user hiện tại;
- `cd /opt`: chuyển vào nơi sẽ chứa source code.

### Tại sao lại dùng `/opt`

Vì đây là chỗ khá phổ biến để đặt ứng dụng tự quản lý trên Linux.

---

## 10. Bước 6 - Clone repo

Chạy:

```bash
git clone <YOUR_REPO_URL> easyhr
cd /opt/easyhr
```

Ví dụ:

```bash
git clone https://github.com/your-org/Recruitment_AI_assistant.git easyhr
cd /opt/easyhr
```

### Ý nghĩa

- `git clone ... easyhr`: tải source code về VM;
- `cd /opt/easyhr`: vào thư mục project.

### Kiểm tra nhanh

```bash
ls
```

Bạn nên thấy các thư mục như:

- `backend`
- `frontend`
- `docker`
- `docs`
- `docker-compose.yml`

---

## 11. Bước 7 - Tạo file `.env`

Guide này giả định app đọc env từ file `.env` ở root repo.

Nếu repo đã có `.env.example`, bạn có thể copy:

```bash
cp .env.example .env
```

Nếu không có hoặc không đủ, hãy tạo file mới:

```bash
nano .env
```

### Nội dung tối thiểu nên có

Bạn có thể bắt đầu với cấu hình kiểu:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=recruitment_db

SECRET_KEY=replace-this-with-a-long-random-secret

LLM_PROVIDER=shopaikey
SHOPAIKEY_API_KEY=your_shopaikey_api_key_here
SHOPAIKEY_MODEL_NAME=llama-3.1-8b
SHOPAIKEY_BASE_URL=https://api.shopaikey.com/v1

MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_SECURE=false
MINIO_REGION=us-east-1
MINIO_RESUME_BUCKET=resumes
MINIO_PRESIGNED_GET_EXPIRY_SECONDS=3600

APP_UI_LANGUAGE=en
VITE_UI_LANGUAGE=en

FRONTEND_BASE_URL=http://34.123.45.67
BACKEND_CORS_ORIGINS=["http://34.123.45.67"]
GOOGLE_REDIRECT_URI=http://34.123.45.67/api/v1/auth/google/callback

GOOGLE_OAUTH_SCOPES=openid email profile https://www.googleapis.com/auth/gmail.send
GOOGLE_OAUTH_ACCESS_TYPE=offline
GOOGLE_OAUTH_PROMPT=consent
GOOGLE_TOKEN_ENCRYPTION_KEY=
GMAIL_SEND_ENABLED=false
GMAIL_SEND_TIMEOUT_SECONDS=20
```

Lưu file trong `nano`:

- nhấn `Ctrl + O`, Enter để lưu
- nhấn `Ctrl + X` để thoát

### Việc bạn đang làm ở bước này là gì

Bạn đang cấp các biến cấu hình để backend và frontend build đúng môi trường.

### Giải thích các biến quan trọng

`FRONTEND_BASE_URL`

- backend dùng giá trị này khi cần tạo link quay lại frontend;
- hiện tại dùng IP nên đặt là `http://34.123.45.67`.

`BACKEND_CORS_ORIGINS`

- cho phép frontend ở IP này gọi API backend;
- nếu sai, browser có thể chặn request vì CORS.

`GOOGLE_REDIRECT_URI`

- nếu bạn dùng Google OAuth, đây là URL callback mà Google sẽ gọi về;
- lúc dùng IP thì đặt theo IP;
- sau này có domain, giá trị này sẽ phải đổi lại.

`SECRET_KEY`

- backend dùng để ký token và một số dữ liệu nhạy cảm;
- không nên để mặc định kiểu `super-secret-key`.

---

## 12. Bước 8 - Xem nhanh `docker-compose.yml` hiện đang làm gì

Trước khi chạy, bạn nên hiểu repo hiện có các service sau:

- `db`
- `redis`
- `minio`
- `backend`
- `worker`
- `frontend`

Trong guide này:

- file `docker-compose.yml` là base cho local/dev;
- file `docker-compose.prod.yml` là override cho production;
- ở production vẫn là service `frontend`, nhưng service này được override sang `target: production`;
- `nginx` ở host chỉ làm reverse proxy vào frontend container ở `127.0.0.1:5173` và backend ở `127.0.0.1:8000`.

### Vì sao tách thành `docker-compose.yml` và `docker-compose.prod.yml`

Vì `frontend` trong compose hiện tại đang chạy kiểu dev:

- dùng Vite
- bind port `5173`
- phù hợp local/dev hơn production

Trong khi đó override production:

- build ở stage production của [frontend/Dockerfile](/C:/Users/Admin/Desktop/Recruitment_AI_assistant/frontend/Dockerfile:18)
- bake `VITE_API_BASE_URL=/api/v1` vào bundle
- tự serve file tĩnh qua `nginx` bên trong container ở port `5173`
- không cần bind-mount source code hay chạy `npm run dev`

---

## 13. Bước 9 - Chạy Docker stack cho production

Chạy lệnh sau trong thư mục repo:

```bash
cd /opt/easyhr
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build db redis minio backend worker frontend
```

### Giải thích đầy đủ

`docker compose up`

- build image nếu cần rồi chạy các service cần cho production.

`-f docker-compose.yml -f docker-compose.prod.yml`

- nạp file base trước, rồi áp override production lên trên.
- vẫn giữ một service tên `frontend`, nhưng runtime production sẽ thay thế runtime dev.
- frontend sẽ được build với `VITE_API_BASE_URL=/api/v1`.
- container sẽ serve app ở port `5173` trên host.

### Kiểm tra frontend production container đã lên chưa

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps frontend
curl http://127.0.0.1:5173
```
- `docker compose ... ps frontend` phải hiện container đang chạy.
- `curl http://127.0.0.1:5173` phải trả ra HTML của frontend app.

### Nếu build lỗi thì làm gì

Xem log lỗi ngay trong terminal. Thường sẽ là:

- thiếu dependency;
- lỗi TypeScript;
- lỗi biến env.

Trong trường hợp đó, sửa lỗi rồi chạy lại đúng lệnh build phía trên.

---

## 14. Bước 10 - Chạy backend stack bằng Docker Compose

Chạy:

```bash
cd /opt/easyhr
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d db redis minio backend worker frontend
```

### Ý nghĩa

`docker compose up`

- khởi động các service theo file `docker-compose.yml`.

`-d`

- chạy nền, không chiếm terminal.

`db redis minio backend worker frontend`

- chỉ chạy các service cần cho production;
- không chạy runtime dev của frontend;
- chạy service `frontend` với override production để container tự serve app ở port `5173`.

### Kiểm tra container đã lên chưa

```bash
docker compose ps
```

Bạn muốn thấy các service ở trạng thái kiểu:

- `Up`
- hoặc `running`

### Xem log backend

```bash
docker compose logs -f backend
```

### Ý nghĩa

- `logs -f` là xem log liên tục;
- rất hữu ích để biết app đang chết vì env, DB, migration hay module lỗi.

Thoát log bằng:

```bash
Ctrl + C
```

### Test backend trực tiếp trên VM

Chạy:

```bash
curl http://127.0.0.1:8000/
```

Nếu ổn, bạn sẽ thấy JSON kiểu:

```json
{"message":"Welcome to Recruitment AI Assistant API"}
```

### Tại sao test bằng `127.0.0.1:8000`

Vì đây là backend đang publish trong chính VM.

Ta test trực tiếp backend trước, để nếu lỗi thì biết lỗi ở backend chứ chưa phải ở `nginx`.

---

## 15. Bước 11 - Viết cấu hình `nginx`

Tạo file config:

```bash
sudo nano /etc/nginx/sites-available/easyhr
```

Dán nội dung sau:

```nginx
server {
    listen 80;
    server_name 34.123.45.67;

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

    location / {
        proxy_pass http://127.0.0.1:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location = /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Lưu ý trước khi lưu file:

- `server_name` có thể là IP hoặc domain, nhưng nếu khai báo nhiều host thì viết cách nhau bằng dấu cách, không dùng dấu phẩy. Ví dụ: `server_name easyhr.site www.easyhr.site;`
- config này không còn dùng `root` của host để serve frontend.
- frontend được serve bởi service `frontend` đang chạy với override production ở `127.0.0.1:5173`.
- tên file `/etc/nginx/sites-available/easyhr` chỉ là tên file config; nó không quyết định app nằm ở đâu.

Lưu file:

- `Ctrl + O`, Enter
- `Ctrl + X`

### Ý nghĩa của config này

`listen 80`

- `nginx` nghe ở cổng HTTP chuẩn.

`server_name 34.123.45.67`

- áp config này khi request đi vào IP đó.
- nếu dùng domain, thay bằng domain thật; nếu có nhiều domain thì viết cách nhau bằng dấu cách.

`location /api/`

- mọi request bắt đầu bằng `/api/` sẽ được chuyển vào backend FastAPI.

`location /`

- mọi request frontend sẽ được chuyển vào service `frontend` ở `127.0.0.1:5173`.
- SPA fallback `/index.html` được xử lý bởi `nginx` bên trong container frontend.

`proxy_set_header ...`

- chuyển tiếp thông tin request thật cho backend;
- đặc biệt hữu ích khi sau này cần log IP hoặc biết request ban đầu là HTTP/HTTPS.

---

## 16. Bước 12 - Enable site `nginx`

Chạy các lệnh sau:

```bash
sudo ln -s /etc/nginx/sites-available/easyhr /etc/nginx/sites-enabled/easyhr
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

### Ý nghĩa của từng lệnh

`sudo ln -s ...sites-available... ...sites-enabled...`

- bật site vừa tạo.

`sudo rm -f /etc/nginx/sites-enabled/default`

- bỏ config mặc định của `nginx`;
- tránh xung đột với site mới.

`sudo nginx -t`

- kiểm tra config `nginx` có hợp lệ không trước khi reload.

`sudo systemctl reload nginx`

- nạp lại config mới mà không cần reboot máy.

### Nếu `nginx -t` báo lỗi

Đừng reload vội. Hãy:

```bash
sudo nano /etc/nginx/sites-available/easyhr
```

sửa lỗi rồi chạy lại:

```bash
sudo nginx -t
```

---

## 17. Bước 13 - Test app qua IP

Từ máy của bạn, mở trình duyệt:

```text
http://34.123.45.67
```

Bạn cũng nên test thêm:

```text
http://34.123.45.67/docs
http://34.123.45.67/openapi.json
```

### Mỗi URL dùng để kiểm tra gì

`http://34.123.45.67`

- kiểm tra frontend đã được `nginx` serve chưa.

`http://34.123.45.67/docs`

- kiểm tra `nginx` đã reverse proxy vào backend chưa.

`http://34.123.45.67/openapi.json`

- kiểm tra backend API response qua proxy có ổn không.

### Test bằng terminal cũng được

Từ máy local hoặc chính VM:

```bash
curl http://34.123.45.67
curl http://34.123.45.67/docs
curl http://34.123.45.67/openapi.json
```

---

## 18. Bước 14 - Kiểm tra frontend có gọi được API chưa

Mở Developer Tools trong browser:

- Chrome: `F12`
- tab `Network`

Sau đó thao tác vài chức năng trong app và xem request có đi tới:

- `/api/v1/...`

hay không.

### Điều bạn muốn thấy

- request đi đến `http://34.123.45.67/api/v1/...`
- không phải `http://localhost:8000/...`
- không phải `http://127.0.0.1:8000/...`

### Tại sao điều này quan trọng

Vì nếu frontend build sai `VITE_API_BASE_URL`, browser của người dùng sẽ cố gọi API về `localhost`, mà `localhost` của họ không phải server của bạn.

---

## 19. Bước 15 - Những lệnh kiểm tra quan trọng khi có lỗi

Nếu app không lên, chạy lần lượt:

### Kiểm tra container

```bash
cd /opt/easyhr
docker compose ps
```

### Xem log backend

```bash
cd /opt/easyhr
docker compose logs --tail=200 backend
```

### Xem log worker

```bash
cd /opt/easyhr
docker compose logs --tail=200 worker
```

### Xem log frontend production

```bash
cd /opt/easyhr
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail=200 frontend
```

### Xem trạng thái nginx

```bash
sudo systemctl status nginx
```

### Xem log nginx

```bash
sudo tail -n 200 /var/log/nginx/error.log
sudo tail -n 200 /var/log/nginx/access.log
```

### Kiểm tra backend trực tiếp

```bash
curl http://127.0.0.1:8000/
```

### Kiểm tra frontend production trực tiếp

```bash
curl http://127.0.0.1:5173
```

### Ý nghĩa

Đây là bộ lệnh tách lỗi theo từng lớp:

- Docker có chạy không
- backend có sống không
- `nginx` có đọc đúng config không
- frontend production container có đang serve app không

---

## 20. Bước 16 - Khi bạn sửa code thì update thế nào

### Trường hợp A - Bạn sửa frontend

Chạy:

```bash
cd /opt/easyhr
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build frontend
```

### Ý nghĩa

`git pull`

- lấy code mới nhất về VM.

`docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build frontend`

- build lại image production của frontend;
- chạy lại container `frontend` với runtime production và bundle mới;
- không cần build tay `dist` trên host nữa.

### Trường hợp B - Bạn sửa backend

Chạy:

```bash
cd /opt/easyhr
git pull
docker compose up -d --build backend worker
```

### Ý nghĩa

- rebuild image backend nếu code/dependency thay đổi;
- chạy lại `backend` và `worker` theo code mới.

### Trường hợp C - Bạn đổi cả frontend lẫn backend

Chạy:

```bash
cd /opt/easyhr
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build frontend backend worker
```

---

## 21. Bước 17 - Những điểm bạn cần nhớ về IP

Ở giai đoạn này, app của bạn đang chạy theo IP:

- `http://34.123.45.67`

Điều đó có nghĩa là:

- chưa có HTTPS chuẩn;
- Google OAuth bằng IP có thể hoạt động hạn chế hoặc không phải cấu hình bạn muốn giữ lâu dài;
- sau này có domain, bạn nên đổi:
  - `FRONTEND_BASE_URL`
  - `BACKEND_CORS_ORIGINS`
  - `GOOGLE_REDIRECT_URI`
  - cấu hình `server_name` trong `nginx`

### Kết luận thực tế

Deploy bằng IP là bước tốt để xác minh:

- VM chạy được không
- Docker stack có ổn không
- `nginx` proxy có đúng không
- frontend build có đúng base URL không

Sau khi các phần này ổn, việc chuyển sang domain + HTTPS sẽ dễ hơn nhiều.

---

## 22. Lỗi hay gặp và cách hiểu nhanh

### Lỗi 1 - Mở IP không thấy gì

Kiểm tra:

```bash
sudo systemctl status nginx
sudo nginx -t
```

Có thể là:

- `nginx` chưa chạy
- config sai
- firewall chưa mở cổng `80`

### Lỗi 2 - Vào frontend được nhưng API hỏng

Kiểm tra:

```bash
curl http://127.0.0.1:8000/
docker compose logs --tail=200 backend
```

Có thể là:

- backend chết;
- env sai;
- `nginx` proxy `/api/` cấu hình chưa đúng.

### Lỗi 3 - Frontend vẫn gọi `localhost:8000`

Nguyên nhân thường là build sai biến:

- `VITE_API_BASE_URL`

Bạn cần build lại bằng đúng lệnh:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build frontend
```

### Lỗi 4 - Mở route như `/dashboard` bị 404

Nguyên nhân:

- frontend đang chạy sai runtime, hoặc image production cũ/chưa được rebuild.

Bạn nên rebuild lại frontend production:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build frontend
```

SPA fallback `index.html` trong kiến trúc này được xử lý bởi `nginx` bên trong production image của frontend.

---

## 23. Tóm tắt ngắn gọn

Flow triển khai trong guide này là:

1. Tạo VM
2. Cài Docker + nginx
3. Clone repo
4. Tạo `.env` dùng IP của VM
5. Chạy Docker stack production bằng Docker Compose
6. Cấu hình `nginx` proxy frontend và API
7. Kiểm tra frontend ở `127.0.0.1:5173` và backend ở `127.0.0.1:8000`
8. Truy cập app qua `http://<VM_IP>`

Đây là cách đơn giản nhất để xác minh toàn bộ hệ thống chạy được trước khi bạn mua domain.

Khi bạn có domain, tôi sẽ bổ sung tiếp phần:

- thay IP bằng domain;
- mở cổng `443`;
- cài `certbot`;
- bật HTTPS;
- kiểm tra lại OAuth redirect URI.
