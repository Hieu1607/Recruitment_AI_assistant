# EasyHR Thesis Detail Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mở rộng báo cáo đồ án EasyHR từ khoảng 38 trang lên hơn 60 trang bằng cách bổ sung bảng biểu, hình vẽ, ảnh giao diện và phần đánh giá có căn cứ từ hệ thống hiện tại.

**Architecture:** Giữ nguyên khung báo cáo LaTeX hiện có, mở rộng từng chương theo từng gói nhỏ để dễ biên dịch và kiểm tra. Mỗi gói tập trung vào một nhóm nội dung riêng, ưu tiên bảng và hình trước, sau đó mới viết thêm nội dung giải thích.

**Tech Stack:** LaTeX `report`, `subfiles`, `graphicx`, `tikz`, `biblatex`, Python script tạo hình trong `EasyHR_DATN_LaTeX_Report/generate_thesis_figures.py`, TeX Live `latexmk`.

---

## Nguyên Tắc Viết Nội Dung

- Giữ giọng văn gần với các file đồ án hiện tại: rõ ràng, trực tiếp, phù hợp với sinh viên.
- Không dùng quá nhiều từ tiếng Anh trong cùng một đoạn.
- Khi bắt buộc dùng thuật ngữ tiếng Anh, mở ngoặc giải thích ngắn ở lần xuất hiện đầu. Ví dụ: API (giao diện lập trình ứng dụng), frontend (phần giao diện), backend (phần xử lý phía máy chủ), LLM (mô hình ngôn ngữ lớn).
- Không thêm nội dung chung chung chỉ để tăng số trang. Mỗi bảng, hình hoặc đoạn mới cần gắn với một chức năng thật của EasyHR.
- Sau mỗi gói phải biên dịch `DoAn.tex` và xem lại PDF trước khi làm gói tiếp theo.

## Hiện Trạng Tài Liệu

- File chính: `EasyHR_DATN_LaTeX_Report/DoAn.tex`.
- PDF hiện tại: `EasyHR_DATN_LaTeX_Report/DoAn.pdf`.
- Nội dung chính hiện có khoảng 6 chương và 2 phụ lục.
- Danh mục hiện có 6 hình trong chương 4 và 2 bảng trong thân đồ án.
- Phụ lục hiện có 4 bảng use case.
- Chương 2, 3 và 4 còn nhiều chỗ có thể làm chi tiết hơn bằng bảng, sơ đồ và ảnh giao diện thật.
- Chương 4 đang có câu nói một số hình là hình tạm. Khi thay ảnh thật, cần bỏ hoặc sửa câu này.

## Dữ Liệu Và Nguồn Tham Khảo Nội Bộ

- Cấu trúc backend API: `backend/src/api/v1/api.py`.
- Các endpoint chính: `backend/src/api/v1/endpoints/`.
- Các model dữ liệu: `backend/src/models/`.
- Các schema trao đổi dữ liệu: `backend/src/schemas/`.
- Các route frontend: `frontend/src/routes/index.ts`.
- Hướng dẫn chạy hệ thống: `QUICKSTART.md`.
- Cấu hình triển khai local: `docker-compose.yml`.
- Log phân tích CV: `logs/resume_parsing/`.
- Log hỏi đáp ứng viên: `logs/langgraph/`.
- Log chấm điểm: `logs/scoring/`.
- Hình hiện có: `EasyHR_DATN_LaTeX_Report/Hinhve/`.
- Hình TikZ hiện có: `EasyHR_DATN_LaTeX_Report/Tikz/`.

## Lệnh Biên Dịch Chuẩn

Chạy từ thư mục `EasyHR_DATN_LaTeX_Report`:

```powershell
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error -synctex=1 DoAn.tex
```

Kết quả mong muốn:

- Lệnh trả về exit code `0`.
- File `DoAn.pdf` được cập nhật.
- Không có lỗi LaTeX làm dừng build.
- Các cảnh báo nhỏ như `Overfull \hbox` hoặc `fancyhdr \headheight` có thể xử lý dần sau, nhưng không được bỏ qua lỗi hình, bảng, label hoặc file thiếu.

---

### Task 1: Gói 0 - Chuẩn Hóa Nền LaTeX Trước Khi Mở Rộng

**Mục tiêu:** Làm sạch phần nền để các gói sau dễ thêm bảng, hình và ảnh giao diện.

**Dự kiến tăng trang:** 0 đến 1 trang.

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/DoAn.tex`
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/4_Ket_qua_thuc_nghiem.tex`

- [ ] Kiểm tra lại các package đang dùng cho bảng và hình: `graphicx`, `array`, `multirow`, `pdflscape`, `caption`, `subcaption`, `tikz`.
- [ ] Giữ nguyên các package đã có nếu không có lỗi build.
- [ ] Sửa câu trong chương 4 nói rằng hình đang là hình tạm sau khi đã có ảnh hoặc sơ đồ thật.
- [ ] Chuẩn hóa cách đặt label cho hình và bảng theo dạng dễ đọc:

```latex
\label{fig:easyhr-architecture}
\label{tab:api-groups}
```

- [ ] Biên dịch `DoAn.tex`.
- [ ] Mở PDF và kiểm tra mục lục, danh mục hình vẽ, danh mục bảng biểu.

---

### Task 2: Gói 1 - Mở Rộng Chương 2 Về Khảo Sát Và Yêu Cầu

**Mục tiêu:** Làm chương 2 rõ hơn về người dùng, chức năng, yêu cầu và phạm vi hệ thống.

**Dự kiến tăng trang:** 4 đến 6 trang.

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/2_Khao_sat.tex`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_quy_trinh_tuyen_dung_hien_tai.tikz`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_quy_trinh_de_xuat.tikz`

- [ ] Thêm bảng phân nhóm người dùng và quyền hạn.

Nội dung bảng nên gồm:

| Nhóm người dùng | Vai trò trong hệ thống | Thao tác chính |
| --- | --- | --- |
| Admin | Quản lý tài khoản và cấu hình | Quản lý người dùng, cấu hình hệ thống |
| Recruiter | Thực hiện nghiệp vụ tuyển dụng | Tải CV, tạo công việc, chấm điểm, shortlist, liên hệ |
| Viewer | Theo dõi dữ liệu | Xem ứng viên, xem kết quả, không sửa dữ liệu quan trọng |

- [ ] Thêm bảng yêu cầu chức năng chi tiết.

Các nhóm chức năng cần có:

- Quản lý công việc tuyển dụng.
- Quản lý mô tả công việc.
- Tải lên và phân tích CV.
- Chấm điểm ứng viên.
- Hỏi đáp trên tập ứng viên.
- Tạo danh sách rút gọn.
- Gửi liên hệ ứng viên.
- Tạo lời mời phỏng vấn.
- Xem báo cáo phỏng vấn.

- [ ] Thêm bảng yêu cầu phi chức năng.

Các nhóm yêu cầu cần có:

- Bảo mật và phân quyền.
- Hiệu năng khi xử lý nhiều CV.
- Khả năng theo dõi trạng thái xử lý.
- Khả năng triển khai bằng Docker.
- Tính dễ dùng của giao diện.
- Khả năng kiểm tra lại kết quả AI.

- [ ] Thêm hình mô tả quy trình tuyển dụng hiện tại.

Gợi ý luồng:

```text
Tạo yêu cầu tuyển dụng -> Nhận CV -> Đọc CV thủ công -> So sánh với JD -> Chọn ứng viên -> Liên hệ -> Phỏng vấn
```

Trong đó JD là job description (mô tả công việc).

- [ ] Thêm hình mô tả quy trình EasyHR đề xuất.

Gợi ý luồng:

```text
Tạo job -> Tải CV -> Phân tích CV -> Chấm điểm -> Hỏi đáp / lọc ứng viên -> Shortlist -> Outreach -> Interview
```

Trong đó outreach là bước liên hệ ứng viên.

- [ ] Viết đoạn nối sau các bảng để giải thích vì sao EasyHR tập trung vào giai đoạn từ nhận CV đến sau sàng lọc.
- [ ] Biên dịch `DoAn.tex`.
- [ ] Kiểm tra chương 2 có bảng và hình không bị tràn khỏi trang.

---

### Task 3: Gói 2 - Mở Rộng Chương 3 Về Công Nghệ Sử Dụng

**Mục tiêu:** Làm rõ vì sao từng công nghệ được chọn và công nghệ đó phục vụ phần nào của EasyHR.

**Dự kiến tăng trang:** 3 đến 5 trang.

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/3_Cong_nghe.tex`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_client_server_request.tikz`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_docker_compose.tikz`

- [ ] Thêm bảng tổng hợp công nghệ.

Các cột nên gồm:

| Nhóm | Công nghệ | Vai trò trong EasyHR |
| --- | --- | --- |
| Giao diện | React, Vite | Hiển thị màn hình tuyển dụng và gọi API |
| Máy chủ | FastAPI, Python | Xử lý nghiệp vụ và kết nối service |
| Dữ liệu | PostgreSQL | Lưu hồ sơ, công việc, điểm số, phỏng vấn |
| Hàng đợi | Redis, Celery | Chuẩn bị cho tác vụ nền |
| Lưu file | MinIO | Lưu CV PDF |
| Triển khai | Docker Compose | Chạy nhiều thành phần cùng lúc |
| AI | LLM provider | Phân tích CV, chấm điểm, hỏi đáp |

- [ ] Thêm bảng so sánh ngắn lý do chọn PostgreSQL thay vì chỉ dùng file hoặc bảng tính.
- [ ] Thêm sơ đồ request (yêu cầu từ giao diện đến máy chủ).

Gợi ý luồng:

```text
Người dùng -> Frontend -> API backend -> Service -> Database / LLM / Storage -> Frontend
```

- [ ] Thêm sơ đồ Docker Compose.

Các thành phần cần có:

- `frontend`.
- `backend`.
- `worker`.
- `db`.
- `redis`.
- `minio`.

- [ ] Bổ sung mục giải thích LLM theo văn phong sinh viên: LLM giúp đọc và sinh văn bản, nhưng kết quả cần lưu kèm lý do để người dùng kiểm tra lại.
- [ ] Bổ sung đoạn giải thích API (giao diện lập trình ứng dụng) và REST (kiểu thiết kế API dùng các phương thức HTTP như GET, POST, PATCH, DELETE).
- [ ] Biên dịch `DoAn.tex`.
- [ ] Kiểm tra chương 3 không biến thành phần lý thuyết rời rạc; mỗi công nghệ phải có liên hệ trực tiếp với EasyHR.

---

### Task 4: Gói 3 - Mở Rộng Chương 4 Về Thiết Kế Kiến Trúc Và Dữ Liệu

**Mục tiêu:** Làm chương 4 trở thành phần chính của báo cáo, có đủ kiến trúc, dữ liệu, API, giao diện và luồng nghiệp vụ.

**Dự kiến tăng trang:** 6 đến 8 trang.

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/4_Ket_qua_thuc_nghiem.tex`
- Modify: `EasyHR_DATN_LaTeX_Report/generate_thesis_figures.py`
- Create or replace images in: `EasyHR_DATN_LaTeX_Report/Hinhve/`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_api_backend_layers.tikz`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_frontend_routes.tikz`

- [ ] Mở rộng phần kiến trúc tổng thể bằng bảng trách nhiệm thành phần.

Các thành phần nên có:

- Frontend.
- Backend API.
- Service xử lý nghiệp vụ.
- PostgreSQL.
- Redis.
- Celery worker.
- MinIO.
- LLM provider.

- [ ] Mở rộng mô hình dữ liệu bằng bảng thực thể chính.

Các thực thể cần nhắc:

- `Job`.
- `ResumeDocument`.
- `CandidateProfile`.
- `JobDescription`.
- `MatchRun`.
- `MatchResult`.
- `QuerySession`.
- `QueryTurn`.
- `ShortlistCollection`.
- `ShortlistItem`.
- `OutreachMessage`.
- `InterviewTemplate`.
- `InterviewInvitation`.
- `InterviewSession`.
- `InterviewReport`.

- [ ] Thêm bảng API đầy đủ hơn.

Các nhóm API cần có:

- `/api/v1/auth`.
- `/api/v1/jobs`.
- `/api/v1/public`.
- `/api/v1/upload`.
- `/api/v1/job-descriptions`.
- `/api/v1/score`.
- `/api/v1/chat`.
- `/api/v1/notifications`.
- `/api/v1/shortlist`.
- `/api/v1/interview-questions`.
- `/api/v1/interview-templates`.
- `/api/v1/interview-reports`.
- `/api/v1/outreach`.
- `/api/v1/outreach-assets`.

- [ ] Thêm sơ đồ tổ chức backend theo lớp.

Gợi ý:

```text
Endpoint -> Schema -> Service -> Model -> Database
```

Schema là lớp mô tả dữ liệu vào ra; model là lớp biểu diễn dữ liệu lưu trong cơ sở dữ liệu.

- [ ] Thêm sơ đồ route frontend.

Các màn hình chính:

- Login.
- Dashboard.
- Jobs.
- Candidates.
- Job descriptions.
- Scoring.
- Chat.
- Shortlists.
- Outreach.
- Interviews.
- Settings.
- Public apply.
- Public interview.

- [ ] Thêm ảnh giao diện thật thay cho ảnh pending nếu hệ thống đang chạy được.

Ảnh nên chụp:

- Danh sách ứng viên.
- Chi tiết ứng viên.
- Kết quả chấm điểm.
- Chat hỏi đáp ứng viên.
- Shortlist hoặc outreach.
- Màn hình phỏng vấn hoặc báo cáo phỏng vấn.

- [ ] Mỗi ảnh giao diện phải có đoạn giải thích ngắn ngay sau hình.
- [ ] Biên dịch `DoAn.tex`.
- [ ] Kiểm tra danh mục hình vẽ và danh mục bảng biểu sau khi thêm nhiều hình/bảng.

---

### Task 5: Gói 4 - Mở Rộng Chương 4 Về Kiểm Thử Và Đánh Giá

**Mục tiêu:** Biến phần kiểm thử và đánh giá thành phần có số liệu, có bảng và có căn cứ từ log thật.

**Dự kiến tăng trang:** 4 đến 6 trang.

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/4_Ket_qua_thuc_nghiem.tex`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_scoring_trace_timeline.tikz`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_test_scope.tikz`

- [ ] Thêm bảng môi trường chạy thử.

Nội dung nên có:

- Docker Compose.
- Backend ở `localhost:8000`.
- Frontend ở `localhost:5173`.
- PostgreSQL ở `localhost:5432`.
- Redis ở `localhost:6379`.
- MinIO ở `localhost:9000` và console ở `localhost:9001`.

- [ ] Thêm bảng nhóm kiểm thử backend.

Dựa trên thư mục `backend/tests`, nhóm kiểm thử nên gồm:

- Xác thực và tài khoản.
- Upload và phân tích CV.
- Chấm điểm ứng viên.
- Chat theo job.
- Shortlist.
- Outreach.
- Public job application.
- Interview template, invitation, public interview và report.
- Notification.

- [ ] Thêm bảng nhóm kiểm thử frontend.

Dựa trên `frontend/tests/e2e`, nhóm kiểm thử nên gồm:

- Workspace smoke test.
- Public apply.
- Chat và candidate PDF panel.
- Shortlist layout.
- Interview voice flow.
- Localization tiếng Việt.
- Notification preference.

- [ ] Thêm bảng kết quả log phân tích CV.

Dữ liệu đã đọc được:

- Resume parsing có nhiều bản ghi thành công theo chế độ `text`.
- Có một số bản ghi `vision`, nghĩa là hệ thống có hướng xử lý CV dạng ảnh hoặc cần nhận diện hình ảnh.
- Có cả bản ghi thất bại, nên có thể dùng để trình bày nhu cầu trace log.

- [ ] Thêm bảng kết quả log chat.

Dữ liệu đã đọc được:

- LangGraph job chat có 40 bản ghi thành công.
- Có 1 bản ghi lỗi.

Giải thích LangGraph là cơ chế điều phối các bước xử lý câu hỏi của người dùng trong chức năng chat.

- [ ] Thêm bảng hoặc timeline cho scoring trace.

Các event tiêu biểu trong một lần chấm điểm:

- `run_started`.
- `job_description_prepared`.
- `rubric_extraction_attempt`.
- `rubric_extraction_completed`.
- `rubric_normalized`.
- `adaptive_batch_plan_created`.
- `candidate_batch_started`.
- `semantic_scoring_started`.
- `candidate_scored`.
- `run_completed`.

- [ ] Viết phần nhận xét sau đánh giá.

Các ý nên có:

- Hệ thống đã có cơ chế ghi log cho các tác vụ dùng AI.
- Log giúp kiểm tra lỗi và giải thích quá trình xử lý.
- Việc đánh giá hiện vẫn ở mức thử nghiệm, chưa phải kiểm thử trên dữ liệu tuyển dụng thật quy mô lớn.

- [ ] Biên dịch `DoAn.tex`.
- [ ] Kiểm tra các bảng dài. Nếu bảng tràn ngang, chuyển sang `pdflscape` hoặc giảm nội dung từng cột.

---

### Task 6: Gói 5 - Mở Rộng Chương 5 Về Đóng Góp Nổi Bật

**Mục tiêu:** Làm rõ phần đóng góp của đồ án bằng các cặp vấn đề, giải pháp và ý nghĩa.

**Dự kiến tăng trang:** 3 đến 4 trang.

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/5_Giai_phap_dong_gop.tex`
- Optional create: `EasyHR_DATN_LaTeX_Report/Tikz/easyhr_dong_gop_tong_quan.tikz`

- [ ] Thêm bảng tổng hợp đóng góp.

Các đóng góp nên có:

- Tổ chức dữ liệu tuyển dụng theo luồng.
- Chấm điểm ứng viên có giải thích.
- Hỏi đáp trên tập ứng viên theo từng công việc.
- Kết nối shortlist, outreach và phỏng vấn.
- Ghi log cho các bước AI để dễ kiểm tra.

- [ ] Với mỗi đóng góp, giữ cấu trúc hiện tại:

```text
Vấn đề đặt ra -> Giải pháp được áp dụng -> Đóng góp đạt được
```

- [ ] Thêm bảng so sánh trước và sau khi có EasyHR.

Ví dụ:

| Vấn đề | Cách làm thủ công | Cách EasyHR hỗ trợ |
| --- | --- | --- |
| Đọc nhiều CV | Mở từng file và ghi chú | Trích xuất thành hồ sơ có cấu trúc |
| So sánh với JD | Tự đọc và tự chấm | Chấm theo tiêu chí và lưu giải thích |
| Tìm ứng viên | Lọc thủ công | Hỏi đáp bằng ngôn ngữ tự nhiên |

- [ ] Thêm sơ đồ nối tiếp sau sàng lọc.

Gợi ý:

```text
Scoring -> Shortlist -> Outreach -> Interview invitation -> Public interview -> Interview report
```

- [ ] Viết rõ giới hạn của đóng góp, tránh mô tả quá mức như một sản phẩm hoàn chỉnh.
- [ ] Biên dịch `DoAn.tex`.
- [ ] Đọc lại chương 5 để đảm bảo văn phong không quá quảng cáo sản phẩm.

---

### Task 7: Gói 6 - Mở Rộng Phụ Lục Và Hoàn Thiện

**Mục tiêu:** Đưa các chi tiết dài ra phụ lục để thân bài dễ đọc nhưng vẫn đầy đủ thông tin.

**Dự kiến tăng trang:** 3 đến 5 trang.

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/Phu_luc_A.tex`
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/Phu_luc_B.tex`
- Optional create: `EasyHR_DATN_LaTeX_Report/Chuong/Phu_luc_C.tex`
- Optional modify: `EasyHR_DATN_LaTeX_Report/DoAn.tex`

- [ ] Mở rộng phụ lục use case.

Thêm các use case:

- Ứng viên nộp CV qua link công khai.
- Nhà tuyển dụng tạo shortlist từ kết quả chat.
- Nhà tuyển dụng tạo nội dung outreach.
- Nhà tuyển dụng gửi lời mời phỏng vấn.
- Ứng viên hoàn thành phỏng vấn công khai.
- Nhà tuyển dụng xem báo cáo phỏng vấn.

- [ ] Mở rộng phụ lục cài đặt và chạy thử.

Nội dung nên có:

- Các biến môi trường quan trọng từ `.env.example`.
- Các địa chỉ truy cập.
- Lệnh chạy migration.
- Lệnh seed tài khoản ban đầu.
- Lệnh xem log backend, worker, frontend.

- [ ] Nếu phụ lục quá dài, tạo `Phu_luc_C.tex` cho API và kiểm thử.
- [ ] Nếu tạo `Phu_luc_C.tex`, thêm chương phụ lục mới vào `DoAn.tex`.
- [ ] Biên dịch `DoAn.tex`.
- [ ] Kiểm tra mục lục phụ lục và số trang cuối.

---

### Task 8: Kiểm Tra Tổng Thể Sau Khi Đạt Hơn 60 Trang

**Mục tiêu:** Đảm bảo báo cáo dài hơn nhưng vẫn đọc được, không chỉ tăng số lượng trang.

**Files:**
- Verify: `EasyHR_DATN_LaTeX_Report/DoAn.tex`
- Verify: `EasyHR_DATN_LaTeX_Report/DoAn.pdf`
- Verify: `EasyHR_DATN_LaTeX_Report/DoAn.log`

- [ ] Chạy biên dịch sạch:

```powershell
cd "C:\Users\Admin\Desktop\Recruitment_AI_assistant\EasyHR_DATN_LaTeX_Report"
latexmk -norc -g -pdf -interaction=nonstopmode -halt-on-error -synctex=1 DoAn.tex
```

- [ ] Kiểm tra PDF có hơn 60 trang.
- [ ] Kiểm tra mục lục có thứ tự hợp lý.
- [ ] Kiểm tra danh mục hình vẽ không còn caption tạm.
- [ ] Kiểm tra danh mục bảng biểu không có bảng bị tràn rõ ràng.
- [ ] Tìm các chữ thể hiện nội dung còn tạm trong báo cáo:

```powershell
rg -n "hình tạm|ảnh tạm|pending" EasyHR_DATN_LaTeX_Report
```

Expected: không còn các từ này trong nội dung xuất bản, trừ khi nằm trong file kế hoạch.

- [ ] Kiểm tra các lỗi LaTeX nghiêm trọng:

```powershell
rg -n "LaTeX Error|Undefined control sequence|File .* not found|Citation .* undefined|Reference .* undefined" EasyHR_DATN_LaTeX_Report/DoAn.log
```

Expected: không có lỗi nghiêm trọng. Nếu còn reference hoặc citation undefined, chạy lại `latexmk` hoặc sửa label/citation.

- [ ] Đọc nhanh các chương 2, 3, 4 để đảm bảo giọng văn thống nhất: dễ hiểu, ít thuật ngữ tiếng Anh, có giải thích khi cần.

---

## Thứ Tự Thực Hiện Khuyến Nghị

1. Làm Task 1 để làm sạch nền.
2. Làm Task 2 cho chương 2 rồi biên dịch và kiểm tra.
3. Làm Task 3 cho chương 3 rồi biên dịch và kiểm tra.
4. Làm Task 4 cho phần thiết kế chương 4 rồi biên dịch và kiểm tra.
5. Làm Task 5 cho kiểm thử và đánh giá rồi biên dịch và kiểm tra.
6. Làm Task 6 cho chương 5 rồi biên dịch và kiểm tra.
7. Làm Task 7 cho phụ lục rồi biên dịch và kiểm tra.
8. Làm Task 8 để kiểm tra tổng thể.

Không nên làm tất cả các task trong một lần vì báo cáo có nhiều bảng và hình. Làm từng gói sẽ dễ phát hiện lỗi bố cục hơn.

## Tiêu Chí Hoàn Thành

- `DoAn.pdf` có hơn 60 trang.
- Chương 2 có thêm bảng yêu cầu và sơ đồ quy trình.
- Chương 3 có bảng công nghệ và sơ đồ triển khai.
- Chương 4 có thêm thiết kế dữ liệu, API, giao diện, kiểm thử và đánh giá log.
- Chương 5 có phần đóng góp rõ ràng hơn.
- Phụ lục có đủ use case và hướng dẫn chạy thử.
- File LaTeX biên dịch thành công bằng TeX Live `latexmk`.
- Văn phong vẫn phù hợp với sinh viên và không lạm dụng thuật ngữ tiếng Anh.
