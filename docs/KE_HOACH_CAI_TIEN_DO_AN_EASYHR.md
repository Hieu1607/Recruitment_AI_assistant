# Ke hoach cai tien do an EasyHR

## 1. Muc tieu cua ban cai tien

Tai lieu nay de xuat ke hoach cai tien bao cao do an tot nghiep EasyHR dua tren ba nguon chinh:

- Khung kien truc do an mau SoICT trong `docs/KHUNG_KIEN_TRUC_DO_AN_MAU_SOICT.md`.
- Noi dung bao cao hien co trong `EasyHR_DATN_LaTeX_Report`.
- Chuc nang thuc te cua repo `Recruitment_AI_assistant`, bao gom frontend, backend, log, test va cac tai lieu ky thuat noi bo.

Muc tieu khong chi la tang so trang, ma la lam bao cao co tinh thuyet phuc hon theo dung tinh than cua mau SoICT: tach bach bai toan, yeu cau, cong nghe, thiet ke, trien khai, danh gia va dong gop ky thuat. Giong van can giu chuyen nghiep, ro rang, phu hop voi sinh vien bao ve do an: tranh quang cao san pham qua da, tranh noi chung chung, moi nhan dinh quan trong nen gan voi chuc nang that, file source, anh man hinh, log hoac test.

Ket qua mong muon sau khi cai tien:

- Bao cao co cau truc gan voi khung 6 chuong cua mau SoICT.
- Chuong 2 co phan tich yeu cau sau hon, co use case ro hon.
- Chuong 4 tro thanh chuong trong tam, co kien truc, module, API, du lieu, giao dien, kiem thu va story su dung san pham that.
- Chuong 5 neu ro dong gop ky thuat, dac biet ve scoring co giai thich, job-first workspace, public apply, interview flow va trace log cho AI.
- Cac noi dung can hinh anh duoc danh dau bang placeholder va ghi chu cu the de bo sung sau.
- Co mot story huong dan su dung di kem anh chup man hinh san pham that, theo tung thao tac nhu tao job, cap nhat CV, tao/chon JD, cham diem, chat, shortlist, outreach va phong van.

## 2. Hien trang bao cao hien co

Bao cao hien tai da co nen tang tot:

- File chinh: `EasyHR_DATN_LaTeX_Report/DoAn.tex`.
- Cac chuong chinh: `Chuong/1_Gioi_thieu.tex` den `Chuong/6_Ket_luan.tex`.
- Phu luc hien co: `Phu_luc_A.tex` cho use case va `Phu_luc_B.tex` cho cai dat/chay thu.
- Danh muc hinh da co cac hinh quy trinh, kien truc, mo hinh du lieu, giao dien ung vien va ket qua cham diem.
- Chuong 2 da co khao sat hien trang, nhom nguoi dung, nhom chuc nang, yeu cau phi chuc nang.
- Chuong 3 da co frontend, backend, PostgreSQL, Redis/Celery, Docker va LLM.
- Chuong 4 da co kien truc tong the, mo hinh du lieu, API, frontend, luong xu ly CV va scoring.
- Chuong 5 da co cac dong gop theo dang "van de - giai phap - dong gop".

Tuy nhien, bao cao van con cac diem nen cai tien:

- Noi dung Chuong 2 chua du sau so voi khung mau SoICT, dac biet thieu use case phan ra theo tung nhom chuc nang va dac ta use case chi tiet trong than chuong.
- Chuong 4 moi dung o muc mo ta he thong, chua co story minh hoa thao tac that tren san pham.
- Anh giao dien trong `EasyHR_DATN_LaTeX_Report/Hinhve` con file dang pending nhu `easyhr_giao_dien_ung_vien_pending.png` va `easyhr_ket_qua_cham_diem_pending.png`.
- Noi dung chua khai thac het chuc nang da co trong repo: job-first workspace, public application link/QR, public apply, voice interview, interview invitation, interview report, Google/Gmail, notifications, rubric-based scoring va trace log.
- Phan kiem thu/danh gia o Chuong 4 con ngan, chua tan dung duoc test suite va log that trong `logs/`.
- Phu luc cai dat con ngan, chua du gia tri de nguoi khac chay lai he thong va chup man hinh.

## 3. Dinh huong cau truc sau cai tien

Khuyen nghi giu nguyen 6 chuong chinh theo mau SoICT, nhung dieu chinh noi dung ben trong nhu sau.

### 3.1. Chuong 1 - Gioi thieu de tai

Giu cau truc hien co, chi bo sung nhe de phan anh dung san pham thuc te:

- `1.1 Dat van de`: neu ro bai toan khong chi la doc CV, ma la lien ket nhieu buoc tu tao job den ung tuyen, sang loc, lien he va phong van.
- `1.2 Muc tieu va pham vi de tai`: cap nhat muc tieu gom job workspace, public apply, scoring, chat, shortlist, outreach va interview.
- `1.3 Dinh huong giai phap`: nhan manh he thong web nhieu lop, job-first, AI co giai thich, co trace log.
- `1.4 Bo cuc do an`: cap nhat neu them story huong dan su dung trong Chuong 4 va phu luc moi.

Muc tang uoc luong toi thieu: 1-2 trang. Khong can gioi han neu bo sung them vi du, bang hoac lap luan co lien quan truc tiep den EasyHR.

### 3.2. Chuong 2 - Khao sat va phan tich yeu cau

Day la chuong can mo rong ro nhat ve phan tich nghiep vu. Nen sua theo khung mau:

- `2.1 Khao sat hien trang`
  - Quy trinh tuyen dung thu cong.
  - Kho khan khi CV, JD, shortlist, email va phong van nam o nhieu noi.
  - So sanh ngan voi cach dung file/folder/bang tinh va cac ATS co san.

- `2.2 Tong quan chuc nang`
  - Bieu do use case tong quat.
  - Use case phan ra theo nhom:
    - Quan ly tai khoan va vai tro.
    - Quan ly job workspace.
    - Quan ly CV va ho so ung vien.
    - Quan ly mo ta cong viec.
    - Cham diem ung vien.
    - AI chat tren tap ung vien.
    - Shortlist va outreach.
    - Public apply.
    - Interview template, invitation, public interview va report.

- `2.3 Dac ta chuc nang`
  - Khong chi tom tat bang, nen chon 6-8 use case trong tam de dac ta theo form chuan:
    - Ma use case.
    - Ten use case.
    - Tac nhan.
    - Tien dieu kien.
    - Hau dieu kien.
    - Luong su kien chinh.
    - Luong thay the/ngoai le.
  - Use case nen dua vao than chuong:
    - UC-01 Tao job tuyen dung.
    - UC-02 Tao/cap nhat link ung tuyen cong khai.
    - UC-03 Ung vien nop CV qua link cong khai.
    - UC-04 Nha tuyen dung tai len va cap nhat CV.
    - UC-05 Tao/cap nhat JD cho job.
    - UC-06 Cham diem ung vien theo JD.
    - UC-07 Hoi dap tren tap ung vien va tao shortlist.
    - UC-08 Gui loi moi phong van va xem bao cao.

- `2.4 Yeu cau phi chuc nang`
  - Bao mat va phan quyen.
  - Hieu nang khi xu ly nhieu CV.
  - Kha nang giai thich ket qua AI.
  - Kha nang truy vet log.
  - Kha nang trien khai bang Docker.
  - Kha nang su dung cua giao dien.

Muc tang uoc luong toi thieu: 6-8 trang. Co the dai hon neu dac ta use case va bang yeu cau giup nguoi doc hieu ro pham vi he thong.

### 3.3. Chuong 3 - Cong nghe su dung

Chuong 3 hien co dung huong, nhung nen viet chi tiet hon theo cau hoi: cong nghe la gi, vi sao chon, ap dung o thanh phan nao.

Bo sung cac muc:

- `3.1 Kien truc ung dung web va REST API`.
- `3.2 Frontend: React, Vite, TypeScript, Playwright`.
- `3.3 Backend: FastAPI, Python, SQLAlchemy/Alembic, Pydantic`.
- `3.4 Co so du lieu va luu tru: PostgreSQL, MinIO`.
- `3.5 Hang doi va tac vu nen: Redis, Celery`.
- `3.6 AI service: LLM provider, prompt, rubric, trace log`.
- `3.7 Trien khai va van hanh local: Docker Compose`.
- `3.8 Kiem thu: pytest, Playwright, frontend build/typecheck`.

Noi dung quan trong can them:

- Giai thich vi sao PostgreSQL phu hop voi du lieu co quan he nhu job, CV, candidate profile, match result, shortlist, interview.
- Giai thich vi sao can MinIO de luu file CV thay vi chi luu path trong database.
- Giai thich Redis/Celery la nen tang de chuyen cac tac vu nang sang xu ly bat dong bo.
- Giai thich LLM chi dong vai tro ho tro, backend van giu quyen chuan hoa rubric, tinh diem tong va pass/fail.

Muc tang uoc luong toi thieu: 3-5 trang. Co the dai hon neu moi cong nghe duoc gan voi thanh phan he thong va ly do lua chon cu the.

### 3.4. Chuong 4 - Thiet ke, trien khai va danh gia he thong

Day nen la chuong dai va quan trong nhat. De xuat cau truc moi:

- `4.1 Thiet ke kien truc`
  - Lua chon kien truc client-server.
  - Kien truc tong the.
  - Kien truc job-first workspace.
  - Phan chia thanh phan frontend, backend, service, database, storage, worker, LLM.

- `4.2 Thiet ke chi tiet`
  - Thiet ke backend theo lop: endpoint, schema, service, model, database.
  - Thiet ke frontend theo route va shared app shell.
  - Thiet ke co so du lieu.
  - Thiet ke API theo nhom nghiep vu.
  - Thiet ke trace log cho cac tac vu AI.

- `4.3 Xay dung ung dung`
  - Luong tao job va job workspace.
  - Luong public application link/QR.
  - Luong nop CV cong khai va upload CV noi bo.
  - Luong xu ly CV.
  - Luong tao/cap nhat JD.
  - Luong cham diem locked-rubric.
  - Luong AI chat va shortlist.
  - Luong outreach va interview.
  - `4.3.x Story huong dan su dung EasyHR tren san pham thuc te`.

- `4.4 Kiem thu va danh gia`
  - Kiem thu backend.
  - Kiem thu frontend E2E.
  - Danh gia log resume parsing.
  - Danh gia log LangGraph/job chat.
  - Danh gia scoring trace.
  - Gioi han cua danh gia.

- `4.5 Trien khai`
  - Moi truong Docker Compose.
  - Bien moi truong quan trong.
  - Dia chi truy cap.
  - Quy trinh build, migration, seed user.

Muc tang uoc luong toi thieu: 12-18 trang. Rieng Chuong 4 khong nen dat tran do dai, vi day la chuong trong tam va can chua tron story huong dan su dung kem anh san pham that.

### 3.5. Chuong 5 - Cac giai phap va dong gop noi bat

Chuong 5 khong nen lap lai Chuong 4. Nen viet thanh cac mini case-study ky thuat:

- `5.1 Mo hinh job-first workspace`
  - Van de: du lieu ung vien/JD/scoring bi lan giua nhieu dot tuyen.
  - Giai phap: moi job la mot khong gian lam viec, resume/JD/scoring/chat duoc scope theo job.
  - Ket qua: de quan ly, de phan quyen, de mo rong public apply.

- `5.2 Public application link va fallback thong tin ung vien`
  - Van de: ung vien can nop CV khong can tai khoan, PDF co the parse thieu ten/email.
  - Giai phap: token public, form ten/email, fallback vao CandidateProfile.
  - Ket qua: ho tro luong ung tuyen thuc te hon.

- `5.3 Cham diem ung vien theo locked rubric`
  - Van de: neu LLM tu cham diem tu do, kho kiem soat va kho giai thich.
  - Giai phap: LLM trich rubric, backend chuan hoa, backend cham measurable criteria, LLM chi cham semantic criteria, backend tinh tong va pass/fail.
  - Ket qua: diem co cau truc, co evidence, de audit hon.

- `5.4 Trace log cho cac tac vu AI`
  - Van de: AI co the loi, rate limit, tra JSON sai hoac ket qua bat thuong.
  - Giai phap: log resume parsing, job chat va scoring theo trace id.
  - Ket qua: co bang chung de debug va danh gia he thong.

- `5.5 Lien ket sau sang loc`
  - Van de: sang loc xong van can shortlist, outreach, interview va report.
  - Giai phap: ket noi cac module nay trong cung he thong.
  - Ket qua: EasyHR co tinh toan dien hon mot cong cu doc CV don le.

Muc tang uoc luong toi thieu: 5-7 trang. Co the dai hon neu moi dong gop duoc viet nhu mot case-study co van de, giai phap, ket qua va gioi han.

### 3.6. Chuong 6 - Ket luan va huong phat trien

Cap nhat lai ket luan theo nhung gi da bo sung:

- Ket qua dat duoc:
  - Nguyen mau web hoat dong.
  - Quan ly job, CV, JD, scoring, chat, shortlist, outreach, interview.
  - Co AI scoring giai thich duoc.
  - Co log va test lam bang chung.

- Han che:
  - Chua co danh gia tren tap du lieu tuyen dung lon va co nhan.
  - Mot so tac vu phu thuoc LLM ben ngoai.
  - Xu ly nen can hoan thien hon cho production.
  - Bao mat/phan quyen can duoc harden neu dung voi du lieu that.

- Huong phat trien:
  - Async processing hoan chinh bang Celery.
  - Real-time progress cho upload/scoring.
  - Dashboard thong ke tuyen dung.
  - Feedback loop de cai thien prompt/scoring.
  - Tich hop lich phong van, email that va ATS ben ngoai.

Muc tang uoc luong toi thieu: 2-3 trang. Khong can keo dai bang noi dung lap lai; chi mo rong khi co ket qua, han che hoac huong phat trien cu the.

## 4. Vi tri phu hop cho story huong dan su dung

Nen dua story huong dan su dung vao Chuong 4, khong dua vao Chuong 2 hay Chuong 5.

Ly do:

- Chuong 2 la phan yeu cau, khong phai noi minh hoa san pham da xay.
- Chuong 3 la cong nghe, khong phu hop voi anh thao tac giao dien.
- Chuong 5 nen tap trung vao dong gop ky thuat, neu chen story dai se lam loang mach lap luan.
- Chuong 4 trong khung SoICT co muc `4.3.3 Minh hoa cac chuc nang chinh`, day la vi tri tu nhien nhat de dat walkthrough co anh chup man hinh.

De xuat cu the:

- Trong Chuong 4 them muc:
  - `4.3.4 Kich ban su dung minh hoa tren he thong EasyHR`
- Dua toan bo story huong dan su dung vao Chuong 4, khong tach sang phu luc chi vi ly do do dai.
- Co the chia `4.3.4` thanh cac muc con `4.3.4.1`, `4.3.4.2`, ... de nguoi doc theo doi duoc tung buoc.
- Moi buoc trong story nen co 4 phan: muc dich thao tac, thao tac tren giao dien, ket qua he thong hien thi, va y nghia ky thuat/nghiep vu cua buoc do.
- Anh chup man hinh nen dat ngay sau doan mo ta buoc tuong ung, khong gom tat ca anh vao cuoi chuong.
- Phu luc chi nen dung cho noi dung bo tro nhu lenh cai dat, bien moi truong, bang use case dai hoac checklist chup anh; khong nen day walkthrough chinh ra phu luc.

Nguyen tac viet chi tiet nhung khong vo nghia:

- Chi tiet duoc giu lai neu no giup nguoi doc hieu duoc can bam vao dau, du lieu nao duoc nhap, man hinh nao hien ket qua gi, API/log/model nao lien quan hoac vi sao buoc do quan trong trong quy trinh tuyen dung.
- Chi tiet nen bo neu chi lap lai caption cua hinh, chi khen giao dien dep, chi dien giai mot nut bam hien nhien ma khong co y nghia nghiep vu, hoac chi them cau chung chung nhu "he thong rat tien loi".
- Moi anh can co doan giai thich rieng, nhung doan giai thich khong nen chi mo ta lai toan bo nhung gi nguoi doc da thay trong anh. Nen tap trung vao thong tin quan trong: trang thai, truong du lieu, ket qua xu ly, loi co the gap, va moi lien he voi module backend/frontend.
- Neu mot buoc co nhieu anh, nen dat theo thu tu thao tac that: truoc khi nhap, sau khi nhap, sau khi he thong xu ly. Khong can cat bo anh chi vi nhieu, mien la moi anh giai thich mot trang thai khac nhau.

## 5. Story huong dan su dung de xuat

Story nen viet theo goc nhin nha tuyen dung, tu luc tao dot tuyen den sau phong van. Ten goi y:

`Kich ban minh hoa: nha tuyen dung tao dot tuyen AI Engineer va sang loc ung vien bang EasyHR`.

Trong Chuong 4, moi buoc story nen viet theo mau sau:

- **Boi canh**: nguoi dung dang o vai tro nao, dang can hoan thanh viec gi trong quy trinh tuyen dung.
- **Thao tac tren giao dien**: vao muc nao trong sidebar/top bar, bam nut nao, nhap truong nao, chon file/ung vien/JD nao.
- **Ket qua hien thi**: man hinh hien bang, badge, modal, toast, trang thai, diem so, danh sach hay report nao.
- **Xu ly ben trong he thong**: route frontend, nhom API, service/model/log lien quan neu can giai thich.
- **Ngoai le can nhac**: truong hop thieu du lieu, file sai dinh dang, link public het hieu luc, scoring loi, LLM rate limit, hoac phien chat khong co ung vien.
- **Hinh minh hoa**: anh chup man hinh duoc dat ngay sau doan giai thich buoc do.

Cach viet nen uu tien cau cu the. Vi du nen viet "Sau khi bam `Upload resumes`, he thong hien modal chon file PDF va danh sach tung file da chon" thay vi "Nguoi dung co the upload CV mot cach de dang". Neu mot buoc khong co lien he voi du lieu, API, log, UI state hoac quyet dinh nghiep vu, buoc do khong nen dua vao story.

### Buoc 1. Dang nhap va vao workspace

Noi dung can viet:

- Nha tuyen dung truy cap frontend tai `http://localhost:5173`.
- Dang nhap bang tai khoan duoc seed hoac tai khoan da tao.
- He thong hien AppShell gom sidebar, top bar, job switcher va cac muc dieu huong.
- Neu dung tai khoan seed, ghi ro day la tai khoan demo, khong phai thong tin production.
- Giai thich ngan ve AppShell: day la khung giao dien dung chung cho cac man hinh quan tri nhu Jobs, Candidates, Scoring, AI Chat, Shortlists, Outreach va Interviews.
- Thanh phan lien quan: route `/login`, route `/dashboard`, auth/session frontend, backend auth endpoint.
- Ngoai le can nhac: dang nhap sai thong tin thi hien loi, token het han thi quay lai login.

Anh can co:

- `Hinh 4.x. Man hinh dang nhap EasyHR`
- `Hinh 4.x. Dashboard sau khi dang nhap`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh man hinh dang nhap EasyHR}}
\caption{Man hinh dang nhap vao he thong EasyHR}
\label{fig:story-login}
\end{figure}
```

Ghi chu can chup sau:

- Chup man hinh login hoac dashboard that.
- Che thong tin nhay cam neu co token/email that.

### Buoc 2. Tao job tuyen dung

Noi dung can viet:

- Tu sidebar chon `Jobs`.
- Bam nut tao job moi.
- Nhap ten vi tri, vi du `AI Engineer`.
- Luu job va dat job nay lam workspace dang lam viec.
- Giai thich y nghia: moi job gom tap CV, JD, scoring, chat va cac buoc sau sang loc rieng.
- Neu man hinh co job switcher, mo ta cach job vua tao xuat hien trong switcher hoac danh sach job.
- Neu job co status, ghi ro trang thai ban dau nen la `active`.
- Thanh phan lien quan: route `/jobs`, `/jobs/new` hoac `/jobs/:jobId/edit`; backend `/api/v1/jobs`; model `Job`.
- Ngoai le can nhac: ten job trong, user chua co quyen tao job, hoac job bi archive thi khong nen dung lam workspace dang hoat dong.

Anh can co:

- `Hinh 4.x. Danh sach job`
- `Hinh 4.x. Form tao hoac cap nhat job`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh thao tac tao job AI Engineer}}
\caption{Nha tuyen dung tao job moi cho vi tri AI Engineer}
\label{fig:story-create-job}
\end{figure}
```

### Buoc 3. Cau hinh link ung tuyen cong khai

Noi dung can viet:

- Trong man hinh job, mo khu vuc application link.
- He thong hien public apply URL va QR code.
- Nha tuyen dung co the copy link, tai QR, bat/tat link hoac cap nhat loi nhan cho ung vien.
- Giai thich y nghia token public: ung vien khong can dang nhap, nhung khong lo job id noi bo.
- Neu co nut rotate link, giai thich tac dung: vo hieu hoa link cu va tao link moi khi link bi lo hoac dot tuyen thay doi.
- Neu co toggle enable/disable, giai thich day la cach khoa tam thoi kenh nop CV public.
- Thanh phan lien quan: application link card trong frontend, backend `/api/v1/jobs/{job_id}/application-link`, public token trong bang `jobs`.
- Ngoai le can nhac: link disabled/rotated thi ung vien mo link cu se gap trang bao loi hoac het hieu luc.

Anh can co:

- `Hinh 4.x. The public application link va QR`
- `Hinh 4.x. Loi nhan ung vien trong cau hinh job`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh public application link va QR trong man hinh job}}
\caption{Cau hinh link ung tuyen cong khai cho job}
\label{fig:story-public-link}
\end{figure}
```

### Buoc 4. Ung vien nop CV qua link public

Noi dung can viet:

- Ung vien mo link `/apply/:token`.
- He thong hien ten job, loi nhan cua nha tuyen dung, form ho ten, email va upload PDF.
- Ung vien chon file PDF va gui.
- He thong dua CV vao dung job, dung ho ten/email ung vien nhap lam fallback neu parse CV thieu thong tin.
- Mo ta ro cac truong bat buoc: ho ten, email, file PDF.
- Neu upload thanh cong, man hinh can hien trang thai da nop ho so thanh cong, khong hien du lieu noi bo cua job hay nha tuyen dung.
- Thanh phan lien quan: route public `/apply/:token`, backend `/api/v1/public/jobs/{token}`, `/api/v1/public/jobs/{token}/resumes`, model `ResumeDocument`, `CandidateProfile`.
- Ngoai le can nhac: token sai, link bi tat, email khong hop le, file khong phai PDF, PDF parse loi.

Anh can co:

- `Hinh 4.x. Trang ung tuyen cong khai`
- `Hinh 4.x. Trang nop CV thanh cong`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh ung vien nop CV tren trang public apply}}
\caption{Ung vien nop CV qua link cong khai cua job}
\label{fig:story-public-apply}
\end{figure}
```

### Buoc 5. Nha tuyen dung upload hoac cap nhat CV trong Candidates

Noi dung can viet:

- Tu sidebar chon `Candidates`.
- Bam `Upload resumes`.
- Chon mot hoac nhieu file PDF.
- Theo doi trang thai upload/parse: queued, processing, processed hoac failed.
- Neu can cap nhat CV, mo chi tiet ung vien hoac thao tac upload lai theo job.
- Neu he thong ho tro batch upload, ghi ro moi file nen co dong trang thai rieng de nha tuyen dung biet file nao thanh cong/file nao loi.
- Neu co candidate detail, mo ta cac truong da parse: ho ten, email, kinh nghiem, ky nang, hoc van, du an, chung chi.
- Neu co PDF viewer, nhan manh viec doi chieu profile da trich xuat voi file goc de kiem tra ket qua AI.
- Thanh phan lien quan: route `/candidates`, `/candidates/:id`; backend `/api/v1/upload`, job-scoped resume APIs; service `resume_service`; logs `logs/resume_parsing`.
- Ngoai le can nhac: PDF rong text thi co the dung vision fallback; LLM rate limit hoac parse failed thi can hien trang thai loi.

Anh can co:

- `Hinh 4.x. Man hinh Candidates`
- `Hinh 4.x. Modal upload resumes`
- `Hinh 4.x. Trang thai parse CV`
- `Hinh 4.x. Chi tiet ung vien sau khi parse`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh modal upload CV va trang thai xu ly}}
\caption{Nha tuyen dung tai len CV va theo doi trang thai xu ly}
\label{fig:story-upload-cv}
\end{figure}
```

### Buoc 6. Tao hoac cap nhat mo ta cong viec

Noi dung can viet:

- Tu sidebar chon `Job Descriptions` hoac vao workspace job.
- Tao JD cho vi tri `AI Engineer`.
- Nhap mo ta cong viec, yeu cau ky nang, kinh nghiem, hoc van va cac tieu chi uu tien.
- Luu JD lam dau vao cho scoring.
- Neu co hidden recruiter criteria, giai thich day la phan tieu chi noi bo co the dung cho scoring nhung khong nhat thiet cong khai cho ung vien.
- Neu JD gan voi job hien tai, ghi ro viec scoring/chat chi nen dung ung vien trong job do.
- Thanh phan lien quan: route `/job-descriptions`, `/job-descriptions/new`, `/job-descriptions/:id/edit`; backend `/api/v1/job-descriptions`; model `JobDescription`.
- Ngoai le can nhac: JD rong, JD qua ngan, hoac job chua co ung vien thi scoring sau do khong co du lieu.

Anh can co:

- `Hinh 4.x. Danh sach job descriptions`
- `Hinh 4.x. Man hinh soan JD`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh man hinh soan mo ta cong viec AI Engineer}}
\caption{Tao mo ta cong viec lam dau vao cho cham diem ung vien}
\label{fig:story-jd}
\end{figure}
```

### Buoc 7. Cham diem ung vien

Noi dung can viet:

- Tu sidebar chon `Scoring`.
- Chon JD cua job hien tai.
- Chon tap ung vien hoac de he thong cham toan bo ung vien trong job.
- Cau hinh threshold, batch size va trong so cac muc nhu skills, experience, education, projects, summary.
- Bam bat dau cham diem.
- He thong tao rubric, chuan hoa tieu chi, cham semantic criteria va luu ket qua.
- Mo ta ro ket qua can hien: tong so ung vien, so ung vien dat nguong, diem trung binh, diem cao nhat, bang xep hang ung vien, diem thanh phan va rationale.
- Giai thich ngan locked-rubric: LLM trich rubric tu JD, backend chuan hoa rubric, backend tinh diem tong va pass/fail de khong phu thuoc hoan toan vao LLM.
- Neu co row expand hoac detail drawer, ghi ro nguoi dung co the xem criterion, weight, score, weighted score va evidence summary.
- Thanh phan lien quan: route `/scoring`, `/scoring/:matchRunId`; backend `/api/v1/score` hoac job-scoped scoring endpoint; service `score_candidate`; model `MatchRun`, `MatchResult`; logs `logs/scoring`.
- Ngoai le can nhac: chua co JD, chua co candidate processed, tong trong so khong hop le, LLM tra JSON sai, rubric extraction failed, run failed.

Anh can co:

- `Hinh 4.x. Man hinh cau hinh scoring`
- `Hinh 4.x. Ket qua scoring`
- `Hinh 4.x. Chi tiet diem thanh phan va rationale`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh ket qua cham diem ung vien theo JD}}
\caption{Ket qua cham diem ung vien voi diem tong, diem thanh phan va ly do danh gia}
\label{fig:story-scoring-result}
\end{figure}
```

### Buoc 8. Hoi dap bang AI chat va tao shortlist

Noi dung can viet:

- Tu sidebar chon `AI Chat`.
- Dat cau hoi nhu: "Ung vien nao phu hop nhat voi vi tri AI Engineer?".
- He thong tra loi dua tren tap ung vien cua job hien tai.
- Nha tuyen dung tao shortlist tu ket qua chat hoac tu ket qua scoring.
- Mo `Shortlists` de xem collection da tao.
- Neu cau hoi co ket qua ung vien, mo ta cach he thong hien so luong ung vien phu hop va card/link den tung ung vien.
- Neu cau hoi khong lien quan tuyen dung, ghi ro he thong nen phan hoi trong pham vi cho phep thay vi tra loi lan man.
- Khi tao shortlist, mo ta ten collection, nguon tao collection, so luong item va cach mo chi tiet collection.
- Thanh phan lien quan: route `/chat`, `/chat/:sessionId`, `/shortlists`, `/shortlists/collections/:id`; backend `/api/v1/chat`, `/api/v1/shortlist`; model `QuerySession`, `QueryTurn`, `ShortlistCollection`, `ShortlistItem`; logs `logs/langgraph`.
- Ngoai le can nhac: job khong co candidate, session het han, chat loi do LLM/provider, shortlist trung ten hoac ung vien da ton tai trong collection.

Anh can co:

- `Hinh 4.x. Man hinh AI Chat`
- `Hinh 4.x. Cau tra loi co danh sach ung vien phu hop`
- `Hinh 4.x. Shortlist collection`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh AI Chat va nut tao shortlist tu ket qua}}
\caption{Hoi dap tren tap ung vien va tao danh sach rut gon}
\label{fig:story-chat-shortlist}
\end{figure}
```

### Buoc 9. Soan outreach va gui loi moi phong van

Noi dung can viet:

- Tu shortlist hoac chi tiet ung vien, chon thao tac tao outreach.
- He thong luu draft email, subject, body va trang thai gui.
- Voi ung vien phu hop, nha tuyen dung tao interview template/invitation.
- He thong sinh link phong van public cho ung vien.
- Neu co Gmail integration, phan biet ro giua draft/mark-as-sent va gui email that; khong nen viet nhu da gui email production neu he thong dang chi luu trang thai.
- Mo ta cac truong outreach quan trong: nguoi nhan, subject, body, content source, sent status, sent at.
- Mo ta invitation quan trong: template, candidate, job, token/link public, trang thai pending/completed/revoked neu co.
- Thanh phan lien quan: route `/outreach`, `/interviews`, `/interviews/templates`, `/interviews/templates/:id`; backend `/api/v1/outreach`, interview template/invitation APIs; model `OutreachMessage`, `InterviewTemplate`, `InterviewInvitation`.
- Ngoai le can nhac: ung vien thieu email, Gmail chua cau hinh, invitation bi revoke, template chua co cau hoi.

Anh can co:

- `Hinh 4.x. Man hinh Outreach`
- `Hinh 4.x. Tao loi moi phong van`
- `Hinh 4.x. Danh sach interview invitations`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh outreach draft va loi moi phong van}}
\caption{Chuan bi noi dung lien he va loi moi phong van cho ung vien}
\label{fig:story-outreach-interview-invite}
\end{figure}
```

### Buoc 10. Ung vien hoan thanh phong van va nha tuyen dung xem report

Noi dung can viet:

- Ung vien mo link phong van public `/interviews/:token`.
- He thong hien cau hoi/phien phong van.
- Ung vien hoan thanh cac cau tra loi.
- Nha tuyen dung mo report trong `Interviews` de xem tom tat, transcript hoac danh gia.
- Neu he thong co voice flow, mo ta trang thai bat dau phien, hien cau hoi, ghi nhan cau tra loi, hoan thanh phien va tao report.
- Neu co TTS/STT hoac transcript, chi viet dung muc do repo co ho tro; neu chua co anh that thi de placeholder va ghi chu can chup sau.
- Mo ta report can co: thong tin ung vien, job, thoi diem hoan thanh, tom tat cau tra loi, nhan xet hoac recommendation neu co.
- Thanh phan lien quan: route public `/interviews/:token`, route report `/interviews/reports/:interviewSessionId`; backend public interview APIs, interview report APIs; model `InterviewSession`, `InterviewReport`.
- Ngoai le can nhac: token het han/bi revoke, ung vien thoat giua chung, report chua tao xong, audio/voice provider loi.

Anh can co:

- `Hinh 4.x. Public interview shell`
- `Hinh 4.x. Bao cao phong van`

Placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: anh bao cao phong van sau khi ung vien hoan thanh}}
\caption{Bao cao phong van duoc tao sau khi ung vien hoan thanh phien phong van}
\label{fig:story-interview-report}
\end{figure}
```

## 6. Danh sach hinh anh can bo sung

Tat ca muc duoi day tam thoi dung placeholder trong spec/LaTeX, sau do thay bang anh chup man hinh that.

| Ma hinh | Noi dung | Vi tri khuyen nghi | File anh de tao sau |
| --- | --- | --- | --- |
| IMG-01 | Login hoac dashboard sau khi dang nhap | Chuong 4.3.4 | `Hinhve/easyhr_story_01_login.png` |
| IMG-02 | Danh sach job va tao job AI Engineer | Chuong 4.3.4 | `Hinhve/easyhr_story_02_create_job.png` |
| IMG-03 | Public application link va QR | Chuong 4.3.4 | `Hinhve/easyhr_story_03_public_link.png` |
| IMG-04 | Trang public apply cua ung vien | Chuong 4.3.4 | `Hinhve/easyhr_story_04_public_apply.png` |
| IMG-05 | Candidates list | Chuong 4.2 hoac 4.3.4 | `Hinhve/easyhr_story_05_candidates.png` |
| IMG-06 | Upload resumes modal | Chuong 4.3.4 | `Hinhve/easyhr_story_06_upload_cv.png` |
| IMG-07 | Candidate detail / parsed profile | Chuong 4.3.4 | `Hinhve/easyhr_story_07_candidate_detail.png` |
| IMG-08 | Create/edit JD | Chuong 4.3.4 | `Hinhve/easyhr_story_08_jd_editor.png` |
| IMG-09 | Scoring setup | Chuong 4.3.4 | `Hinhve/easyhr_story_09_scoring_setup.png` |
| IMG-10 | Scoring results | Chuong 4.3.4 va Chuong 5.3 | `Hinhve/easyhr_story_10_scoring_results.png` |
| IMG-11 | Scoring result detail/rationale | Chuong 5.3 | `Hinhve/easyhr_story_11_scoring_rationale.png` |
| IMG-12 | AI Chat | Chuong 4.3.4 va Chuong 5.5 | `Hinhve/easyhr_story_12_chat.png` |
| IMG-13 | Shortlist collection | Chuong 4.3.4 | `Hinhve/easyhr_story_13_shortlist.png` |
| IMG-14 | Outreach draft | Chuong 4.3.4 | `Hinhve/easyhr_story_14_outreach.png` |
| IMG-15 | Interview template/invitation | Chuong 4.3.4 | `Hinhve/easyhr_story_15_interview_invitation.png` |
| IMG-16 | Public interview | Chuong 4.3.4 | `Hinhve/easyhr_story_16_public_interview.png` |
| IMG-17 | Interview report | Chuong 4.3.4 | `Hinhve/easyhr_story_17_interview_report.png` |
| IMG-18 | Swagger/API docs hoac Docker services | Phu luc B | `Hinhve/easyhr_appendix_api_or_docker.png` |

Quy uoc khi chup anh:

- Chup anh man hinh san pham that, uu tien du lieu demo khong nhay cam.
- Che token public, email ca nhan, API key, UUID neu khong can hien thi.
- Moi anh can co caption noi ro nguoi dung dang lam gi va ket qua nao tren man hinh la quan trong.
- Khong nen dua qua nhieu anh lien tiep khong co doan giai thich; moi anh can co 1 doan 3-5 cau sau hinh.

## 7. Bang chung ky thuat nen dua vao Chuong 4

### 7.1. Source code va route/API

Co the dua vao bang tong hop:

- Frontend route map: `frontend/src/routes/index.ts`.
- Backend router: `backend/src/api/v1/api.py`.
- Data model: `docs/data-model.md`.
- Quickstart va Docker: `QUICKSTART.md`, `docker-compose.yml`.
- Scoring architecture: `docs/SCORING_ARCHITECTURE_AFTER_REFACTOR.md`.
- Feature/test inventory: `docs/FEATURE_TEST_PLAN.md`.

Noi dung nen trinh bay:

- Bang route frontend theo nhom man hinh.
- Bang API backend theo nhom nghiep vu.
- Bang thuc the du lieu chinh.
- Bang service/module chinh cua backend.

### 7.2. Test evidence

Can dua vao Chuong 4.4 bang kiem thu:

- Backend tests:
  - Auth/account.
  - Job/public application link.
  - Public apply.
  - Resume parsing fallback.
  - Score endpoint/service.
  - Job chat.
  - Shortlist.
  - Outreach.
  - Interview template/public/report.
  - Notifications.

- Frontend E2E:
  - Workspace smoke.
  - Public apply.
  - Resume upload batching.
  - Chat sidebars/candidate PDF panel.
  - Shortlist layout.
  - Interview voice MVP.
  - Localization tieng Viet.
  - Notification preferences.

Nen viet theo huong: "Trong pham vi do an, kiem thu duoc dung de chung minh cac luong chinh khong bi loi hoi quy khi he thong co nhieu module lien ket."

### 7.3. Log evidence

Tu log hien co, co the dua vao bang minh hoa sau khi xac minh lai lan cuoi:

- `logs/resume_parsing/index.jsonl`: 85 trace, 66 success, 19 failed, trong do co 61 mode `text` va 4 mode `vision`.
- `logs/langgraph/index.jsonl`: 41 trace job chat, 40 success, 1 error.
- `logs/scoring`: 13 scoring trace files, 511 events; cac event tieu bieu gom `run_started`, `job_description_prepared`, `rubric_extraction_attempt`, `rubric_normalized`, `adaptive_batch_plan_created`, `candidate_scored`, `run_completed`, `run_failed`.

Luu y: cac con so nay la snapshot cua repo tai thoi diem lap ke hoach. Khi viet bao cao chinh thuc, nen chay lai lenh thong ke va cap nhat bang de tranh sai lech.

## 8. Noi dung placeholder can ghi vao spec/LaTeX

Khi chua co anh that, nen dung placeholder ro rang thay vi de file `pending` ma khong giai thich. Mau placeholder:

```latex
\begin{figure}[H]
\centering
\fbox{\parbox[c][0.32\textheight][c]{0.9\textwidth}{\centering Placeholder: [mo ta anh can bo sung]}}
\caption{[Caption du kien]}
\label{fig:[label-du-kien]}
\end{figure}
```

Ngay duoi placeholder, them ghi chu trong comment LaTeX:

```latex
% TODO(Screenshot): Thay placeholder nay bang anh chup man hinh that tu frontend.
% Yeu cau anh: che email/token/API key; chup o viewport desktop; du lieu demo phai nhat quan voi story AI Engineer.
```

Neu muon tranh TODO xuat hien trong ban nop, co the dung comment LaTeX nhu tren vi comment se khong hien trong PDF. Trong file ke hoach thi van nen liet ke day du cac TODO nay.

## 9. Ke hoach thuc hien theo giai doan

### Giai doan 1. Chuan hoa cau truc va dat cho placeholder

Files:

- `EasyHR_DATN_LaTeX_Report/DoAn.tex`
- `EasyHR_DATN_LaTeX_Report/Chuong/4_Ket_qua_thuc_nghiem.tex`

Viec can lam:

- Cap nhat muc luc/bo cuc de `4.3.4 Kich ban su dung minh hoa tren he thong EasyHR` co cac muc con ro rang.
- Doi ten cac anh pending hoac thay bang placeholder co ghi chu.
- Them skeleton day du cho muc `4.3.4 Kich ban su dung minh hoa tren he thong EasyHR`, gom toan bo 10 buoc story.
- Dat placeholder ngay trong Chuong 4 cho tat ca anh can chup, khong day anh walkthrough sang phu luc.
- Neu can tao them phu luc, chi dung cho checklist chup anh, lenh chay he thong hoac bang use case dai; khong dung phu luc de rut gon story chinh.

Tieu chi hoan thanh:

- LaTeX build thanh cong.
- Muc luc hien dung cac muc moi.
- Khong con caption mo ho nhu "pending" trong PDF.
- Chuong 4 da co day du khung story, ke ca khi anh that chua duoc thay vao.

### Giai doan 2. Mo rong Chuong 2

Files:

- `EasyHR_DATN_LaTeX_Report/Chuong/2_Khao_sat.tex`
- Co the them TikZ use case trong `EasyHR_DATN_LaTeX_Report/Tikz/`

Viec can lam:

- Them bang so sanh quy trinh thu cong, ATS thong thuong va EasyHR.
- Them use case tong quat va use case theo nhom.
- Dua 6-8 use case chinh vao than chuong.
- Chuyen cac use case chi tiet con lai sang Phu luc A.

Tieu chi hoan thanh:

- Chuong 2 doc nhu phan phan tich nghiep vu, khong phai danh sach tinh nang.
- Moi use case co tac nhan, dieu kien, luong chinh va ngoai le.

### Giai doan 3. Mo rong Chuong 3

Files:

- `EasyHR_DATN_LaTeX_Report/Chuong/3_Cong_nghe.tex`

Viec can lam:

- Viet lai cac muc cong nghe theo ba cau hoi: la gi, vi sao chon, ap dung o dau.
- Bo sung bang cong nghe day du hon.
- Bo sung vai tro cua TypeScript, SQLAlchemy/Alembic, MinIO, Playwright, pytest.
- Bo sung giai thich LLM va cac gioi han can kiem soat.

Tieu chi hoan thanh:

- Khong bien Chuong 3 thanh tong quan ly thuyet chung chung.
- Moi cong nghe deu gan voi mot thanh phan EasyHR.

### Giai doan 4. Mo rong Chuong 4 va story su dung

Files:

- `EasyHR_DATN_LaTeX_Report/Chuong/4_Ket_qua_thuc_nghiem.tex`
- `EasyHR_DATN_LaTeX_Report/Hinhve/`

Viec can lam:

- Them mo hinh job-first workspace.
- Them bang API va route day du hon.
- Them bang thuc the du lieu chinh.
- Them muc trace log cho AI tasks.
- Them story su dung 10 buoc nhu Muc 5 cua ke hoach nay.
- Dat placeholder cho toan bo anh story.
- Moi buoc story phai neu ro: muc dich, thao tac tren UI, du lieu nhap vao, ket qua hien thi, va thanh phan he thong lien quan.
- Bo cac cau khong tao gia tri nhu "giao dien than thien", "he thong hoat dong hieu qua" neu khong co anh, log, test hoac vi du cu the di kem.

Tieu chi hoan thanh:

- Story cho nguoi doc hinh dung duoc can bam vao dau, man hinh nao hien noi dung gi, ket qua cua moi buoc la gi.
- Chuong 4 co du bang, hinh, screenshot va phan danh gia.
- Toan bo walkthrough nam trong Chuong 4, khong can doc phu luc moi hieu duoc quy trinh san pham.

### Giai doan 5. Mo rong Chuong 5

Files:

- `EasyHR_DATN_LaTeX_Report/Chuong/5_Giai_phap_dong_gop.tex`

Viec can lam:

- Viet lai cac dong gop thanh 5 case-study ky thuat.
- Dua locked-rubric scoring vao dong gop rieng.
- Dua trace log vao dong gop rieng.
- Dua public apply/job-first vao dong gop rieng.
- Them bang "truoc va sau EasyHR".

Tieu chi hoan thanh:

- Chuong 5 khong lap lai thao tac giao dien.
- Moi dong gop co van de, giai phap, ket qua va gioi han.

### Giai doan 6. Mo rong phu luc va hoan thien

Files:

- `EasyHR_DATN_LaTeX_Report/Chuong/Phu_luc_A.tex`
- `EasyHR_DATN_LaTeX_Report/Chuong/Phu_luc_B.tex`

Viec can lam:

- Mo rong Phu luc A voi day du use case.
- Mo rong Phu luc B voi huong dan Docker, migration, seed, env, logs.
- Neu tao Phu luc C, chi dua cac bang bo tro nhu checklist chup anh, danh sach du lieu demo, mapping anh-sang-file, hoac log command tham khao.
- Khong dua story chi tiet vao Phu luc C; story chi tiet phai o Chuong 4.

Tieu chi hoan thanh:

- Nguoi doc co the dua vao phu luc de chay thu va tai hien story.
- Phu luc bo sung kha nang tai hien, nhung khong thay the cho phan minh hoa chinh trong Chuong 4.

### Giai doan 7. Chup anh that va thay placeholder

Viec can lam:

- Chay he thong bang Docker Compose.
- Seed user va chuan bi du lieu demo nhat quan:
  - Job: `AI Engineer`.
  - Mot JD tuong ung.
  - 5-10 CV demo.
  - Mot scoring run.
  - Mot chat session.
  - Mot shortlist.
  - Mot outreach draft.
  - Mot interview invitation va report.
- Chup anh theo danh sach IMG-01 den IMG-18.
- Thay placeholder bang `\includegraphics`.

Tieu chi hoan thanh:

- Khong con placeholder trong PDF ban nop.
- Tat ca anh co caption va duoc tham chieu hop ly trong text.
- Anh khong lo thong tin nhay cam.

### Giai doan 8. Kiem tra tong the

Lenh bien dich khuyen nghi:

```powershell
cd "C:\Users\Admin\Desktop\Recruitment_AI_assistant\EasyHR_DATN_LaTeX_Report"
latexmk -norc -g -pdf -interaction=nonstopmode -halt-on-error -synctex=1 DoAn.tex
```

Lenh tim noi dung tam:

```powershell
rg -n "pending|Placeholder|TODO\\(Screenshot\\)|hinh tam|anh tam" EasyHR_DATN_LaTeX_Report
```

Lenh tim loi LaTeX nghiem trong:

```powershell
rg -n "LaTeX Error|Undefined control sequence|File .* not found|Citation .* undefined|Reference .* undefined" EasyHR_DATN_LaTeX_Report/DoAn.log
```

Tieu chi hoan thanh:

- PDF build thanh cong.
- Muc luc, danh muc hinh va danh muc bang cap nhat dung.
- Khong co hinh pending trong ban nop cuoi.
- Chuong 4 la chuong trong tam.
- Chuong 5 doc nhu phan dong gop ky thuat, khong phai danh sach tinh nang.
- Van phong thong nhat, chuyen nghiep, phu hop voi sinh vien.

## 10. Uu tien thuc hien

Neu thoi gian han che, nen lam theo thu tu sau:

1. Them story huong dan su dung vao Chuong 4 voi placeholder anh.
2. Mo rong Chuong 4 ve architecture, API, data model, testing va log.
3. Mo rong Chuong 5 ve locked-rubric scoring, trace log va job-first workspace.
4. Mo rong Chuong 2 ve use case.
5. Mo rong Phu luc A/B/C.
6. Chup anh that va thay placeholder.
7. Bien dich va doc soat tong the.

Phan story nen lam som vi no quyet dinh du lieu demo, danh sach screenshot va mach minh hoa san pham. Neu de den cuoi moi them story, cac anh chup se de bi roi rac va khong tao thanh mot quy trinh thuyet phuc.
