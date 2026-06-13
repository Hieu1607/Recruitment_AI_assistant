# Thesis Structure Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Làm cho báo cáo LaTeX của EasyHR có khung chương mục giống một đồ án tốt nghiệp chuẩn hơn và vẫn giữ văn phong sinh viên.

**Architecture:** Giữ nguyên file gốc và chỉ chỉnh cấu trúc trình bày trong các chương 2 đến 6. Các thay đổi tập trung vào tên chương, phân cấp mục, câu nối giữa các mục và cách nhóm nội dung để mục lục nhìn chặt chẽ hơn.

**Tech Stack:** LaTeX `report`, `subfiles`, `biblatex`

---

### Task 1: Chốt khung chương và tên chương

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/DoAn.tex`

- [ ] Đổi tên chương 3 và chương 4 theo khung gần với mẫu tham chiếu hơn.
- [ ] Giữ nguyên số lượng chương lớn để không làm vỡ cấu trúc tổng thể của báo cáo.

### Task 2: Tái cấu trúc chương 2 và 3

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/2_Khao_sat.tex`
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/3_Cong_nghe.tex`

- [ ] Tách chương 2 thành các nhóm khảo sát, tổng quan chức năng, đặc tả chức năng, phi chức năng.
- [ ] Tách chương 3 thành kiến trúc web, frontend, backend, dữ liệu, triển khai và AI.
- [ ] Giữ văn phong tiếng Việt đơn giản, tránh nhồi quá nhiều thuật ngữ.

### Task 3: Tái cấu trúc chương 4, 5, 6

**Files:**
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/4_Ket_qua_thuc_nghiem.tex`
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/5_Giai_phap_dong_gop.tex`
- Modify: `EasyHR_DATN_LaTeX_Report/Chuong/6_Ket_luan.tex`

- [ ] Tách chương 4 thành thiết kế kiến trúc, thiết kế chi tiết, xây dựng, kiểm thử, triển khai.
- [ ] Chuẩn hóa chương 5 theo mạch vấn đề, giải pháp, đóng góp.
- [ ] Gộp chương 6 theo mạch kết luận, hạn chế, hướng phát triển.

### Task 4: Biên dịch và kiểm tra

**Files:**
- Verify: `EasyHR_DATN_LaTeX_Report/DoAn.tex`

- [ ] Chạy biên dịch LaTeX cho file chính.
- [ ] Đọc log để xác nhận thay đổi không làm hỏng build.
- [ ] Báo lại phần nào đã đổi và phần nào chưa đụng tới.
