# Thesis Structure Refresh Design

## Goal

Làm mới cấu trúc báo cáo LaTeX trong `EasyHR_DATN_LaTeX_Report` để khung chương mục bám sát một đồ án tốt nghiệp chuẩn hơn, nhưng vẫn giữ nguyên đề tài EasyHR và văn phong gần với cách sinh viên trình bày.

## Scope

- Giữ nguyên bộ khung cấp chương hiện có trong file `DoAn.tex`.
- Tăng mức phân cấp bên trong các chương 2, 3, 4, 5, 6 bằng `section`, `subsection` khi cần.
- Điều chỉnh tên một số chương để gần hơn với mẫu tham chiếu.
- Giữ cách diễn đạt tiếng Việt đơn giản, chỉ dùng tiếng Anh cho thuật ngữ chuyên ngành thật sự cần thiết và có giải thích ngắn khi phù hợp.
- Không thay đổi đề tài, không viết lại toàn bộ báo cáo theo nội dung của file PDF tham chiếu.

## Reference Structure

Khung mục lục tham chiếu cho lần chỉnh này:

- Chương 2 theo nhịp: khảo sát hiện trạng, tổng quan chức năng, đặc tả chức năng, yêu cầu phi chức năng.
- Chương 3 theo nhịp: kiến trúc web, công nghệ frontend, công nghệ backend, dữ liệu và triển khai, AI.
- Chương 4 theo nhịp: thiết kế kiến trúc, thiết kế chi tiết, xây dựng hệ thống, kiểm thử, triển khai.
- Chương 5 theo nhịp: mỗi đóng góp lớn nên có mạch vấn đề, giải pháp, đóng góp hoặc ý nghĩa thực tế.
- Chương 6 theo nhịp: kết luận, hạn chế, hướng phát triển.

## Content Style

- Ưu tiên câu ngắn, rõ, có tính báo cáo sinh viên.
- Tránh lạm dụng từ tiếng Anh trong câu văn.
- Không dùng khái niệm quá học thuật nếu có thể thay bằng cách nói trực tiếp hơn.
- Khi nhắc thuật ngữ chuyên ngành như API, REST, LLM, client-server, có thể mở ngoặc giải thích ngắn ở lần xuất hiện đầu.

## Files To Change

- `EasyHR_DATN_LaTeX_Report/DoAn.tex`
- `EasyHR_DATN_LaTeX_Report/Chuong/2_Khao_sat.tex`
- `EasyHR_DATN_LaTeX_Report/Chuong/3_Cong_nghe.tex`
- `EasyHR_DATN_LaTeX_Report/Chuong/4_Ket_qua_thuc_nghiem.tex`
- `EasyHR_DATN_LaTeX_Report/Chuong/5_Giai_phap_dong_gop.tex`
- `EasyHR_DATN_LaTeX_Report/Chuong/6_Ket_luan.tex`

## Verification

- Biên dịch `EasyHR_DATN_LaTeX_Report/DoAn.tex`.
- Kiểm tra lỗi build, lỗi mục lục, lỗi môi trường bảng/hình và cảnh báo cấu trúc rõ ràng.
