Tôi muốn xây dựng 1 trợ lý AI agent tuyển dụng dưới hình thức 1 website. 
Trợ lý có khả năng nhận các file resume pdf đầu vào, lọc dữ liệu, thông qua LLM tạo thành form dữ liệu để import vào database. 

Các trường dữ liệu ít nhất cần :
[
  {{
    "name": "họ và tên đầy đủ",
    "phone": "số điện thoại",
    "email": "địa chỉ email",
    "location": "địa điểm hiện tại hoặc null, chuẩn hóa dưới dạng tên tỉnh, tên tỉnh + quốc gia hoặc chỉ quốc gia",
    "contact": "thông tin liên hệ khác hoặc null",
    "current_job_title": "chức danh công việc gần nhất hoặc null",
    "educated": true hoặc false,
    "ever studied abroad": true hoặc false,
    "major": "ngành học hoặc null",
    "cpa": "điểm GPA/CPA hoặc null",
    "education": "nội dung học vấn chi tiết hoặc null",
    "experience": "nội dung kinh nghiệm làm việc chi tiết hoặc null",
    "experiment_years": "tổng số năm kinh nghiệm hoặc null",
    "skills": "các kỹ năng hoặc null",
    "languages": "các ngoại ngữ (không phải ngôn ngữ code) hoặc null",
    "projects": "các dự án đã tham gia hoặc null",
    "summary": "tóm tắt bản thân hoặc null",
    "achievements": "thành tích nổi bật hoặc null",
    "publications": "công trình nghiên cứu/bài báo hoặc null",
    "certifications": "chứng chỉ hoặc null",
    "references": "người tham chiếu hoặc null",
    "other": "thông tin khác liên quan hoặc null"
  }}
]

Người dùng sau đó có khả năng nhập job description vào để kiếm tra độ tương thích giữa từng CV và JD .
Người dùng cũng có thể hỏi đáp về các CV đó và yêu cầu trợ lý lọc ra nếu cần. Đây là chức năng quan trọng và phức tạp nhất. Ví dụ:
- Có bao nhiêu người tốt nghiệp Bách Khoa Hà Nội.
- Bao nhiêu người biết Python và trên 3 năm kinh nghiệm.
- Những người quê ở Bắc Ninh tốt nghiệp đại học Sư Phạm và có chứng chỉ SMO gồm những ai ?

Sau khi trả lời các câu hỏi, trợ lý có thể show ra từng CV (1 widget bên phải ô cửa sổ chat chẳng hạn), đồng thời lưu lại kết quả các CV đã được lọc khi người dùng yêu cầu. Trợ lý cũng có thể hỗ trợ gửi email đến cho các ứng viên, mail sinh bởi AI hoặc theo mẫu do người dùng cung cấp.

Khi người dùng yêu cầu, có thể tạo bộ câu hỏi phỏng vấn từ đầu vào là 1 CV và 1 job description được cung cấp.

