Sử dụng PyMuPDF để lọc text từ resumes
Gửi text đến LLM để phân loại và lưu trữ vào bảng dữ liệu. Nếu người dùng gửi kèm job description, gửi batch text các resumes kết hợp với JD đến LLM để chấm điểm và lưu vào database. Có thể cho người dùng chỉnh sửa trọng số chấm điểm cho các mục khác nhau trong CV.
Sử dụng langGraph để xây dựng agent với các tool:
1. Tool search thông qua SQL với các trường dữ liệu đơn giản như tên người, địa chỉ, đã tốt nghiệp chưa
2. Tool search thông qua LLM (gửi câu hỏi và các sections được trích xuất từ resumes như skills, education...) để trả lời các câu hỏi không dễ trả lời thông qua SQL
3. Tool điều phối, nhận câu hỏi từ người dùng và quyết định dùng tool 1 hay 2, dùng bao nhiêu lần theo thứ tự nào, câu hỏi cho mỗi tool sẽ như thế nào.
4. Tool ghi các resumes đã được lọc ra collection riêng.
5. Tool gửi email nếu được yêu cầu thông qua chatbot.
Agent cần có memory, có fallback hợp lý.

Sau khi lọc các resumes ra collection, có thể xây dựng bộ câu hỏi phỏng vấn cho từng ứng viên thông qua LLM.

Giao diện đẹp, chuyên nghiệp.