# HỆ THỐNG TRUY VẤN VĂN BẢN QUY PHẠM PHÁP LUẬT (RAG)

Đây là một dự án mẫu xây dựng hệ thống hỏi-đáp về văn bản pháp luật Việt Nam sử dụng mô hình RAG (Retrieval-Augmented Generation) với LangChain, ChromaDB và OpenAI GPT API.

## Cài đặt

1.  **Clone repository (hoặc tạo các file từ script):**
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2.  **Tạo môi trường ảo (khuyến khích):**
    ```bash
    python -m venv venv
    ```
    -   Trên Windows: `venv\Scripts\activate`
    -   Trên macOS/Linux: `source venv/bin/activate`

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Thiết lập API Key:**
    -   Mở file `.env` và dán OpenAI API key của bạn vào.

5.  **Thêm dữ liệu:**
    -   Tạo thư mục `data` (nếu chưa có).
    -   Copy các file văn bản pháp luật (định dạng PDF) vào thư mục `data`.

## Sử dụng

Chạy file `app.py` để thực hiện các chức năng:

1.  **Chỉ mục hóa dữ liệu (Lần đầu tiên hoặc khi có dữ liệu mới):**
    -   Trong file `app.py`, đảm bảo dòng `create_vector_db()` được bỏ comment.
    -   Chạy script: `python app.py`
    -   Quá trình này sẽ đọc các file PDF, chia nhỏ, tạo embedding và lưu vào ChromaDB trong thư mục `chroma_db_phapluat`.
    -   Sau khi chạy xong, bạn có thể comment lại dòng `create_vector_db()` để không phải chạy lại.

2.  **Hỏi-đáp và truy vấn:**
    -   Trong file `app.py`, đảm bảo phần "TRUY VẤN" được bỏ comment.
    -   Thay đổi câu hỏi trong biến `question` thành câu hỏi bạn muốn hỏi.
    -   Chạy script: `python app.py`
    -   Hệ thống sẽ trả về câu trả lời dựa trên dữ liệu đã được chỉ mục.