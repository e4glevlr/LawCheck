# convert_to_text.py

import os
from langchain_community.document_loaders import PyPDFLoader
import re

# --- CẤU HÌNH ---
# Thư mục chứa các file PDF nguồn
PDF_SOURCE_DIR = "data/"
# Thư mục để lưu các file .txt kết quả
TEXT_OUTPUT_DIR = "data_text_output/"


def clean_extracted_text(text: str) -> str:
    """
    Hàm này thực hiện một số bước làm sạch cơ bản trên văn bản được trích xuất.
    Bạn có thể tùy chỉnh thêm các quy tắc làm sạch ở đây.
    """
    # Tách văn bản thành các dòng
    lines = text.split('\n')

    cleaned_lines = []
    for line in lines:
        # Loại bỏ khoảng trắng thừa ở đầu và cuối dòng
        stripped_line = line.strip()

        # Bỏ qua các dòng trống
        if not stripped_line:
            continue

        # Bỏ qua các dòng chỉ là số (thường là số trang)
        if stripped_line.isdigit():
            continue

        # Ví dụ: Bỏ qua các dòng header/footer phổ biến (bạn có thể thêm các mẫu khác)
        # if "Luật Doanh nghiệp" in stripped_line or "CỔNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ" in stripped_line:
        #     continue

        cleaned_lines.append(stripped_line)

    # Nối các dòng đã làm sạch lại với nhau
    return "\n".join(cleaned_lines)


def convert_pdfs_to_text():
    """
    Quét qua thư mục nguồn, chuyển đổi từng file PDF thành file TXT
    và lưu vào thư mục đầu ra.
    """
    print("Bắt đầu quá trình chuyển đổi PDF sang Text...")

    # 1. Kiểm tra và tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(TEXT_OUTPUT_DIR):
        os.makedirs(TEXT_OUTPUT_DIR)
        print(f"Đã tạo thư mục đầu ra: '{TEXT_OUTPUT_DIR}'")

    # 2. Lấy danh sách các file PDF trong thư mục nguồn
    try:
        pdf_files = [f for f in os.listdir(PDF_SOURCE_DIR) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        print(f"Lỗi: Thư mục nguồn '{PDF_SOURCE_DIR}' không tồn tại. Vui lòng tạo thư mục và thêm file PDF.")
        return

    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong '{PDF_SOURCE_DIR}'.")
        return

    print(f"Tìm thấy {len(pdf_files)} file PDF để xử lý.")

    success_count = 0
    failure_count = 0

    # 3. Lặp qua từng file PDF và xử lý
    for filename in pdf_files:
        pdf_path = os.path.join(PDF_SOURCE_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(TEXT_OUTPUT_DIR, output_filename)

        print(f"\nĐang xử lý: '{pdf_path}'")

        try:
            # Sử dụng PyPDFLoader để tải và trích xuất văn bản
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            # Nối nội dung của tất cả các trang lại
            full_text = "\n\n".join([page.page_content for page in pages])

            # Làm sạch văn bản
            cleaned_text = clean_extracted_text(full_text)

            # Ghi văn bản đã làm sạch vào file .txt
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"  -> Thành công! Đã lưu vào: '{output_path}'")
            success_count += 1

        except Exception as e:
            print(f"  -> Lỗi! Không thể xử lý file '{filename}'. Lý do: {e}")
            failure_count += 1

    # 4. In thông báo tổng kết
    print("\n--- QUÁ TRÌNH HOÀN TẤT ---")
    print(f"Tổng số file đã xử lý: {len(pdf_files)}")
    print(f"Thành công: {success_count}")
    print(f"Thất bại: {failure_count}")
    print(f"Bạn có thể xem kết quả trong thư mục: '{TEXT_OUTPUT_DIR}'")


if __name__ == "__main__":
    convert_pdfs_to_text()