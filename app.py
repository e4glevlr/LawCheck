# app.py
import os
import boto3
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# --- PHẦN CẤU HÌNH ---
# Người dùng cần điền các thông tin này

# 1. Đường dẫn đến thư mục chứa các file văn bản đã tải về (.doc, .docx, .pdf, .txt)
DATA_PATH = "data/"

# 2. Tên của mô hình embedding trên Hugging Face Hub
EMBEDDING_MODEL_NAME = "huyydangg/DEk21_hcmute_embedding"

# 3. Thông tin kết nối đến Amazon OpenSearch Serverless
#    Bạn có thể tìm thấy URL này trong trang quản lý Collection trên AWS Console.
OPENSEARCH_URL = "https://12g32kt3h1787gxv7t14.eu-north-1.aoss.amazonaws.com"

# 4. Tên của chỉ mục (index) bạn muốn tạo trong OpenSearch để lưu trữ dữ liệu
INDEX_NAME = "lawcheck"

# 5. Region của AWS nơi bạn tạo OpenSearch Collection
AWS_REGION = "eu-north-1"  # Ví dụ: "ap-southeast-1"


def get_aws_credentials():
    session = boto3.Session()
    credentials = session.get_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        AWS_REGION,
        'aoss',
        session_token=credentials.token
    )


def index_documents_to_opensearch():
    print("--- BẮT ĐẦU QUY TRÌNH CHỈ MỤC HÓA DỮ LIỆU ---")

    print(f"\n[BƯỚC 1/4] Đang tải tài liệu từ thư mục: '{DATA_PATH}'...")

    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.doc",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True,
        use_multithreading=True
    )

    try:
        documents = loader.load()
        if not documents:
            print(f"Lỗi: Không tìm thấy hoặc không thể tải bất kỳ file .doc nào trong '{DATA_PATH}'.")
            print("Vui lòng kiểm tra lại đường dẫn và định dạng file.")
            return
        print(f"-> Đã tải thành công {len(documents)} tài liệu.")
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải tài liệu: {e}")
        print(
            "Gợi ý: Hãy chắc chắn bạn đã cài đặt các thư viện trong `requirements.txt` mới, đặc biệt là `unstructured`.")
        return

    # --- BƯỚC 2: CHIA NHỎ VĂN BẢN (SPLITTING) ---
    print(f"\n[BƯỚC 2/4] Đang chia nhỏ các tài liệu thành các đoạn văn bản (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"-> Đã chia {len(documents)} tài liệu thành {len(chunks)} chunks.")

    # --- BƯỚC 3: KHỞI TẠO MÔ HÌNH EMBEDDING ---
    print(f"\n[BƯỚC 3/4] Đang tải mô hình embedding: '{EMBEDDING_MODEL_NAME}'...")
    print("(Quá trình này có thể mất vài phút cho lần chạy đầu tiên)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("-> Tải mô hình embedding thành công.")

    # --- BƯỚC 4: TẢI DỮ LIỆU LÊN AMAZON OPENSEARCH ---
    print(f"\n[BƯỚC 4/4] Đang tạo embeddings và tải lên OpenSearch index: '{INDEX_NAME}'...")

    try:
        if "your-opensearch-endpoint" in OPENSEARCH_URL or "your-aws-region" in AWS_REGION:
            print("\n!!! LỖI CẤU HÌNH !!!")
            print("Vui lòng cập nhật các biến OPENSEARCH_URL và AWS_REGION trong file app.py.")
            return

        aws_auth = get_aws_credentials()

        docsearch = OpenSearchVectorSearch.from_documents(
            documents=chunks,
            embedding=embeddings,
            opensearch_url=OPENSEARCH_URL,
            http_auth=aws_auth,  # Sử dụng xác thực IAM
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            index_name=INDEX_NAME,
            engine="faiss",
        )
        print("\n--- HOÀN TẤT ---")
        print(f"-> Đã chỉ mục hóa thành công {len(chunks)} chunks vào OpenSearch.")
        print(f"-> Tên Index: {INDEX_NAME}")
        print(f"-> Endpoint: {OPENSEARCH_URL}")

    except Exception as e:
        print(f"\n!!! LỖI KHI KẾT NỐI HOẶC TẢI DỮ LIỆU LÊN OPENSEARCH !!!")
        print(f"Lỗi chi tiết: {e}")
        print("\n**Gợi ý khắc phục:**")
        print("1. Kiểm tra lại thông tin OPENSEARCH_URL và AWS_REGION có chính xác không.")
        print(
            "2. Đảm bảo máy của bạn đã được cấu hình AWS credentials đúng cách (chạy `aws configure` với IAM User đã được cấp quyền trong Data Access Policy).")
        print("3. Kiểm tra cấu hình Network của OpenSearch Collection, đảm bảo nó cho phép truy cập từ IP của bạn.")


if __name__ == '__main__':
    index_documents_to_opensearch()
