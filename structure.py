import os

# --- NỘI DUNG CÁC FILE SẼ ĐƯỢC TẠO ---

# Nội dung cho file requirements.txt
requirements_content = """
langchain
langchain-openai
openai
chromadb
pypdf
tiktoken
python-dotenv
"""

# Nội dung cho file .env.example (mẫu)
env_example_content = """
# Đổi tên file này thành .env và điền API key của bạn vào đây
# Ví dụ: OPENAI_API_KEY="sk-..."
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
"""

# Nội dung cho file .gitignore
gitignore_content = """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environment
venv/
*.venv/
env/
ENV/

# Database
chroma_db_phapluat/

# Secrets
.env

# IDE files
.idea/
.vscode/
"""

# Nội dung cho file README.md
readme_content = """
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
    -   Trên Windows: `venv\\Scripts\\activate`
    -   Trên macOS/Linux: `source venv/bin/activate`

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Thiết lập API Key:**
    -   Đổi tên file `.env.example` thành `.env`.
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
"""

# Nội dung cho file app.py (file chính của ứng dụng)
app_content = """
import os
from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Tải các biến môi trường từ file .env
load_dotenv()

# --- CÁC HẰNG SỐ CẤU HÌNH ---
DATA_PATH = "data/"
CHROMA_PATH = "chroma_db_phapluat"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_LLM_MODEL = "gpt-4o"

def create_vector_db():
    """

"""
print("Bắt đầu quá trình chỉ mục hóa dữ liệu...")

# 1. Tải tài liệu
loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
documents = loader.load()
if not documents:
    print("Không tìm thấy file PDF nào trong thư mục 'data'. Vui lòng thêm file và thử lại.")
    return

print(f"Đã tải {len(documents)} trang từ các file PDF.")

# 2. Chia nhỏ văn bản
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Đã chia tài liệu thành {len(chunks)} chunks.")

# 3. Tạo và lưu trữ vào ChromaDB
print("Đang tạo embeddings và lưu vào ChromaDB...")
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)
print(f"Đã lưu thành công {db._collection.count()} chunks vào '{CHROMA_PATH}'.")
print("--- QUÁ TRÌNH CHỈ MỤC HÓA HOÀN TẤT ---")


def get_rag_chain():
"""

"""
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# 1. Tạo Retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# 2. Tạo Prompt Template
PROMPT_TEMPLATE = \"\"\"
Bạn là một trợ lý AI chuyên về pháp luật Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác và khách quan.
Hãy trả lời câu hỏi của người dùng DỰA VÀO VÀ CHỈ DỰA VÀO phần ngữ cảnh được cung cấp dưới đây.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

HƯỚNG DẪN:
- Nếu ngữ cảnh không chứa thông tin để trả lời câu hỏi, hãy nói rõ: "Dựa trên các văn bản được cung cấp, tôi không tìm thấy thông tin để trả lời câu hỏi này."
- Đưa ra câu trả lời trực tiếp, rõ ràng.
- Sau câu trả lời, hãy trích dẫn nguồn tin từ ngữ cảnh (nếu có thể).
- Trả lời bằng tiếng Việt.

CÂU TRẢ LỜI CỦA BẠN:
\"\"\"
prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

# 3. Khởi tạo LLM
llm = ChatOpenAI(model_name=OPENAI_LLM_MODEL, temperature=0.1)

# 4. Tạo chuỗi RAG
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
return rag_chain


if __name__ == '__main__':
# --- CHỨC NĂNG 1: CHỈ MỤC HÓA DỮ LIỆU ---
# Bỏ comment dòng dưới đây để chạy lần đầu hoặc khi có dữ liệu mới.
# Sau khi chạy xong, hãy comment lại để không phải chạy lại mỗi lần truy vấn.

# create_vector_db()

# --- CHỨC NĂNG 2: TRUY VẤN ---
# Đảm bảo bạn đã chạy create_vector_db() ít nhất một lần.
rag_chain = get_rag_chain()

# Đặt câu hỏi của bạn ở đây
question = "Doanh nghiệp tư nhân có tư cách pháp nhân không theo Luật Doanh nghiệp 2020?"

print(f"\\nĐang truy vấn: '{question}'")
response = rag_chain.invoke(question)

print("\\n--- CÂU TRẢ LỜI ---")
print(response)
print("---------------------\\n")

# Ví dụ câu hỏi khác
question_2 = "Công ty TNHH một thành viên do tổ chức làm chủ sở hữu có cơ cấu tổ chức như thế nào?"
print(f"\\nĐang truy vấn: '{question_2}'")
response_2 = rag_chain.invoke(question_2)

print("\\n--- CÂU TRẢ LỜI ---")
print(response_2)
print("---------------------\\n")
"""


# --- SCRIPT CHÍNH ĐỂ TẠO CẤU TRÚC DỰ ÁN ---

def create_project_structure():
    """Tạo các thư mục và file cho dự án."""
    project_files = {
        "data/.placeholder": "",  # Tạo thư mục data
        "requirements.txt": requirements_content,
        ".env.example": env_example_content,
        ".gitignore": gitignore_content,
        "README.md": readme_content,
        "app.py": app_content
    }

    print("Bắt đầu tạo cấu trúc dự án...")

    for file_path, content in project_files.items():
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Đã tạo thư mục: {dir_name}/")

        if not file_path.endswith('.placeholder'):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content.strip())
            print(f"Đã tạo file: {file_path}")

    # Xóa file placeholder nếu có
    if os.path.exists("data/.placeholder"):
        os.remove("data/.placeholder")

    print("\n--- HOÀN TẤT ---")
    print("Cấu trúc dự án đã được tạo thành công.")
    print("\nCác bước tiếp theo:")
    print("1. Cài đặt các thư viện: pip install -r requirements.txt")
    print("2. Đổi tên '.env.example' thành '.env' và điền API key của bạn.")
    print("3. Thêm các file PDF văn bản pháp luật vào thư mục 'data/'.")
    print("4. Chạy 'python app.py' để bắt đầu (nhớ bỏ comment hàm create_vector_db() cho lần chạy đầu tiên).")


if __name__ == "__main__":
    create_project_structure()