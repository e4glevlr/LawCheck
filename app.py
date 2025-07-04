# app.py

import os
from dotenv import load_dotenv

# LangChain Components
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Tải các biến môi trường từ file .env (chứa OPENAI_API_KEY)
load_dotenv()

# --- CÁC HẰNG SỐ CẤU HÌNH ---
DATA_PATH = "data/"
CHROMA_PATH = "chroma_db_phapluat"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_LLM_MODEL = "gpt-4o"


def create_vector_db():
    """
    Hàm này thực hiện quá trình Indexing:
    1. Tải các tài liệu PDF từ thư mục DATA_PATH.
    2. Chia nhỏ các tài liệu thành các chunks.
    3. Tạo embeddings cho từng chunk.
    4. Lưu trữ các chunks và embeddings vào ChromaDB.

    Hàm này chỉ cần chạy MỘT LẦN hoặc khi có văn bản pháp luật mới.
    """
    print("Bắt đầu quá trình chỉ mục hóa dữ liệu...")

    # 1. Tải tài liệu
    # Tải tất cả các file PDF trong thư mục và các thư mục con của nó
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        print(f"Lỗi: Không tìm thấy file PDF nào trong thư mục '{DATA_PATH}'.")
        print("Vui lòng thêm ít nhất một file PDF vào thư mục 'data' và thử lại.")
        return

    print(f"Đã tải thành công {len(documents)} trang từ các file PDF.")

    # 2. Chia nhỏ văn bản thành các chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Kích thước mỗi chunk (tính bằng ký tự)
        chunk_overlap=200  # Số ký tự chồng lấp để đảm bảo tính liên tục
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia tài liệu thành {len(chunks)} chunks nhỏ.")

    # 3. Tạo embeddings và lưu trữ vào ChromaDB
    print("Đang tạo embeddings và lưu vào ChromaDB (quá trình này có thể mất vài phút)...")
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    # Tạo và lưu DB vào thư mục CHROMA_PATH
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Đã lưu thành công {db._collection.count()} chunks vào '{CHROMA_PATH}'.")
    print("--- QUÁ TRÌNH CHỈ MỤC HÓA HOÀN TẤT ---")


def get_rag_chain():
    """
    Hàm này tạo ra một chuỗi RAG (Retrieval-Augmented Generation) để xử lý truy vấn.
    Nó sẽ kết nối VectorDB (Retriever) với mô hình ngôn ngữ (LLM).
    """
    # Khởi tạo mô hình embedding (cần thiết để tải DB)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    # Tải VectorDB đã được lưu từ trước
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 1. Tạo Retriever từ VectorDB
    # Retriever có nhiệm vụ tìm kiếm các chunks văn bản liên quan trong DB
    retriever = vector_db.as_retriever(
        search_type="similarity",  # Các loại khác: "mmr", "similarity_score_threshold"
        search_kwargs={"k": 5}  # Lấy về 5 chunks liên quan nhất cho mỗi truy vấn
    )

    # 2. Tạo Prompt Template
    # Đây là "linh hồn" của hệ thống RAG, hướng dẫn LLM cách trả lời.
    PROMPT_TEMPLATE = """
Bạn là một trợ lý AI chuyên về pháp luật Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác và khách quan.
Hãy trả lời câu hỏi của người dùng DỰA VÀO VÀ CHỈ DỰA VÀO phần ngữ cảnh được cung cấp dưới đây.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

HƯỚNG DẪN:
- Nếu ngữ cảnh không chứa thông tin để trả lời câu hỏi, hãy nói rõ ràng: "Dựa trên các văn bản được cung cấp, tôi không tìm thấy thông tin để trả lời cho câu hỏi này."
- Đưa ra câu trả lời trực tiếp, rõ ràng và súc tích.
- Trích dẫn các điều, khoản liên quan nếu có trong ngữ cảnh.
- Luôn trả lời bằng tiếng Việt.

CÂU TRẢ LỜI CỦA BẠN:
"""
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # 3. Khởi tạo mô hình LLM
    # Sử dụng temperature thấp (0.1) để câu trả lời bám sát sự thật, ít sáng tạo
    llm = ChatOpenAI(model_name=OPENAI_LLM_MODEL, temperature=0.1)

    # 4. Tạo chuỗi RAG bằng LangChain Expression Language (LCEL)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


# --- ĐIỂM BẮT ĐẦU CỦA CHƯƠNG TRÌNH ---
if __name__ == '__main__':
    # KIỂM TRA API KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("Lỗi: Không tìm thấy OPENAI_API_KEY. Vui lòng tạo file .env và thêm key của bạn vào.")
    else:
        # --- CHỨC NĂNG 1: CHỈ MỤC HÓA DỮ LIỆU ---
        # Bỏ comment dòng dưới đây để chạy lần đầu hoặc khi có dữ liệu mới.
        # Sau khi chạy xong, hãy comment lại để không phải chạy lại mỗi lần truy vấn.

        # create_vector_db()

        # --- CHỨC NĂNG 2: TRUY VẤN VÀ HỎI ĐÁP ---
        # Đảm bảo bạn đã chạy create_vector_db() ít nhất một lần.
        if not os.path.exists(CHROMA_PATH):
            print(f"Lỗi: Không tìm thấy cơ sở dữ liệu tại '{CHROMA_PATH}'.")
            print("Vui lòng chạy hàm 'create_vector_db()' trước tiên (bỏ comment dòng đó trong code).")
        else:
            rag_chain = get_rag_chain()

            print("\nHệ thống hỏi đáp pháp luật đã sẵn sàng. Nhập 'exit' để thoát.")
            while True:
                question = input("\n[BẠN HỎI]: ")
                if question.lower() == 'exit':
                    break
                if not question.strip():
                    continue

                print("\n[AI ĐANG TRẢ LỜI]...")
                response = rag_chain.invoke(question)
                print(response)