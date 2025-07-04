# ui_app.py

import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Components (copy từ app.py)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Tải các biến môi trường từ file .env
load_dotenv()

# --- CÁC HẰNG SỐ CẤU HÌNH (copy từ app.py) ---
CHROMA_PATH = "chroma_db_phapluat"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_LLM_MODEL = "gpt-4o"


# --- LOGIC CỐT LÕI CỦA RAG ---

# Sử dụng cache của Streamlit để không phải tải lại mô hình và DB mỗi lần người dùng tương tác
# Hàm này chỉ chạy một lần và kết quả được lưu lại.
@st.cache_resource
def get_rag_chain():
    """
    Tạo và trả về chuỗi RAG. Hàm này được cache để tăng hiệu suất.
    """
    if not os.path.exists(CHROMA_PATH):
        st.error(
            f"Lỗi: Không tìm thấy cơ sở dữ liệu tại '{CHROMA_PATH}'. Vui lòng chạy file `app.py` với chức năng `create_vector_db()` trước.")
        return None

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

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
- Đưa vị trí có thể tìm thấy trong PDF

- Luôn trả lời bằng tiếng Việt.

CÂU TRẢ LỜI CỦA BẠN:
"""
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model_name=OPENAI_LLM_MODEL, temperature=0.1)

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


# --- XÂY DỰNG GIAO DIỆN STREAMLIT ---

# Thiết lập tiêu đề trang
st.set_page_config(page_title="Hỏi-Đáp Pháp Luật", page_icon="⚖️")

st.title("⚖️ Hệ thống Hỏi-Đáp Văn bản Quy phạm Pháp luật")
st.write("Chào mừng bạn! Đặt câu hỏi về pháp luật và AI sẽ tìm câu trả lời từ các văn bản được cung cấp.")

# Lấy chuỗi RAG đã được cache
rag_chain = get_rag_chain()

# Tạo ô nhập câu hỏi
question = st.text_input(
    "**Nhập câu hỏi của bạn vào đây:**",
    placeholder="Ví dụ: Doanh nghiệp tư nhân có tư cách pháp nhân không?",
)

# Khi người dùng nhập câu hỏi và nhấn Enter
if question and rag_chain:
    # Hiển thị spinner trong khi chờ xử lý
    with st.spinner("AI đang tìm kiếm và tổng hợp câu trả lời..."):
        try:
            response = rag_chain.invoke(question)
            # Hiển thị câu trả lời
            st.success("**Câu trả lời từ AI:**")
            st.markdown(response)
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")
elif not rag_chain:
    # Nếu không tải được rag_chain (do DB chưa có)
    st.warning("Vui lòng đảm bảo bạn đã tạo cơ sở dữ liệu vector trước khi sử dụng giao diện này.")