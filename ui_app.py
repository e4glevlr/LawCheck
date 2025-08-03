# ui_app.py

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

# LangChain Components (copy từ app.py)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

# Tải các biến môi trường từ file .env
load_dotenv()

# --- CÁC HẰNG SỐ CẤU HÌNH (copy từ app.py) ---
CHROMA_PATH = "chroma_db_phapluat"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_LLM_MODEL = "gpt-4.1-nano"


# --- LOGIC CỐT LÕI CỦA RAG ---

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

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

    rewrite_question_prompt = ChatPromptTemplate.from_template(
        """
        Dựa vào câu hỏi đầu vào, hãy viết lại nó thành một câu hỏi đầy đủ và rõ ràng hơn để có thể tìm kiếm thông tin trong một cơ sở dữ liệu vector.
        Ví dụ:
        - Đầu vào: "Hợp đồng lao động" -> Đầu ra: "Hợp đồng lao động là gì?"
        - Đầu vào: "tội phạm" -> Đầu ra: "Tội phạm được định nghĩa như thế nào theo pháp luật hình sự?"
        - Đầu vào: "nghĩa vụ của người lao động" -> Đầu ra: "Người lao động có những nghĩa vụ cơ bản nào?"

        Câu hỏi đầu vào: "{question}"
        Câu hỏi được viết lại:
        """
    )

    llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0.1)
    question_rewriter = rewrite_question_prompt | llm | StrOutputParser()

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
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    answer_chain = prompt | llm | StrOutputParser()

    final_rag_chain = RunnablePassthrough.assign(
        # Viết lại câu hỏi
        rewritten_question=RunnableLambda(lambda x: question_rewriter.invoke({"question": x["question"]}))
    ).assign(
        # Dùng câu hỏi đã viết lại để tìm kiếm nguồn (sources)
        sources=RunnableLambda(lambda x: retriever.invoke(x["rewritten_question"]))
    ).assign(
        # Dùng nguồn và câu hỏi gốc để tạo câu trả lời
        answer=lambda x: answer_chain.invoke({"context": format_docs(x["sources"]), "question": x["question"]})
    )
    return final_rag_chain


# --- XÂY DỰNG GIAO DIỆN STREAMLIT ---

# Thiết lập tiêu đề trang
st.set_page_config(page_title="Hỏi-Đáp Pháp Luật", page_icon="⚖️")

st.title("⚖️ Hệ thống Hỏi-Đáp Văn bản Quy phạm Pháp luật")
st.write("Chào mừng bạn! Đặt câu hỏi về pháp luật và AI sẽ tìm câu trả lời từ các văn bản được cung cấp.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Lấy chuỗi RAG đã được cache
rag_chain = get_rag_chain()
if not rag_chain:
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Xem nguồn"):
                for source in message["sources"]:
                    st.info(f"Nguồn: {os.path.basename(source.metadata.get('source', 'N/A'))} - Trang: {source.metadata.get('page', 'N/A')}")

if prompt := st.chat_input("Nhập câu hỏi của bạn vào đây: **"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI đang tổng hợp câu trả lời"):
            # try:
            response = rag_chain.invoke({"question": prompt})
            print(type(response))
            print(response)
            answer = response["answer"]
            sources = response["sources"]
            st.markdown(answer)
            print("sources:", sources)

            no_answer = "Dựa trên các văn bản được cung cấp, tôi không tìm thấy thông tin để trả lời cho câu hỏi này"
            if sources and not answer.strip().startswith(no_answer):
                with st.expander("Xem nguồn"):
                    for source in sources:
                        st.info(f"Nguồn: {os.path.basename(source.metadata.get('source', 'N/A'))} - Trang: {source.metadata.get('page', 'N/A')}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )
            # except Exception as e:
            #     print(e)
