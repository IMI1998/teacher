import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="استاد MIT", layout="wide")

st.markdown("""
<style>
    body { direction: rtl; text-align: right; }
    .stChatMessage { direction: rtl; text-align: right; }
    .stMarkdown { direction: rtl; text-align: right; }
    .stMarkdown > div > p { direction: rtl; text-align: right; }
    .stSpinner { direction: rtl; text-align: right; }
    div { direction: rtl; text-align: right; }
    p { direction: rtl; text-align: right; }
    h1, h2, h3, h4, h5, h6 { direction: rtl; text-align: right; }
    li { direction: rtl; text-align: right; }
    code { direction: ltr; text-align: left; unicode-bidi: embed; }  /* برای کدهای انگلیسی LTR نگه داریم */
</style>
""", unsafe_allow_html=True)

st.title("🎓 استاد خصوصی MIT - نسخه پایدار")

with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password")
    uploaded_file = st.file_uploader("کتاب PDF را آپلود کنید", type="pdf")

    if st.button("پاک کردن حافظه گفتگو"):
        st.session_state.histories = {}
        st.rerun()

# ---------------------------------------------------
# PDF → Vectorstore
# ---------------------------------------------------
@st.cache_resource
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.getvalue())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # خرد کردن متن PDF
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # پاکسازی متن‌ها از newline
    for doc in chunks:
        doc.page_content = doc.page_content.replace("\n", " ")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.remove(temp_path)
    return vectorstore

# ---------------------------------------------------
# حافظه گفتگو
# ---------------------------------------------------
if "histories" not in st.session_state:
    st.session_state.histories = {}

def get_history(session_id):
    if session_id not in st.session_state.histories:
        st.session_state.histories[session_id] = InMemoryChatMessageHistory()
    return st.session_state.histories[session_id]

# ---------------------------------------------------
# Prompt Template حرفه‌ای
# ---------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a top-tier MIT professor AND an Iranian Konkur (entrance exam) instructor.
Your mission is TEACHING, not just answering and translating.

You must ALWAYS:
1. Detect and analyze examples inside the retrieved PDF context.
2. If the context contains an example, solve it step-by-step like a Konkur teacher.
3. Extract formulas, definitions, and key points.
4. Warn the student about common misconceptions and traps.
5. Produce 1–3 NEW similar practice problems with answers.
6. Use Persian for teaching. Use English only for technical terms.
7. When answering:
   - بخش ۱: خلاصه مفهوم اصلی
   - بخش ۲: تحلیل خط به خط محتوای PDF مربوطه
   - بخش ۳: تحلیل کامل مثال‌های موجود در PDF
   - بخش ۴: مثال‌های جدید مشابه برای تمرین
   - بخش ۵: نکات کنکوری، دام‌ها، روش میان‌بر
   - بخش 6: ارادئه روش های تست زنی سریع روش های روز دنیا و سریع این روش ها میتواند پیشنهاد خودت یا دیگران باشد
8.In-depth and conceptual teaching is very important. Even the smallest concepts should not be left out.
9.Within the lesson, there may be exercises and problems that you must solve for me and explain fully so that I can learn completely.
10.Imagine that I know nothing and I expect you to teach me everything from scratch, completely and comprehensively, by providing test and conceptual tips. Nothing should be left out.
Your teaching style must be:
- precise
- structured
- exam-oriented
- clear and deep

CONTEXT FROM BOOK:
{context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user",
     """
پرسش دانشجو:
{question}
""")
])

# ---------------------------------------------------
# ساخت chain RAG + LLM
# ---------------------------------------------------
def build_chain(vectorstore, api_key):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.3
    )

    chain = (
        {
            "context": lambda x: "\n\n".join(
                 doc.page_content for doc in retriever.invoke(x["question"])
                ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

# ---------------------------------------------------
# اجرای چت
# ---------------------------------------------------
if uploaded_file and api_key:

    vectorstore = process_pdf(uploaded_file)
    chat = build_chain(vectorstore, api_key)

    st.success("کتاب پردازش شد. سوال خود را بپرسید.")

    session_id = "student"
    history = get_history(session_id)

    # نمایش تاریخچه چت
    for msg in history.messages:
        with st.chat_message("assistant" if msg.type == "ai" else "user"):
            st.markdown(msg.content, unsafe_allow_html=True)  # از markdown برای پشتیبانی بهتر RTL استفاده کن

    # دریافت سوال جدید
    if question := st.chat_input("سوال خود را بپرسید..."):
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("در حال فکر کردن..."):
                result = chat.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": session_id}}
                )
                st.markdown(result.content, unsafe_allow_html=True)  # از markdown برای RTL بهتر

else:

    st.info("لطفاً API Key و PDF را وارد کنید.")

