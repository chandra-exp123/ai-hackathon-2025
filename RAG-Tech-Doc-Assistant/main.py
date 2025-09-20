import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ==============================
# 🔑 Load Environment Variables
# ==============================
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# ==============================
# 🎨 Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="RAG Technical Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

# ==============================
# 🎨 Load External CSS
# ==============================
def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ==============================
# 🚀 Page Header
# ==============================
st.markdown('<div class="title">📚 RAG Technical Documentation Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask a question about your technical documents and get an AI-powered, context-aware answer.</div>', unsafe_allow_html=True)

# ==============================
# 📦 Load Vectorstore
# ==============================
@st.cache_resource
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore()

# ==============================
# 🤖 LLM & RAG Chain Setup
# ==============================
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = """
Use the following context to answer the question at the end. 
If you don't know the answer, just say you don't know. 
Do not make up an answer.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# ==============================
# 💬 User Interaction
# ==============================
query = st.text_input("🔍 Enter your question here:", placeholder="e.g., How does authentication work in this system?")

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("Get Answer", use_container_width=True)

# ==============================
# 📖 Response Section
# ==============================
if submit:
    if query:
        with st.spinner("⚡ Generating answer..."):
            result = qa_chain({"query": query})

        st.subheader("✅ Answer")
        st.markdown(f"<div class='answer-box'>{result['result']}</div>", unsafe_allow_html=True)

        st.subheader("📂 Sources")
        for i, source in enumerate(result["source_documents"], start=1):
            st.markdown(f"<div class='source-box'>🔗 {i}. {source.metadata['source']}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a question.")
