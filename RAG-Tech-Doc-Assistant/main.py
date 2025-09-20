import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# Check if the GOOGLE_API_KEY is set
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Set up the Streamlit page
st.set_page_config(page_title="RAG Technical Documentation Assistant", page_icon=":books:")
st.title("RAG Technical Documentation Assistant")
st.write("Ask a question about your technical documents and get a context-aware answer.")

# Load the vector store
@st.cache_resource
def load_vectorstore():
    """
    Loads the FAISS vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore()

# Set up the LLM and the RAG chain
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

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

# Get user input and display the response
query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": query})
            st.subheader("Answer:")
            st.write(result["result"])
            st.subheader("Sources:")
            for source in result["source_documents"]:
                st.write(f"- {source.metadata['source']}")
    else:
        st.warning("Please enter a question.")
