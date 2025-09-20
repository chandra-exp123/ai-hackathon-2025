import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Check if the GOOGLE_API_KEY is set
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

def ingest_docs():
    """
    Ingests documents from the 'docs' directory, processes them, and creates a FAISS vector store.
    """
    # Create the 'vectorstore' directory if it doesn't exist
    if not os.path.exists("vectorstore"):
        os.makedirs("vectorstore")

    # Load documents from the 'docs' directory
    documents = []
    for filename in os.listdir("docs"):
        filepath = os.path.join("docs", filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith(".html"):
            loader = UnstructuredHTMLLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(filepath)
            documents.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and FAISS vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Save the vector store
    vectorstore.save_local("vectorstore")

if __name__ == "__main__":
    ingest_docs()