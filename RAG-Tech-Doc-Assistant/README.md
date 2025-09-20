# RAG-Based Technical Documentation Assistant

This project is a RAG (Retrieval-Augmented Generation) system that helps users find information and generate responses from large technical documentation repositories.

## Setup

1. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

2. **Create a .env file:**

   Create a `.env` file in the root of the project and add your Google API key:

   ```
   GOOGLE_API_KEY="YOUR_API_KEY_HERE"
   ```

3. **Add documents:**

   Place your technical documents (PDF, HTML, or Markdown) in the `docs` directory.

4. **Create the vector store:**

   ```bash
   python ingest.py
   ```

5. **Run the Streamlit application:**

   ```bash
   streamlit run main.py
   ```

## Usage

Once the application is running, you can ask questions about your technical documents in the text input field and get context-aware answers with source attribution.